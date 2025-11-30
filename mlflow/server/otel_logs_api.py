"""
OpenTelemetry Logs REST API endpoints for MLflow FastAPI server.

This module implements the OpenTelemetry Protocol (OTLP) REST API for ingesting logs,
primarily designed to receive telemetry from Codex CLI and convert them into MLflow traces.

Codex CLI emits structured log events (e.g., codex.conversation_starts, codex.user_prompt,
codex.tool_result) that are converted into MLflow trace spans for visualization in the
MLflow UI.
"""

import hashlib
import logging
import time
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
    ExportLogsServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.trace import Status as OTelStatus
from opentelemetry.trace import StatusCode as OTelStatusCode

from mlflow.entities.span import Span, SpanType
from mlflow.server.handlers import _get_tracking_store
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import build_otel_context, dump_span_attribute_value
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER, OTLP_LOGS_PATH

_logger = logging.getLogger(__name__)

# Create FastAPI router for OTel logs endpoint
otel_logs_router = APIRouter(prefix=OTLP_LOGS_PATH, tags=["OpenTelemetry Logs"])

# Codex event type constants
CODEX_EVENT_CONVERSATION_STARTS = "codex.conversation_starts"
CODEX_EVENT_API_REQUEST = "codex.api_request"
CODEX_EVENT_SSE_EVENT = "codex.sse_event"
CODEX_EVENT_USER_PROMPT = "codex.user_prompt"
CODEX_EVENT_TOOL_DECISION = "codex.tool_decision"
CODEX_EVENT_TOOL_RESULT = "codex.tool_result"
CODEX_EVENT_SANDBOX_ASSESSMENT = "codex.sandbox_assessment"

# Map Codex event types to MLflow span types
CODEX_EVENT_TO_SPAN_TYPE = {
    CODEX_EVENT_CONVERSATION_STARTS: SpanType.CHAIN,
    CODEX_EVENT_API_REQUEST: SpanType.LLM,
    CODEX_EVENT_SSE_EVENT: SpanType.CHAIN,
    CODEX_EVENT_USER_PROMPT: SpanType.CHAIN,
    CODEX_EVENT_TOOL_DECISION: SpanType.CHAIN,
    CODEX_EVENT_TOOL_RESULT: SpanType.TOOL,
    CODEX_EVENT_SANDBOX_ASSESSMENT: SpanType.CHAIN,
}


def _decode_log_anyvalue(pb_any_value: AnyValue) -> Any:
    """Decode an OTel protobuf AnyValue from log records.

    Args:
        pb_any_value: The OTel protobuf AnyValue message to decode.

    Returns:
        The decoded value.
    """
    value_type = pb_any_value.WhichOneof("value")
    if not value_type:
        return None

    if value_type == "array_value":
        return [_decode_log_anyvalue(v) for v in pb_any_value.array_value.values]
    elif value_type == "kvlist_value":
        return {kv.key: _decode_log_anyvalue(kv.value) for kv in pb_any_value.kvlist_value.values}
    else:
        return getattr(pb_any_value, value_type)


def _generate_trace_id_from_conversation_id(conversation_id: str) -> int:
    """Generate a consistent trace ID from a Codex conversation ID.

    Uses SHA-256 hash to ensure consistent IDs across multiple log batches
    from the same conversation.
    """
    hash_bytes = hashlib.sha256(conversation_id.encode()).digest()[:16]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def _generate_span_id(conversation_id: str, event_name: str, timestamp: int, index: int) -> int:
    """Generate a unique span ID for an event.

    Combines conversation ID, event type, timestamp, and index to ensure uniqueness.
    """
    unique_str = f"{conversation_id}:{event_name}:{timestamp}:{index}"
    hash_bytes = hashlib.sha256(unique_str.encode()).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def _generate_mlflow_trace_id(otel_trace_id: int) -> str:
    """Generate MLflow trace ID from OTel trace ID."""
    return f"tr-{otel_trace_id:032x}"


class CodexLogToSpanConverter:
    """Converts Codex CLI OTLP log records to MLflow spans.

    Codex CLI emits structured log events that represent different phases
    of a conversation. This converter groups events by conversation ID and
    creates a hierarchical trace structure suitable for MLflow visualization.
    """

    def __init__(self):
        # Group events by conversation_id
        self._events_by_conversation: dict[str, list[dict]] = defaultdict(list)

    def add_log_record(
        self,
        attributes: dict[str, Any],
        timestamp_ns: int,
        body: str | None = None,
        resource_attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add a log record to be converted into spans.

        Args:
            attributes: Log record attributes containing event data.
            timestamp_ns: Log record timestamp in nanoseconds.
            body: Optional log record body.
            resource_attributes: Optional resource attributes.
        """
        event_name = attributes.get("event.name")
        conversation_id = attributes.get("conversation.id")

        if not event_name or not conversation_id:
            _logger.debug(
                f"Skipping log record without event.name or conversation.id: {attributes}"
            )
            return

        event_data = {
            "event_name": event_name,
            "timestamp_ns": timestamp_ns,
            "attributes": attributes,
            "body": body,
            "resource_attributes": resource_attributes or {},
        }

        self._events_by_conversation[str(conversation_id)].append(event_data)

    def convert_to_spans(self) -> dict[str, list[Span]]:
        """Convert accumulated log records into MLflow spans.

        Returns:
            Dictionary mapping trace_id to list of spans for that trace.
        """
        spans_by_trace: dict[str, list[Span]] = {}

        for conversation_id, events in self._events_by_conversation.items():
            if not events:
                continue

            # Sort events by timestamp
            events.sort(key=lambda e: e["timestamp_ns"])

            otel_trace_id = _generate_trace_id_from_conversation_id(conversation_id)
            mlflow_trace_id = _generate_mlflow_trace_id(otel_trace_id)

            # Create root span from first conversation_starts event or use first event
            root_event = next(
                (e for e in events if e["event_name"] == CODEX_EVENT_CONVERSATION_STARTS),
                events[0],
            )

            root_span_id = _generate_span_id(conversation_id, "root", root_event["timestamp_ns"], 0)

            # Calculate conversation time bounds
            start_time_ns = min(e["timestamp_ns"] for e in events)
            end_time_ns = max(e["timestamp_ns"] for e in events)

            # Create root conversation span
            root_span = self._create_span(
                name=f"Codex Conversation: {conversation_id[:8]}",
                otel_trace_id=otel_trace_id,
                span_id=root_span_id,
                parent_span_id=None,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                mlflow_trace_id=mlflow_trace_id,
                span_type=SpanType.AGENT,
                attributes=self._extract_conversation_attributes(root_event),
            )

            trace_spans = [root_span]

            # Create child spans for each event
            for idx, event in enumerate(events):
                event_name = event["event_name"]
                child_span_id = _generate_span_id(
                    conversation_id, event_name, event["timestamp_ns"], idx
                )

                # Calculate span duration from duration_ms attribute if available
                duration_ms = event["attributes"].get("duration_ms")
                if duration_ms is not None:
                    try:
                        duration_ns = int(float(str(duration_ms))) * 1_000_000
                        event_end_time_ns = event["timestamp_ns"]
                        event_start_time_ns = event_end_time_ns - duration_ns
                    except (ValueError, TypeError):
                        event_start_time_ns = event["timestamp_ns"]
                        event_end_time_ns = event["timestamp_ns"] + 1_000_000  # 1ms default
                else:
                    event_start_time_ns = event["timestamp_ns"]
                    event_end_time_ns = event["timestamp_ns"] + 1_000_000  # 1ms default

                span_type = CODEX_EVENT_TO_SPAN_TYPE.get(event_name, SpanType.UNKNOWN)
                span_name = self._get_span_name(event)

                child_span = self._create_span(
                    name=span_name,
                    otel_trace_id=otel_trace_id,
                    span_id=child_span_id,
                    parent_span_id=root_span_id,
                    start_time_ns=event_start_time_ns,
                    end_time_ns=event_end_time_ns,
                    mlflow_trace_id=mlflow_trace_id,
                    span_type=span_type,
                    attributes=self._extract_span_attributes(event),
                    inputs=self._extract_inputs(event),
                    outputs=self._extract_outputs(event),
                )
                trace_spans.append(child_span)

            spans_by_trace[mlflow_trace_id] = trace_spans

        return spans_by_trace

    def _create_span(
        self,
        name: str,
        otel_trace_id: int,
        span_id: int,
        parent_span_id: int | None,
        start_time_ns: int,
        end_time_ns: int,
        mlflow_trace_id: str,
        span_type: str,
        attributes: dict[str, Any] | None = None,
        inputs: Any = None,
        outputs: Any = None,
    ) -> Span:
        """Create an MLflow Span from the given parameters."""
        span_attributes = {
            SpanAttributeKey.REQUEST_ID: dump_span_attribute_value(mlflow_trace_id),
            SpanAttributeKey.SPAN_TYPE: dump_span_attribute_value(span_type),
        }

        if inputs is not None:
            span_attributes[SpanAttributeKey.INPUTS] = dump_span_attribute_value(inputs)

        if outputs is not None:
            span_attributes[SpanAttributeKey.OUTPUTS] = dump_span_attribute_value(outputs)

        if attributes:
            for key, value in attributes.items():
                if key not in (SpanAttributeKey.REQUEST_ID, SpanAttributeKey.SPAN_TYPE):
                    span_attributes[key] = dump_span_attribute_value(value)

        otel_span = OTelReadableSpan(
            name=name,
            context=build_otel_context(otel_trace_id, span_id),
            parent=build_otel_context(otel_trace_id, parent_span_id) if parent_span_id else None,
            start_time=start_time_ns,
            end_time=end_time_ns,
            attributes=span_attributes,
            status=OTelStatus(OTelStatusCode.OK),
            resource=_OTelResource.get_empty(),
            events=[],
        )

        return Span(otel_span)

    def _get_span_name(self, event: dict) -> str:
        """Get a descriptive span name for the event."""
        event_name = event["event_name"]
        attrs = event["attributes"]

        if event_name == CODEX_EVENT_USER_PROMPT:
            prompt_length = attrs.get("prompt_length", "")
            return f"User Prompt ({prompt_length} chars)"
        elif event_name == CODEX_EVENT_API_REQUEST:
            attempt = attrs.get("attempt", 1)
            status = attrs.get("http.response.status_code", "")
            return f"API Request (attempt {attempt}, status {status})"
        elif event_name == CODEX_EVENT_SSE_EVENT:
            kind = attrs.get("event.kind", "unknown")
            return f"SSE: {kind}"
        elif event_name == CODEX_EVENT_TOOL_RESULT:
            tool_name = attrs.get("tool_name", "unknown")
            success = attrs.get("success", "")
            return f"Tool: {tool_name} (success={success})"
        elif event_name == CODEX_EVENT_TOOL_DECISION:
            tool_name = attrs.get("tool_name", "unknown")
            decision = attrs.get("decision", "")
            return f"Tool Decision: {tool_name} ({decision})"
        elif event_name == CODEX_EVENT_CONVERSATION_STARTS:
            model = attrs.get("model", "unknown")
            return f"Conversation Start ({model})"
        elif event_name == CODEX_EVENT_SANDBOX_ASSESSMENT:
            call_id = attrs.get("call_id", "")[:8] if attrs.get("call_id") else ""
            return f"Sandbox Assessment ({call_id})"
        else:
            return event_name.replace("codex.", "").replace("_", " ").title()

    def _extract_conversation_attributes(self, event: dict) -> dict[str, Any]:
        """Extract conversation-level attributes from the root event."""
        attrs = event["attributes"]
        result = {}

        # Include key conversation metadata
        for key in [
            "model",
            "slug",
            "app.version",
            "provider_name",
            "approval_policy",
            "sandbox_policy",
            "mcp_servers",
            "active_profile",
            "reasoning_effort",
            "reasoning_summary",
        ]:
            if key in attrs:
                result[key] = attrs[key]

        # Include resource attributes
        if event.get("resource_attributes"):
            for key, value in event["resource_attributes"].items():
                if key not in result:
                    result[f"resource.{key}"] = value

        return result

    def _extract_span_attributes(self, event: dict) -> dict[str, Any]:
        """Extract relevant attributes for a span."""
        attrs = event["attributes"]
        result = {}

        # Copy relevant attributes, excluding some internal ones
        excluded_keys = {"event.name", "event.timestamp", "conversation.id"}
        for key, value in attrs.items():
            if key not in excluded_keys:
                result[key] = value

        return result

    def _extract_inputs(self, event: dict) -> Any:
        """Extract input values for a span."""
        event_name = event["event_name"]
        attrs = event["attributes"]

        if event_name == CODEX_EVENT_USER_PROMPT:
            return {"prompt": attrs.get("prompt", "[REDACTED]")}
        elif event_name == CODEX_EVENT_TOOL_RESULT:
            return {"arguments": attrs.get("arguments", "")}
        elif event_name == CODEX_EVENT_API_REQUEST:
            return {"model": attrs.get("model", "")}
        return None

    def _extract_outputs(self, event: dict) -> Any:
        """Extract output values for a span."""
        event_name = event["event_name"]
        attrs = event["attributes"]

        if event_name == CODEX_EVENT_TOOL_RESULT:
            return {
                "output": attrs.get("output", ""),
                "success": attrs.get("success", ""),
            }
        elif event_name == CODEX_EVENT_SSE_EVENT:
            result = {}
            for key in ["input_token_count", "output_token_count", "cached_token_count"]:
                if key in attrs:
                    result[key] = attrs[key]
            return result or None
        elif event_name == CODEX_EVENT_API_REQUEST:
            return {
                "status_code": attrs.get("http.response.status_code"),
                "duration_ms": attrs.get("duration_ms"),
            }
        return None


@otel_logs_router.post("", status_code=200)
async def export_logs(
    request: Request,
    x_mlflow_experiment_id: str = Header(..., alias=MLFLOW_EXPERIMENT_ID_HEADER),
    content_type: str = Header(None),
) -> Response:
    """
    Export log records to MLflow, converting them to traces.

    This endpoint accepts OTLP/HTTP protobuf log export requests from Codex CLI
    and converts them into MLflow traces for visualization.

    Args:
        request: OTel ExportLogsServiceRequest in protobuf format
        x_mlflow_experiment_id: Required header containing the experiment ID
        content_type: Content-Type header from the request

    Returns:
        FastAPI Response with ExportLogsServiceResponse in protobuf format

    Raises:
        HTTPException: If the request is invalid or span logging fails
    """
    # Validate Content-Type header
    if content_type != "application/x-protobuf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Content-Type: {content_type}. Expected: application/x-protobuf",
        )

    body = await request.body()
    parsed_request = ExportLogsServiceRequest()

    try:
        parsed_request.ParseFromString(body)

        if not parsed_request.resource_logs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid OpenTelemetry protobuf format - no logs found",
            )

    except DecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OpenTelemetry protobuf format",
        )

    # Convert log records to spans
    converter = CodexLogToSpanConverter()

    for resource_log in parsed_request.resource_logs:
        # Extract resource attributes
        resource_attributes = {
            attr.key: _decode_log_anyvalue(attr.value) for attr in resource_log.resource.attributes
        }

        for scope_log in resource_log.scope_logs:
            for log_record in scope_log.log_records:
                # Extract log record attributes
                attributes = {
                    attr.key: _decode_log_anyvalue(attr.value) for attr in log_record.attributes
                }

                # Get timestamp (use observed_time_unix_nano as fallback)
                timestamp_ns = log_record.time_unix_nano or log_record.observed_time_unix_nano
                if not timestamp_ns:
                    timestamp_ns = int(time.time() * 1_000_000_000)

                # Get body if it's a string
                body_value = None
                if log_record.body.WhichOneof("value") == "string_value":
                    body_value = log_record.body.string_value

                converter.add_log_record(
                    attributes=attributes,
                    timestamp_ns=timestamp_ns,
                    body=body_value,
                    resource_attributes=resource_attributes,
                )

    # Convert logs to spans and store them
    spans_by_trace = converter.convert_to_spans()

    if spans_by_trace:
        store = _get_tracking_store()
        errors = {}

        for trace_id, spans in spans_by_trace.items():
            try:
                store.log_spans(x_mlflow_experiment_id, spans)
            except NotImplementedError:
                store_name = store.__class__.__name__
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail=f"REST OTLP log ingestion is not supported by {store_name}",
                )
            except Exception as e:
                errors[trace_id] = e

        if errors:
            error_msg = "\n".join(
                [f"Trace {trace_id}: {error}" for trace_id, error in errors.items()]
            )
            raise HTTPException(
                status_code=422,
                detail=f"Failed to log Codex traces: {error_msg}",
            )

    # Return protobuf response as per OTLP specification
    response_message = ExportLogsServiceResponse()
    response_bytes = response_message.SerializeToString()
    return Response(
        content=response_bytes,
        media_type="application/x-protobuf",
        status_code=200,
    )
