"""
Tests for the OpenTelemetry Logs API endpoint.

This tests the Codex CLI integration that converts OTLP logs to MLflow traces.
"""

import time
import uuid

import pytest
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource as OTelResource

from mlflow.server.otel_logs_api import (
    CODEX_EVENT_API_REQUEST,
    CODEX_EVENT_CONVERSATION_STARTS,
    CODEX_EVENT_SSE_EVENT,
    CODEX_EVENT_TOOL_RESULT,
    CODEX_EVENT_USER_PROMPT,
    CodexLogToSpanConverter,
    _decode_log_anyvalue,
    _generate_mlflow_trace_id,
    _generate_span_id,
    _generate_trace_id_from_conversation_id,
)


class TestDecodeLogAnyvalue:
    def test_string_value(self):
        value = AnyValue(string_value="test")
        assert _decode_log_anyvalue(value) == "test"

    def test_int_value(self):
        value = AnyValue(int_value=42)
        assert _decode_log_anyvalue(value) == 42

    def test_bool_value(self):
        value = AnyValue(bool_value=True)
        assert _decode_log_anyvalue(value) is True

    def test_double_value(self):
        value = AnyValue(double_value=3.14)
        assert _decode_log_anyvalue(value) == 3.14

    def test_none_value(self):
        value = AnyValue()
        assert _decode_log_anyvalue(value) is None


class TestTraceIdGeneration:
    def test_consistent_trace_id(self):
        """Same conversation ID should produce same trace ID."""
        conv_id = "test-conversation-123"
        trace_id_1 = _generate_trace_id_from_conversation_id(conv_id)
        trace_id_2 = _generate_trace_id_from_conversation_id(conv_id)
        assert trace_id_1 == trace_id_2

    def test_different_conversation_ids(self):
        """Different conversation IDs should produce different trace IDs."""
        trace_id_1 = _generate_trace_id_from_conversation_id("conv-1")
        trace_id_2 = _generate_trace_id_from_conversation_id("conv-2")
        assert trace_id_1 != trace_id_2

    def test_mlflow_trace_id_format(self):
        """MLflow trace ID should have tr- prefix."""
        otel_trace_id = 12345
        mlflow_trace_id = _generate_mlflow_trace_id(otel_trace_id)
        assert mlflow_trace_id.startswith("tr-")

    def test_span_id_uniqueness(self):
        """Different events should produce different span IDs."""
        conv_id = "test-conv"
        span_id_1 = _generate_span_id(conv_id, "event1", 1000, 0)
        span_id_2 = _generate_span_id(conv_id, "event2", 1000, 0)
        span_id_3 = _generate_span_id(conv_id, "event1", 2000, 0)
        span_id_4 = _generate_span_id(conv_id, "event1", 1000, 1)

        assert span_id_1 != span_id_2
        assert span_id_1 != span_id_3
        assert span_id_1 != span_id_4


class TestCodexLogToSpanConverter:
    def test_add_log_record_without_event_name(self):
        """Log records without event.name should be skipped."""
        converter = CodexLogToSpanConverter()
        converter.add_log_record(
            attributes={"conversation.id": "test"},
            timestamp_ns=int(time.time() * 1e9),
        )
        spans = converter.convert_to_spans()
        assert len(spans) == 0

    def test_add_log_record_without_conversation_id(self):
        """Log records without conversation.id should be skipped."""
        converter = CodexLogToSpanConverter()
        converter.add_log_record(
            attributes={"event.name": "codex.user_prompt"},
            timestamp_ns=int(time.time() * 1e9),
        )
        spans = converter.convert_to_spans()
        assert len(spans) == 0

    def test_single_event_conversion(self):
        """A single event should create a root span and one child span."""
        converter = CodexLogToSpanConverter()
        conv_id = str(uuid.uuid4())
        now = int(time.time() * 1e9)

        converter.add_log_record(
            attributes={
                "event.name": CODEX_EVENT_USER_PROMPT,
                "conversation.id": conv_id,
                "prompt": "Hello world",
                "prompt_length": "11",
            },
            timestamp_ns=now,
        )

        spans_by_trace = converter.convert_to_spans()
        assert len(spans_by_trace) == 1

        trace_id, spans = list(spans_by_trace.items())[0]
        assert trace_id.startswith("tr-")
        # Root span + 1 child span
        assert len(spans) == 2

    def test_multiple_events_same_conversation(self):
        """Multiple events in same conversation should be grouped."""
        converter = CodexLogToSpanConverter()
        conv_id = str(uuid.uuid4())
        base_time = int(time.time() * 1e9)

        # Add multiple events
        events = [
            (CODEX_EVENT_CONVERSATION_STARTS, {"model": "gpt-4o", "provider_name": "openai"}),
            (CODEX_EVENT_USER_PROMPT, {"prompt": "test", "prompt_length": "4"}),
            (CODEX_EVENT_API_REQUEST, {"attempt": "1", "duration_ms": "500"}),
            (CODEX_EVENT_SSE_EVENT, {"event.kind": "response.completed"}),
        ]

        for i, (event_name, attrs) in enumerate(events):
            converter.add_log_record(
                attributes={
                    "event.name": event_name,
                    "conversation.id": conv_id,
                    **attrs,
                },
                timestamp_ns=base_time + i * 1_000_000_000,  # 1 second apart
            )

        spans_by_trace = converter.convert_to_spans()
        assert len(spans_by_trace) == 1

        trace_id, spans = list(spans_by_trace.items())[0]
        # Root span + 4 child spans
        assert len(spans) == 5

    def test_multiple_conversations(self):
        """Events from different conversations should create separate traces."""
        converter = CodexLogToSpanConverter()
        now = int(time.time() * 1e9)

        conv_ids = [str(uuid.uuid4()) for _ in range(3)]

        for conv_id in conv_ids:
            converter.add_log_record(
                attributes={
                    "event.name": CODEX_EVENT_USER_PROMPT,
                    "conversation.id": conv_id,
                    "prompt": "test",
                },
                timestamp_ns=now,
            )

        spans_by_trace = converter.convert_to_spans()
        assert len(spans_by_trace) == 3

    def test_tool_result_span(self):
        """Tool result events should extract inputs and outputs."""
        converter = CodexLogToSpanConverter()
        conv_id = str(uuid.uuid4())
        now = int(time.time() * 1e9)

        converter.add_log_record(
            attributes={
                "event.name": CODEX_EVENT_TOOL_RESULT,
                "conversation.id": conv_id,
                "tool_name": "shell",
                "call_id": "call-123",
                "arguments": '{"cmd": "ls -la"}',
                "output": "file1.txt\nfile2.txt",
                "success": "true",
                "duration_ms": "150",
            },
            timestamp_ns=now,
        )

        spans_by_trace = converter.convert_to_spans()
        trace_id, spans = list(spans_by_trace.items())[0]

        # Find the tool span (not the root span)
        tool_span = next(s for s in spans if "Tool: shell" in s.name)
        assert tool_span is not None
        assert tool_span.span_type == "TOOL"

    def test_span_hierarchy(self):
        """All child spans should have the root span as parent."""
        converter = CodexLogToSpanConverter()
        conv_id = str(uuid.uuid4())
        now = int(time.time() * 1e9)

        for i, event_name in enumerate(
            [CODEX_EVENT_USER_PROMPT, CODEX_EVENT_API_REQUEST, CODEX_EVENT_SSE_EVENT]
        ):
            converter.add_log_record(
                attributes={
                    "event.name": event_name,
                    "conversation.id": conv_id,
                },
                timestamp_ns=now + i * 1_000_000,
            )

        spans_by_trace = converter.convert_to_spans()
        trace_id, spans = list(spans_by_trace.items())[0]

        # Find root span (no parent)
        root_span = next(s for s in spans if s.parent_id is None)

        # All other spans should have root as parent
        child_spans = [s for s in spans if s.parent_id is not None]
        for child in child_spans:
            assert child.parent_id == root_span.span_id


def _create_test_log_request(
    events: list[tuple[str, str, dict]],
) -> ExportLogsServiceRequest:
    """Helper to create a test ExportLogsServiceRequest.

    Args:
        events: List of (event_name, conversation_id, additional_attrs) tuples.
    """
    request = ExportLogsServiceRequest()
    resource_logs = request.resource_logs.add()

    resource_logs.resource.CopyFrom(
        OTelResource(
            attributes=[
                KeyValue(key="service.name", value=AnyValue(string_value="codex")),
                KeyValue(key="service.version", value=AnyValue(string_value="1.0.0")),
            ]
        )
    )

    scope_logs = resource_logs.scope_logs.add()

    for event_name, conv_id, attrs in events:
        log_record = scope_logs.log_records.add()
        log_record.time_unix_nano = int(time.time() * 1e9)

        # Add required attributes
        log_record.attributes.add(key="event.name", value=AnyValue(string_value=event_name))
        log_record.attributes.add(key="conversation.id", value=AnyValue(string_value=conv_id))

        # Add additional attributes
        for key, value in attrs.items():
            log_record.attributes.add(key=key, value=AnyValue(string_value=str(value)))

    return request


class TestExportLogsEndpoint:
    @pytest.fixture
    def client(self, monkeypatch):
        """Create a test client for the FastAPI app."""
        # Allow testserver host for testing
        monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "testserver,localhost")
        monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "*")

        from fastapi.testclient import TestClient

        # Create a minimal Flask app for testing
        from flask import Flask

        from mlflow.server.fastapi_app import create_fastapi_app

        flask_app = Flask(__name__)
        fastapi_app = create_fastapi_app(flask_app)
        return TestClient(fastapi_app)

    def test_invalid_content_type(self, client):
        """Should reject requests with invalid content type."""
        response = client.post(
            "/v1/logs",
            content=b"test",
            headers={
                "Content-Type": "application/json",
                "x-mlflow-experiment-id": "1",
            },
        )
        assert response.status_code == 400
        assert "Invalid Content-Type" in response.json()["detail"]

    def test_missing_experiment_id(self, client):
        """Should require experiment ID header."""
        response = client.post(
            "/v1/logs",
            content=b"test",
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert response.status_code == 422  # FastAPI validation error

    def test_invalid_protobuf(self, client):
        """Should reject invalid protobuf data."""
        response = client.post(
            "/v1/logs",
            content=b"invalid protobuf data",
            headers={
                "Content-Type": "application/x-protobuf",
                "x-mlflow-experiment-id": "1",
            },
        )
        assert response.status_code == 400

    def test_empty_logs(self, client):
        """Should reject requests with no logs."""
        request = ExportLogsServiceRequest()
        response = client.post(
            "/v1/logs",
            content=request.SerializeToString(),
            headers={
                "Content-Type": "application/x-protobuf",
                "x-mlflow-experiment-id": "1",
            },
        )
        assert response.status_code == 400
        assert "no logs found" in response.json()["detail"]
