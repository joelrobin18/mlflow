"""
Export translator for converting MLflow spans to OpenTelemetry GenAI semantic conventions.

This module provides a wrapping SpanExporter that transforms MLflow span attributes
to follow the OpenTelemetry Semantic Conventions for GenAI systems before export.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/

Usage
-----
To enable GenAI semantic conventions when exporting traces via OTLP, set the
following environment variable:

    export MLFLOW_OTLP_TRACES_EXPORT_SCHEMA=genai

This will transform the following MLflow attributes to GenAI conventions:

Span Type Mapping:
    - mlflow.spanType: CHAT_MODEL -> gen_ai.operation.name: chat
    - mlflow.spanType: LLM -> gen_ai.operation.name: text_completion
    - mlflow.spanType: EMBEDDING -> gen_ai.operation.name: embeddings
    - mlflow.spanType: AGENT -> gen_ai.operation.name: invoke_agent
    - mlflow.spanType: TOOL -> gen_ai.operation.name: execute_tool

Input/Output Mapping:
    - mlflow.spanInputs -> gen_ai.input.messages (for non-tool spans)
    - mlflow.spanOutputs -> gen_ai.output.messages (for non-tool spans)
    - mlflow.spanInputs -> gen_ai.tool.call.arguments (for TOOL spans)
    - mlflow.spanOutputs -> gen_ai.tool.call.result (for TOOL spans)

Token Usage Mapping:
    - mlflow.chat.tokenUsage.input_tokens -> gen_ai.usage.input_tokens
    - mlflow.chat.tokenUsage.output_tokens -> gen_ai.usage.output_tokens

Example
-------
.. code-block:: bash

    # Set the OTLP endpoint and enable GenAI schema
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
    export MLFLOW_OTLP_TRACES_EXPORT_SCHEMA=genai

    # Run your MLflow tracing code
    python my_genai_app.py

The exported traces will use GenAI semantic conventions, making them compatible
with observability tools that understand OpenTelemetry GenAI standards.
"""

import json
import logging
from typing import Any, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

_logger = logging.getLogger(__name__)


class GenAiAttributeKey:
    """
    OpenTelemetry GenAI Semantic Convention attribute keys.

    Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
    """

    # Operation name attribute
    OPERATION_NAME = "gen_ai.operation.name"

    # Token usage attributes
    INPUT_TOKENS = "gen_ai.usage.input_tokens"
    OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # Input/Output attributes
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"

    # Tool call attributes
    TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    TOOL_CALL_RESULT = "gen_ai.tool.call.result"

    # Model attributes
    REQUEST_MODEL = "gen_ai.request.model"
    RESPONSE_MODEL = "gen_ai.response.model"

    # Agent attributes
    AGENT_ID = "gen_ai.agent.id"
    AGENT_NAME = "gen_ai.agent.name"

    # System attributes
    SYSTEM = "gen_ai.system"


class GenAiOperationName:
    """
    GenAI semantic convention operation names.

    Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
    """

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    EXECUTE_TOOL = "execute_tool"
    GENERATE_CONTENT = "generate_content"


# Mapping from MLflow SpanType to GenAI operation.name
MLFLOW_TYPE_TO_GENAI_OPERATION = {
    SpanType.CHAT_MODEL: GenAiOperationName.CHAT,
    SpanType.LLM: GenAiOperationName.TEXT_COMPLETION,
    SpanType.EMBEDDING: GenAiOperationName.EMBEDDINGS,
    SpanType.AGENT: GenAiOperationName.INVOKE_AGENT,
    SpanType.TOOL: GenAiOperationName.EXECUTE_TOOL,
}


class GenAiSchemaSpanExporter(SpanExporter):
    """
    A wrapping SpanExporter that transforms MLflow span attributes to GenAI semantic conventions.

    This exporter wraps another SpanExporter (e.g., OTLPSpanExporter) and transforms
    span attributes from MLflow's format to OpenTelemetry GenAI semantic conventions
    before delegating the actual export.

    Example usage:
        >>> from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        >>> base_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
        >>> genai_exporter = GenAiSchemaSpanExporter(base_exporter)
    """

    def __init__(self, base_exporter: SpanExporter):
        """
        Initialize the GenAI schema span exporter.

        Args:
            base_exporter: The underlying SpanExporter to delegate export operations to.
        """
        self._base_exporter = base_exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Transform spans to GenAI semantic conventions and export them.

        Args:
            spans: The spans to export.

        Returns:
            The result of the export operation.
        """
        transformed_spans = [self._transform_span(span) for span in spans]
        return self._base_exporter.export(transformed_spans)

    def shutdown(self) -> None:
        """Shutdown the underlying exporter."""
        self._base_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the underlying exporter."""
        return self._base_exporter.force_flush(timeout_millis)

    def _transform_span(self, span: ReadableSpan) -> ReadableSpan:
        """
        Transform a single span's attributes to GenAI semantic conventions.

        Args:
            span: The span to transform.

        Returns:
            A new span with transformed attributes.
        """
        try:
            # Get original attributes and create new transformed attributes
            original_attrs = dict(span.attributes or {})
            new_attrs = self._transform_attributes(original_attrs)

            # Create a new ReadableSpan with transformed attributes
            return _ReadableSpanWithNewAttributes(span, new_attrs)
        except Exception as e:
            _logger.debug(f"Failed to transform span attributes to GenAI schema: {e}")
            return span

    def _transform_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """
        Transform MLflow span attributes to GenAI semantic conventions.

        Args:
            attributes: The original MLflow span attributes.

        Returns:
            Transformed attributes following GenAI semantic conventions.
        """
        new_attrs = {}

        # Copy all original attributes first (they may include GenAI conventions already)
        for key, value in attributes.items():
            # Skip MLflow-specific attributes that will be transformed
            if key in (
                SpanAttributeKey.SPAN_TYPE,
                SpanAttributeKey.INPUTS,
                SpanAttributeKey.OUTPUTS,
                SpanAttributeKey.CHAT_USAGE,
            ):
                continue
            new_attrs[key] = value

        # Transform span type to gen_ai.operation.name
        if SpanAttributeKey.SPAN_TYPE in attributes:
            span_type = self._decode_json_value(attributes[SpanAttributeKey.SPAN_TYPE])
            if span_type and span_type in MLFLOW_TYPE_TO_GENAI_OPERATION:
                new_attrs[GenAiAttributeKey.OPERATION_NAME] = MLFLOW_TYPE_TO_GENAI_OPERATION[
                    span_type
                ]
            # Also keep the original span type for reference
            new_attrs[SpanAttributeKey.SPAN_TYPE] = attributes[SpanAttributeKey.SPAN_TYPE]

        # Transform inputs
        if SpanAttributeKey.INPUTS in attributes:
            inputs = self._decode_json_value(attributes[SpanAttributeKey.INPUTS])
            span_type = self._decode_json_value(
                attributes.get(SpanAttributeKey.SPAN_TYPE, f'"{SpanType.UNKNOWN}"')
            )

            if span_type == SpanType.TOOL:
                # For tool spans, use tool call arguments
                new_attrs[GenAiAttributeKey.TOOL_CALL_ARGUMENTS] = self._encode_json_value(inputs)
            else:
                # For other spans, use input messages
                new_attrs[GenAiAttributeKey.INPUT_MESSAGES] = self._encode_json_value(inputs)

        # Transform outputs
        if SpanAttributeKey.OUTPUTS in attributes:
            outputs = self._decode_json_value(attributes[SpanAttributeKey.OUTPUTS])
            span_type = self._decode_json_value(
                attributes.get(SpanAttributeKey.SPAN_TYPE, f'"{SpanType.UNKNOWN}"')
            )

            if span_type == SpanType.TOOL:
                # For tool spans, use tool call result
                new_attrs[GenAiAttributeKey.TOOL_CALL_RESULT] = self._encode_json_value(outputs)
            else:
                # For other spans, use output messages
                new_attrs[GenAiAttributeKey.OUTPUT_MESSAGES] = self._encode_json_value(outputs)

        # Transform token usage
        if SpanAttributeKey.CHAT_USAGE in attributes:
            token_usage = self._decode_json_value(attributes[SpanAttributeKey.CHAT_USAGE])
            if isinstance(token_usage, dict):
                if TokenUsageKey.INPUT_TOKENS in token_usage:
                    new_attrs[GenAiAttributeKey.INPUT_TOKENS] = token_usage[
                        TokenUsageKey.INPUT_TOKENS
                    ]
                if TokenUsageKey.OUTPUT_TOKENS in token_usage:
                    new_attrs[GenAiAttributeKey.OUTPUT_TOKENS] = token_usage[
                        TokenUsageKey.OUTPUT_TOKENS
                    ]

        return new_attrs

    def _decode_json_value(self, value: Any) -> Any:
        """Decode a JSON-encoded value if it's a string."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return value

    def _encode_json_value(self, value: Any) -> str:
        """Encode a value as JSON string."""
        if isinstance(value, str):
            return value
        return json.dumps(value)


class _ReadableSpanWithNewAttributes(ReadableSpan):
    """
    A wrapper around ReadableSpan that allows overriding attributes.

    This class delegates all properties to the wrapped span except for attributes,
    which are replaced with the provided new attributes.
    """

    def __init__(self, wrapped_span: ReadableSpan, new_attributes: dict[str, Any]):
        """
        Initialize the wrapper.

        Args:
            wrapped_span: The original span to wrap.
            new_attributes: The new attributes to use instead of the original ones.
        """
        self._wrapped_span = wrapped_span
        self._new_attributes = new_attributes

    @property
    def name(self) -> str:
        return self._wrapped_span.name

    @property
    def context(self):
        return self._wrapped_span.context

    @property
    def kind(self):
        return self._wrapped_span.kind

    @property
    def parent(self):
        return self._wrapped_span.parent

    @property
    def start_time(self) -> int:
        return self._wrapped_span.start_time

    @property
    def end_time(self) -> int:
        return self._wrapped_span.end_time

    @property
    def status(self):
        return self._wrapped_span.status

    @property
    def attributes(self):
        return self._new_attributes

    @property
    def events(self):
        return self._wrapped_span.events

    @property
    def links(self):
        return self._wrapped_span.links

    @property
    def resource(self):
        return self._wrapped_span.resource

    @property
    def instrumentation_info(self):
        return self._wrapped_span.instrumentation_info

    @property
    def instrumentation_scope(self):
        return self._wrapped_span.instrumentation_scope

    def get_span_context(self):
        return self._wrapped_span.get_span_context()

    def is_recording(self) -> bool:
        return False  # ReadableSpan is always not recording
