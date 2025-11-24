"""
Export converter for OpenTelemetry GenAI Semantic Conventions.

This module converts MLflow span and metric attributes to the OpenTelemetry
GenAI Semantic Conventions format when exporting via OTLP.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

import logging
from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.otel.export.base import OtelExportSchemaConverter

_logger = logging.getLogger(__name__)


class GenAiExportConverter(OtelExportSchemaConverter):
    """
    Converter for exporting to OpenTelemetry GenAI Semantic Conventions.

    This converter transforms MLflow-specific attributes to GenAI semantic
    convention attributes while preserving the original attributes for compatibility.
    """

    # Mapping from MLflow span types to GenAI operation names
    # Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
    MLFLOW_TYPE_TO_GENAI_OPERATION = {
        SpanType.CHAT_MODEL: "chat",
        SpanType.LLM: "text_completion",
        SpanType.EMBEDDING: "embeddings",
        SpanType.TOOL: "execute_tool",
        SpanType.AGENT: "invoke_agent",
    }

    def convert_span_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """
        Convert MLflow span attributes to GenAI semantic convention attributes.

        This method adds GenAI-compliant attributes alongside existing MLflow attributes.
        Original MLflow attributes are preserved for backward compatibility.

        Args:
            attributes: Dictionary of MLflow span attributes

        Returns:
            Dictionary with both original and GenAI-converted attributes
        """
        converted = attributes.copy()

        # Convert span type to gen_ai.operation.name
        if SpanAttributeKey.SPAN_TYPE in attributes:
            span_type = self._parse_json_attribute(attributes[SpanAttributeKey.SPAN_TYPE])
            if genai_operation := self.MLFLOW_TYPE_TO_GENAI_OPERATION.get(span_type):
                converted["gen_ai.operation.name"] = genai_operation

        # Convert token usage to gen_ai.usage.* attributes
        if SpanAttributeKey.CHAT_USAGE in attributes:
            token_usage = self._parse_json_attribute(attributes[SpanAttributeKey.CHAT_USAGE])
            # Handle double-encoded JSON (MLflow encodes attributes twice in some cases)
            if isinstance(token_usage, str):
                token_usage = self._parse_json_attribute(token_usage)
            if isinstance(token_usage, dict):
                if TokenUsageKey.INPUT_TOKENS in token_usage:
                    converted["gen_ai.usage.input_tokens"] = int(
                        token_usage[TokenUsageKey.INPUT_TOKENS]
                    )
                if TokenUsageKey.OUTPUT_TOKENS in token_usage:
                    converted["gen_ai.usage.output_tokens"] = int(
                        token_usage[TokenUsageKey.OUTPUT_TOKENS]
                    )

        # Convert inputs to gen_ai.input.messages or gen_ai.tool.call.arguments
        if SpanAttributeKey.INPUTS in attributes:
            inputs = self._parse_json_attribute(attributes[SpanAttributeKey.INPUTS])
            # Handle double-encoded JSON
            if isinstance(inputs, str):
                inputs = self._parse_json_attribute(inputs)
            span_type = self._parse_json_attribute(
                attributes.get(SpanAttributeKey.SPAN_TYPE, SpanType.UNKNOWN)
            )

            if span_type == SpanType.TOOL:
                converted["gen_ai.tool.call.arguments"] = inputs
            else:
                converted["gen_ai.input.messages"] = inputs

        # Convert outputs to gen_ai.output.messages or gen_ai.tool.call.result
        if SpanAttributeKey.OUTPUTS in attributes:
            outputs = self._parse_json_attribute(attributes[SpanAttributeKey.OUTPUTS])
            # Handle double-encoded JSON
            if isinstance(outputs, str):
                outputs = self._parse_json_attribute(outputs)
            span_type = self._parse_json_attribute(
                attributes.get(SpanAttributeKey.SPAN_TYPE, SpanType.UNKNOWN)
            )

            if span_type == SpanType.TOOL:
                converted["gen_ai.tool.call.result"] = outputs
            else:
                converted["gen_ai.output.messages"] = outputs

        return converted

    def convert_metric_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """
        Convert MLflow metric attributes to GenAI semantic convention attributes.

        Args:
            attributes: Dictionary of MLflow metric attributes

        Returns:
            Dictionary with converted metric attributes
        """
        converted = attributes.copy()

        # Convert span_type to gen_ai.operation.name for metric dimensions
        if "span_type" in attributes:
            span_type = attributes["span_type"]
            if genai_operation := self.MLFLOW_TYPE_TO_GENAI_OPERATION.get(span_type):
                converted["gen_ai.operation.name"] = genai_operation

        # Map span_status to standard error.type attribute
        if "span_status" in attributes:
            status = attributes["span_status"]
            if status == "ERROR":
                converted["error.type"] = "error"

        return converted

    def get_metric_name(self, mlflow_metric_name: str) -> str:
        """
        Convert MLflow metric name to GenAI semantic convention metric name.

        Args:
            mlflow_metric_name: MLflow metric name (e.g., "mlflow.trace.span.duration")

        Returns:
            GenAI semantic convention metric name (e.g., "gen_ai.client.operation.duration")
        """
        # Map MLflow span duration metric to GenAI operation duration metric
        # Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
        if mlflow_metric_name == "mlflow.trace.span.duration":
            return "gen_ai.client.operation.duration"

        return mlflow_metric_name
