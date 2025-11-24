"""
Integration tests for GenAI Semantic Conventions export.

Tests the end-to-end flow of exporting spans and metrics with GenAI schema.
"""

import pytest

from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import dump_span_attribute_value


class TestGenAiSchemaIntegration:
    """Integration tests for GenAI schema export."""

    def test_span_attribute_conversion_with_genai_schema(self, monkeypatch):
        """Test that span attributes are converted when env var is set."""
        from mlflow.entities.span import Span
        from mlflow.environment_variables import OTEL_EXPORTER_OTLP_TRACES_SCHEMA

        # Set environment variable to use GenAI schema
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_SCHEMA.name, "genai")

        # Create test attributes
        attributes = {
            SpanAttributeKey.SPAN_TYPE: dump_span_attribute_value(SpanType.CHAT_MODEL),
            SpanAttributeKey.CHAT_USAGE: dump_span_attribute_value(
                {
                    TokenUsageKey.INPUT_TOKENS: 100,
                    TokenUsageKey.OUTPUT_TOKENS: 50,
                    TokenUsageKey.TOTAL_TOKENS: 150,
                }
            ),
        }

        # Test the conversion directly using the static method
        converted = Span._apply_export_schema_conversion(attributes)

        # Verify GenAI attributes are present
        assert "gen_ai.operation.name" in converted
        assert converted["gen_ai.operation.name"] == "chat"
        assert "gen_ai.usage.input_tokens" in converted
        assert converted["gen_ai.usage.input_tokens"] == 100
        assert "gen_ai.usage.output_tokens" in converted
        assert converted["gen_ai.usage.output_tokens"] == 50

        # Verify original attributes are preserved
        assert SpanAttributeKey.SPAN_TYPE in converted
        assert SpanAttributeKey.CHAT_USAGE in converted

    def test_span_attribute_conversion_without_genai_schema(self, monkeypatch):
        """Test that attributes are not converted when env var is not set."""
        from mlflow.entities.span import Span
        from mlflow.environment_variables import OTEL_EXPORTER_OTLP_TRACES_SCHEMA

        # Ensure env var is not set
        monkeypatch.delenv(OTEL_EXPORTER_OTLP_TRACES_SCHEMA.name, raising=False)

        attributes = {
            SpanAttributeKey.SPAN_TYPE: dump_span_attribute_value(SpanType.CHAT_MODEL),
            SpanAttributeKey.CHAT_USAGE: dump_span_attribute_value(
                {
                    TokenUsageKey.INPUT_TOKENS: 100,
                    TokenUsageKey.OUTPUT_TOKENS: 50,
                    TokenUsageKey.TOTAL_TOKENS: 150,
                }
            ),
        }

        converted = Span._apply_export_schema_conversion(attributes)

        # Verify GenAI attributes are NOT present
        assert "gen_ai.operation.name" not in converted
        assert "gen_ai.usage.input_tokens" not in converted
        assert "gen_ai.usage.output_tokens" not in converted

        # Verify attributes are unchanged
        assert converted == attributes

    def test_metrics_conversion_with_genai_schema(self, monkeypatch):
        """Test that metrics use GenAI schema when env var is set."""
        from mlflow.environment_variables import OTEL_EXPORTER_OTLP_METRICS_SCHEMA
        from mlflow.tracing.processor.otel_metrics_mixin import OtelMetricsMixin

        # Set environment variable to use GenAI schema
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_METRICS_SCHEMA.name, "genai")

        # Test metric name conversion
        metric_name = OtelMetricsMixin._get_metric_name("mlflow.trace.span.duration")
        assert metric_name == "gen_ai.client.operation.duration"

        # Test metric attributes conversion
        attributes = {
            "span_type": SpanType.CHAT_MODEL,
            "span_status": "ERROR",
            "experiment_id": "123",
        }
        converted = OtelMetricsMixin._apply_metrics_schema_conversion(attributes)

        assert "gen_ai.operation.name" in converted
        assert converted["gen_ai.operation.name"] == "chat"
        assert "error.type" in converted
        assert converted["error.type"] == "error"

    def test_metrics_without_genai_schema(self, monkeypatch):
        """Test that metrics use default schema when env var is not set."""
        from mlflow.environment_variables import OTEL_EXPORTER_OTLP_METRICS_SCHEMA
        from mlflow.tracing.processor.otel_metrics_mixin import OtelMetricsMixin

        # Ensure env var is not set
        monkeypatch.delenv(OTEL_EXPORTER_OTLP_METRICS_SCHEMA.name, raising=False)

        # Test metric name is unchanged
        metric_name = OtelMetricsMixin._get_metric_name("mlflow.trace.span.duration")
        assert metric_name == "mlflow.trace.span.duration"

        # Test metric attributes are unchanged
        attributes = {
            "span_type": SpanType.CHAT_MODEL,
            "span_status": "ERROR",
        }
        converted = OtelMetricsMixin._apply_metrics_schema_conversion(attributes)

        assert "gen_ai.operation.name" not in converted
        assert "error.type" not in converted
        assert converted == attributes

    def test_tool_span_attribute_conversion(self, monkeypatch):
        """Test that TOOL spans use correct GenAI attributes."""
        from mlflow.entities.span import Span
        from mlflow.environment_variables import OTEL_EXPORTER_OTLP_TRACES_SCHEMA

        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_SCHEMA.name, "genai")

        attributes = {
            SpanAttributeKey.SPAN_TYPE: dump_span_attribute_value(SpanType.TOOL),
            SpanAttributeKey.INPUTS: dump_span_attribute_value({"query": "search term"}),
            SpanAttributeKey.OUTPUTS: dump_span_attribute_value(
                {"results": ["item1", "item2"]}
            ),
        }

        converted = Span._apply_export_schema_conversion(attributes)

        # Verify TOOL uses execute_tool operation
        assert converted["gen_ai.operation.name"] == "execute_tool"

        # Verify TOOL uses tool.call.arguments/result instead of input/output.messages
        assert "gen_ai.tool.call.arguments" in converted
        assert "gen_ai.tool.call.result" in converted
        assert "gen_ai.input.messages" not in converted
        assert "gen_ai.output.messages" not in converted
