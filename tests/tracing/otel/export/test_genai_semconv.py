"""
Tests for GenAI Semantic Conventions export converter.
"""

import pytest

from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.otel.export.genai_semconv import GenAiExportConverter


@pytest.fixture
def converter():
    """Create a GenAiExportConverter instance for testing."""
    return GenAiExportConverter()


class TestGenAiExportConverter:
    """Test suite for GenAI export converter."""

    def test_convert_span_type_chat_model(self, converter):
        """Test conversion of CHAT_MODEL span type to GenAI operation name."""
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.CHAT_MODEL}"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.operation.name" in converted
        assert converted["gen_ai.operation.name"] == "chat"
        # Original attribute should be preserved
        assert SpanAttributeKey.SPAN_TYPE in converted

    def test_convert_span_type_llm(self, converter):
        """Test conversion of LLM span type to GenAI operation name."""
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.LLM}"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.operation.name" in converted
        assert converted["gen_ai.operation.name"] == "text_completion"

    def test_convert_span_type_embedding(self, converter):
        """Test conversion of EMBEDDING span type to GenAI operation name."""
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.EMBEDDING}"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.operation.name" in converted
        assert converted["gen_ai.operation.name"] == "embeddings"

    def test_convert_span_type_tool(self, converter):
        """Test conversion of TOOL span type to GenAI operation name."""
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.TOOL}"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.operation.name" in converted
        assert converted["gen_ai.operation.name"] == "execute_tool"

    def test_convert_span_type_agent(self, converter):
        """Test conversion of AGENT span type to GenAI operation name."""
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.AGENT}"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.operation.name" in converted
        assert converted["gen_ai.operation.name"] == "invoke_agent"

    def test_convert_span_type_unsupported(self, converter):
        """Test that unsupported span types don't add GenAI operation name."""
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.CHAIN}"',
        }
        converted = converter.convert_span_attributes(attributes)

        # Should not add gen_ai.operation.name for unsupported types
        assert "gen_ai.operation.name" not in converted
        # Original attribute should still be there
        assert SpanAttributeKey.SPAN_TYPE in converted

    def test_convert_token_usage(self, converter):
        """Test conversion of token usage to GenAI attributes."""
        token_usage = {
            TokenUsageKey.INPUT_TOKENS: 100,
            TokenUsageKey.OUTPUT_TOKENS: 50,
            TokenUsageKey.TOTAL_TOKENS: 150,
        }
        attributes = {
            SpanAttributeKey.CHAT_USAGE: f'"{{\\"input_tokens\\": 100, \\"output_tokens\\": 50, \\"total_tokens\\": 150}}"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.usage.input_tokens" in converted
        assert converted["gen_ai.usage.input_tokens"] == 100
        assert "gen_ai.usage.output_tokens" in converted
        assert converted["gen_ai.usage.output_tokens"] == 50
        # Original attribute should be preserved
        assert SpanAttributeKey.CHAT_USAGE in converted

    def test_convert_inputs_for_chat_model(self, converter):
        """Test conversion of inputs to gen_ai.input.messages for chat model."""
        inputs = [{"role": "user", "content": "Hello"}]
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.CHAT_MODEL}"',
            SpanAttributeKey.INPUTS: f'"[{{\\"role\\": \\"user\\", \\"content\\": \\"Hello\\"}}]"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.input.messages" in converted
        # Should not use tool.call.arguments for chat model
        assert "gen_ai.tool.call.arguments" not in converted

    def test_convert_inputs_for_tool(self, converter):
        """Test conversion of inputs to gen_ai.tool.call.arguments for tool."""
        inputs = {"query": "search term"}
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.TOOL}"',
            SpanAttributeKey.INPUTS: f'"{{\\"query\\": \\"search term\\"}}"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.tool.call.arguments" in converted
        # Should not use input.messages for tool
        assert "gen_ai.input.messages" not in converted

    def test_convert_outputs_for_chat_model(self, converter):
        """Test conversion of outputs to gen_ai.output.messages for chat model."""
        outputs = [{"role": "assistant", "content": "Hi there!"}]
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.CHAT_MODEL}"',
            SpanAttributeKey.OUTPUTS: f'"[{{\\"role\\": \\"assistant\\", \\"content\\": \\"Hi there!\\"}}]"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.output.messages" in converted
        # Should not use tool.call.result for chat model
        assert "gen_ai.tool.call.result" not in converted

    def test_convert_outputs_for_tool(self, converter):
        """Test conversion of outputs to gen_ai.tool.call.result for tool."""
        outputs = {"results": ["item1", "item2"]}
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.TOOL}"',
            SpanAttributeKey.OUTPUTS: f'"{{\\"results\\": [\\"item1\\", \\"item2\\"]}}"',
        }
        converted = converter.convert_span_attributes(attributes)

        assert "gen_ai.tool.call.result" in converted
        # Should not use output.messages for tool
        assert "gen_ai.output.messages" not in converted

    def test_convert_complete_span_attributes(self, converter):
        """Test conversion of a complete set of span attributes."""
        attributes = {
            SpanAttributeKey.SPAN_TYPE: f'"{SpanType.CHAT_MODEL}"',
            SpanAttributeKey.CHAT_USAGE: f'"{{\\"input_tokens\\": 100, \\"output_tokens\\": 50, \\"total_tokens\\": 150}}"',
            SpanAttributeKey.INPUTS: f'"[{{\\"role\\": \\"user\\", \\"content\\": \\"Hello\\"}}]"',
            SpanAttributeKey.OUTPUTS: f'"[{{\\"role\\": \\"assistant\\", \\"content\\": \\"Hi!\\"}}]"',
            "custom_attribute": "custom_value",
        }
        converted = converter.convert_span_attributes(attributes)

        # Check all conversions happened
        assert converted["gen_ai.operation.name"] == "chat"
        assert converted["gen_ai.usage.input_tokens"] == 100
        assert converted["gen_ai.usage.output_tokens"] == 50
        assert "gen_ai.input.messages" in converted
        assert "gen_ai.output.messages" in converted

        # Check original attributes preserved
        assert SpanAttributeKey.SPAN_TYPE in converted
        assert SpanAttributeKey.CHAT_USAGE in converted
        assert SpanAttributeKey.INPUTS in converted
        assert SpanAttributeKey.OUTPUTS in converted
        assert converted["custom_attribute"] == "custom_value"

    def test_convert_metric_attributes(self, converter):
        """Test conversion of metric attributes to GenAI format."""
        attributes = {
            "span_type": SpanType.CHAT_MODEL,
            "span_status": "ERROR",
            "experiment_id": "123",
            "root": True,
        }
        converted = converter.convert_metric_attributes(attributes)

        assert "gen_ai.operation.name" in converted
        assert converted["gen_ai.operation.name"] == "chat"
        assert "error.type" in converted
        assert converted["error.type"] == "error"
        # Original attributes should be preserved
        assert "span_type" in converted
        assert "span_status" in converted

    def test_convert_metric_attributes_success_status(self, converter):
        """Test that non-ERROR status doesn't add error.type."""
        attributes = {
            "span_type": SpanType.LLM,
            "span_status": "OK",
        }
        converted = converter.convert_metric_attributes(attributes)

        assert "gen_ai.operation.name" in converted
        assert "error.type" not in converted

    def test_get_metric_name(self, converter):
        """Test conversion of metric names to GenAI format."""
        assert (
            converter.get_metric_name("mlflow.trace.span.duration")
            == "gen_ai.client.operation.duration"
        )

    def test_get_metric_name_unknown(self, converter):
        """Test that unknown metric names are returned unchanged."""
        assert converter.get_metric_name("custom.metric") == "custom.metric"

    def test_empty_attributes(self, converter):
        """Test that empty attributes dict is handled correctly."""
        attributes = {}
        converted = converter.convert_span_attributes(attributes)

        assert converted == {}

    def test_none_token_usage(self, converter):
        """Test handling of None token usage values."""
        attributes = {
            SpanAttributeKey.CHAT_USAGE: "null",
        }
        converted = converter.convert_span_attributes(attributes)

        # Should not add token usage attributes if value is null
        assert "gen_ai.usage.input_tokens" not in converted
        assert "gen_ai.usage.output_tokens" not in converted
