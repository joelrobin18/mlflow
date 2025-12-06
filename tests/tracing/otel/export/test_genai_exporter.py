"""Tests for the GenAI schema span exporter."""

import json
from unittest import mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, TraceFlags

from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.otel.export.genai_exporter import (
    MLFLOW_TYPE_TO_GENAI_OPERATION,
    GenAiAttributeKey,
    GenAiOperationName,
    GenAiSchemaSpanExporter,
)


@pytest.fixture
def mock_base_exporter():
    exporter = mock.MagicMock()
    exporter.export.return_value = SpanExportResult.SUCCESS
    exporter.shutdown.return_value = None
    exporter.force_flush.return_value = True
    return exporter


@pytest.fixture
def mock_span():
    span = mock.MagicMock(spec=ReadableSpan)
    span.name = "test_span"
    span.context = mock.MagicMock(spec=SpanContext)
    span.context.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
    span.context.span_id = 0x1234567890ABCDEF
    span.context.trace_flags = TraceFlags(TraceFlags.SAMPLED)
    span.parent = None
    span.start_time = 1000000000
    span.end_time = 2000000000
    span.status = mock.MagicMock()
    span.status.status_code = mock.MagicMock()
    span.status.status_code.name = "OK"
    span.events = []
    span.links = []
    span.resource = mock.MagicMock()
    span.instrumentation_info = None
    span.instrumentation_scope = None
    span.kind = mock.MagicMock()
    span.get_span_context.return_value = span.context
    return span


class TestGenAiSchemaSpanExporter:
    def test_export_delegates_to_base_exporter(self, mock_base_exporter, mock_span):
        mock_span.attributes = {}
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        result = exporter.export([mock_span])

        assert result == SpanExportResult.SUCCESS
        mock_base_exporter.export.assert_called_once()

    def test_shutdown_delegates_to_base_exporter(self, mock_base_exporter):
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.shutdown()

        mock_base_exporter.shutdown.assert_called_once()

    def test_force_flush_delegates_to_base_exporter(self, mock_base_exporter):
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        result = exporter.force_flush(timeout_millis=5000)

        assert result is True
        mock_base_exporter.force_flush.assert_called_once_with(5000)


class TestSpanTypeTransformation:
    @pytest.mark.parametrize(
        ("mlflow_span_type", "expected_operation"),
        [
            (SpanType.CHAT_MODEL, GenAiOperationName.CHAT),
            (SpanType.LLM, GenAiOperationName.TEXT_COMPLETION),
            (SpanType.EMBEDDING, GenAiOperationName.EMBEDDINGS),
            (SpanType.AGENT, GenAiOperationName.INVOKE_AGENT),
            (SpanType.TOOL, GenAiOperationName.EXECUTE_TOOL),
        ],
    )
    def test_span_type_to_operation_name(
        self, mock_base_exporter, mock_span, mlflow_span_type, expected_operation
    ):
        mock_span.attributes = {
            SpanAttributeKey.SPAN_TYPE: json.dumps(mlflow_span_type),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        # Get the transformed span from the export call
        exported_spans = mock_base_exporter.export.call_args[0][0]
        assert len(exported_spans) == 1
        transformed_span = exported_spans[0]
        assert transformed_span.attributes[GenAiAttributeKey.OPERATION_NAME] == expected_operation

    def test_unmapped_span_type_preserved(self, mock_base_exporter, mock_span):
        mock_span.attributes = {
            SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.CHAIN),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        transformed_span = exported_spans[0]
        # CHAIN type doesn't have a mapping, so no operation name is set
        assert GenAiAttributeKey.OPERATION_NAME not in transformed_span.attributes


class TestInputOutputTransformation:
    def test_inputs_transformed_to_input_messages(self, mock_base_exporter, mock_span):
        inputs = [{"role": "user", "content": "Hello"}]
        mock_span.attributes = {
            SpanAttributeKey.INPUTS: json.dumps(inputs),
            SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.LLM),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        transformed_span = exported_spans[0]
        assert GenAiAttributeKey.INPUT_MESSAGES in transformed_span.attributes
        assert json.loads(transformed_span.attributes[GenAiAttributeKey.INPUT_MESSAGES]) == inputs

    def test_outputs_transformed_to_output_messages(self, mock_base_exporter, mock_span):
        outputs = [{"role": "assistant", "content": "Hi there!"}]
        mock_span.attributes = {
            SpanAttributeKey.OUTPUTS: json.dumps(outputs),
            SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.LLM),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        transformed_span = exported_spans[0]
        assert GenAiAttributeKey.OUTPUT_MESSAGES in transformed_span.attributes
        assert json.loads(transformed_span.attributes[GenAiAttributeKey.OUTPUT_MESSAGES]) == outputs

    def test_tool_inputs_transformed_to_tool_call_arguments(self, mock_base_exporter, mock_span):
        inputs = {"param1": "value1", "param2": 42}
        mock_span.attributes = {
            SpanAttributeKey.INPUTS: json.dumps(inputs),
            SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.TOOL),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        transformed_span = exported_spans[0]
        assert GenAiAttributeKey.TOOL_CALL_ARGUMENTS in transformed_span.attributes
        assert (
            json.loads(transformed_span.attributes[GenAiAttributeKey.TOOL_CALL_ARGUMENTS]) == inputs
        )

    def test_tool_outputs_transformed_to_tool_call_result(self, mock_base_exporter, mock_span):
        outputs = {"result": "success", "data": [1, 2, 3]}
        mock_span.attributes = {
            SpanAttributeKey.OUTPUTS: json.dumps(outputs),
            SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.TOOL),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        transformed_span = exported_spans[0]
        assert GenAiAttributeKey.TOOL_CALL_RESULT in transformed_span.attributes
        assert (
            json.loads(transformed_span.attributes[GenAiAttributeKey.TOOL_CALL_RESULT]) == outputs
        )


class TestTokenUsageTransformation:
    def test_token_usage_transformed(self, mock_base_exporter, mock_span):
        token_usage = {
            TokenUsageKey.INPUT_TOKENS: 100,
            TokenUsageKey.OUTPUT_TOKENS: 50,
            TokenUsageKey.TOTAL_TOKENS: 150,
        }
        mock_span.attributes = {
            SpanAttributeKey.CHAT_USAGE: json.dumps(token_usage),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        transformed_span = exported_spans[0]
        assert transformed_span.attributes[GenAiAttributeKey.INPUT_TOKENS] == 100
        assert transformed_span.attributes[GenAiAttributeKey.OUTPUT_TOKENS] == 50

    def test_partial_token_usage_transformed(self, mock_base_exporter, mock_span):
        token_usage = {
            TokenUsageKey.INPUT_TOKENS: 75,
        }
        mock_span.attributes = {
            SpanAttributeKey.CHAT_USAGE: json.dumps(token_usage),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        transformed_span = exported_spans[0]
        assert transformed_span.attributes[GenAiAttributeKey.INPUT_TOKENS] == 75
        assert GenAiAttributeKey.OUTPUT_TOKENS not in transformed_span.attributes


class TestNonMlflowAttributesPreserved:
    def test_custom_attributes_preserved(self, mock_base_exporter, mock_span):
        mock_span.attributes = {
            "custom.attribute": "custom_value",
            "another.attribute": 42,
            SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.LLM),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        transformed_span = exported_spans[0]
        assert transformed_span.attributes["custom.attribute"] == "custom_value"
        assert transformed_span.attributes["another.attribute"] == 42

    def test_existing_genai_attributes_preserved(self, mock_base_exporter, mock_span):
        mock_span.attributes = {
            GenAiAttributeKey.REQUEST_MODEL: "gpt-4",
            GenAiAttributeKey.SYSTEM: "openai",
            SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.LLM),
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        transformed_span = exported_spans[0]
        assert transformed_span.attributes[GenAiAttributeKey.REQUEST_MODEL] == "gpt-4"
        assert transformed_span.attributes[GenAiAttributeKey.SYSTEM] == "openai"


class TestMappingCompleteness:
    def test_mlflow_type_to_genai_operation_mapping(self):
        expected_mappings = {
            SpanType.CHAT_MODEL: GenAiOperationName.CHAT,
            SpanType.LLM: GenAiOperationName.TEXT_COMPLETION,
            SpanType.EMBEDDING: GenAiOperationName.EMBEDDINGS,
            SpanType.AGENT: GenAiOperationName.INVOKE_AGENT,
            SpanType.TOOL: GenAiOperationName.EXECUTE_TOOL,
        }
        assert MLFLOW_TYPE_TO_GENAI_OPERATION == expected_mappings


class TestEdgeCases:
    def test_empty_attributes(self, mock_base_exporter, mock_span):
        mock_span.attributes = {}
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        assert len(exported_spans) == 1

    def test_none_attributes(self, mock_base_exporter, mock_span):
        mock_span.attributes = None
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        # Should not raise an exception
        mock_base_exporter.export.assert_called_once()

    def test_malformed_json_attributes(self, mock_base_exporter, mock_span):
        mock_span.attributes = {
            SpanAttributeKey.SPAN_TYPE: "not-valid-json{",
        }
        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span])

        # Should not raise an exception
        mock_base_exporter.export.assert_called_once()

    def test_multiple_spans(self, mock_base_exporter, mock_span):
        mock_span.attributes = {
            SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.LLM),
        }
        mock_span2 = mock.MagicMock(spec=ReadableSpan)
        mock_span2.attributes = {
            SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.TOOL),
        }
        mock_span2.name = "test_span_2"
        mock_span2.context = mock_span.context
        mock_span2.parent = None
        mock_span2.start_time = mock_span.start_time
        mock_span2.end_time = mock_span.end_time
        mock_span2.status = mock_span.status
        mock_span2.events = []
        mock_span2.links = []
        mock_span2.resource = mock_span.resource
        mock_span2.instrumentation_info = None
        mock_span2.instrumentation_scope = None
        mock_span2.kind = mock_span.kind
        mock_span2.get_span_context.return_value = mock_span.context

        exporter = GenAiSchemaSpanExporter(mock_base_exporter)

        exporter.export([mock_span, mock_span2])

        exported_spans = mock_base_exporter.export.call_args[0][0]
        assert len(exported_spans) == 2
        assert (
            exported_spans[0].attributes[GenAiAttributeKey.OPERATION_NAME]
            == GenAiOperationName.TEXT_COMPLETION
        )
        assert (
            exported_spans[1].attributes[GenAiAttributeKey.OPERATION_NAME]
            == GenAiOperationName.EXECUTE_TOOL
        )
