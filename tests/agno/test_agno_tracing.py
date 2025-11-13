from unittest.mock import MagicMock, Mock, patch

import pytest
from agno.agent import Agent
from agno.exceptions import ModelProviderError
from agno.models.anthropic import Claude
from agno.tools.function import Function, FunctionCall
from anthropic.types import Message, TextBlock, Usage

import mlflow
import mlflow.agno
from mlflow.entities import SpanType
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import TokenUsageKey

from tests.tracing.helper import get_traces, purge_traces


def _create_message(content):
    return Message(
        id="1",
        model="claude-sonnet-4-20250514",
        content=[TextBlock(text=content, type="text")],
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=5, output_tokens=7, total_tokens=12),
    )


@pytest.fixture
def simple_agent():
    return Agent(
        model=Claude(id="claude-sonnet-4-20250514"),
        instructions="Be concise.",
        markdown=True,
    )


def test_run_simple_autolog(simple_agent):
    mlflow.agno.autolog()

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_message("Paris")
    with patch.object(Claude, "get_client", return_value=mock_client):
        resp = simple_agent.run("Capital of France?")
    assert resp.content == "Paris"

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 5,
        TokenUsageKey.OUTPUT_TOKENS: 7,
        TokenUsageKey.TOTAL_TOKENS: 12,
    }
    spans = traces[0].data.spans
    assert len(spans) == 2
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].name == "Agent.run"
    assert spans[0].inputs == {"message": "Capital of France?"}
    assert spans[0].outputs["content"] == "Paris"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].name == "Claude.invoke"
    # Agno add system message to the input messages, so validate the last message
    assert spans[1].inputs["messages"][-1]["content"] == "Capital of France?"
    assert spans[1].outputs["content"][0]["text"] == "Paris"

    purge_traces()

    mlflow.agno.autolog(disable=True)
    with patch.object(Claude, "get_client", return_value=mock_client):
        simple_agent.run("Again?")
    assert get_traces() == []


def test_run_failure_tracing(simple_agent):
    mlflow.agno.autolog()

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("bang")
    with patch.object(Claude, "get_client", return_value=mock_client):
        with pytest.raises(ModelProviderError, match="bang"):
            simple_agent.run("fail")

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    assert trace.info.token_usage is None
    spans = trace.data.spans
    assert spans[0].name == "Agent.run"
    assert spans[1].name == "Claude.invoke"
    assert spans[1].status.status_code == SpanStatusCode.ERROR
    assert spans[1].status.description == "ModelProviderError: bang"


@pytest.mark.asyncio
async def test_arun_simple_autolog(simple_agent):
    mlflow.agno.autolog()

    async def _mock_create(*args, **kwargs):
        return _create_message("Paris")

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = _mock_create
    with patch.object(Claude, "get_async_client", return_value=mock_client):
        resp = await simple_agent.arun("Capital of France?")

    assert resp.content == "Paris"

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 5,
        TokenUsageKey.OUTPUT_TOKENS: 7,
        TokenUsageKey.TOTAL_TOKENS: 12,
    }
    spans = traces[0].data.spans
    assert len(spans) == 2
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].name == "Agent.arun"
    assert spans[0].inputs == {"message": "Capital of France?"}
    assert spans[0].outputs["content"] == "Paris"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].name == "Claude.ainvoke"
    # Agno add system message to the input messages, so validate the last message
    assert spans[1].inputs["messages"][-1]["content"] == "Capital of France?"
    assert spans[1].outputs["content"][0]["text"] == "Paris"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
async def test_failure_tracing(simple_agent, is_async):
    mlflow.agno.autolog()

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("bang")
    mock_method = "get_async_client" if is_async else "get_client"
    with patch.object(Claude, mock_method, return_value=mock_client):
        with pytest.raises(ModelProviderError, match="bang"):  # noqa: PT012
            if is_async:
                await simple_agent.arun("fail")
            else:
                simple_agent.run("fail")

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    assert trace.info.token_usage is None
    spans = trace.data.spans
    assert spans[0].name == "Agent.run" if not is_async else "Agent.arun"
    assert spans[1].name == "Claude.invoke" if not is_async else "Claude.ainvoke"
    assert spans[1].status.status_code == SpanStatusCode.ERROR
    assert spans[1].status.description == "ModelProviderError: bang"


def test_function_execute_tracing():
    def dummy(x):
        return x + 1

    fc = FunctionCall(function=Function.from_callable(dummy, name="dummy"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    result = fc.execute()
    assert result.result == 2

    spans = get_traces()[0].data.spans
    assert len(spans) == 1
    span = spans[0]
    assert span.span_type == SpanType.TOOL
    assert span.name == "dummy"
    assert span.inputs == {"x": 1}
    assert span.attributes["entrypoint"] is not None
    assert span.outputs["result"] == 2


@pytest.mark.asyncio
async def test_function_aexecute_tracing():
    async def dummy(x):
        return x + 1

    fc = FunctionCall(function=Function.from_callable(dummy, name="dummy"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    result = await fc.aexecute()
    assert result.result == 2

    spans = get_traces()[0].data.spans
    assert len(spans) == 1
    span = spans[0]
    assert span.span_type == SpanType.TOOL
    assert span.name == "dummy"
    assert span.inputs == {"x": 1}
    assert span.attributes["entrypoint"] is not None
    assert span.outputs["result"] == 2


def test_function_execute_failure_tracing():
    from agno.exceptions import AgentRunException

    def boom(x):
        raise AgentRunException("bad")

    fc = FunctionCall(function=Function.from_callable(boom, name="boom"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    with pytest.raises(AgentRunException, match="bad"):
        fc.execute()

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    span = trace.data.spans[0]
    assert span.span_type == SpanType.TOOL
    assert span.status.status_code == SpanStatusCode.ERROR
    assert span.inputs == {"x": 1}
    assert span.outputs is None


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
async def test_agno_and_anthropic_autolog_single_trace(simple_agent, is_async):
    mlflow.agno.autolog()
    mlflow.anthropic.autolog()

    client = "AsyncAPIClient" if is_async else "SyncAPIClient"
    with patch(f"anthropic._base_client.{client}.post", return_value=_create_message("Paris")):
        if is_async:
            await simple_agent.arun("hi")
        else:
            simple_agent.run("hi")

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].name == "Agent.arun" if is_async else "Agent.run"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].name == "Claude.ainvoke" if is_async else "Claude.invoke"
    assert spans[2].span_type == SpanType.CHAT_MODEL
    assert spans[2].name == "AsyncMessages.create" if is_async else "Messages.create"


def test_is_agno_v2_detection():
    """Test that Agno version detection works correctly."""
    from mlflow.agno.autolog import _is_agno_v2

    # Mock agno module with V2 version
    with patch("mlflow.agno.autolog.agno") as mock_agno:
        mock_agno.__version__ = "2.0.0"
        assert _is_agno_v2() is True

        mock_agno.__version__ = "2.1.0"
        assert _is_agno_v2() is True

        mock_agno.__version__ = "1.9.9"
        assert _is_agno_v2() is False

        mock_agno.__version__ = "1.7.0"
        assert _is_agno_v2() is False


def test_otel_instrumentation_setup_for_v2():
    """Test that OTel instrumentation is set up for Agno V2."""
    from mlflow.agno.autolog import _setup_otel_instrumentation

    # Mock all the required imports for OTel setup
    mock_tracer_provider = Mock()
    mock_instrumentor = Mock()

    with (
        patch("mlflow.agno.autolog._is_agno_v2", return_value=True),
        patch("mlflow.agno.autolog.trace"),
        patch("mlflow.agno.autolog.OTLPSpanExporter") as mock_exporter,
        patch("mlflow.agno.autolog.TracerProvider", return_value=mock_tracer_provider),
        patch("mlflow.agno.autolog.BatchSpanProcessor"),
        patch("mlflow.agno.autolog.AgnoInstrumentor", return_value=mock_instrumentor),
        patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"),
        patch("mlflow.tracking.fluent._get_experiment_id", return_value="0"),
    ):
        # Reset the global state
        import mlflow.agno.autolog

        mlflow.agno.autolog._otel_instrumentation_setup = False
        mlflow.agno.autolog._agno_instrumentor = None

        _setup_otel_instrumentation()

        # Verify that OTLPSpanExporter was called with correct parameters
        mock_exporter.assert_called_once_with(
            endpoint="http://localhost:5000/v1/traces",
            headers={"x-mlflow-experiment-id": "0"},
        )

        # Verify that instrumentor was called
        mock_instrumentor.instrument.assert_called_once()

        # Verify that the instrumentor was stored
        assert mlflow.agno.autolog._agno_instrumentor is mock_instrumentor


def test_otel_uninstrumentation_for_v2():
    """Test that OTel uninstrumentation works for Agno V2."""
    from mlflow.agno.autolog import _uninstrument_otel

    mock_instrumentor = Mock()

    # Set the global state to simulate that instrumentation was set up
    import mlflow.agno.autolog

    mlflow.agno.autolog._otel_instrumentation_setup = True
    mlflow.agno.autolog._agno_instrumentor = mock_instrumentor

    _uninstrument_otel()

    # Verify that uninstrument was called
    mock_instrumentor.uninstrument.assert_called_once()

    # Verify that the flag was reset
    assert mlflow.agno.autolog._otel_instrumentation_setup is False


def test_otel_uninstrumentation_skipped_when_not_setup():
    """Test that uninstrumentation is skipped when OTel was never set up."""
    # Reset the global state
    import mlflow.agno.autolog
    from mlflow.agno.autolog import _uninstrument_otel

    mlflow.agno.autolog._otel_instrumentation_setup = False
    mlflow.agno.autolog._agno_instrumentor = None

    # Should not raise any errors, just skip uninstrumentation
    _uninstrument_otel()

    # Verify the flag is still False
    assert mlflow.agno.autolog._otel_instrumentation_setup is False


def test_autolog_v2_uses_otel():
    """Test that autolog uses OTel for Agno V2."""
    with (
        patch("mlflow.agno._is_agno_v2", return_value=True),
        patch("mlflow.agno._setup_otel_instrumentation") as mock_setup,
        patch("mlflow.agno._uninstrument_otel") as mock_uninstrument,
    ):
        # Test with log_traces=True
        mlflow.agno.autolog(log_traces=True)
        mock_setup.assert_called_once()
        mock_uninstrument.assert_not_called()

        # Reset mocks
        mock_setup.reset_mock()
        mock_uninstrument.reset_mock()

        # Test with disable=True
        mlflow.agno.autolog(disable=True)
        mock_uninstrument.assert_called_once()
        mock_setup.assert_not_called()

        # Reset mocks
        mock_setup.reset_mock()
        mock_uninstrument.reset_mock()

        # Test with log_traces=False
        mlflow.agno.autolog(log_traces=False)
        mock_uninstrument.assert_called_once()
        mock_setup.assert_not_called()


def test_cleanup_callback_registered_for_v2():
    """Test that cleanup callback is registered when OTel instrumentation is set up."""
    from mlflow.agno.autolog import _setup_otel_instrumentation

    mock_instrumentor = Mock()

    with (
        patch("mlflow.agno.autolog.trace"),
        patch("mlflow.agno.autolog.OTLPSpanExporter"),
        patch("mlflow.agno.autolog.TracerProvider"),
        patch("mlflow.agno.autolog.BatchSpanProcessor"),
        patch("mlflow.agno.autolog.AgnoInstrumentor", return_value=mock_instrumentor),
        patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"),
        patch("mlflow.tracking.fluent._get_experiment_id", return_value="0"),
        patch("mlflow.utils.autologging_utils.safety.register_cleanup_callback") as mock_register,
    ):
        # Reset the global state
        import mlflow.agno.autolog

        mlflow.agno.autolog._otel_instrumentation_setup = False
        mlflow.agno.autolog._agno_instrumentor = None

        _setup_otel_instrumentation()

        # Verify that cleanup callback was registered
        mock_register.assert_called_once()
        call_args = mock_register.call_args
        assert call_args[0][0] == "agno"  # Integration name
        # Verify the callback is our cleanup function
        assert callable(call_args[0][1])


def test_disable_true_calls_cleanup_via_decorator():
    """Test that disable=True triggers cleanup through the decorator's revert_patches."""
    from mlflow.utils.autologging_utils.safety import (
        _AUTOLOGGING_CLEANUP_CALLBACKS,
        register_cleanup_callback,
        revert_patches,
    )

    # Setup: Register a mock cleanup callback
    cleanup_called = []

    def mock_cleanup():
        cleanup_called.append(True)

    register_cleanup_callback("test_integration", mock_cleanup)

    # Verify callback is registered
    assert "test_integration" in _AUTOLOGGING_CLEANUP_CALLBACKS

    # Call revert_patches (this is what happens when disable=True)
    revert_patches("test_integration")

    # Verify cleanup was called
    assert len(cleanup_called) == 1

    # Verify callback was removed after revert
    assert "test_integration" not in _AUTOLOGGING_CLEANUP_CALLBACKS
