"""Tests for Codex CLI tracing functionality - verifying trace and span creation."""

import json
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

import mlflow
from mlflow.codex_cli.tracing import (
    extract_message_content,
    get_latest_session_file,
    parse_timestamp,
    process_session_file,
    read_session_file,
)
from mlflow.entities import SpanType


@pytest.fixture
def temp_sessions_dir(tmp_path):
    """Create a temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


@pytest.fixture
def mock_codex_config(temp_sessions_dir):
    """Mock Codex configuration."""
    with mock.patch("mlflow.codex_cli.tracing.get_sessions_dir", return_value=temp_sessions_dir):
        with mock.patch("mlflow.codex_cli.tracing.is_tracing_enabled", return_value=True):
            with mock.patch("mlflow.codex_cli.tracing.setup_mlflow"):
                yield temp_sessions_dir


def create_session_file(sessions_dir: Path, session_id: str, items: list[dict]) -> Path:
    """Helper to create a mock Codex session JSONL file."""
    session_file = sessions_dir / f"{session_id}.jsonl"
    with open(session_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    return session_file


def test_read_session_file(temp_sessions_dir):
    """Test reading a session JSONL file."""
    items = [
        {"role": "user", "content": "Hello", "timestamp": 1234567890},
        {"role": "assistant", "content": "Hi there!", "timestamp": 1234567891},
    ]

    session_file = create_session_file(temp_sessions_dir, "test-session", items)
    loaded_items = read_session_file(session_file)

    assert len(loaded_items) == 2
    assert loaded_items[0]["role"] == "user"
    assert loaded_items[1]["role"] == "assistant"


def test_extract_message_content_string():
    """Test extracting content from string message."""
    content = extract_message_content("Hello world")
    assert content == "Hello world"


def test_extract_message_content_dict_with_content():
    """Test extracting content from dict with content field."""
    message = {"content": "Test message"}
    content = extract_message_content(message)
    assert content == "Test message"


def test_extract_message_content_dict_with_text_array():
    """Test extracting content from dict with text array."""
    message = {
        "content": [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
    }
    content = extract_message_content(message)
    assert content == "Part 1\nPart 2"


def test_parse_timestamp():
    """Test timestamp parsing."""
    # Test with seconds
    ts_seconds = 1234567890
    ns = parse_timestamp(ts_seconds)
    assert ns == 1234567890000000000

    # Test with milliseconds
    ts_ms = 1234567890000
    ns = parse_timestamp(ts_ms)
    assert ns == 1234567890000000000


def test_get_latest_session_file(mock_codex_config):
    """Test finding the latest session file."""
    # Create multiple session files with different timestamps
    session1 = create_session_file(mock_codex_config, "session1", [{"role": "user", "content": "Test 1"}])
    time.sleep(0.01)
    session2 = create_session_file(mock_codex_config, "session2", [{"role": "user", "content": "Test 2"}])

    latest = get_latest_session_file()

    assert latest is not None
    assert latest == session2


def test_get_latest_session_file_no_sessions(mock_codex_config):
    """Test when no session files exist."""
    latest = get_latest_session_file()
    assert latest is None


@pytest.mark.usefixtures("mock_codex_config")
def test_process_session_file_creates_trace(temp_sessions_dir):
    """Test that processing a session file creates a trace with correct structure."""
    # Create a simple session with user message and assistant response
    session_items = [
        {
            "role": "user",
            "content": "Write a hello world function",
            "timestamp": 1700000000000,
        },
        {
            "role": "assistant",
            "content": "Here's a hello world function:\ndef hello():\n    print('Hello, World!')",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "timestamp": 1700000001000,
        },
    ]

    session_file = create_session_file(temp_sessions_dir, "test-session", session_items)

    # Process the session
    trace = process_session_file(session_file, "test-session")

    # Verify trace was created
    assert trace is not None
    assert trace.info.request_id is not None

    # Verify trace metadata
    assert "test-session" in str(trace.info.trace_metadata.get("mlflow.trace.session", ""))
    assert trace.info.trace_metadata.get("mlflow.trace.source") == "codex_cli"

    # Verify trace has spans
    assert trace.data is not None
    assert len(trace.data.spans) > 0

    # Find parent span
    parent_span = next((s for s in trace.data.spans if s.parent_id is None), None)
    assert parent_span is not None
    assert parent_span.name == "codex_cli_session"
    assert parent_span.span_type == SpanType.AGENT


@pytest.mark.usefixtures("mock_codex_config")
def test_process_session_file_creates_llm_spans(temp_sessions_dir):
    """Test that LLM spans are created correctly with proper attributes."""
    session_items = [
        {
            "role": "user",
            "content": "Explain async/await",
            "timestamp": 1700000000000,
        },
        {
            "role": "assistant",
            "content": "Async/await is a pattern for asynchronous programming...",
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 50,
            },
            "timestamp": 1700000002000,
        },
    ]

    session_file = create_session_file(temp_sessions_dir, "llm-test", session_items)
    trace = process_session_file(session_file, "llm-test")

    assert trace is not None

    # Find LLM spans
    llm_spans = [s for s in trace.data.spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) >= 1

    llm_span = llm_spans[0]
    assert "llm_call" in llm_span.name
    assert llm_span.inputs.get("model") == "gpt-4"
    assert llm_span.attributes.get("input_tokens") == 15
    assert llm_span.attributes.get("output_tokens") == 50
    assert "Async/await" in str(llm_span.outputs.get("response", ""))


@pytest.mark.usefixtures("mock_codex_config")
def test_process_session_file_creates_tool_spans(temp_sessions_dir):
    """Test that tool spans are created correctly from tool calls."""
    session_items = [
        {
            "role": "user",
            "content": "Read the config file",
            "timestamp": 1700000000000,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "config.json"}),
                    },
                }
            ],
            "timestamp": 1700000001000,
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"api_key": "test123", "timeout": 30}',
            "timestamp": 1700000002000,
        },
        {
            "role": "assistant",
            "content": "The config file contains an API key and timeout setting.",
            "model": "gpt-4",
            "timestamp": 1700000003000,
        },
    ]

    session_file = create_session_file(temp_sessions_dir, "tool-test", session_items)
    trace = process_session_file(session_file, "tool-test")

    assert trace is not None

    # Find tool spans
    tool_spans = [s for s in trace.data.spans if s.span_type == SpanType.TOOL]
    assert len(tool_spans) >= 1

    tool_span = tool_spans[0]
    assert "tool_read_file" in tool_span.name
    assert tool_span.attributes.get("tool_name") == "read_file"
    assert tool_span.attributes.get("tool_id") == "call_123"
    assert "config.json" in str(tool_span.inputs)
    assert "api_key" in str(tool_span.outputs.get("result", ""))


@pytest.mark.usefixtures("mock_codex_config")
def test_process_session_file_multiple_llm_calls(temp_sessions_dir):
    """Test session with multiple LLM calls creates multiple spans."""
    session_items = [
        {"role": "user", "content": "First question", "timestamp": 1700000000000},
        {
            "role": "assistant",
            "content": "First answer",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            "timestamp": 1700000001000,
        },
        {"role": "user", "content": "Second question", "timestamp": 1700000002000},
        {
            "role": "assistant",
            "content": "Second answer",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 8, "completion_tokens": 15},
            "timestamp": 1700000003000,
        },
    ]

    session_file = create_session_file(temp_sessions_dir, "multi-llm", session_items)
    trace = process_session_file(session_file, "multi-llm")

    assert trace is not None

    # Should have 2 LLM spans
    llm_spans = [s for s in trace.data.spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) == 2

    assert "llm_call_1" in llm_spans[0].name
    assert "llm_call_2" in llm_spans[1].name


@pytest.mark.usefixtures("mock_codex_config")
def test_process_session_file_complex_workflow(temp_sessions_dir):
    """Test complex session with multiple LLM calls and tool invocations."""
    session_items = [
        {
            "role": "user",
            "content": "Refactor the authentication module",
            "timestamp": 1700000000000,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "auth.py"}),
                    },
                }
            ],
            "timestamp": 1700000001000,
        },
        {
            "role": "tool",
            "tool_call_id": "call_read_1",
            "content": "def login(user, pwd): ...",
            "timestamp": 1700000002000,
        },
        {
            "role": "assistant",
            "content": "I'll refactor the authentication module to use async/await...",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "timestamp": 1700000003000,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_write_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": "auth.py", "content": "async def login..."}),
                    },
                }
            ],
            "timestamp": 1700000004000,
        },
        {
            "role": "tool",
            "tool_call_id": "call_write_1",
            "content": "File written successfully",
            "timestamp": 1700000005000,
        },
    ]

    session_file = create_session_file(temp_sessions_dir, "complex-workflow", session_items)
    trace = process_session_file(session_file, "complex-workflow")

    assert trace is not None

    # Verify we have parent span
    parent_span = next((s for s in trace.data.spans if s.parent_id is None), None)
    assert parent_span is not None
    assert parent_span.span_type == SpanType.AGENT

    # Verify we have LLM spans
    llm_spans = [s for s in trace.data.spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) >= 1

    # Verify we have tool spans
    tool_spans = [s for s in trace.data.spans if s.span_type == SpanType.TOOL]
    assert len(tool_spans) == 2

    # Verify tool span names
    tool_names = [s.name for s in tool_spans]
    assert any("read_file" in name for name in tool_names)
    assert any("write_file" in name for name in tool_names)

    # Verify all child spans reference parent
    child_spans = [s for s in trace.data.spans if s.parent_id is not None]
    assert all(s.parent_id == parent_span.span_id for s in child_spans)


@pytest.mark.usefixtures("mock_codex_config")
def test_process_session_file_empty_session(temp_sessions_dir):
    """Test that empty session files are handled gracefully."""
    session_file = create_session_file(temp_sessions_dir, "empty", [])
    trace = process_session_file(session_file, "empty")

    assert trace is None


@pytest.mark.usefixtures("mock_codex_config")
def test_process_session_file_no_user_message(temp_sessions_dir):
    """Test that sessions without user messages are handled gracefully."""
    session_items = [
        {
            "role": "assistant",
            "content": "Hello",
            "timestamp": 1700000000000,
        }
    ]

    session_file = create_session_file(temp_sessions_dir, "no-user", session_items)
    trace = process_session_file(session_file, "no-user")

    assert trace is None


def test_process_session_file_nonexistent(mock_codex_config):
    """Test processing nonexistent file returns None."""
    trace = process_session_file("/nonexistent/file.jsonl", "test")
    assert trace is None


@pytest.mark.usefixtures("mock_codex_config")
def test_trace_has_correct_timestamps(temp_sessions_dir):
    """Test that trace timestamps are correctly set from session data."""
    start_time = 1700000000000
    end_time = 1700000010000

    session_items = [
        {"role": "user", "content": "Test", "timestamp": start_time},
        {
            "role": "assistant",
            "content": "Response",
            "model": "gpt-4",
            "timestamp": end_time,
        },
    ]

    session_file = create_session_file(temp_sessions_dir, "timestamp-test", session_items)
    trace = process_session_file(session_file, "timestamp-test")

    assert trace is not None

    # Verify parent span timing
    parent_span = next((s for s in trace.data.spans if s.parent_id is None), None)
    assert parent_span is not None

    # Start time should match first item
    expected_start_ns = int(start_time * 1000000)  # Convert ms to ns
    assert parent_span.start_time_ns == expected_start_ns
