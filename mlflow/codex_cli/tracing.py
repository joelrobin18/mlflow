"""MLflow tracing integration for Codex CLI interactions."""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.codex_cli.config import (
    get_env_var,
    get_sessions_dir,
    is_tracing_enabled,
)
from mlflow.entities import SpanType
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking.fluent import _get_trace_exporter

# ============================================================================
# CONSTANTS
# ============================================================================

NANOSECONDS_PER_MS = 1e6
NANOSECONDS_PER_S = 1e9
MAX_PREVIEW_LENGTH = 1000

# Codex session item types
ITEM_TYPE_USER = "user"
ITEM_TYPE_ASSISTANT = "assistant"
ITEM_TYPE_TOOL_CALL = "tool_call"
ITEM_TYPE_TOOL_OUTPUT = "tool_output"

# Custom logging level
CODEX_TRACING_LEVEL = logging.WARNING - 5


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging() -> logging.Logger:
    """Set up logging for Codex tracing."""
    log_dir = Path.home() / ".codex" / "mlflow"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    log_file = log_dir / "codex_tracing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logging.addLevelName(CODEX_TRACING_LEVEL, "CODEX_TRACING")
    logger.setLevel(CODEX_TRACING_LEVEL)
    logger.propagate = False

    return logger


_MODULE_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Get the configured module logger."""
    global _MODULE_LOGGER
    if _MODULE_LOGGER is None:
        _MODULE_LOGGER = setup_logging()
    return _MODULE_LOGGER


# ============================================================================
# MLFLOW SETUP
# ============================================================================


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and experiment."""
    if not is_tracing_enabled():
        return

    mlflow.set_tracking_uri(get_env_var(MLFLOW_TRACKING_URI.name))

    experiment_id = get_env_var(MLFLOW_EXPERIMENT_ID.name)
    experiment_name = get_env_var(MLFLOW_EXPERIMENT_NAME.name)

    try:
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
        elif experiment_name:
            mlflow.set_experiment(experiment_name)
    except Exception as e:
        get_logger().warning("Failed to set experiment: %s", e)


# ============================================================================
# SESSION FILE PROCESSING
# ============================================================================


def read_session_file(session_path: str | Path) -> list[dict[str, Any]]:
    """Read and parse a Codex session JSONL file.

    Args:
        session_path: Path to the session file

    Returns:
        List of session items
    """
    items = []
    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    get_logger().warning("Failed to parse session line: %s", e)
    return items


def extract_message_content(message: dict[str, Any]) -> str:
    """Extract text content from a message.

    Args:
        message: Message dictionary

    Returns:
        Text content
    """
    if isinstance(message, str):
        return message

    if isinstance(message, dict):
        # Handle different message formats
        if "content" in message:
            content = message["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "\n".join(text_parts)

        if "text" in message:
            return message["text"]

    return str(message)


def parse_timestamp(ts: Any) -> int:
    """Parse timestamp to nanoseconds.

    Args:
        ts: Timestamp in various formats

    Returns:
        Nanoseconds since epoch
    """
    if isinstance(ts, (int, float)):
        # Assume milliseconds if less than a typical Unix timestamp in seconds
        if ts < 1e10:
            return int(ts * NANOSECONDS_PER_S)
        if ts < 1e13:
            return int(ts * NANOSECONDS_PER_MS)
        return int(ts)

    # Current time as fallback
    return int(time.time() * NANOSECONDS_PER_S)


# ============================================================================
# TRACE CREATION
# ============================================================================


def process_session_file(
    session_path: str | Path, session_id: str | None = None
) -> mlflow.entities.Trace | None:
    """Process a Codex session file and create an MLflow trace.

    Args:
        session_path: Path to the Codex session file
        session_id: Optional session identifier

    Returns:
        MLflow trace object if successful, None otherwise
    """
    try:
        session_path = Path(session_path)
        if not session_path.exists():
            get_logger().warning("Session file not found: %s", session_path)
            return None

        items = read_session_file(session_path)
        if not items:
            get_logger().warning("Empty session file: %s", session_path)
            return None

        if not session_id:
            # Use filename as session ID
            session_id = session_path.stem

        get_logger().log(CODEX_TRACING_LEVEL, "Processing session: %s", session_id)

        # Find first user message for trace input
        first_user_message = None
        for item in items:
            if item.get("role") == "user" or item.get("type") == ITEM_TYPE_USER:
                first_user_message = extract_message_content(item)
                break

        if not first_user_message:
            get_logger().warning("No user message found in session")
            return None

        # Create parent span for the conversation
        start_time = parse_timestamp(items[0].get("timestamp", time.time()))
        parent_span = mlflow.start_span_no_context(
            name="codex_cli_session",
            inputs={"prompt": first_user_message},
            start_time_ns=start_time,
            span_type=SpanType.AGENT,
        )

        # Process session items and create spans
        _process_session_items(parent_span, items, start_time)

        # Find final response for trace output
        final_response = _find_final_response(items)

        # Set trace metadata and previews
        try:
            with InMemoryTraceManager.get_instance().get_trace(
                parent_span.trace_id
            ) as in_memory_trace:
                if first_user_message:
                    in_memory_trace.info.request_preview = first_user_message[:MAX_PREVIEW_LENGTH]
                if final_response:
                    in_memory_trace.info.response_preview = final_response[:MAX_PREVIEW_LENGTH]
                in_memory_trace.info.trace_metadata = {
                    **in_memory_trace.info.trace_metadata,
                    TraceMetadataKey.TRACE_SESSION: session_id,
                    TraceMetadataKey.TRACE_USER: os.environ.get("USER", ""),
                    "mlflow.trace.source": "codex_cli",
                    "mlflow.trace.session_file": str(session_path),
                }
        except Exception as e:
            get_logger().warning("Failed to update trace metadata: %s", e)

        # End parent span
        end_time = parse_timestamp(items[-1].get("timestamp", time.time()))
        if end_time <= start_time:
            end_time = start_time + int(10 * NANOSECONDS_PER_S)

        parent_span.set_outputs(
            {"response": final_response or "Session completed", "status": "completed"}
        )
        parent_span.end(end_time_ns=end_time)

        # Flush async logging if enabled to ensure trace is exported to backend
        try:
            if hasattr(_get_trace_exporter(), "_async_queue"):
                mlflow.flush_trace_async_logging()
        except Exception as e:
            get_logger().debug("Failed to flush trace async logging: %s", e)

        get_logger().log(CODEX_TRACING_LEVEL, "Created trace: %s", parent_span.trace_id)

        # Retrieve trace from backend to verify it was persisted
        try:
            trace = mlflow.get_trace(parent_span.trace_id)
            if trace is None:
                get_logger().error(
                    "Trace was created but could not be retrieved from backend. "
                    "Check MLflow tracking URI configuration."
                )
            return trace
        except Exception as e:
            get_logger().error(
                "Failed to retrieve trace from backend: %s. "
                "The trace may not be visible in the MLflow UI. "
                "Check your tracking URI configuration.",
                e,
            )
            return None

    except Exception as e:
        get_logger().error("Error processing session file: %s", e, exc_info=True)
        return None


def _process_session_items(parent_span, items: list[dict[str, Any]], base_time: int) -> None:
    """Process session items and create spans for LLM calls and tools.

    Args:
        parent_span: Parent span for the session
        items: List of session items
        base_time: Base timestamp in nanoseconds
    """
    llm_call_count = 0
    tool_call_map = {}  # Map tool call IDs to their data

    for i, item in enumerate(items):
        item_type = item.get("type") or item.get("role")
        timestamp = parse_timestamp(item.get("timestamp", base_time + i * 100 * NANOSECONDS_PER_MS))

        if item_type == "assistant" or item_type == ITEM_TYPE_ASSISTANT:
            # LLM response
            llm_call_count += 1
            content = extract_message_content(item)

            # Check for tool calls in the message
            tool_calls = item.get("tool_calls", [])

            if content and not tool_calls:
                # Pure LLM response without tool calls
                _create_llm_span(
                    parent_span,
                    llm_call_count,
                    content,
                    item,
                    timestamp,
                )
            elif tool_calls:
                # Assistant requested tool calls
                for tool_call in tool_calls:
                    tool_id = tool_call.get("id")
                    if tool_id:
                        tool_call_map[tool_id] = {
                            "call": tool_call,
                            "timestamp": timestamp,
                        }

        elif item_type == ITEM_TYPE_TOOL_CALL:
            # Alternative format for tool calls
            tool_id = item.get("id")
            if tool_id:
                tool_call_map[tool_id] = {
                    "call": item,
                    "timestamp": timestamp,
                }

        elif item_type == ITEM_TYPE_TOOL_OUTPUT or item_type == "tool":
            # Tool execution result
            tool_id = item.get("tool_call_id") or item.get("id")
            if tool_id and tool_id in tool_call_map:
                tool_data = tool_call_map[tool_id]
                _create_tool_span(
                    parent_span,
                    tool_data["call"],
                    item,
                    tool_data["timestamp"],
                    timestamp,
                )


def _create_llm_span(
    parent_span,
    call_number: int,
    content: str,
    item: dict[str, Any],
    timestamp: int,
) -> None:
    """Create a span for an LLM call.

    Args:
        parent_span: Parent span
        call_number: LLM call number
        content: Response content
        item: Session item
        timestamp: Timestamp in nanoseconds
    """
    model = item.get("model", "unknown")
    usage = item.get("usage", {})

    llm_span = mlflow.start_span_no_context(
        name=f"llm_call_{call_number}",
        parent_span=parent_span,
        span_type=SpanType.LLM,
        start_time_ns=timestamp,
        inputs={"model": model},
        attributes={
            "model": model,
            "input_tokens": usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
        },
    )

    llm_span.set_outputs({"response": content})
    # Estimate duration (500ms default)
    duration = int(500 * NANOSECONDS_PER_MS)
    llm_span.end(end_time_ns=timestamp + duration)


def _create_tool_span(
    parent_span,
    tool_call: dict[str, Any],
    tool_output: dict[str, Any],
    start_time: int,
    end_time: int,
) -> None:
    """Create a span for a tool call.

    Args:
        parent_span: Parent span
        tool_call: Tool call data
        tool_output: Tool output data
        start_time: Start timestamp in nanoseconds
        end_time: End timestamp in nanoseconds
    """
    # Extract tool information
    tool_name = tool_call.get("function", {}).get("name") or tool_call.get("name", "unknown_tool")
    tool_args = tool_call.get("function", {}).get("arguments") or tool_call.get("arguments", {})

    # Parse arguments if they're a JSON string
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            pass

    # Extract tool output
    output_content = tool_output.get("content") or tool_output.get("output", "")

    tool_span = mlflow.start_span_no_context(
        name=f"tool_{tool_name}",
        parent_span=parent_span,
        span_type=SpanType.TOOL,
        start_time_ns=start_time,
        inputs=tool_args if isinstance(tool_args, dict) else {"arguments": tool_args},
        attributes={
            "tool_name": tool_name,
            "tool_id": tool_call.get("id", ""),
        },
    )

    tool_span.set_outputs({"result": output_content})
    tool_span.end(end_time_ns=end_time)


def _find_final_response(items: list[dict[str, Any]]) -> str | None:
    """Find the final assistant response in the session.

    Args:
        items: List of session items

    Returns:
        Final response text or None
    """
    for item in reversed(items):
        item_type = item.get("type") or item.get("role")
        if item_type == "assistant" or item_type == ITEM_TYPE_ASSISTANT:
            content = extract_message_content(item)
            if content and content.strip():
                return content
    return None


# ============================================================================
# SESSION MONITORING
# ============================================================================


def get_latest_session_file() -> Path | None:
    """Get the most recently modified session file.

    Returns:
        Path to latest session file or None
    """
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return None

    # Find all .jsonl files
    session_files = list(sessions_dir.glob("*.jsonl"))
    if not session_files:
        return None

    # Return most recently modified
    return max(session_files, key=lambda p: p.stat().st_mtime)


def process_latest_session() -> mlflow.entities.Trace | None:
    """Process the most recent Codex session file.

    Returns:
        MLflow trace if successful, None otherwise
    """
    if not is_tracing_enabled():
        get_logger().debug("Tracing not enabled")
        return None

    setup_mlflow()

    latest_file = get_latest_session_file()
    if not latest_file:
        get_logger().warning("No session files found")
        return None

    return process_session_file(latest_file)
