"""
Comprehensive Codex Session Tracer that reads session JSONL files.

The session files contain complete conversation data including:
- User messages
- Assistant responses (with reasoning)
- Tool calls (Read, Write, Bash, etc.) with inputs and outputs
- Token usage information
- Model information
- Session metadata

Each session is stored in ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.fluent import start_span

_logger = logging.getLogger(__name__)


class CodexSessionTracer:
    """Comprehensive tracer for Codex session files."""

    def __init__(
        self,
        sessions_dir: Path | None = None,
        experiment_id: str | None = None,
        tracking_uri: str | None = None,
    ):
        """
        Initialize the session tracer.

        Args:
            sessions_dir: Path to sessions directory (default: ~/.codex/sessions)
            experiment_id: MLflow experiment ID
            tracking_uri: MLflow tracking URI
        """
        if sessions_dir:
            self.sessions_dir = Path(sessions_dir)
        else:
            self.sessions_dir = Path.home() / ".codex" / "sessions"

        self.experiment_id = experiment_id

        # Persistent tracking file to avoid duplicate traces
        self.codex_home = Path.home() / ".codex"
        self.tracking_file = self.codex_home / ".mlflow_session_turns"

        # Load previously processed turn counts per session
        # Maps session file path -> number of turns processed
        self.session_turn_counts: dict[str, int] = self._load_session_turn_counts()

        # Set tracking URI (default to localhost if not provided)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to http://localhost:5000 for traces
            mlflow.set_tracking_uri("http://localhost:5000")

        if experiment_id:
            try:
                mlflow.set_experiment(experiment_id=experiment_id)
                _logger.info(f"Using experiment ID: {experiment_id}")
            except Exception as e:
                _logger.warning(f"Could not set experiment: {e}")

        _logger.info(f"Watching sessions directory: {self.sessions_dir}")
        _logger.info(f"Previously tracked sessions: {len(self.session_turn_counts)}")

    def _load_session_turn_counts(self) -> dict[str, int]:
        """Load the mapping of session paths to processed turn counts."""
        if self.tracking_file.exists():
            try:
                turn_counts = {}
                with open(self.tracking_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Format: session_path|turn_count
                        parts = line.split("|")
                        if len(parts) == 2:
                            session_path, count_str = parts
                            try:
                                turn_counts[session_path] = int(count_str)
                            except ValueError:
                                _logger.warning(f"Invalid turn count in tracking file: {line}")
                        elif len(parts) == 1:
                            # Old format compatibility: treat as 1 turn processed
                            turn_counts[parts[0]] = 1
                return turn_counts
            except Exception as e:
                _logger.warning(f"Could not load tracking file: {e}")
        return {}

    def _save_session_turn_count(self, session_path: str, turn_count: int) -> None:
        """Save or update the turn count for a session."""
        try:
            # Ensure directory exists
            self.tracking_file.parent.mkdir(parents=True, exist_ok=True)

            # Read all current entries
            all_counts = self._load_session_turn_counts()

            # Update the count for this session
            all_counts[session_path] = turn_count

            # Write all entries back
            with open(self.tracking_file, "w") as f:
                for path, count in all_counts.items():
                    f.write(f"{path}|{count}\n")

        except Exception as e:
            _logger.warning(f"Could not save to tracking file: {e}")

    def process_session_file(self, session_file: Path) -> None:
        """Process a session file and create MLflow traces for any new turns."""
        session_path = str(session_file)

        # Wait for file to be stable (fully written by Codex)
        # Check file size twice with a delay to ensure it's not being written

        try:
            size1 = session_file.stat().st_size
            time.sleep(0.5)  # Wait 500ms
            size2 = session_file.stat().st_size

            if size1 != size2:
                # File is still being written, skip for now
                _logger.debug(f"File still being written: {session_file.name}, will retry")
                return
        except Exception as e:
            _logger.warning(f"Could not check file stability for {session_file.name}: {e}")
            return

        try:
            # Load all events from the session file
            with open(session_file) as f:
                events = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            events.append(event)
                        except json.JSONDecodeError as e:
                            _logger.error(f"Invalid JSON in {session_file}: {e}")

            if not events:
                return

            # Get number of turns already processed for this session
            turns_processed = self.session_turn_counts.get(session_path, 0)

            # Create traces for new turns only
            result = self._create_traces_for_new_turns(session_file, events, turns_processed)

            if result and result.get("turns_processed") > turns_processed:
                new_turn_count = result["turns_processed"]

                # Force trace to be committed to backend by accessing it
                if result.get("last_trace_id"):
                    try:
                        from mlflow.tracing.fluent import get_trace

                        get_trace(result["last_trace_id"])
                    except Exception as e:
                        _logger.debug(f"Could not force trace commit: {e}")
                        time.sleep(0.1)

                # Update turn count and save
                self.session_turn_counts[session_path] = new_turn_count
                self._save_session_turn_count(session_path, new_turn_count)
                _logger.info(
                    f"Updated session {session_file.name}: {new_turn_count} turns processed"
                )

        except Exception as e:
            _logger.error(f"Error processing {session_file}: {e}", exc_info=True)

    def _create_traces_for_new_turns(
        self, session_file: Path, events: list[dict[str, Any]], turns_already_processed: int
    ) -> dict[str, Any] | None:
        """Create MLflow traces for new turns only. Returns Dict with turns_processed
        and last_trace_id.
        """
        try:
            # Extract session metadata
            session_meta = None
            turn_context = None

            for event in events:
                if event.get("type") == "session_meta":
                    session_meta = event.get("payload", {})
                elif event.get("type") == "turn_context":
                    turn_context = event.get("payload", {})

            if not session_meta:
                _logger.warning(f"No session metadata found in {session_file}")
                return None

            session_id = session_meta.get("id", "unknown")

            # Group events by conversation turns
            turns = self._group_events_into_turns(events)
            total_turns = len(turns)

            if not turns:
                _logger.warning(f"No conversation turns found in session {session_id[:8]}")
                return None

            # Only process new turns (turns we haven't seen before)
            new_turns = turns[turns_already_processed:]

            if not new_turns:
                _logger.debug(
                    f"No new turns in session {session_id[:8]} ({total_turns} total, "
                    f"{turns_already_processed} already processed)"
                )
                return {"turns_processed": turns_already_processed, "last_trace_id": None}

            _logger.info(
                f"Processing {len(new_turns)} new turns for session {session_id[:8]} (total:"
                f"{total_turns}, previously: {turns_already_processed})"
            )

            # Create a trace for each new turn and capture the last trace ID
            last_trace_id = None
            for i, turn_events in enumerate(new_turns):
                turn_idx = turns_already_processed + i  # Global turn index
                _logger.info(
                    f"Processing turn {turn_idx + 1}/{total_turns} for session {session_id[:8]}..."
                )
                trace_id = self._create_turn_trace(
                    session_id, turn_idx, turn_events, session_meta, turn_context
                )
                if trace_id:
                    last_trace_id = trace_id

            _logger.info(f"✅ Completed {len(new_turns)} new traces for session {session_id[:8]}")

            return {"turns_processed": total_turns, "last_trace_id": last_trace_id}

        except Exception as e:
            _logger.error(f"Error creating traces from events: {e}", exc_info=True)
            return None

    def _group_events_into_turns(self, events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group events into conversation turns."""
        turns = []
        current_turn = []

        for event in events:
            event_type = event.get("type")
            payload_type = event.get("payload", {}).get("type") if "payload" in event else None

            # Skip metadata events
            if event_type in ["session_meta", "turn_context"]:
                continue

            # User message starts a new turn
            if payload_type == "user_message":
                # Append previous turn if exists
                if current_turn:
                    turns.append(current_turn)
                # Start new turn
                current_turn = [event]
            else:
                # Add to current turn (only if we're in a turn)
                if current_turn:
                    current_turn.append(event)

        # Add final turn if exists
        if current_turn:
            turns.append(current_turn)

        return turns

    def _create_turn_trace(
        self,
        session_id: str,
        turn_idx: int,
        turn_events: list[dict[str, Any]],
        session_meta: dict[str, Any],
        turn_context: dict[str, Any] | None,
    ) -> str | None:
        """Create a complete MLflow trace for a single conversation turn. Returns trace ID."""
        try:
            # Extract turn information
            user_message = None
            agent_messages = []
            reasoning_blocks = []
            tool_calls = []
            token_usage = None

            for event in turn_events:
                payload = event.get("payload", {})
                payload_type = payload.get("type")

                if payload_type == "user_message":
                    user_message = payload.get("message", "")
                elif payload_type == "agent_message":
                    agent_messages.append(payload.get("message", ""))
                elif payload_type == "agent_reasoning":
                    reasoning_blocks.append(payload.get("text", ""))
                elif payload_type == "custom_tool_call":
                    tool_calls.append(payload)
                elif payload_type == "function_call":
                    # Also capture function_call events (shell commands, etc.)
                    tool_calls.append(payload)
                elif payload_type == "token_count":
                    token_usage = payload.get("info")

            if not user_message:
                return None

            # Create root span for the turn
            turn_name = f"Codex Turn {turn_idx + 1}"
            timestamp = turn_events[0].get("timestamp", datetime.now().isoformat())
            trace_id = None

            with start_span(
                name=turn_name,
                span_type=SpanType.AGENT,
            ) as turn_span:
                # Capture trace ID from span
                trace_id = turn_span.request_id
                # Set turn inputs
                turn_span.set_inputs(
                    {
                        "prompt": user_message,
                        "session_id": session_id,
                        "turn_index": turn_idx,
                    }
                )

                # Set turn attributes
                turn_span.set_attribute("session_id", session_id)
                turn_span.set_attribute("turn_index", turn_idx)
                turn_span.set_attribute("timestamp", timestamp)

                if session_meta:
                    turn_span.set_attribute("cwd", session_meta.get("cwd", ""))
                    turn_span.set_attribute("cli_version", session_meta.get("cli_version", ""))
                    turn_span.set_attribute("originator", session_meta.get("originator", ""))

                if turn_context:
                    turn_span.set_attribute("model", turn_context.get("model", ""))
                    turn_span.set_attribute(
                        "approval_policy", turn_context.get("approval_policy", "")
                    )
                    turn_span.set_attribute(
                        "sandbox_mode", turn_context.get("sandbox_policy", {}).get("mode", "")
                    )

                # Add token usage
                if token_usage:
                    total_usage = token_usage.get("total_token_usage", {})
                    last_usage = token_usage.get("last_token_usage", {})

                    turn_span.set_attribute("input_tokens", last_usage.get("input_tokens", 0))
                    turn_span.set_attribute("output_tokens", last_usage.get("output_tokens", 0))
                    turn_span.set_attribute(
                        "cached_tokens", last_usage.get("cached_input_tokens", 0)
                    )
                    turn_span.set_attribute(
                        "reasoning_tokens", last_usage.get("reasoning_output_tokens", 0)
                    )
                    turn_span.set_attribute("total_tokens", last_usage.get("total_tokens", 0))

                    turn_span.set_attribute(
                        "cumulative_input_tokens", total_usage.get("input_tokens", 0)
                    )
                    turn_span.set_attribute(
                        "cumulative_output_tokens", total_usage.get("output_tokens", 0)
                    )

                # Create reasoning spans
                for idx, reasoning in enumerate(reasoning_blocks):
                    with start_span(
                        name=f"Reasoning Block {idx + 1}",
                        span_type=SpanType.CHAT_MODEL,
                    ) as reasoning_span:
                        reasoning_span.set_inputs({"context": user_message[:200]})
                        reasoning_span.set_outputs({"reasoning": reasoning})

                # Create tool call spans
                tool_outputs = []
                for tool_call in tool_calls:
                    tool_type = tool_call.get("type")
                    tool_name = tool_call.get("name", "unknown")
                    call_id = tool_call.get("call_id", "")

                    # Handle different tool call formats
                    if tool_type == "custom_tool_call":
                        # Custom tool calls have input and status fields
                        tool_input = tool_call.get("input", "")
                        status = tool_call.get("status", "unknown")
                        output_type = "custom_tool_call_output"
                    else:
                        # function_call has arguments field
                        tool_input = tool_call.get("arguments", "")
                        status = "completed"  # function_calls don't have explicit status
                        output_type = "function_call_output"

                    # Find corresponding output
                    tool_output = None
                    for event in turn_events:
                        payload = event.get("payload", {})
                        if payload.get("type") == output_type and payload.get("call_id") == call_id:
                            tool_output = payload.get("output", "")
                            break

                    with start_span(
                        name=f"Tool: {tool_name}",
                        span_type=SpanType.TOOL,
                    ) as tool_span:
                        tool_span.set_inputs(
                            {
                                "tool_name": tool_name,
                                "input": tool_input,
                            }
                        )

                        if tool_output:
                            try:
                                # Try to parse JSON output
                                output_data = (
                                    json.loads(tool_output)
                                    if isinstance(tool_output, str)
                                    else tool_output
                                )
                                tool_span.set_outputs(output_data)
                            except json.JSONDecodeError:
                                tool_span.set_outputs({"output": str(tool_output)})

                        tool_span.set_attribute("call_id", call_id)
                        tool_span.set_attribute("status", status)
                        tool_span.set_attribute("tool_type", tool_type)

                    tool_outputs.append(
                        {
                            "tool": tool_name,
                            "status": status,
                        }
                    )

                # Set final turn outputs
                final_response = agent_messages[-1] if agent_messages else "No response"
                turn_span.set_outputs(
                    {
                        "response": final_response,
                        "all_responses": agent_messages,
                        "tool_calls": tool_outputs,
                        "num_reasoning_blocks": len(reasoning_blocks),
                    }
                )

            _logger.info(f"✅ Created trace for turn {turn_idx + 1} of session {session_id[:8]}")

            return trace_id

        except Exception as e:
            _logger.error(
                f"Error creating turn trace for turn {turn_idx + 1}, session {session_id[:8]}: {e}",
                exc_info=True,
            )
            raise  # Re-raise to see full stack trace

    def watch(self, poll_interval: float = 2.0) -> None:
        """
        Watch the sessions directory for new session files.

        Args:
            poll_interval: How often to check for new files (seconds)
        """
        _logger.info("Starting session watcher...")
        _logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

        if not self.sessions_dir.exists():
            _logger.warning(f"Sessions directory not found: {self.sessions_dir}")
            _logger.info("Waiting for directory to be created...")
            _logger.info("Run 'codex' to create sessions.")

        try:
            while True:
                if self.sessions_dir.exists():
                    # Find all session files
                    session_files = sorted(
                        self.sessions_dir.rglob("rollout-*.jsonl"),
                        key=lambda p: p.stat().st_mtime,
                    )

                    for session_file in session_files:
                        # Process all files - they'll check internally if there are new turns
                        # Wait a bit to ensure file is complete
                        time.sleep(1)
                        self.process_session_file(session_file)

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            _logger.info("\n✅ Stopping tracer...")
            _logger.info(f"Tracked {len(self.session_turn_counts)} sessions total")

    def process_existing(self) -> None:
        """Process all existing session files in the sessions directory."""
        if not self.sessions_dir.exists():
            _logger.warning(f"Sessions directory not found: {self.sessions_dir}")
            return

        _logger.info("Processing existing sessions...")

        session_files = sorted(
            self.sessions_dir.rglob("rollout-*.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )

        count = 0
        for session_file in session_files:
            self.process_session_file(session_file)
            count += 1

        _logger.info(f"✅ Processed {count} existing sessions")


class SessionState:
    """Track the state of an active session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] | None = None


def run_session_tracer(
    experiment_id: str | None = None,
    tracking_uri: str | None = None,
    sessions_dir: str | None = None,
    process_existing: bool = False,
):
    """
    Run the comprehensive Codex session tracer.

    Args:
        experiment_id: MLflow experiment ID
        tracking_uri: MLflow tracking URI
        sessions_dir: Path to sessions directory
        process_existing: Whether to process existing sessions first
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create tracer
    sessions_path = Path(sessions_dir) if sessions_dir else None
    tracer = CodexSessionTracer(
        sessions_dir=sessions_path,
        experiment_id=experiment_id,
        tracking_uri=tracking_uri,
    )

    _logger.info("=" * 70)
    _logger.info("🚀 Codex Comprehensive Session Tracer for MLflow")
    _logger.info("=" * 70)
    _logger.info(f"\nSessions directory: {tracer.sessions_dir}")
    _logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    _logger.info(f"Experiment ID: {experiment_id or 'default'}")
    _logger.info("\n" + "=" * 70)

    # Process existing sessions if requested
    if process_existing:
        tracer.process_existing()
        _logger.info()

    # Watch for new sessions
    _logger.info("\n👀 Watching for new Codex sessions...")
    _logger.info("Run 'codex' in another terminal to create traces.")
    _logger.info("Press Ctrl+C to stop\n")
    _logger.info("=" * 70 + "\n")

    try:
        tracer.watch()
    except KeyboardInterrupt:
        _logger.info("\n\n✅ Tracer stopped.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Codex Session Tracer")
    parser.add_argument("--experiment-id", type=str, help="MLflow experiment ID")
    parser.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--sessions-dir", type=str, help="Path to sessions directory")
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="Process existing sessions before watching",
    )

    args = parser.parse_args()

    run_session_tracer(
        experiment_id=args.experiment_id,
        tracking_uri=args.tracking_uri,
        sessions_dir=args.sessions_dir,
        process_existing=args.process_existing,
    )
