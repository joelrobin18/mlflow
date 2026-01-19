"""Hook handlers for Opencode integration with MLflow.

This module provides CLI-invocable hook handlers that can be called by
Opencode's experimental.hook.session_completed configuration.

Usage:
    python -m mlflow.opencode.hooks session_completed

The hooks read session data from stdin (JSON format) and create MLflow traces.
"""

import json
import sys
from typing import Any

from mlflow.opencode.tracing import (
    OPENCODE_TRACING_LEVEL,
    get_logger,
    is_tracing_enabled,
    process_session,
    setup_mlflow,
)


def read_hook_input() -> dict[str, Any]:
    """Read JSON input from stdin for hook processing.

    Returns:
        Parsed JSON data from stdin
    """
    try:
        input_data = sys.stdin.read()
        if not input_data.strip():
            return {}
        return json.loads(input_data)
    except json.JSONDecodeError as e:
        get_logger().error("Failed to parse hook input: %s", e)
        return {}


def session_completed_handler() -> None:
    """CLI handler for session_completed hook.

    This handler is invoked by Opencode when a session completes.
    It reads session data from stdin and creates an MLflow trace.

    Expected stdin format:
    {
        "sessionID": "...",
        "session": { session info },
        "messages": [ message list ]
    }
    """
    if not is_tracing_enabled():
        return

    try:
        setup_mlflow()
        hook_data = read_hook_input()

        session_id = hook_data.get("sessionID")
        session_info = hook_data.get("session", {})
        messages = hook_data.get("messages", [])

        if not session_id:
            get_logger().warning("No session ID in hook data")
            return

        get_logger().log(
            OPENCODE_TRACING_LEVEL, "Processing session_completed hook for session: %s", session_id
        )

        trace = process_session(session_id, session_info, messages)

        if trace is not None:
            get_logger().log(
                OPENCODE_TRACING_LEVEL,
                "Created trace %s for session %s",
                trace.info.trace_id,
                session_id,
            )

    except Exception as e:
        get_logger().error("Error in session_completed hook: %s", e, exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for hook CLI invocation."""
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python -m mlflow.opencode.hooks <hook_name>\n")
        sys.exit(1)

    hook_name = sys.argv[1]

    if hook_name == "session_completed":
        session_completed_handler()
    else:
        sys.stderr.write(f"Unknown hook: {hook_name}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
