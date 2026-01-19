"""MLflow tracing plugin for Opencode.

This module provides an Opencode plugin that automatically traces conversations
to MLflow. The plugin hooks into Opencode's event system and creates MLflow
traces when sessions become idle (conversation completed).

Usage:
    Add "mlflow-opencode-tracing" to your opencode.json plugin list, or use:
    mlflow autolog opencode

The plugin can also be used programmatically:

```python
from mlflow.opencode.plugin import MLflowTracingPlugin

# Register the plugin with Opencode
plugin = MLflowTracingPlugin
```
"""

from typing import Any

from mlflow.opencode.tracing import (
    get_logger,
    is_tracing_enabled,
    process_session,
    setup_mlflow,
)


class MLflowTracingHooks:
    """MLflow tracing hooks for Opencode plugin system."""

    def __init__(self, client: Any, directory: str, worktree: str):
        """Initialize the MLflow tracing hooks.

        Args:
            client: Opencode SDK client
            directory: Current working directory
            worktree: Git worktree path
        """
        self.client = client
        self.directory = directory
        self.worktree = worktree
        self._setup_complete = False

    def _ensure_setup(self) -> bool:
        """Ensure MLflow is configured. Returns True if tracing is enabled."""
        if not self._setup_complete:
            if is_tracing_enabled():
                setup_mlflow()
            self._setup_complete = True
        return is_tracing_enabled()

    async def event(self, input: dict[str, Any]) -> None:
        """Handle Opencode events and create traces on session.idle.

        Args:
            input: Event data containing {"event": Event}
        """
        if not self._ensure_setup():
            return

        event = input.get("event", {})
        event_type = event.get("type")

        # Create trace when session becomes idle (conversation completed)
        if event_type == "session.idle":
            properties = event.get("properties", {})
            session_id = properties.get("sessionID")
            if session_id:
                await self._process_session(session_id)

    async def _process_session(self, session_id: str) -> None:
        """Process a completed session and create MLflow trace.

        Args:
            session_id: Opencode session ID
        """
        try:
            # Fetch session info
            session_response = await self._fetch_session(session_id)
            if session_response is None:
                return

            # Fetch session messages
            messages = await self._fetch_messages(session_id)
            if messages is None:
                return

            # Create MLflow trace
            process_session(session_id, session_response, messages)

        except Exception as e:
            get_logger().error("Failed to process session %s: %s", session_id, e)

    async def _fetch_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch session info from Opencode server."""
        try:
            response = self.client.session.get(id=session_id)
            if hasattr(response, "error") and response.error:
                get_logger().error("Failed to fetch session: %s", response.error)
                return None
            return response.data if hasattr(response, "data") else response
        except Exception as e:
            get_logger().error("Error fetching session: %s", e)
            return None

    async def _fetch_messages(self, session_id: str) -> list[dict[str, Any]] | None:
        """Fetch session messages from Opencode server."""
        try:
            response = self.client.session.messages(id=session_id)
            if hasattr(response, "error") and response.error:
                get_logger().error("Failed to fetch messages: %s", response.error)
                return None
            return response.data if hasattr(response, "data") else response
        except Exception as e:
            get_logger().error("Error fetching messages: %s", e)
            return None


async def MLflowTracingPlugin(ctx: dict[str, Any]) -> dict[str, Any]:
    """Opencode plugin factory function for MLflow tracing.

    This is the main entry point for the Opencode plugin system.
    It returns a hooks object that Opencode will use to capture events.

    Args:
        ctx: Plugin context containing:
            - client: Opencode SDK client
            - project: Project information
            - directory: Current directory
            - worktree: Git worktree path
            - serverUrl: Opencode server URL
            - $: Shell utility

    Returns:
        Hooks dictionary with event handlers
    """
    client = ctx.get("client")
    directory = ctx.get("directory", "")
    worktree = ctx.get("worktree", "")

    hooks = MLflowTracingHooks(client, directory, worktree)

    return {
        "event": hooks.event,
    }


# Export the plugin for Opencode to discover
__all__ = ["MLflowTracingPlugin", "MLflowTracingHooks"]
