"""Claude Code / Codex integration for MLflow.

This module provides automatic tracing of Claude Code (Codex) conversations to MLflow.

Usage:
    mlflow autolog codex [directory] [options]
    # OR
    mlflow autolog claude [directory] [options]

After setup, use the regular 'claude' command and traces will be automatically captured.
All LLM calls, tool invocations, and conversation context are traced to MLflow.

To enable tracing for the Claude Agent SDK, use `mlflow.anthropic.autolog()`.

Example:

```python
import mlflow.anthropic
from claude_agent_sdk import ClaudeSDKClient

mlflow.anthropic.autolog()

async with ClaudeSDKClient() as client:
    await client.query("What is the capital of France?")

    async for message in client.receive_response():
        print(message)
```
"""
