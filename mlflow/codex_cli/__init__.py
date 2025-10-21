"""OpenAI Codex CLI integration for MLflow.

This module provides automatic tracing integration between OpenAI Codex CLI and MLflow.

Codex CLI is an open-source command-line coding agent from OpenAI that runs in your terminal
and can read, modify, and run code on your machine. It uses GPT-4, GPT-5, or GPT-5-Codex models.

Usage:
    mlflow autolog codex [options]

After setup, use the regular 'codex' command and traces will be automatically captured
from session files stored in ~/.codex/sessions.

Example:

```bash
# Enable Codex tracing
mlflow autolog codex

# Use Codex normally
codex "write a function to parse JSON"

# View traces in MLflow UI
mlflow ui
```

Architecture:
    - Codex CLI stores sessions in ~/.codex/sessions as JSONL files
    - MLflow processes these session files to create traces
    - Each trace captures LLM calls, tool invocations, and conversation context
    - Traces are automatically created after each Codex session completes

For more information about Codex CLI:
    - GitHub: https://github.com/openai/codex
    - Docs: https://developers.openai.com/codex/
"""
