"""OpenAI Codex CLI integration for MLflow.

This module provides automatic tracing integration between OpenAI Codex CLI and MLflow.

Codex CLI is an open-source command-line coding agent from OpenAI that runs in your terminal
and can read, modify, and run code on your machine. It uses GPT-4, GPT-5, or GPT-5-Codex models.

## Quick Start

```bash
# 1. Enable tracing
mlflow autolog codex

# 2. Use Codex normally
codex "write a function to parse JSON"

# 3. Process the session
mlflow autolog codex --trace-latest

# 4. View traces
mlflow ui
```

## All Commands

- `mlflow autolog codex` - Enable tracing (default)
- `mlflow autolog codex --status` - Check current status
- `mlflow autolog codex --trace-latest` - Process latest session
- `mlflow autolog codex --disable` - Disable tracing

Architecture:
    - Codex CLI stores sessions in ~/.codex/sessions as JSONL files
    - MLflow processes these session files to create traces
    - Each trace captures LLM calls, tool invocations, and conversation context
    - Traces are automatically created after each Codex session completes

For more information about Codex CLI:
    - GitHub: https://github.com/openai/codex
    - Docs: https://developers.openai.com/codex/
"""
