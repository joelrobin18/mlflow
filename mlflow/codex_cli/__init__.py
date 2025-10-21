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

## Architecture

- **Sessions**: Codex CLI stores sessions in `~/.codex/sessions` as JSONL files
- **Processing**: MLflow reads session files and creates structured traces
- **Storage**: Traces saved to `./mlruns` by default (standard MLflow directory)
- **Viewing**: Run `mlflow ui` to see traces with LLM calls and tool invocations

## What Gets Traced

Each Codex session creates a hierarchical trace:
- Parent AGENT span: Overall session
- LLM child spans: Each GPT model call with token usage
- Tool child spans: File operations with inputs/outputs
- Full conversation context and metadata

For more information:
- Codex CLI: https://github.com/openai/codex
- MLflow Tracing: https://mlflow.org/docs/latest/llms/tracing/
"""
