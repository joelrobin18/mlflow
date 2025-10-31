"""
The ``mlflow.codex`` module provides integration for tracing Codex CLI sessions with MLflow.

Codex CLI is an AI coding agent from OpenAI that runs locally on your computer.
This integration enables automatic tracing of LLM calls, tool calls, and agentic workflows
when using Codex CLI.

This implementation uses Codex's history.jsonl file for tracing, which is much more
reliable than OpenTelemetry.
"""

from mlflow.codex.autolog import autolog
from mlflow.codex.constant import FLAVOR_NAME

__all__ = ["autolog", "FLAVOR_NAME"]
