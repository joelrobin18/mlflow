# MLflow Claude Code / Codex Integration

This module provides automatic tracing integration between Claude Code (Codex) and MLflow.

## Module Structure

- **`config.py`** - Configuration management (settings files, environment variables)
- **`hooks.py`** - Claude Code hook setup and management
- **`cli.py`** - MLflow CLI commands (`mlflow autolog claude` or `mlflow autolog codex`)
- **`tracing.py`** - Core tracing logic and processors
- **`hooks/`** - Hook implementation handlers

## Installation

```bash
pip install mlflow
```

## Usage

Set up Claude Code / Codex tracing in any project directory:

```bash
# Set up tracing in current directory (both commands are equivalent)
mlflow autolog claude
# OR
mlflow autolog codex

# Set up tracing in specific directory
mlflow autolog codex ~/my-project

# Set up with custom tracking URI
mlflow autolog codex -u file://./custom-mlruns
mlflow autolog codex -u sqlite:///mlflow.db

# Set up with Databricks
mlflow autolog codex -u databricks -e 123456789

# Check status
mlflow autolog codex --status

# Disable tracing
mlflow autolog codex --disable
```

## How it Works

1. **Setup**: The `mlflow autolog codex` (or `mlflow autolog claude`) command configures Claude Code hooks in a `.claude/settings.json` file
2. **Automatic Tracing**: When you use the `claude` command in the configured directory, your conversations are automatically traced to MLflow
3. **View Traces**: Use `mlflow ui` to view your conversation traces with detailed LLM calls and tool usage

## Configuration

The setup creates two types of configuration:

### Claude Code Hooks

- **PostToolUse**: Captures tool usage during conversations
- **Stop**: Processes complete conversations into MLflow traces

### Environment Variables

- `MLFLOW_CLAUDE_TRACING_ENABLED=true`: Enables tracing
- `MLFLOW_TRACKING_URI`: Where to store traces (defaults to local `.claude/mlflow/runs`)
- `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`: Which experiment to use

## Examples

### Basic Local Setup

```bash
mlflow autolog codex
cd .
claude "help me write a function"
mlflow ui
```

### Databricks Integration

```bash
mlflow autolog codex -u databricks -e 123456789
claude "analyze this data"
# View traces in Databricks
```

### Custom Project Setup

```bash
mlflow autolog codex ~/my-ai-project -u sqlite:///mlflow.db -n "My AI Project"
cd ~/my-ai-project
claude "refactor this code"
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Troubleshooting

### Check Status

```bash
mlflow autolog codex --status
```

### Disable Tracing

```bash
mlflow autolog codex --disable
```

### View Raw Configuration

The configuration is stored in `.claude/settings.json`:

```bash
cat .claude/settings.json
```

## Requirements

- Python 3.10+ (required by MLflow)
- MLflow installed (`pip install mlflow`)
- Claude Code CLI installed
