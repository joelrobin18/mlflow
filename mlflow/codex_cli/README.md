# MLflow Codex CLI Integration

This module provides automatic tracing integration between OpenAI Codex CLI and MLflow.

## Overview

Codex CLI is an open-source command-line coding agent from OpenAI that runs in your terminal and can read, modify, and run code on your machine. It uses GPT-4, GPT-5, or GPT-5-Codex models.

This integration automatically traces Codex CLI sessions to MLflow, capturing:
- User prompts
- LLM responses with token usage
- Tool calls and results
- Full conversation context

## Installation

```bash
# Install MLflow
pip install mlflow

# Install Codex CLI
npm i -g @openai/codex
# OR
brew install --cask codex
```

## Quick Start

### 1. Enable Tracing

```bash
# Enable with local storage
mlflow codex-cli enable

# Enable with custom tracking URI
mlflow codex-cli enable -u file://./mlruns

# Enable with Databricks
mlflow codex-cli enable -u databricks -e 123456789
```

### 2. Use Codex CLI Normally

```bash
codex "write a function to parse JSON files"
```

### 3. Process Session and View Trace

```bash
# Process the latest session
mlflow codex-cli trace-latest

# View traces in MLflow UI
mlflow ui
```

## How It Works

1. **Session Storage**: Codex CLI stores conversation sessions in `~/.codex/sessions` as JSONL files
2. **MLflow Processing**: MLflow reads these session files and creates structured traces
3. **Trace Creation**: Each session becomes a trace with hierarchical spans:
   - Parent span: `codex_cli_session` (AGENT type)
   - Child spans: Individual LLM calls and tool invocations

## Commands

### Enable Tracing

```bash
mlflow codex-cli enable [OPTIONS]
```

Options:
- `-u, --tracking-uri TEXT`: MLflow tracking URI
- `-e, --experiment-id TEXT`: MLflow experiment ID
- `-n, --experiment-name TEXT`: MLflow experiment name
- `-s, --sessions-dir TEXT`: Custom Codex sessions directory

### Check Status

```bash
mlflow codex-cli status
```

Shows current configuration and number of session files found.

### Process Latest Session

```bash
mlflow codex-cli trace-latest
```

Manually process the most recent Codex session file and create a trace.

### Disable Tracing

```bash
mlflow codex-cli disable
```

## Configuration

Configuration is stored in `~/.codex/mlflow_config.json`:

```json
{
  "environment": {
    "MLFLOW_CODEX_TRACING_ENABLED": "true",
    "MLFLOW_TRACKING_URI": "file:///Users/you/.codex/mlflow",
    "MLFLOW_EXPERIMENT_NAME": "codex-sessions"
  },
  "sessions_dir": "/Users/you/.codex/sessions"
}
```

## Trace Structure

Each Codex session creates a trace with the following structure:

```
📊 Trace: codex_cli_session
├── Inputs: {prompt: "user's initial request"}
├── Outputs: {response: "final assistant response", status: "completed"}
├── Metadata:
│   ├── session: <session_id>
│   ├── source: "codex_cli"
│   └── session_file: <path_to_session_file>
│
├── 💬 llm_call_1 (SpanType.LLM)
│   ├── Inputs: {model: "gpt-4"}
│   ├── Outputs: {response: "..."}
│   └── Attributes: {input_tokens: N, output_tokens: M}
│
├── 🔧 tool_read_file (SpanType.TOOL)
│   ├── Inputs: {path: "file.py"}
│   ├── Outputs: {result: "file contents"}
│   └── Attributes: {tool_name: "read_file", tool_id: "..."}
│
├── 💬 llm_call_2 (SpanType.LLM)
│   └── ...
│
└── 🔧 tool_write_file (SpanType.TOOL)
    └── ...
```

## Examples

### Basic Local Setup

```bash
# Enable tracing
mlflow codex-cli enable

# Use Codex
codex "create a FastAPI endpoint for user authentication"

# Process session
mlflow codex-cli trace-latest

# View in UI
mlflow ui
```

### Databricks Integration

```bash
# Enable with Databricks backend
mlflow codex-cli enable \
  -u databricks \
  -e 123456789

# Use Codex
codex "refactor this code to use async/await"

# Process and view in Databricks
mlflow codex-cli trace-latest
```

### Custom Sessions Directory

```bash
# If you've configured Codex with a custom sessions directory
mlflow codex-cli enable -s /custom/path/to/sessions

# Check status
mlflow codex-cli status
```

## Troubleshooting

### No Session Files Found

If you see "No session files found":

1. Check that Codex CLI is installed: `codex --version`
2. Run at least one Codex session: `codex "hello"`
3. Verify sessions directory exists: `ls ~/.codex/sessions`

### Tracing Not Working

1. Check status: `mlflow codex-cli status`
2. View logs: `cat ~/.codex/mlflow/codex_tracing.log`
3. Verify configuration: `cat ~/.codex/mlflow_config.json`

### Session Files Format

Codex session files are JSONL (JSON Lines) format. Each line contains a session item (message, tool call, etc.). If processing fails, check the session file format:

```bash
head ~/.codex/sessions/<session-file>.jsonl
```

## Limitations

- **Post-hoc Processing**: Traces are created after the session completes, not in real-time
- **Manual Trigger**: You need to run `mlflow codex-cli trace-latest` to process sessions
- **Latest Only**: The `trace-latest` command only processes the most recent session

## Future Enhancements

Potential improvements:
- Real-time tracing via MCP (Model Context Protocol) server integration
- Automatic background watcher for new session files
- Batch processing of multiple sessions
- Resume session tracing for continued conversations

## Related Links

- [Codex CLI GitHub](https://github.com/openai/codex)
- [Codex CLI Documentation](https://developers.openai.com/codex/)
- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html)

## Platform Support

- ✅ macOS
- ✅ Linux
- ⚠️ Windows (via WSL - experimental)

## Requirements

- Python 3.10+
- MLflow installed
- Codex CLI installed and authenticated
- ChatGPT Plus, Pro, Team, Edu, or Enterprise plan (or OpenAI API key)
