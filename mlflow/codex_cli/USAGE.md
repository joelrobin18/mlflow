# Codex CLI Auto-Tracing - Quick Start

## Command: `mlflow autolog codex`

This single command handles all Codex CLI tracing operations through flags.

### Enable Tracing (Default)

```bash
# Basic setup
mlflow autolog codex

# With custom tracking URI
mlflow autolog codex -u file://./mlruns

# With Databricks
mlflow autolog codex -u databricks -e 123456789

# With custom sessions directory
mlflow autolog codex -s ~/my-codex-sessions
```

### Check Status

```bash
mlflow autolog codex --status
```

### Process Latest Session

```bash
mlflow autolog codex --trace-latest
```

### Disable Tracing

```bash
mlflow autolog codex --disable
```

## Complete Workflow

```bash
# 1. Enable tracing
mlflow autolog codex

# 2. Use Codex CLI normally
codex "write a function to parse JSON"
codex "add error handling"
codex "write unit tests"

# 3. Process each session after it completes
mlflow autolog codex --trace-latest

# 4. View traces
mlflow ui
```

## What Gets Traced?

Each Codex session creates a hierarchical trace:

```
📊 Trace: codex_cli_session (AGENT)
├── 💬 llm_call_1 (LLM) - with token usage
├── 🔧 tool_read_file (TOOL) - with inputs/outputs
├── 💬 llm_call_2 (LLM)
└── 🔧 tool_write_file (TOOL)
```

**Captured Data:**
- User prompts
- LLM responses with token usage
- Tool calls (read, write, execute, etc.)
- Tool results
- Full conversation context
- Session metadata

## Requirements

- MLflow installed
- Codex CLI installed (`npm i -g @openai/codex`)
- ChatGPT Plus/Pro/Team/Enterprise plan (or API key)

## Configuration

- **Config file**: `~/.codex/mlflow_config.json`
- **Sessions dir**: `~/.codex/sessions` (Codex CLI default)
- **Traces dir**: `./mlruns` (standard MLflow directory)
- **Logs**: `~/.codex/mlflow/codex_tracing.log`

## Why ./mlruns?

Traces are saved to `./mlruns` by default (instead of `~/.codex/mlflow`) because:

✅ **Standard MLflow location** - Matches other MLflow integrations
✅ **No extra flags needed** - Just run `mlflow ui` to see traces
✅ **Per-project isolation** - Each project directory has its own traces
✅ **Consistent workflow** - Same as OpenAI, LangChain, Anthropic autolog
