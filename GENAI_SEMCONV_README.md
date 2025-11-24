# Testing OpenTelemetry GenAI Semantic Conventions in MLflow

This guide shows how to test the new OpenTelemetry GenAI Semantic Conventions support in MLflow's OTLP export functionality.

## Overview

MLflow now supports exporting traces and metrics using OpenTelemetry GenAI Semantic Conventions when configured via environment variables. This enables standardized telemetry data across different GenAI instrumentation libraries.

## Prerequisites

1. **Install MLflow with dependencies**:
   ```bash
   uv sync
   uv pip install -r requirements/test-requirements.txt
   ```

2. **Set up an OTLP collector** (optional - for end-to-end testing):
   - Option A: Use the test collector (requires Docker)
   - Option B: Use any OTLP-compatible backend (Jaeger, Grafana Tempo, etc.)

## Quick Test: Unit Tests

Run the test suite to verify the implementation:

```bash
# Test the GenAI converter
uv run pytest tests/tracing/otel/export/test_genai_semconv.py -v

# Test the integration
uv run pytest tests/tracing/otel/export/test_genai_integration.py -v

# Run all export tests
uv run pytest tests/tracing/otel/export/ -v
```

Expected output: All 23 tests should pass ✅

## Test Scenario 1: Trace Export with GenAI Schema

### Step 1: Create a test script

Create `test_genai_export.py`:

```python
"""
Test script for GenAI Semantic Conventions export.
"""

import mlflow
from mlflow.entities.span import SpanType
import time


@mlflow.trace(span_type=SpanType.CHAT_MODEL, name="chat_completion")
def call_llm(messages):
    """Simulate an LLM call."""
    time.sleep(0.1)

    # Set token usage (simulated)
    mlflow.set_span_chat_messages(
        inputs=messages,
        outputs=[{"role": "assistant", "content": "Hello! How can I help you?"}]
    )

    # This would normally come from the LLM response
    # For testing, we'll set it manually via span attributes
    return "Hello! How can I help you?"


@mlflow.trace(span_type=SpanType.TOOL, name="search_tool")
def search_database(query):
    """Simulate a tool call."""
    time.sleep(0.05)
    return {"results": ["result1", "result2", "result3"]}


@mlflow.trace(span_type=SpanType.AGENT, name="agent_workflow")
def run_agent():
    """Main agent workflow."""
    # First, call the LLM
    messages = [{"role": "user", "content": "Hello!"}]
    response = call_llm(messages)

    # Then use a tool
    search_results = search_database("test query")

    # Final LLM call with results
    final_messages = messages + [
        {"role": "assistant", "content": response},
        {"role": "user", "content": f"Search results: {search_results}"}
    ]
    final_response = call_llm(final_messages)

    return final_response


if __name__ == "__main__":
    print("Starting GenAI Semantic Conventions test...")

    # Set up experiment
    mlflow.set_experiment("genai_semconv_test")

    # Run the agent
    result = run_agent()

    print(f"Result: {result}")
    print("\nTrace created! Check your OTLP collector for GenAI semantic convention attributes.")
    print("\nExpected attributes in spans:")
    print("  - gen_ai.operation.name: 'chat', 'execute_tool', 'invoke_agent'")
    print("  - gen_ai.usage.input_tokens: <number>")
    print("  - gen_ai.usage.output_tokens: <number>")
    print("  - gen_ai.input.messages: <messages>")
    print("  - gen_ai.output.messages: <messages>")
    print("  - gen_ai.tool.call.arguments: <tool args>")
    print("  - gen_ai.tool.call.result: <tool results>")
```

### Step 2: Configure environment for GenAI export

```bash
# Enable GenAI semantic conventions for traces
export OTEL_EXPORTER_OTLP_TRACES_SCHEMA=genai

# Enable GenAI semantic conventions for metrics
export OTEL_EXPORTER_OTLP_METRICS_SCHEMA=genai

# Configure OTLP endpoint (example with local collector)
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4317

# Optional: Use HTTP protocol instead of gRPC
# export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
# export OTEL_EXPORTER_OTLP_METRICS_PROTOCOL=http/protobuf
```

### Step 3: Run the test

```bash
uv run python test_genai_export.py
```

## Test Scenario 2: Compare Default vs GenAI Schema

### Create a comparison script

Create `compare_schemas.py`:

```python
"""
Compare default MLflow schema vs GenAI schema export.
"""

import os
import json
from mlflow.entities.span import Span, SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import dump_span_attribute_value


def test_schema_conversion():
    """Test and compare schema conversion."""

    # Create test attributes
    attributes = {
        SpanAttributeKey.SPAN_TYPE: dump_span_attribute_value(SpanType.CHAT_MODEL),
        SpanAttributeKey.CHAT_USAGE: dump_span_attribute_value({
            TokenUsageKey.INPUT_TOKENS: 100,
            TokenUsageKey.OUTPUT_TOKENS: 50,
            TokenUsageKey.TOTAL_TOKENS: 150,
        }),
        SpanAttributeKey.INPUTS: dump_span_attribute_value([
            {"role": "user", "content": "What is machine learning?"}
        ]),
        SpanAttributeKey.OUTPUTS: dump_span_attribute_value([
            {"role": "assistant", "content": "Machine learning is..."}
        ]),
    }

    print("Original MLflow attributes:")
    print(json.dumps(attributes, indent=2))
    print("\n" + "="*60 + "\n")

    # Test without GenAI schema
    os.environ.pop('OTEL_EXPORTER_OTLP_TRACES_SCHEMA', None)
    default_attrs = Span._apply_export_schema_conversion(attributes)

    print("Default export (MLflow schema only):")
    print(f"Number of attributes: {len(default_attrs)}")
    print("Attributes:")
    for key in sorted(default_attrs.keys()):
        print(f"  - {key}")
    print("\n" + "="*60 + "\n")

    # Test with GenAI schema
    os.environ['OTEL_EXPORTER_OTLP_TRACES_SCHEMA'] = 'genai'
    genai_attrs = Span._apply_export_schema_conversion(attributes)

    print("GenAI export (MLflow + GenAI schema):")
    print(f"Number of attributes: {len(genai_attrs)}")
    print("Attributes:")
    for key in sorted(genai_attrs.keys()):
        print(f"  - {key}")

    print("\n" + "="*60 + "\n")
    print("New GenAI attributes added:")
    genai_only = set(genai_attrs.keys()) - set(default_attrs.keys())
    for key in sorted(genai_only):
        value = genai_attrs[key]
        if isinstance(value, (int, str)) and not isinstance(value, bool):
            print(f"  - {key}: {value}")
        else:
            print(f"  - {key}: <{type(value).__name__}>")


if __name__ == "__main__":
    test_schema_conversion()
```

Run the comparison:

```bash
uv run python compare_schemas.py
```

Expected output shows the additional GenAI attributes added when the schema is enabled.

## Test Scenario 3: Metrics Export with GenAI Schema

### Create a metrics test script

Create `test_genai_metrics.py`:

```python
"""
Test GenAI semantic conventions for metrics export.
"""

from mlflow.entities.span import SpanType
from mlflow.tracing.processor.otel_metrics_mixin import OtelMetricsMixin
import os


def test_metric_conversion():
    """Test metric name and attribute conversion."""

    print("Testing GenAI Semantic Conventions for Metrics\n")
    print("="*60)

    # Test without GenAI schema
    print("\n1. Default MLflow Schema:")
    os.environ.pop('OTEL_EXPORTER_OTLP_METRICS_SCHEMA', None)

    metric_name = OtelMetricsMixin._get_metric_name("mlflow.trace.span.duration")
    print(f"   Metric name: {metric_name}")

    attributes = {
        "span_type": SpanType.CHAT_MODEL,
        "span_status": "ERROR",
        "experiment_id": "123",
        "root": True,
    }
    converted = OtelMetricsMixin._apply_metrics_schema_conversion(attributes)
    print(f"   Attributes: {list(converted.keys())}")

    # Test with GenAI schema
    print("\n2. GenAI Schema:")
    os.environ['OTEL_EXPORTER_OTLP_METRICS_SCHEMA'] = 'genai'

    metric_name = OtelMetricsMixin._get_metric_name("mlflow.trace.span.duration")
    print(f"   Metric name: {metric_name}")

    converted = OtelMetricsMixin._apply_metrics_schema_conversion(attributes)
    print(f"   Attributes: {list(converted.keys())}")

    print("\n" + "="*60)
    print("\nGenAI Metric Mapping:")
    print("  MLflow: mlflow.trace.span.duration")
    print("  GenAI:  gen_ai.client.operation.duration")
    print("\nGenAI Attribute Mappings:")
    print("  span_type (CHAT_MODEL) -> gen_ai.operation.name (chat)")
    print("  span_status (ERROR) -> error.type (error)")


if __name__ == "__main__":
    test_metric_conversion()
```

Run the metrics test:

```bash
uv run python test_genai_metrics.py
```

## Test Scenario 4: End-to-End with OTLP Collector

### Step 1: Start an OTLP collector (Docker)

```bash
# Create a collector config
cat > otel-collector-config.yaml <<EOF
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  debug:
    verbosity: detailed

  # Optional: Export to Jaeger
  # jaeger:
  #   endpoint: jaeger:14250
  #   tls:
  #     insecure: true

processors:
  batch:

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
EOF

# Start the collector
docker run -d --name otel-collector \
  -p 4317:4317 \
  -p 4318:4318 \
  -v $(pwd)/otel-collector-config.yaml:/etc/otel-collector-config.yaml \
  otel/opentelemetry-collector:latest \
  --config=/etc/otel-collector-config.yaml
```

### Step 2: Configure and run MLflow with GenAI export

```bash
# Enable GenAI schema
export OTEL_EXPORTER_OTLP_TRACES_SCHEMA=genai
export OTEL_EXPORTER_OTLP_METRICS_SCHEMA=genai

# Point to the collector
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4317

# Run your test script
uv run python test_genai_export.py
```

### Step 3: View the exported data

```bash
# View collector logs to see GenAI attributes
docker logs otel-collector

# Look for attributes like:
#   - gen_ai.operation.name
#   - gen_ai.usage.input_tokens
#   - gen_ai.usage.output_tokens
#   - gen_ai.input.messages
#   - gen_ai.output.messages
```

### Step 4: Cleanup

```bash
docker stop otel-collector
docker rm otel-collector
```

## Environment Variables Reference

| Variable | Values | Description |
|----------|--------|-------------|
| `OTEL_EXPORTER_OTLP_TRACES_SCHEMA` | `genai` or unset | Export traces with GenAI semantic conventions |
| `OTEL_EXPORTER_OTLP_METRICS_SCHEMA` | `genai` or unset | Export metrics with GenAI semantic conventions |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | URL | OTLP endpoint for traces |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | URL | OTLP endpoint for metrics |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | URL | Fallback OTLP endpoint (for both) |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | `grpc` or `http/protobuf` | Protocol for trace export (default: `grpc`) |
| `OTEL_EXPORTER_OTLP_METRICS_PROTOCOL` | `grpc` or `http/protobuf` | Protocol for metrics export (default: `grpc`) |

## GenAI Semantic Conventions Mappings

### Span Type Mappings

| MLflow Span Type | GenAI Operation Name |
|-----------------|---------------------|
| `CHAT_MODEL` | `chat` |
| `LLM` | `text_completion` |
| `EMBEDDING` | `embeddings` |
| `TOOL` | `execute_tool` |
| `AGENT` | `invoke_agent` |

### Attribute Mappings

| MLflow Attribute | GenAI Attribute | Notes |
|-----------------|-----------------|-------|
| `mlflow.spanType` | `gen_ai.operation.name` | Based on span type mapping |
| `mlflow.chat.tokenUsage.input_tokens` | `gen_ai.usage.input_tokens` | Token count |
| `mlflow.chat.tokenUsage.output_tokens` | `gen_ai.usage.output_tokens` | Token count |
| `mlflow.spanInputs` | `gen_ai.input.messages` | For LLM/CHAT spans |
| `mlflow.spanOutputs` | `gen_ai.output.messages` | For LLM/CHAT spans |
| `mlflow.spanInputs` | `gen_ai.tool.call.arguments` | For TOOL spans |
| `mlflow.spanOutputs` | `gen_ai.tool.call.result` | For TOOL spans |

### Metric Mappings

| MLflow Metric | GenAI Metric |
|--------------|-------------|
| `mlflow.trace.span.duration` | `gen_ai.client.operation.duration` |

**Metric Attributes:**

| MLflow Attribute | GenAI Attribute |
|-----------------|-----------------|
| `span_type` | `gen_ai.operation.name` |
| `span_status: ERROR` | `error.type: error` |

## Verification Checklist

After running tests, verify:

- ✅ All 23 unit/integration tests pass
- ✅ Traces exported with `gen_ai.operation.name` attribute
- ✅ Token usage exported as `gen_ai.usage.*` attributes
- ✅ Input/output messages use correct GenAI attributes
- ✅ TOOL spans use `gen_ai.tool.call.*` attributes
- ✅ Metrics use `gen_ai.client.operation.duration` name
- ✅ Original MLflow attributes preserved (backward compatibility)
- ✅ No errors when schema env vars are not set (default behavior)

## Troubleshooting

### Issue: Attributes not converting

**Check:**
1. Environment variables are set correctly
2. Value is exactly `"genai"` (case-insensitive)
3. Variables are exported before running Python

```bash
# Verify env vars
echo $OTEL_EXPORTER_OTLP_TRACES_SCHEMA
echo $OTEL_EXPORTER_OTLP_METRICS_SCHEMA
```

### Issue: OTLP export not working

**Check:**
1. OTLP collector is running and accessible
2. Endpoint URL is correct
3. Network connectivity

```bash
# Test collector connectivity
curl -v http://localhost:4317
```

### Issue: Tests failing

**Check:**
1. All dependencies installed: `uv sync`
2. Running from the mlflow repository root
3. Python version >= 3.10

## Additional Resources

- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [GenAI Spans Specification](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [GenAI Metrics Specification](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [MLflow Tracing Documentation](https://mlflow.org/docs/latest/llms/tracing/index.html)

## Feedback

If you encounter any issues or have suggestions, please open an issue in the MLflow repository.
