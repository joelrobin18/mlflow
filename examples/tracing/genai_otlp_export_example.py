"""
Example: Exporting MLflow Traces with OpenTelemetry GenAI Semantic Conventions

This example demonstrates how to use MLflow's tracing capabilities with OTLP export
using the OpenTelemetry GenAI semantic conventions. This allows MLflow traces to be
exported in a standardized format that is compatible with observability tools that
understand the OpenTelemetry GenAI specifications.

Prerequisites:
    pip install mlflow opentelemetry-exporter-otlp-proto-grpc

Running this example:
    1. Start an OTLP collector (e.g., Jaeger with OTLP support):
       docker run -d --name jaeger \
         -e COLLECTOR_OTLP_ENABLED=true \
         -p 16686:16686 \
         -p 4317:4317 \
         jaegertracing/all-in-one:latest

    2. Set the environment variables:
       export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
       export MLFLOW_OTLP_TRACES_EXPORT_SCHEMA=genai

    3. Run this script:
       python genai_otlp_export_example.py

    4. View traces in Jaeger UI at http://localhost:16686

The exported traces will include GenAI semantic convention attributes like:
    - gen_ai.operation.name (instead of mlflow.spanType)
    - gen_ai.input.messages / gen_ai.output.messages
    - gen_ai.usage.input_tokens / gen_ai.usage.output_tokens
    - gen_ai.tool.call.arguments / gen_ai.tool.call.result (for tool spans)
"""

import json
import os
import time

import mlflow
from mlflow.entities.span import SpanType


def setup_environment():
    """Set up environment variables for GenAI OTLP export."""
    # Set OTLP endpoint (adjust to your collector's address)
    if "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in os.environ:
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:4317"

    # Enable GenAI semantic conventions for export
    os.environ["MLFLOW_OTLP_TRACES_EXPORT_SCHEMA"] = "genai"

    # Optional: Enable metrics export with GenAI schema
    os.environ["MLFLOW_OTLP_METRICS_EXPORT_SCHEMA"] = "genai"

    print("Environment configured for GenAI OTLP export:")
    print(f"  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: {os.environ.get('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT')}")
    print(f"  MLFLOW_OTLP_TRACES_EXPORT_SCHEMA: {os.environ.get('MLFLOW_OTLP_TRACES_EXPORT_SCHEMA')}")
    print()


# ============================================================================
# Example 1: Simple LLM-like function with MLflow tracing
# ============================================================================


@mlflow.trace(span_type=SpanType.LLM)
def generate_text(prompt: str, max_tokens: int = 100) -> dict:
    """
    Simulates an LLM text generation call.

    When exported with GenAI schema:
    - mlflow.spanType: LLM -> gen_ai.operation.name: text_completion
    - mlflow.spanInputs -> gen_ai.input.messages
    - mlflow.spanOutputs -> gen_ai.output.messages
    """
    # Simulate processing time
    time.sleep(0.1)

    # Get current span and set attributes
    span = mlflow.get_current_active_span()
    if span:
        # Set inputs (will be transformed to gen_ai.input.messages)
        span.set_inputs({"prompt": prompt, "max_tokens": max_tokens})

        # Simulate token usage (will be transformed to gen_ai.usage.*)
        token_usage = {
            "input_tokens": len(prompt.split()),
            "output_tokens": 25,
            "total_tokens": len(prompt.split()) + 25,
        }
        span.set_attribute("mlflow.chat.tokenUsage", json.dumps(token_usage))

    # Simulated response
    response = {
        "text": f"This is a simulated response to: {prompt[:50]}...",
        "finish_reason": "stop",
    }

    if span:
        span.set_outputs(response)

    return response


# ============================================================================
# Example 2: Chat model with message format
# ============================================================================


@mlflow.trace(span_type=SpanType.CHAT_MODEL)
def chat_completion(messages: list[dict], model: str = "gpt-4") -> dict:
    """
    Simulates a chat completion API call.

    When exported with GenAI schema:
    - mlflow.spanType: CHAT_MODEL -> gen_ai.operation.name: chat
    - Messages are formatted according to GenAI conventions
    """
    time.sleep(0.15)

    span = mlflow.get_current_active_span()
    if span:
        span.set_inputs(messages)
        span.set_attribute("mlflow.chat.tokenUsage", json.dumps({
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 150,
        }))

    # Simulated assistant response
    response = {
        "role": "assistant",
        "content": "I understand you're asking about MLflow tracing. "
                   "MLflow provides comprehensive tracing capabilities for GenAI applications.",
    }

    if span:
        span.set_outputs([response])

    return response


# ============================================================================
# Example 3: Tool/Function call
# ============================================================================


@mlflow.trace(span_type=SpanType.TOOL)
def search_database(query: str, limit: int = 10) -> list[dict]:
    """
    Simulates a tool/function call that an agent might use.

    When exported with GenAI schema:
    - mlflow.spanType: TOOL -> gen_ai.operation.name: execute_tool
    - mlflow.spanInputs -> gen_ai.tool.call.arguments
    - mlflow.spanOutputs -> gen_ai.tool.call.result
    """
    time.sleep(0.05)

    span = mlflow.get_current_active_span()
    if span:
        span.set_inputs({"query": query, "limit": limit})

    # Simulated search results
    results = [
        {"id": 1, "title": "MLflow Tracing Guide", "score": 0.95},
        {"id": 2, "title": "OpenTelemetry Integration", "score": 0.87},
        {"id": 3, "title": "GenAI Best Practices", "score": 0.82},
    ]

    if span:
        span.set_outputs(results)

    return results


# ============================================================================
# Example 4: Embedding generation
# ============================================================================


@mlflow.trace(span_type=SpanType.EMBEDDING)
def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Simulates embedding generation.

    When exported with GenAI schema:
    - mlflow.spanType: EMBEDDING -> gen_ai.operation.name: embeddings
    """
    time.sleep(0.08)

    span = mlflow.get_current_active_span()
    if span:
        span.set_inputs(texts)
        span.set_attribute("mlflow.chat.tokenUsage", json.dumps({
            "input_tokens": sum(len(t.split()) for t in texts),
            "output_tokens": 0,
            "total_tokens": sum(len(t.split()) for t in texts),
        }))

    # Simulated embeddings (768-dimensional vectors, showing only first 5 dims)
    embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]

    if span:
        span.set_outputs({"embeddings": embeddings, "model": "text-embedding-ada-002"})

    return embeddings


# ============================================================================
# Example 5: Agent workflow combining multiple operations
# ============================================================================


@mlflow.trace(span_type=SpanType.AGENT)
def agent_workflow(user_query: str) -> str:
    """
    Simulates an AI agent workflow that combines multiple operations.

    When exported with GenAI schema:
    - mlflow.spanType: AGENT -> gen_ai.operation.name: invoke_agent
    - Child spans show the complete execution flow with proper GenAI attributes
    """
    span = mlflow.get_current_active_span()
    if span:
        span.set_inputs({"user_query": user_query})

    # Step 1: Generate embeddings for the query
    query_embedding = generate_embeddings([user_query])

    # Step 2: Search for relevant information
    search_results = search_database(user_query, limit=3)

    # Step 3: Generate a response using the context
    context = "\n".join([r["title"] for r in search_results])
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}"},
    ]
    response = chat_completion(messages)

    final_response = response["content"]

    if span:
        span.set_outputs({"response": final_response, "sources": search_results})

    return final_response


# ============================================================================
# Main execution
# ============================================================================


def main():
    """Run all examples to demonstrate GenAI OTLP export."""
    print("=" * 70)
    print("MLflow Tracing with GenAI Semantic Conventions - OTLP Export Example")
    print("=" * 70)
    print()

    # Set up environment
    setup_environment()

    # Set MLflow experiment
    mlflow.set_experiment("genai-otlp-export-demo")

    print("Running examples...\n")

    # Example 1: Simple text generation
    print("1. Text Generation (LLM span type -> gen_ai.operation.name: text_completion)")
    result1 = generate_text("Explain the benefits of observability in AI systems")
    print(f"   Result: {result1['text'][:60]}...")
    print()

    # Example 2: Chat completion
    print("2. Chat Completion (CHAT_MODEL -> gen_ai.operation.name: chat)")
    messages = [
        {"role": "user", "content": "What is MLflow tracing?"}
    ]
    result2 = chat_completion(messages)
    print(f"   Result: {result2['content'][:60]}...")
    print()

    # Example 3: Tool execution
    print("3. Tool Execution (TOOL -> gen_ai.operation.name: execute_tool)")
    result3 = search_database("machine learning monitoring")
    print(f"   Result: Found {len(result3)} results")
    print()

    # Example 4: Embedding generation
    print("4. Embedding Generation (EMBEDDING -> gen_ai.operation.name: embeddings)")
    texts = ["MLflow is great", "OpenTelemetry rocks"]
    result4 = generate_embeddings(texts)
    print(f"   Result: Generated {len(result4)} embeddings")
    print()

    # Example 5: Full agent workflow
    print("5. Agent Workflow (AGENT -> gen_ai.operation.name: invoke_agent)")
    result5 = agent_workflow("How do I implement tracing in my AI application?")
    print(f"   Result: {result5[:60]}...")
    print()

    print("=" * 70)
    print("Examples completed!")
    print()
    print("The traces have been exported to your OTLP collector with GenAI")
    print("semantic conventions. Check your observability tool (e.g., Jaeger)")
    print("to see the traces with attributes like:")
    print("  - gen_ai.operation.name")
    print("  - gen_ai.input.messages / gen_ai.output.messages")
    print("  - gen_ai.usage.input_tokens / gen_ai.usage.output_tokens")
    print("  - gen_ai.tool.call.arguments / gen_ai.tool.call.result")
    print("=" * 70)


if __name__ == "__main__":
    main()
