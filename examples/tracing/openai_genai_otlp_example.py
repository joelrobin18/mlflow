"""
Example: OpenAI Integration with MLflow Tracing and GenAI OTLP Export

This example shows how to use MLflow's autologging with OpenAI and export
traces using OpenTelemetry GenAI semantic conventions.

Prerequisites:
    pip install mlflow openai opentelemetry-exporter-otlp-proto-grpc

Setup:
    1. Start an OTLP collector (e.g., Jaeger):
       docker run -d --name jaeger \
         -e COLLECTOR_OTLP_ENABLED=true \
         -p 16686:16686 \
         -p 4317:4317 \
         jaegertracing/all-in-one:latest

    2. Set environment variables:
       export OPENAI_API_KEY=your-api-key
       export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
       export MLFLOW_OTLP_TRACES_EXPORT_SCHEMA=genai

    3. Run this script:
       python openai_genai_otlp_example.py

    4. View traces at http://localhost:16686
"""

import os


def setup_genai_export():
    """Configure MLflow for GenAI semantic conventions OTLP export."""
    # OTLP endpoint (adjust to your collector)
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317")

    # Enable GenAI semantic conventions
    os.environ["MLFLOW_OTLP_TRACES_EXPORT_SCHEMA"] = "genai"
    os.environ["MLFLOW_OTLP_METRICS_EXPORT_SCHEMA"] = "genai"

    print("GenAI OTLP export configured:")
    print(f"  Endpoint: {os.environ.get('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT')}")
    print(f"  Schema: {os.environ.get('MLFLOW_OTLP_TRACES_EXPORT_SCHEMA')}")
    print()


def run_with_real_openai():
    """Example using the real OpenAI API with MLflow autologging."""
    import mlflow
    from openai import OpenAI

    # Enable MLflow autologging for OpenAI
    mlflow.openai.autolog()

    # Set experiment
    mlflow.set_experiment("openai-genai-otlp-demo")

    # Create OpenAI client
    client = OpenAI()

    print("Running OpenAI chat completion with MLflow tracing...")
    print()

    # Make a chat completion request
    # MLflow will automatically trace this and export with GenAI conventions
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the benefits of using OpenTelemetry for AI observability?"}
        ],
        max_tokens=150,
        temperature=0.7,
    )

    print("Response:")
    print(response.choices[0].message.content)
    print()

    # The trace will be exported with GenAI semantic conventions:
    # - gen_ai.operation.name: "chat"
    # - gen_ai.input.messages: the input messages
    # - gen_ai.output.messages: the response
    # - gen_ai.usage.input_tokens: prompt tokens
    # - gen_ai.usage.output_tokens: completion tokens

    return response


def run_with_mock_openai():
    """Example using a mock OpenAI client for demonstration without API key."""
    import json
    import time

    import mlflow
    from mlflow.entities.span import SpanType

    mlflow.set_experiment("mock-openai-genai-otlp-demo")

    print("Running mock OpenAI example (no API key required)...")
    print()

    @mlflow.trace(name="ChatCompletion", span_type=SpanType.CHAT_MODEL)
    def mock_chat_completion(messages: list[dict], model: str = "gpt-3.5-turbo", **kwargs):
        """Mock OpenAI chat completion that mimics the real API behavior."""
        span = mlflow.get_current_active_span()

        # Set inputs in OpenAI-like format
        if span:
            span.set_inputs(messages)
            span.set_attribute("gen_ai.request.model", model)

        # Simulate API latency
        time.sleep(0.2)

        # Simulate token counting
        input_tokens = sum(len(m.get("content", "").split()) for m in messages)
        output_tokens = 45

        if span:
            span.set_attribute("mlflow.chat.tokenUsage", json.dumps({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }))

        # Simulated response
        response = {
            "id": "chatcmpl-mock123",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "OpenTelemetry provides several key benefits for AI observability:\n\n"
                               "1. **Standardization**: Provides a vendor-neutral standard for "
                               "collecting telemetry data.\n"
                               "2. **Tracing**: Enables end-to-end distributed tracing across "
                               "your AI pipeline.\n"
                               "3. **Metrics**: Collects performance metrics like latency and "
                               "token usage.\n"
                               "4. **Interoperability**: Works with many observability backends "
                               "like Jaeger, Zipkin, and commercial solutions."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        }

        if span:
            span.set_outputs([response["choices"][0]["message"]])

        return response

    @mlflow.trace(name="Embedding", span_type=SpanType.EMBEDDING)
    def mock_embeddings(texts: list[str], model: str = "text-embedding-ada-002"):
        """Mock OpenAI embeddings API."""
        span = mlflow.get_current_active_span()

        if span:
            span.set_inputs(texts)
            span.set_attribute("gen_ai.request.model", model)

        time.sleep(0.1)

        # Mock embeddings
        embeddings = [[0.001 * i for i in range(1536)] for _ in texts]

        input_tokens = sum(len(t.split()) for t in texts)
        if span:
            span.set_attribute("mlflow.chat.tokenUsage", json.dumps({
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "total_tokens": input_tokens,
            }))
            span.set_outputs({"embeddings_count": len(embeddings), "dimensions": 1536})

        return embeddings

    @mlflow.trace(name="RAGPipeline", span_type=SpanType.CHAIN)
    def rag_pipeline(query: str):
        """A simple RAG pipeline demonstrating multiple span types."""
        span = mlflow.get_current_active_span()
        if span:
            span.set_inputs({"query": query})

        # Step 1: Generate embeddings for the query
        query_embedding = mock_embeddings([query])

        # Step 2: Simulate retrieval (tool call)
        @mlflow.trace(name="VectorSearch", span_type=SpanType.TOOL)
        def vector_search(embedding):
            s = mlflow.get_current_active_span()
            if s:
                s.set_inputs({"embedding_dims": len(embedding[0])})
            time.sleep(0.05)
            results = [
                {"doc_id": "doc1", "content": "MLflow provides model tracking...", "score": 0.92},
                {"doc_id": "doc2", "content": "OpenTelemetry is a CNCF project...", "score": 0.88},
            ]
            if s:
                s.set_outputs(results)
            return results

        context_docs = vector_search(query_embedding)

        # Step 3: Generate response with context
        context = "\n".join([doc["content"] for doc in context_docs])
        messages = [
            {"role": "system", "content": f"Use this context to answer: {context}"},
            {"role": "user", "content": query}
        ]
        response = mock_chat_completion(messages)

        final_response = response["choices"][0]["message"]["content"]
        if span:
            span.set_outputs({"response": final_response, "sources": [d["doc_id"] for d in context_docs]})

        return final_response

    # Run the examples
    print("1. Simple Chat Completion")
    print("-" * 40)
    response = mock_chat_completion([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the benefits of OpenTelemetry for AI?"}
    ])
    print(f"Response: {response['choices'][0]['message']['content'][:100]}...")
    print()

    print("2. Embeddings Generation")
    print("-" * 40)
    embeddings = mock_embeddings(["Hello world", "MLflow is great"])
    print(f"Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions each")
    print()

    print("3. RAG Pipeline (Chain with multiple span types)")
    print("-" * 40)
    rag_response = rag_pipeline("How can I monitor my AI application?")
    print(f"RAG Response: {rag_response[:100]}...")
    print()

    return response


def main():
    """Main function to run the example."""
    print("=" * 70)
    print("OpenAI + MLflow Tracing + GenAI OTLP Export Example")
    print("=" * 70)
    print()

    # Set up GenAI export
    setup_genai_export()

    # Check if OpenAI API key is available
    if os.environ.get("OPENAI_API_KEY"):
        print("OpenAI API key found. Running with real API...")
        try:
            run_with_real_openai()
        except Exception as e:
            print(f"Error with real OpenAI: {e}")
            print("Falling back to mock example...")
            run_with_mock_openai()
    else:
        print("No OPENAI_API_KEY found. Running with mock example...")
        run_with_mock_openai()

    print()
    print("=" * 70)
    print("Traces exported with GenAI semantic conventions!")
    print()
    print("In your observability tool, you should see attributes like:")
    print("  - gen_ai.operation.name: 'chat', 'embeddings', 'execute_tool'")
    print("  - gen_ai.input.messages: Input messages in GenAI format")
    print("  - gen_ai.output.messages: Output messages in GenAI format")
    print("  - gen_ai.usage.input_tokens: Number of input tokens")
    print("  - gen_ai.usage.output_tokens: Number of output tokens")
    print("  - gen_ai.tool.call.arguments: Tool input parameters (for TOOL spans)")
    print("  - gen_ai.tool.call.result: Tool output results (for TOOL spans)")
    print("=" * 70)


if __name__ == "__main__":
    main()
