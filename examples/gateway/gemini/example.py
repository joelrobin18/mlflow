from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"Gemini endpoints: {client.list_endpoints()}\n")
    print(f"Gemini completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "What is the future of artificial intelligence?",
            "temperature": 0.2,
        },
    )
    print(f"Gemini response for completions: {response_completions}\n")

    # Chat request (if you have configured a chat endpoint)
    response_chat = client.predict(
        endpoint="chat",
        inputs={
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm fine, thanks! How can I help you today?"},
            ],
            "temperature": 0.2,
        },
    )
    print(f"Gemini response for chat: {response_chat}\n")

    # Embeddings request
    response_embeddings = client.predict(
        endpoint="embeddings",
        inputs={
            "input": [
                "Describe the main differences between renewable and nonrenewable energy sources."
            ]
        },
    )
    print(f"Gemini response for embeddings: {response_embeddings}\n")


if __name__ == "__main__":
    main()
