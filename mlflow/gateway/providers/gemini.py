import time
from typing import Any

from mlflow.gateway.config import GeminiConfig, RouteConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class GeminiProvider(BaseProvider):
    NAME = "Gemini"
    CONFIG_TYPE = GeminiConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, GeminiConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.gemini_config: GeminiConfig = config.model.config

    async def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Build the full URL with the API key in the query string.
        # Gemini’s endpoint (v1beta) is used instead of PaLM’s v1beta3.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{path}?key={self.gemini_config.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        return await send_request(headers=headers, base_url="", path=url, payload=payload)

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        # Create generationConfig from keys.
        generation_config = {}
        if "max_tokens" in payload:
            generation_config["maxOutputTokens"] = payload.pop("max_tokens")
        if "temperature" in payload:
            generation_config["temperature"] = payload.pop("temperature")
        if "n" in payload:
            generation_config["candidateCount"] = payload.pop("n")
        if "top_p" in payload:
            generation_config["topP"] = payload.pop("top_p")
        if "top_k" in payload:
            generation_config["topK"] = payload.pop("top_k")

        # Convert prompt to messages if messages are not provided.
        messages = payload.pop("messages", None)
        if messages is None and "prompt" in payload:
            messages = [{"role": "user", "content": payload.pop("prompt")}]
        if messages is None or not messages:
            raise AIGatewayException(
                status_code=422, detail="Field 'messages' is required for chat completions"
            )

        # Convert messages to Gemini’s expected "contents" format.
        contents = []
        for msg in messages:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
        # Optionally handle examples or context.
        if "examples" in payload:
            contents.append({"examples": payload.pop("examples")})
        if "context" in payload:
            contents.append({"context": payload.pop("context")})
        payload["contents"] = contents

        # Set the generationConfig field.
        payload["generationConfig"] = generation_config

        path = f"{self.config.model.name}:generateContent"
        resp = await self._request(path, payload)

        return chat.ResponsePayload(
            created=int(time.time()),
            model=self.config.model.name,
            choices=[
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(
                        role=c["content"].get("role", "model"),
                        content=c["content"]["parts"][0]["text"]
                        if c["content"].get("parts")
                        else "",
                    ),
                    finish_reason=c.get("finishReason"),
                )
                for idx, c in enumerate(resp.get("candidates", []))
            ],
            usage=chat.ChatUsage(
                prompt_tokens=resp.get("usageMetadata", {}).get("promptTokenCount"),
                completion_tokens=resp.get("usageMetadata", {}).get("candidatesTokenCount"),
                total_tokens=resp.get("usageMetadata", {}).get("totalTokenCount"),
            ),
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        # Create generationConfig from known keys.
        generation_config = {}
        # Rename "max_tokens" to "maxOutputTokens" if present.
        if "max_tokens" in payload:
            generation_config["maxOutputTokens"] = payload.pop("max_tokens")
        # Move "temperature" if present.
        if "temperature" in payload:
            generation_config["temperature"] = payload.pop("temperature")
        # If candidate count is provided as "n", map it to "candidateCount".
        if "n" in payload:
            generation_config["candidateCount"] = payload.pop("n")
        # Optionally, you can also check for keys like "top_p" or "top_k" if you support them.
        if "top_p" in payload:
            generation_config["topP"] = payload.pop("top_p")
        if "top_k" in payload:
            generation_config["topK"] = payload.pop("top_k")

        # Wrap the prompt text into the expected "contents" structure.
        prompt_text = payload.pop("prompt", "")
        payload["contents"] = [{"role": "user", "parts": [{"text": prompt_text}]}]

        # Set the generationConfig into the payload.
        payload["generationConfig"] = generation_config

        path = f"{self.config.model.name}:generateContent"
        resp = await self._request(path, payload)

        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=self.config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=c["content"]["parts"][0]["text"] if c["content"].get("parts") else "",
                    finish_reason=c.get("finishReason"),
                )
                for idx, c in enumerate(resp.get("candidates", []))
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=resp.get("usageMetadata", {}).get("promptTokenCount"),
                completion_tokens=resp.get("usageMetadata", {}).get("candidatesTokenCount"),
                total_tokens=resp.get("usageMetadata", {}).get("totalTokenCount"),
            ),
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        # For embeddings, assume Gemini still expects "texts" instead of "input".
        # payload = rename_payload_keys(payload, {"input": "texts"})
        path = f"{self.config.model.name}:batchEmbedContents"
        resp = await self._request(path, payload)

        return embeddings.ResponsePayload(
            data=[
                embeddings.EmbeddingObject(
                    embedding=embedding["value"],
                    index=idx,
                )
                for idx, embedding in enumerate(resp.get("embeddings", []))
            ],
            model=self.config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=resp.get("usageMetadata", {}).get("promptTokenCount"),
                total_tokens=resp.get("usageMetadata", {}).get("totalTokenCount"),
            ),
        )
