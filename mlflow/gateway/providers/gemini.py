import time
from typing import Any

from mlflow.gateway.config import GeminiConfig, RouteConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import completions, embeddings


class GeminiAdapter(ProviderAdapter):
    @classmethod
    def chat_to_model(cls, payload, config):
        key_mapping = {
            "stop": "stopSequences",
            "n": "candidateCount",
            "max_tokens": "maxOutputTokens",
        }

        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise AIGatewayException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )

        if "topP" in payload and payload["topP"] > 1:
            raise AIGatewayException(
                status_code=422, detail="topP should be less than or equal to 1"
            )

        payload = rename_payload_keys(payload, key_mapping)

        contents = []
        for message in payload["messages"]:
            role = message["role"]

            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
                message["content"] = f"System: {message['content']}"

            contents.append({"role": role, "parts": [{"text": message["content"]}]})

        gemini_payload = {
            "contents": contents,
        }

        generation_config = {}
        for param in [
            "temperature",
            "topP",
            "stopSequences",
            "candidateCount",
            "maxOutputTokens",
            "topK",
        ]:
            if param in payload:
                generation_config[param] = payload[param]

        if generation_config:
            gemini_payload["generationConfig"] = generation_config

        return gemini_payload

    @classmethod
    def completions_to_model(cls, payload, config):
        chat_payload = {"messages": [{"role": "user", "content": payload.pop("prompt")}], **payload}
        return cls.chat_to_model(chat_payload, config)

    @classmethod
    def model_to_completions(cls, resp, config):
        choices = []

        for idx, candidate in enumerate(resp.get("candidates", [])):
            text = ""
            if "content" in candidate and candidate.get("content", {}).get("parts", {}):
                text = candidate["content"]["parts"][0].get("text", "")

            finish_reason = candidate.get("finishReason", "stop")
            if finish_reason == "MAX_TOKENS":
                finish_reason = "length"

            choices.append(
                completions.Choice(
                    index=idx,
                    text=text,
                    finish_reason=finish_reason,
                )
            )

        usage_metadata = resp.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", None)
        completion_tokens = usage_metadata.get("candidatesTokenCount", None)
        total_tokens = usage_metadata.get("totalTokenCount", None)

        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=config.model.name,
            choices=choices,
            usage=completions.CompletionsUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    @classmethod
    def embeddings_to_model(cls, payload, config):
        # Documentation: (https://ai.google.dev/api/embeddings#v1beta.ContentEmbedding):
        #     ```{"requests": [{
        #   "model": "models/text-embedding-004",
        #   "content": {
        #   "parts":[{
        #     "text": "What is the meaning of life?"}]}, },
        #   {
        #   "model": "models/text-embedding-004",
        #   "content": {
        #   "parts":[{
        #     "text": "How much wood would a woodchuck chuck?"}]}, },
        #   {
        #   "model": "models/text-embedding-004",
        #   "content": {
        #   "parts":[{
        #     "text": "How does the brain work?"}]}, }, ]}
        # ```

        texts = payload["input"]
        if isinstance(texts, str):
            texts = [texts]
        return (
            {"content": {"parts": [{"text": texts[0]}]}}
            if len(texts) == 1
            else {
                "requests": [
                    {"model": f"models/{config.model.name}", "content": {"parts": [{"text": text}]}}
                    for text in texts
                ]
            }
        )

    @classmethod
    def model_to_embeddings(cls, resp, config):
        # Documentation (https://ai.google.dev/api/embeddings#v1beta.ContentEmbedding):
        # ```
        # {
        #   "embeddings": [
        #     {
        #       "values": [
        #         3.25,
        #         0.7685547,
        #         2.65625,
        #         ...
        #         -0.30126953,
        #         -2.3554688,
        #         1.2597656
        #       ]
        #     }
        #   ]
        # }
        # ```

        data = [
            embeddings.EmbeddingObject(embedding=item.get("values", []), index=i)
            for i, item in enumerate(resp.get("embeddings") or [resp.get("embedding", {})])
        ]

        # Create and return response payload directly
        return embeddings.ResponsePayload(
            data=data,
            model=config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=None,
                total_tokens=None,
            ),
        )


class GeminiProvider(BaseProvider):
    NAME = "Gemini"
    CONFIG_TYPE = GeminiConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, GeminiConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.gemini_config: GeminiConfig = config.model.config

    @property
    def headers(self):
        return {"x-goog-api-key": self.gemini_config.gemini_api_key}

    @property
    def base_url(self):
        return "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def adapter_class(self):
        return GeminiAdapter

    def get_endpoint_url(self, route_type: str) -> str:
        if route_type == "llm/v1/chat" or route_type == "llm/v1/completions":
            return f"{self.base_url}/{self.config.model.name}:generateContent"
        elif route_type == "llm/v1/embeddings":
            return f"{self.base_url}/{self.config.model.name}:embedContent"
        else:
            raise ValueError(f"Invalid route type {route_type}")

    async def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        if payload.get("stream", False):
            # TODO: Implement streaming for completions
            raise AIGatewayException(
                status_code=422,
                detail="Streaming is not yet supported for completions with Gemini AI Gateway",
            )

        resp = await self._request(
            f"{self.config.model.name}:generateContent",
            self.adapter_class.completions_to_model(payload, self.config),
        )

        return self.adapter_class.model_to_completions(resp, self.config)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        embedding_payload = self.adapter_class.embeddings_to_model(payload, self.config)

        # Use the batch endpoint if payload contains "requests"
        if "requests" in embedding_payload:
            endpoint_suffix = ":batchEmbedContents"
        else:
            endpoint_suffix = ":embedContent"

        resp = await self._request(
            f"{self.config.model.name}{endpoint_suffix}",
            embedding_payload,
        )
        return self.adapter_class.model_to_embeddings(resp, self.config)
