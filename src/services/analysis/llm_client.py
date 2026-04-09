import json
import logging
from typing import Optional, Dict, Any

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Async HTTP client for vLLM OpenAI-compatible API."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.vllm_base_url.rstrip("/")
        self.model_name = settings.vllm_model_name
        self.api_key = settings.vllm_api_key
        self.timeout = settings.vllm_timeout
        self.max_tokens = settings.vllm_max_tokens
        self.temperature = settings.vllm_temperature
        self.top_p = settings.vllm_top_p
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request to vLLM and return parsed JSON."""
        client = await self._get_client()

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "response_format": {"type": "json_object"},
        }

        response = await client.post("/v1/chat/completions", json=payload)
        if response.status_code >= 400:
            logger.error("vLLM error %d: %s", response.status_code, response.text)
        response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}

        logger.info(
            "vLLM completion: finish_reason=%s, prompt_tokens=%s, "
            "completion_tokens=%s, total_tokens=%s",
            finish_reason,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {content[:500]}")
            if finish_reason == "length":
                requested_max = payload["max_tokens"]
                completion_tokens = usage.get("completion_tokens")
                raise RuntimeError(
                    f"vLLM hit max_tokens={requested_max} "
                    f"(completion_tokens={completion_tokens}) — output was "
                    "truncated before the JSON structure closed. Increase "
                    "VLLM_MAX_TOKENS or shorten the input."
                )
            return self._extract_json(content)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Fallback: extract JSON from a response that may have extra text."""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        raise ValueError("Could not extract valid JSON from LLM response")

    async def health_check(self) -> bool:
        """Check if the vLLM server is reachable."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
