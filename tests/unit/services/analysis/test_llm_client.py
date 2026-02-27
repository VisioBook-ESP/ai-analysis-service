import json

import httpx
import pytest
import respx

from src.services.analysis.llm_client import LLMClient


def _llm_response(content: dict | str) -> dict:
    """Construit une réponse vLLM OpenAI-compatible."""
    if isinstance(content, dict):
        content = json.dumps(content)
    return {"choices": [{"message": {"content": content}}]}


# ---------------------------------------------------------------------------
# chat_completion
# ---------------------------------------------------------------------------

class TestChatCompletion:
    async def test_returns_parsed_json(self, mock_settings):
        payload = {"themes": ["love"], "tone": "melancholic"}
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_llm_response(payload))
            )
            client = LLMClient()
            result = await client.chat_completion("system prompt", "user prompt")
            assert result == payload
            await client.close()

    async def test_sends_correct_payload_structure(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            route = mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_llm_response({"ok": True}))
            )
            client = LLMClient()
            await client.chat_completion("sys", "user")
            body = json.loads(route.calls[0].request.content)
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][0]["content"] == "sys"
            assert body["messages"][1]["role"] == "user"
            assert body["messages"][1]["content"] == "user"
            assert body["response_format"] == {"type": "json_object"}
            await client.close()

    async def test_custom_max_tokens_and_temperature_override_defaults(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            route = mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_llm_response({"ok": True}))
            )
            client = LLMClient()
            await client.chat_completion("sys", "user", max_tokens=128, temperature=0.9)
            body = json.loads(route.calls[0].request.content)
            assert body["max_tokens"] == 128
            assert body["temperature"] == 0.9
            await client.close()

    async def test_temperature_zero_is_sent_not_replaced_by_default(self, mock_settings):
        """temperature=0 est falsy mais doit être envoyé tel quel."""
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            route = mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_llm_response({"ok": True}))
            )
            client = LLMClient()
            await client.chat_completion("sys", "user", temperature=0.0)
            body = json.loads(route.calls[0].request.content)
            assert body["temperature"] == 0.0
            await client.close()

    async def test_fallback_extracts_json_embedded_in_garbage(self, mock_settings):
        raw = 'Here is the result: {"key": "value"} end of response.'
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_llm_response(raw))
            )
            client = LLMClient()
            result = await client.chat_completion("sys", "user")
            assert result == {"key": "value"}
            await client.close()

    async def test_raises_http_status_error_on_500(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(500, text="Internal Server Error")
            )
            client = LLMClient()
            with pytest.raises(httpx.HTTPStatusError):
                await client.chat_completion("sys", "user")
            await client.close()

    async def test_raises_http_status_error_on_401(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(401, text="Unauthorized")
            )
            client = LLMClient()
            with pytest.raises(httpx.HTTPStatusError):
                await client.chat_completion("sys", "user")
            await client.close()

    async def test_raises_value_error_when_no_json_extractable(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_llm_response("no json here at all"))
            )
            client = LLMClient()
            with pytest.raises(ValueError, match="Could not extract valid JSON"):
                await client.chat_completion("sys", "user")
            await client.close()

    async def test_sends_bearer_auth_header(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            route = mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_llm_response({"ok": True}))
            )
            client = LLMClient()
            await client.chat_completion("sys", "user")
            assert route.calls[0].request.headers["authorization"] == "Bearer test-key"
            await client.close()


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    async def test_returns_true_on_200(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            mock.get("/health").mock(return_value=httpx.Response(200))
            client = LLMClient()
            assert await client.health_check() is True
            await client.close()

    async def test_returns_false_on_non_200(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            mock.get("/health").mock(return_value=httpx.Response(503))
            client = LLMClient()
            assert await client.health_check() is False
            await client.close()

    async def test_returns_false_on_connection_error(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            mock.get("/health").mock(side_effect=httpx.ConnectError("unreachable"))
            client = LLMClient()
            assert await client.health_check() is False
            await client.close()

    async def test_returns_false_on_timeout(self, mock_settings):
        with respx.mock(base_url="http://mock-vllm:8000") as mock:
            mock.get("/health").mock(side_effect=httpx.TimeoutException("timeout"))
            client = LLMClient()
            assert await client.health_check() is False
            await client.close()


# ---------------------------------------------------------------------------
# _extract_json (static, testable directement)
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_extracts_json_with_surrounding_text(self):
        result = LLMClient._extract_json('some text {"a": 1, "b": "two"} more text')
        assert result == {"a": 1, "b": "two"}

    def test_extracts_nested_json(self):
        result = LLMClient._extract_json('prefix {"outer": {"inner": 42}} suffix')
        assert result == {"outer": {"inner": 42}}

    def test_raises_on_no_braces(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            LLMClient._extract_json("no json here at all")

    def test_raises_on_malformed_json(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            LLMClient._extract_json("{malformed: json without quotes}")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            LLMClient._extract_json("")
