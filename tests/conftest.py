import pytest

from src.config import settings as settings_module


@pytest.fixture
def mock_settings(monkeypatch):
    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("VLLM_BASE_URL", "http://mock-vllm:8000")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setenv("VLLM_MODEL_NAME", "test-model")
    yield
    settings_module.get_settings.cache_clear()
