import httpx
import openai
import pytest
from types import SimpleNamespace

from guard.llm import GLMClient, LLMClientError, get_client


class _FakeOpenAI:
    instances = []
    create_kwargs = None
    response = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        _FakeOpenAI.instances.append(self)

    @staticmethod
    def _create(**kwargs):
        _FakeOpenAI.create_kwargs = kwargs
        response = _FakeOpenAI.response
        if isinstance(response, Exception):
            raise response
        return response


def _fake_response(content="hello there", model="glm-5.2"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        model=model,
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeOpenAI.instances = []
    _FakeOpenAI.create_kwargs = None
    _FakeOpenAI.response = _fake_response()
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("GUARD_LLM_API_KEY", "test-key")
    return _FakeOpenAI


def test_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("GUARD_LLM_API_KEY", raising=False)
    client = GLMClient()
    with pytest.raises(LLMClientError, match="GUARD_LLM_API_KEY"):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_rejects_empty_messages():
    with pytest.raises(LLMClientError, match="at least one"):
        GLMClient(api_key="k").chat([])


def test_chat_returns_response(fake_openai):
    result = GLMClient(api_key="k").chat([{"role": "user", "content": "hi"}])
    assert result.content == "hello there"
    assert result.model == "glm-5.2"
    assert result.finish_reason == "stop"
    assert result.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


def test_chat_prepends_system_and_forwards_options(fake_openai):
    GLMClient(api_key="k").chat(
        [{"role": "user", "content": "hi"}],
        system="be brief",
        temperature=0.2,
        max_tokens=64,
    )
    kwargs = fake_openai.create_kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "be brief"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hi"}
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 64


def test_env_overrides_honored(fake_openai, monkeypatch):
    monkeypatch.setenv("GUARD_LLM_BASE_URL", "https://example.invalid/v4/")
    monkeypatch.setenv("GUARD_LLM_MODEL", "glm-5.2-air")
    monkeypatch.setenv("GUARD_LLM_TIMEOUT", "12.5")
    client = GLMClient()
    assert client.base_url == "https://example.invalid/v4/"
    assert client.model == "glm-5.2-air"
    client.chat([{"role": "user", "content": "hi"}])
    ctor = fake_openai.instances[-1].kwargs
    assert ctor["base_url"] == "https://example.invalid/v4/"
    assert ctor["timeout"] == 12.5
    assert fake_openai.create_kwargs["model"] == "glm-5.2-air"


def test_api_error_surfaces_as_llm_error(fake_openai):
    request = httpx.Request("POST", "https://api.z.ai/api/paas/v4/chat/completions")
    response = httpx.Response(429, request=request)
    fake_openai.response = openai.APIStatusError(
        "rate limited", response=response, body=None
    )
    with pytest.raises(LLMClientError) as excinfo:
        GLMClient(api_key="k").chat([{"role": "user", "content": "hi"}])
    assert excinfo.value.status_code == 429


def test_empty_choices_raises(fake_openai):
    fake_openai.response = SimpleNamespace(choices=[], model="glm-5.2", usage=None)
    with pytest.raises(LLMClientError, match="no choices"):
        GLMClient(api_key="k").chat([{"role": "user", "content": "hi"}])


def test_get_client_returns_same_instance():
    assert get_client() is get_client()
