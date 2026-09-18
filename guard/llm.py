"""Shared GLM 5.2 client for any module that needs an LLM call.

Thin wrapper around the ``openai`` SDK pointed at Z.ai's OpenAI-compatible
endpoint. Credentials and settings come from the ``GUARD_LLM_*`` environment
variables (see the ``.env`` file at the project root, loaded here via
python-dotenv with ``override=False`` so real environment variables always
win). The underlying SDK client is constructed lazily and thread-safely on
the first ``chat()`` call; a missing API key only fails at call time, so
importing this module (or ``guard``) stays safe offline. Message content is
never logged, matching the project's guardrail conventions.
"""

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("guard.llm")

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4/"
DEFAULT_MODEL = "glm-5.2"
DEFAULT_TIMEOUT = 60.0

API_KEY_VAR = "GUARD_LLM_API_KEY"
BASE_URL_VAR = "GUARD_LLM_BASE_URL"
MODEL_VAR = "GUARD_LLM_MODEL"
TIMEOUT_VAR = "GUARD_LLM_TIMEOUT"


def _load_env_file() -> None:
    """Load ``custom/.env`` without overriding variables already in the shell."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ENV_FILE, override=False)


_load_env_file()


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    finish_reason: str | None
    usage: dict | None


class LLMClientError(RuntimeError):
    """Raised when a GLM call cannot be made or fails (missing key, API error)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GLMClient:
    """Reusable chat-completions client for the GLM API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get(API_KEY_VAR, "")
        self._base_url = base_url or os.environ.get(BASE_URL_VAR, DEFAULT_BASE_URL)
        self._model = model or os.environ.get(MODEL_VAR, DEFAULT_MODEL)
        self._timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get(TIMEOUT_VAR, str(DEFAULT_TIMEOUT)))
        )
        self._lock = threading.Lock()
        self._client = None
        self._openai = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return
            if not self._api_key:
                raise LLMClientError(
                    f"{API_KEY_VAR} is not set; put the Z.ai key in {ENV_FILE.name} "
                    "at the project root or export it as an environment variable"
                )
            try:
                import openai
            except ImportError as exc:
                raise LLMClientError(
                    "the 'openai' package is not installed; "
                    "run: pip install -r requirements.txt"
                ) from exc
            logger.info(
                "llm | creating OpenAI-compatible client (base_url=%s model=%s timeout=%ss)",
                self._base_url,
                self._model,
                self._timeout,
            )
            self._openai = openai
            self._client = openai.OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )

    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """One blocking chat completion; returns content plus usage metadata."""
        if not messages:
            raise LLMClientError("messages must contain at least one message")
        self._ensure_client()

        payload = dict(kwargs)
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        request_messages = (
            [{"role": "system", "content": system}] + list(messages)
            if system
            else list(messages)
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=request_messages,
                **payload,
            )
        except self._openai.APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            logger.warning(
                "llm | API error: status=%s model=%s (%s)",
                status,
                self._model,
                type(exc).__name__,
            )
            raise LLMClientError(
                f"GLM API error (status={status}): {exc}", status_code=status
            ) from exc
        except self._openai.OpenAIError as exc:
            logger.warning(
                "llm | client error: %s model=%s", type(exc).__name__, self._model
            )
            raise LLMClientError(f"GLM client error: {exc}") from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMClientError("GLM response contained no choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        usage_obj = getattr(response, "usage", None)
        usage = None
        if usage_obj is not None:
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
        response_model = getattr(response, "model", None) or self._model
        logger.info(
            "llm | chat complete: model=%s finish=%s total_tokens=%s",
            response_model,
            getattr(choice, "finish_reason", None),
            usage.get("total_tokens") if usage else None,
        )
        return LLMResponse(
            content=getattr(message, "content", None) or "",
            model=response_model,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
        )


_default_lock = threading.Lock()
_default_client: GLMClient | None = None


def get_client() -> GLMClient:
    """Shared env-configured GLMClient (lazy singleton)."""
    global _default_client
    if _default_client is None:
        with _default_lock:
            if _default_client is None:
                _default_client = GLMClient()
    return _default_client
