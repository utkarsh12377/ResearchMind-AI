"""Multi-provider LLM gateway.

Everything above this layer — RAG, the reasoning loops, every agent — calls
`complete()` or `stream()` and never touches a provider SDK. That keeps
provider choice a configuration decision, gives one place to implement
retries, fallback, and token accounting, and makes the whole system testable
without network access by swapping in a fake provider.

Providers are called over HTTP with a shared client rather than through three
vendor SDKs, for the same reasons as the embedding layer: one retry story, one
streaming abstraction, and no dependency disagreements about async transports.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 120.0
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None

    @property
    def was_truncated(self) -> bool:
        """Whether the model stopped at the token limit rather than finishing.

        Callers that parse structured output need to know this: a truncated
        response is often still valid-looking but incomplete.
        """
        return self.finish_reason in {"length", "max_tokens"}


class LLMError(RuntimeError):
    """Raised when a completion cannot be produced."""


class RateLimitError(LLMError):
    """Provider throttled the request. Retryable."""


class LLMProvider(ABC):
    name: str
    default_model: str

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Completion: ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]: ...


class _HttpProvider(LLMProvider):
    endpoint: str

    def __init__(self, api_key: str, *, model: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        if not api_key:
            raise LLMError(
                f"{type(self).__name__} requires an API key. Set it in .env, or set "
                "DEFAULT_LLM_PROVIDER to 'echo' for offline development."
            )
        self.api_key = api_key
        self.model = model or self.default_model
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _body(self, messages, model, temperature, max_tokens, stream):  # noqa: ANN001, ANN202
        raise NotImplementedError

    def _parse(self, payload: dict) -> Completion:
        raise NotImplementedError

    def _parse_stream_chunk(self, data: dict) -> str | None:
        raise NotImplementedError

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Completion:
        model = model or self.model
        body = self._body(messages, model, temperature, max_tokens, stream=False)

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.endpoint, headers=self._headers(), json=body)

                if response.status_code == 429:
                    raise RateLimitError(f"{self.name} rate limited")
                if response.status_code >= 500:
                    raise LLMError(f"{self.name} server error {response.status_code}")
                if response.status_code >= 400:
                    # Bad key, bad request, content filter: retrying cannot help.
                    raise LLMError(
                        f"{self.name} rejected the request ({response.status_code}): "
                        f"{response.text[:300]}"
                    )

                return self._parse(response.json())

            except (RateLimitError, httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
            except LLMError as exc:
                if "server error" not in str(exc):
                    raise
                last_error = exc

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "llm_retry", provider=self.name, attempt=attempt + 1, error=str(last_error)
                )
                await asyncio.sleep(delay)

        raise LLMError(f"{self.name} failed after {MAX_RETRIES} attempts: {last_error}")

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        model = model or self.model
        body = self._body(messages, model, temperature, max_tokens, stream=True)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", self.endpoint, headers=self._headers(), json=body
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode()[:300]
                    raise LLMError(
                        f"{self.name} streaming failed "
                        f"({response.status_code}): {detail}"
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        # Keep-alive comments and partial frames are expected.
                        continue
                    text = self._parse_stream_chunk(chunk)
                    if text:
                        yield text


class OpenAIProvider(_HttpProvider):
    name = "openai"
    default_model = "gpt-4o-mini"
    endpoint = "https://api.openai.com/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _body(self, messages, model, temperature, max_tokens, stream):  # noqa: ANN001, ANN202
        return {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def _parse(self, payload: dict) -> Completion:
        choice = payload["choices"][0]
        usage = payload.get("usage", {})
        return Completion(
            text=choice["message"]["content"] or "",
            model=payload.get("model", self.model),
            provider=self.name,
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
        )

    def _parse_stream_chunk(self, data: dict) -> str | None:
        choices = data.get("choices") or []
        if not choices:
            return None
        return choices[0].get("delta", {}).get("content")


class AnthropicProvider(_HttpProvider):
    name = "anthropic"
    default_model = "claude-sonnet-4-5"
    endpoint = "https://api.anthropic.com/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _body(self, messages, model, temperature, max_tokens, stream):  # noqa: ANN001, ANN202
        # Anthropic takes the system prompt as a top-level field rather than a
        # message role, so it has to be lifted out of the conversation.
        system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        conversation = [m.as_dict() for m in messages if m.role != Role.SYSTEM]

        body: dict[str, object] = {
            "model": model,
            "messages": conversation,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if system:
            body["system"] = system
        return body

    def _parse(self, payload: dict) -> Completion:
        blocks = payload.get("content") or []
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        usage = payload.get("usage", {})
        return Completion(
            text=text,
            model=payload.get("model", self.model),
            provider=self.name,
            finish_reason=payload.get("stop_reason"),
            usage=Usage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
            ),
        )

    def _parse_stream_chunk(self, data: dict) -> str | None:
        if data.get("type") == "content_block_delta":
            return data.get("delta", {}).get("text")
        return None


class GeminiProvider(_HttpProvider):
    name = "gemini"
    default_model = "gemini-2.0-flash"

    def __init__(self, api_key: str, *, model: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        super().__init__(api_key, model=model, timeout=timeout)
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
            f":generateContent?key={self.api_key}"
        )

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _body(self, messages, model, temperature, max_tokens, stream):  # noqa: ANN001, ANN202
        system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        contents = [
            {
                "role": "user" if m.role == Role.USER else "model",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != Role.SYSTEM
        ]
        body: dict[str, object] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _parse(self, payload: dict) -> Completion:
        candidates = payload.get("candidates") or []
        text = ""
        finish_reason = None
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
            finish_reason = candidates[0].get("finishReason")

        usage = payload.get("usageMetadata", {})
        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
            ),
        )

    def _parse_stream_chunk(self, data: dict) -> str | None:
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts) or None


class EchoProvider(LLMProvider):
    """Offline provider for development and tests.

    Returns a deterministic, obviously-synthetic response derived from the
    prompt. It exists so every layer above the gateway — RAG, reflection,
    the agent graph — can be exercised end to end without API keys. It performs
    no reasoning, so any test asserting on reasoning quality must use a real
    provider or a purpose-built scripted fake.
    """

    name = "echo"
    default_model = "echo-1"

    def __init__(self, canned_response: str | None = None) -> None:
        self.canned_response = canned_response
        self.calls: list[list[Message]] = []

    def _respond(self, messages: list[Message]) -> str:
        if self.canned_response is not None:
            return self.canned_response
        last_user = next(
            (m.content for m in reversed(messages) if m.role == Role.USER), ""
        )
        return f"[echo] {last_user[:500]}"

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Completion:
        self.calls.append(list(messages))
        text = self._respond(messages)
        return Completion(
            text=text,
            model=model or self.default_model,
            provider=self.name,
            finish_reason="stop",
            # Rough but consistent with the 4-chars-per-token estimate used
            # elsewhere, so budget accounting behaves realistically offline.
            usage=Usage(
                prompt_tokens=sum(len(m.content) for m in messages) // 4,
                completion_tokens=len(text) // 4,
            ),
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        for word in self._respond(messages).split(" "):
            yield word + " "


class LLMGateway:
    """Routes requests to a primary provider, with optional fallback.

    Fallback covers provider-level outages, which are the failure mode retries
    can't fix. It deliberately does not fall back on 4xx: a malformed request
    or content-filter refusal would fail identically everywhere, and retrying
    it elsewhere just doubles the cost and latency.
    """

    def __init__(self, primary: LLMProvider, fallback: LLMProvider | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.total_usage = Usage()

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Completion:
        try:
            completion = await self.primary.complete(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            )
        except LLMError as exc:
            if self.fallback is None:
                raise
            logger.warning(
                "llm_fallback", primary=self.primary.name, fallback=self.fallback.name,
                error=str(exc),
            )
            completion = await self.fallback.complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )

        self.total_usage = self.total_usage + completion.usage
        logger.info(
            "llm_completion",
            provider=completion.provider,
            model=completion.model,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            truncated=completion.was_truncated,
        )
        return completion

    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        # Streaming has no fallback: tokens already sent to the client cannot be
        # retracted, so a mid-stream switch would produce a spliced answer.
        return self.primary.stream(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )


def build_provider(name: str, settings) -> LLMProvider:  # noqa: ANN001
    name = name.lower()
    if name == "openai":
        return OpenAIProvider(settings.openai_api_key, model=settings.llm_model or None)
    if name == "anthropic":
        return AnthropicProvider(settings.anthropic_api_key, model=settings.llm_model or None)
    if name == "gemini":
        return GeminiProvider(settings.gemini_api_key, model=settings.llm_model or None)
    if name == "echo":
        return EchoProvider()
    raise LLMError(
        f"Unknown LLM provider {name!r}; expected one of: openai, anthropic, gemini, echo"
    )


_gateway: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    global _gateway
    if _gateway is not None:
        return _gateway

    from app.core.config import get_settings

    settings = get_settings()
    primary = build_provider(settings.default_llm_provider, settings)

    fallback = None
    if settings.fallback_llm_provider:
        try:
            fallback = build_provider(settings.fallback_llm_provider, settings)
        except LLMError as exc:
            # A missing fallback key shouldn't prevent the primary from working.
            logger.warning("llm_fallback_unavailable", error=str(exc))

    _gateway = LLMGateway(primary, fallback)
    logger.info(
        "llm_gateway_initialized",
        primary=primary.name,
        fallback=fallback.name if fallback else None,
    )
    return _gateway


def reset_gateway() -> None:
    global _gateway
    _gateway = None
