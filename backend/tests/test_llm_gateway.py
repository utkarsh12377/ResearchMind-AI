import httpx
import pytest

from app.llm.gateway import (
    AnthropicProvider,
    Completion,
    EchoProvider,
    LLMError,
    LLMGateway,
    LLMProvider,
    Message,
    OpenAIProvider,
    Role,
    Usage,
)


class FailingProvider(LLMProvider):
    name = "failing"
    default_model = "fail-1"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or LLMError("provider is down")
        self.calls = 0

    async def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.calls += 1
        raise self.error

    async def stream(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        raise self.error
        yield ""  # pragma: no cover


def test_usage_addition() -> None:
    total = Usage(10, 5) + Usage(3, 2)

    assert total.prompt_tokens == 13
    assert total.completion_tokens == 7
    assert total.total_tokens == 20


@pytest.mark.parametrize(
    ("reason", "expected"), [("length", True), ("max_tokens", True), ("stop", False), (None, False)]
)
def test_truncation_is_detected(reason, expected) -> None:  # noqa: ANN001
    completion = Completion(text="x", model="m", provider="p", finish_reason=reason)

    assert completion.was_truncated is expected


@pytest.mark.asyncio
async def test_echo_provider_returns_a_deterministic_completion() -> None:
    provider = EchoProvider()

    first = await provider.complete([Message(Role.USER, "hello")])
    second = await provider.complete([Message(Role.USER, "hello")])

    assert first.text == second.text
    assert first.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_echo_provider_streams_the_same_text() -> None:
    provider = EchoProvider(canned_response="one two three")

    chunks = [chunk async for chunk in provider.stream([Message(Role.USER, "q")])]

    assert "".join(chunks).strip() == "one two three"


@pytest.mark.asyncio
async def test_gateway_accumulates_usage_across_calls() -> None:
    gateway = LLMGateway(EchoProvider(canned_response="abcd"))

    await gateway.complete([Message(Role.USER, "first")])
    await gateway.complete([Message(Role.USER, "second")])

    assert gateway.total_usage.total_tokens > 0


@pytest.mark.asyncio
async def test_gateway_falls_back_when_the_primary_fails() -> None:
    primary = FailingProvider()
    gateway = LLMGateway(primary, fallback=EchoProvider(canned_response="from fallback"))

    completion = await gateway.complete([Message(Role.USER, "question")])

    assert completion.text == "from fallback"
    assert primary.calls == 1


@pytest.mark.asyncio
async def test_gateway_raises_when_there_is_no_fallback() -> None:
    gateway = LLMGateway(FailingProvider())

    with pytest.raises(LLMError):
        await gateway.complete([Message(Role.USER, "question")])


def test_providers_require_an_api_key() -> None:
    with pytest.raises(LLMError, match="requires an API key"):
        OpenAIProvider("")


def test_anthropic_lifts_system_messages_out_of_the_conversation() -> None:
    provider = AnthropicProvider("key")

    body = provider._body(
        [Message(Role.SYSTEM, "be precise"), Message(Role.USER, "hi")],
        "claude-sonnet-4-5",
        0.2,
        100,
        stream=False,
    )

    # Anthropic takes `system` as a top-level field, not a message role.
    assert body["system"] == "be precise"
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_openai_keeps_system_messages_inline() -> None:
    provider = OpenAIProvider("key")

    body = provider._body(
        [Message(Role.SYSTEM, "be precise"), Message(Role.USER, "hi")],
        "gpt-4o-mini",
        0.2,
        100,
        stream=False,
    )

    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_openai_parses_usage_and_finish_reason() -> None:
    provider = OpenAIProvider("key")

    completion = provider._parse(
        {
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
    )

    assert completion.text == "answer"
    assert completion.usage.prompt_tokens == 12
    assert completion.finish_reason == "stop"


def test_anthropic_concatenates_text_blocks() -> None:
    provider = AnthropicProvider("key")

    completion = provider._parse(
        {
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "text", "text": "part one "},
                {"type": "thinking", "text": "ignored"},
                {"type": "text", "text": "part two"},
            ],
            "usage": {"input_tokens": 9, "output_tokens": 3},
        }
    )

    assert completion.text == "part one part two"
    assert completion.usage.completion_tokens == 3


@pytest.mark.asyncio
async def test_client_errors_are_not_retried(monkeypatch) -> None:  # noqa: ANN001
    attempts = {"count": 0}

    async def fake_post(self, url, headers=None, json=None):  # noqa: ANN001, ANN202
        attempts["count"] += 1
        return httpx.Response(400, text="bad request", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(LLMError, match="rejected the request"):
        await OpenAIProvider("key").complete([Message(Role.USER, "q")])

    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_rate_limits_are_retried(monkeypatch) -> None:  # noqa: ANN001
    attempts = {"count": 0}

    async def fake_post(self, url, headers=None, json=None):  # noqa: ANN001, ANN202
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(429, text="slow down", request=httpx.Request("POST", url))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            request=httpx.Request("POST", url),
        )

    async def no_sleep(_delay):  # noqa: ANN001, ANN202
        return None

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr("app.llm.gateway.asyncio.sleep", no_sleep)

    completion = await OpenAIProvider("key").complete([Message(Role.USER, "q")])

    assert completion.text == "ok"
    assert attempts["count"] == 3
