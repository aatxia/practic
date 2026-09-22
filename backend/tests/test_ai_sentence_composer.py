"""
Tests for app/services/ai_sentence_composer.py. Never hits the real Gemini
API -- httpx.MockTransport (httpx's own documented test mechanism) swaps
out only the network transport, so the real request-building/response-
parsing code in compose_with_ai() still runs, just against a fake handler
instead of a real socket.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import Settings
from app.services.ai_sentence_composer import compose_with_ai


def _settings(api_key: str = "fake-test-key") -> Settings:
    return Settings(gemini_api_key=api_key, gemini_model="gemini-2.0-flash", _env_file=None)


def _client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _gemini_response(text: str) -> httpx.Response:
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


def test_returns_none_immediately_when_no_api_key_is_configured():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _gemini_response("should never be reached")

    result = compose_with_ai(["Я", "хотіти", "вода"], _settings(api_key=""), client=_client_with(handler))

    assert result is None
    assert calls == []  # no network call attempted at all


def test_returns_none_for_an_empty_word_list():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API for an empty word list")

    result = compose_with_ai([], _settings(), client=_client_with(handler))

    assert result is None


def test_returns_the_real_composed_sentence_from_a_successful_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return _gemini_response("Я хочу води.")

    result = compose_with_ai(["Я", "хотіти", "вода"], _settings(), client=_client_with(handler))

    assert result == "Я хочу води."


def test_sends_the_api_key_and_every_word_in_the_request():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        # Parse the real JSON body rather than substring-searching the raw
        # wire bytes: httpx's JSON encoder escapes non-ASCII as \uXXXX in
        # some versions and sends raw UTF-8 in others (both valid JSON,
        # same decoded value) -- a raw substring check is version-dependent
        # and would only pass by accident.
        captured["body"] = json.loads(request.content)
        return _gemini_response("Я хочу води.")

    compose_with_ai(["Я", "хотіти", "вода"], _settings(api_key="my-real-key"), client=_client_with(handler))

    assert "key=my-real-key" in captured["url"]
    prompt_text = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "Я" in prompt_text
    assert "хотіти" in prompt_text
    assert "вода" in prompt_text


def test_strips_surrounding_whitespace_from_the_response_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return _gemini_response("  Я хочу води.  \n")

    result = compose_with_ai(["Я", "хотіти", "вода"], _settings(), client=_client_with(handler))

    assert result == "Я хочу води."


def test_returns_none_for_an_empty_response_text_rather_than_an_empty_string():
    def handler(request: httpx.Request) -> httpx.Response:
        return _gemini_response("   ")

    result = compose_with_ai(["Я"], _settings(), client=_client_with(handler))

    assert result is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="internal error"),
        httpx.Response(429, text="rate limited"),
        httpx.Response(200, json={"candidates": []}),
        httpx.Response(200, json={"unexpected": "shape"}),
        httpx.Response(200, text="not json"),
    ],
)
def test_never_raises_and_returns_none_for_any_kind_of_api_failure(response):
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    result = compose_with_ai(["Я", "хотіти"], _settings(), client=_client_with(handler))

    assert result is None


def test_returns_none_when_the_request_itself_fails_to_send():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)

    result = compose_with_ai(["Я", "хотіти"], _settings(), client=_client_with(handler))

    assert result is None
