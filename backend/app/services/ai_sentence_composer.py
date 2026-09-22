"""
ai_sentence_composer -- optional, OFF-BY-DEFAULT fallback that asks a real
LLM (Google Gemini's free-tier API) to compose a natural Ukrainian sentence
from the same recognized Ukrainian words ml/nlp/gloss_to_text.py's
rule-based composer already tried and couldn't fit a supported pattern for
(compose_sentence raised UnsupportedPatternError/UnknownGlossError).

This is deliberately never the primary path. The rule-based composer is
the only thing allowed to claim "this sentence is grammatically guaranteed
within its coverage" -- it raises rather than guessing (see its own module
docstring). A real LLM call on real recognized words is genuine inference,
not a fabrication -- but it can still pick a wrong case ending, word order,
or (rarely) drop/invent a word, so a caller MUST label its output as
AI-generated and never present it with the rule-based result's confidence
(see app/schemas/translate.py's composed_text_source field, which every
route setting composed_text from here also sets to "ai_fallback").

Disabled by default: requires GEMINI_API_KEY (see .env.example, and
app/core/config.py). With no key configured, compose_with_ai() always
returns None immediately -- no network call, same as the rule-based
composer being unable to compose (never treated as an error by callers).
"""
from __future__ import annotations

import httpx

from app.core.config import Settings

GEMINI_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Explicitly forbids inventing or dropping words -- the LLM's only job is
# grammar (case endings, prepositions, word order, punctuation), not
# composing new content. Words are handed over already in Ukrainian (via
# ml/nlp/gloss_to_text.py's gloss_display_labels(), never the raw internal
# gloss codes like "WANT") so the model isn't asked to translate anything,
# only to grammatically assemble what was already understood.
_PROMPT_TEMPLATE = """Склади ОДНЕ граматично правильне українське речення з цих слів, використавши кожне рівно один раз (можеш відмінювати/дієвідмінювати слова, додавати прийменники та розділові знаки, але НЕ додавай нових значущих слів і не вигадуй зміст):

{words}

Дай відповідь ЛИШЕ самим реченням, без лапок і пояснень."""


def compose_with_ai(words: list[str], settings: Settings, client: httpx.Client | None = None) -> str | None:
    """Real call to Gemini asking it to grammatically compose `words` (a
    list of real Ukrainian words/phrases, e.g. from
    gloss_display_labels()) into one sentence. Returns None -- never
    raises -- when the feature is disabled (no API key), `words` is
    empty, the request fails, or the response is empty/malformed: this is
    a fallback, so its own failure must never break the caller's
    response, exactly like the rule-based composer's own
    None-on-unsupported-pattern contract (see translate.py's routes).

    `client` is injectable so tests can supply a fake transport instead of
    making a real network call -- see
    backend/tests/test_ai_sentence_composer.py."""
    if not settings.gemini_api_key or not words:
        return None

    prompt = _PROMPT_TEMPLATE.format(words="\n".join(f"- {word}" for word in words))
    url = GEMINI_ENDPOINT_TEMPLATE.format(model=settings.gemini_model)

    owns_client = client is None
    http_client = client if client is not None else httpx.Client(timeout=8.0)
    try:
        response = http_client.post(
            url,
            params={"key": settings.gemini_api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
        response.raise_for_status()
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip()
        return text or None
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return None
    finally:
        if owns_client:
            http_client.close()
