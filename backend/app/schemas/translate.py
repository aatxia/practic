from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TextToGlossRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Ukrainian text to translate into a gloss sequence")


class GlossLabel(BaseModel):
    text: str = Field(..., description="A Ukrainian word/phrase, ready to show a user")
    is_fingerspell: bool = Field(
        ..., description="True if this word came from letter-by-letter fingerspelling, not the dictionary"
    )


class GlossToTextRequest(BaseModel):
    gloss_sequence: list[str] = Field(
        ..., min_length=1, description="УЖМ gloss tokens accumulated from confirmed live camera predictions"
    )


class GlossToTextResponse(BaseModel):
    gloss_labels: list[GlossLabel] = Field(
        ..., description="Ukrainian word/phrase per gloss token (PAST tense markers omitted -- not a word)"
    )
    composed_text: str | None = Field(
        None,
        description="Full composed Ukrainian sentence, when the gloss sequence matches a "
        "supported grammatical pattern OR the optional AI fallback (see composed_text_source) "
        "produced one; null when neither did (still shown via gloss_labels)",
    )
    composed_text_source: Literal["rule_based", "ai_fallback"] | None = Field(
        None,
        description="Which engine produced composed_text -- 'rule_based' is grammatically "
        "guaranteed within ml/nlp/gloss_to_text.py's coverage; 'ai_fallback' is a real LLM's "
        "best attempt (app/services/ai_sentence_composer.py) used only when rule_based couldn't "
        "compose a pattern, and should be shown to the user as AI-generated, not verified. "
        "null when composed_text itself is null.",
    )


class TextToGlossResponse(BaseModel):
    gloss_sequence: list[str] = Field(..., description="УЖМ gloss tokens, in signing order")
    # The frontend must never show gloss_sequence's own tokens (internal
    # English identifiers like "WANT"/"CAR") to the user -- these two
    # fields are the actual Ukrainian words to display instead.
    gloss_labels: list[GlossLabel] = Field(
        ..., description="Ukrainian word/phrase per gloss token (PAST tense markers omitted -- not a word)"
    )
    composed_text: str | None = Field(
        None,
        description="Full composed Ukrainian sentence, when the gloss sequence matches a "
        "supported grammatical pattern; null otherwise (still shown via gloss_labels)",
    )
