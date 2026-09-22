"""
POST /translate/text-to-gloss — Phase 14: Ukrainian text -> gloss sequence,
via RuleBasedTranslationService (ml/nlp/text_to_gloss.py). A real (if
narrow-coverage) rule-based parse, not a stub -- an unrecognized word gets a
clear 422 naming it, never a guessed gloss sequence. The frontend displays
the returned gloss_sequence (components/Transcript) and drives the 3D
avatar with it (components/Avatar, Phase 15) -- gloss->pose mapping turned
out to be simple static data, so it lives entirely client-side
(components/Avatar/poses.ts) rather than as a backend service/endpoint.

gloss_sequence's own tokens (e.g. "WANT", "CAR") are internal English
identifiers, not Ukrainian -- never meant for a person to read. Alongside
it, this also returns gloss_labels (the real Ukrainian word per token,
ml/nlp/gloss_to_text.py::gloss_display_labels) and, when the sequence
matches a supported grammatical pattern, composed_text (the actual
Ukrainian sentence, same engine Phase 12 uses for gloss->text). Composing
never turns a successful parse into a 422: an UnsupportedPatternError here
just means composed_text is null, not that the translation failed --
gloss_labels still lets the user see what was understood.

POST /translate/gloss-to-text is the reverse direction -- Phase 20:
composes a full sentence from an accumulated sequence of live-camera-
confirmed glosses (the frontend gathers every WebSocket final_prediction's
gloss while the signer keeps going, then posts the whole sequence here
once they press "Зупинити"/Stop, instead of composing each confirmed sign
alone as it arrives). Same composition engine, same "null composed_text
rather than a guess" honesty rule.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.schemas.translate import (
    GlossLabel,
    GlossToTextRequest,
    GlossToTextResponse,
    TextToGlossRequest,
    TextToGlossResponse,
)
from app.services.ai_sentence_composer import compose_with_ai
from app.services.translation_service import RuleBasedTranslationService
from ml.nlp.gloss_to_text import compose_sentence as _compose_sentence
from ml.nlp.gloss_to_text import gloss_display_labels

router = APIRouter(prefix="/translate", tags=["translate"])

_translation_service = RuleBasedTranslationService()


@router.post("/text-to-gloss", response_model=TextToGlossResponse)
def text_to_gloss(request: TextToGlossRequest) -> TextToGlossResponse:
    try:
        gloss_sequence = _translation_service.text_to_gloss(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        composed_text = _compose_sentence(gloss_sequence)
    except ValueError:
        composed_text = None

    return TextToGlossResponse(
        gloss_sequence=gloss_sequence,
        gloss_labels=[
            GlossLabel(text=text, is_fingerspell=is_fingerspell)
            for text, is_fingerspell in gloss_display_labels(gloss_sequence)
        ],
        composed_text=composed_text,
    )


@router.post("/gloss-to-text", response_model=GlossToTextResponse)
def gloss_to_text(request: GlossToTextRequest) -> GlossToTextResponse:
    """Composes a full Ukrainian sentence from an accumulated sequence of
    confirmed live-camera glosses (frontend: signing continuously, then
    pressing "Зупинити" gathers every final_prediction's gloss seen since
    the last stop/start and posts the whole sequence here at once, instead
    of composing each gloss alone as it arrives). Same composition engine
    as text-to-gloss (Phase 12/ml/nlp/gloss_to_text.py) -- a sequence that
    doesn't match a supported grammatical pattern gets composed_text=None
    from the rule-based engine, never a guessed sentence.

    Only THEN, if configured (GEMINI_API_KEY set), is the optional AI
    fallback (app/services/ai_sentence_composer.py) asked to compose the
    same real understood words instead -- composed_text_source tells the
    caller which one actually produced composed_text, since only
    "rule_based" is grammar-guaranteed; "ai_fallback" must be shown to the
    user as AI-generated, not verified. gloss_labels always shows what was
    understood, regardless of whether either composer succeeded."""
    labels = gloss_display_labels(request.gloss_sequence)

    try:
        composed_text = _compose_sentence(request.gloss_sequence)
        composed_text_source = "rule_based" if composed_text is not None else None
    except ValueError:
        composed_text = None
        composed_text_source = None

    if composed_text is None:
        ai_text = compose_with_ai([text for text, _ in labels], get_settings())
        if ai_text is not None:
            composed_text = ai_text
            composed_text_source = "ai_fallback"

    return GlossToTextResponse(
        gloss_labels=[GlossLabel(text=text, is_fingerspell=is_fingerspell) for text, is_fingerspell in labels],
        composed_text=composed_text,
        composed_text_source=composed_text_source,
    )
