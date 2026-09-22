"""
These tests encode the project's "NO FAKE AI" rule: service interfaces that
depend on an untrained ML model or unbuilt NLP layer must raise a clear,
typed error instead of returning a plausible-looking but fabricated result.
"""
import pytest

from app.services.inference_service import (
    MLNotReadyError,
    NotConfiguredInferenceService,
)
from app.services.translation_service import (
    NotConfiguredError as TranslationNotConfiguredError,
)
from app.services.translation_service import NotConfiguredTranslationService


def test_inference_service_refuses_to_predict_without_a_model():
    service = NotConfiguredInferenceService()
    assert service.is_ready() is False
    with pytest.raises(MLNotReadyError):
        service.predict(landmark_sequence=[[0.0] * 63])


def test_translation_service_refuses_gloss_to_text_before_phase_12():
    service = NotConfiguredTranslationService()
    with pytest.raises(TranslationNotConfiguredError):
        service.gloss_to_text(["I", "WANT", "WATER"])


def test_translation_service_refuses_text_to_gloss_before_phase_14():
    service = NotConfiguredTranslationService()
    with pytest.raises(TranslationNotConfiguredError):
        service.text_to_gloss("Я хочу води.")
