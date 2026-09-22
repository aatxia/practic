"""
TranslationService — abstraction over gloss-sequence -> natural Ukrainian text
(and the reverse direction, text -> gloss sequence for the avatar).

STATUS: both directions are real as of Phase 14. RuleBasedTranslationService
delegates gloss_to_text() to ml/nlp/gloss_to_text.py (Phase 12) and
text_to_gloss() to ml/nlp/text_to_gloss.py (Phase 14) -- genuine (if narrow,
hand-authored lexicon) rule-based engines per master-prompt section 14/17,
sharing one lexicon (ml/nlp/lexicon.py) so the two directions can never
silently disagree about vocabulary.

Later an HF-Transformers-backed implementation can replace either direction
via `NLP_BACKEND=huggingface` in .env without touching callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ml.nlp.gloss_to_text import compose_sentence
from ml.nlp.text_to_gloss import parse_gloss_sequence


class NotConfiguredError(RuntimeError):
    """Raised when a translation direction is requested before it's implemented."""


class TranslationService(ABC):
    @abstractmethod
    def gloss_to_text(self, gloss_sequence: list[str], *, is_question: bool = False) -> str:
        """Gloss sequence (e.g. ['I', 'WANT', 'WATER']) -> natural Ukrainian
        sentence (e.g. 'Я хочу води.'), applying case/number/gender/word-order
        rules -- NOT a naive ' '.join(). Implemented in Phase 12. `is_question`
        (Phase 17) ends the sentence with "?" -- pass it when a non-manual
        question marker (ml/features/facial_grammar.py) was detected."""
        raise NotImplementedError

    @abstractmethod
    def text_to_gloss(self, text: str) -> list[str]:
        """Ukrainian text -> gloss sequence, for driving the avatar.
        Implemented in Phase 14."""
        raise NotImplementedError


class NotConfiguredTranslationService(TranslationService):
    def gloss_to_text(self, gloss_sequence: list[str], *, is_question: bool = False) -> str:
        raise NotConfiguredError(
            "Gloss-to-text NLP is not implemented yet (scheduled: Phase 12). "
            "See docs/architecture.md and PROJECT_STATUS.md."
        )

    def text_to_gloss(self, text: str) -> list[str]:
        raise NotConfiguredError(
            "Text-to-gloss NLP is not implemented yet (scheduled: Phase 14). "
            "See docs/architecture.md and PROJECT_STATUS.md."
        )


class RuleBasedTranslationService(TranslationService):
    """Phase 12 (gloss_to_text) + Phase 14 (text_to_gloss): both real
    rule-based engines sharing ml/nlp/lexicon.py (see ml/nlp/gloss_to_text.py
    and ml/nlp/text_to_gloss.py for exactly which glosses/words/patterns
    they cover -- coverage is intentionally small, since no public
    annotated УЖМ dataset exists yet to build a broader lexicon from). An
    unrecognized gloss/word or an unsupported combination raises a clear
    ValueError subclass rather than guessing.
    """

    def gloss_to_text(self, gloss_sequence: list[str], *, is_question: bool = False) -> str:
        return compose_sentence(gloss_sequence, is_question=is_question)

    def text_to_gloss(self, text: str) -> list[str]:
        return parse_gloss_sequence(text)
