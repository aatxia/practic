"""
text_to_gloss — rule-based Ukrainian text -> УЖМ gloss sequence (section 14,
Phase 14). The reverse of ml/nlp/gloss_to_text.py's compose_sentence(),
built from the same shared lexicon (ml/nlp/lexicon.py) so the two directions
never silently disagree about vocabulary -- anything this project can
compose into text, it can also parse back out of text, and vice versa.

Framework-agnostic (no FastAPI import); wrapped by
backend/app/services/translation_service.py::RuleBasedTranslationService.

Word order is NOT reordered into sign-language grammar (topic-comment,
classifier constructions, etc.) -- that's a substantially harder linguistic
problem this rule-based baseline doesn't attempt. Glosses come out in the
same order their Ukrainian surface forms appeared in the input text. That
happens to already match the gloss order gloss_to_text.py composes (SVO
with preverbal negation, since Ukrainian negation is already preverbal) for
every pattern this project currently supports -- a real limitation for any
future pattern where sign order and spoken order would genuinely differ.

A word in the input that isn't in the lexicon is fingerspelled letter-by-
letter (Phase 16, ml/nlp/fingerspelling.py) rather than dropped or guessed
-- a real fallback deaf signers use for names/loanwords, not a workaround.

A past-tense verb form (e.g. "їхала") expands into TWO gloss tokens, the
TENSE_PAST_GLOSS marker followed by the verb's gloss -- not folded into a
single "past-tense" gloss -- since lexicon.py represents tense as a
separate marker gloss (see its module docstring). A noun in subject
position (e.g. "Машина" in "Машина їхала...") isn't special-cased here at
all: it's just another recognized word, resolved via the same reverse
index nouns already populate for object position (their "nominative" form
was always indexed) -- gloss_to_text.py is what decides whether a noun in
that position forms a valid sentence.
Only a word containing a character with no dactyl handshape (Latin script,
digits, ...) raises UnrecognizedWordError.
"""
from __future__ import annotations

import re

from ml.nlp.fingerspelling import (
    FINGERSPELL_WORD_BOUNDARY,
    UnspellableCharacterError,
    is_fingerspell_gloss,
    spell_word,
)
from ml.nlp.lexicon import (
    ADVERBS,
    NEGATION_GLOSS,
    NEGATION_PARTICLE,
    NOUNS,
    PRONOUNS,
    STANDALONE,
    TENSE_PAST_GLOSS,
    VERBS,
)

_TRAILING_PUNCTUATION = re.compile(r"[.,!?;:]+$")


class UnrecognizedWordError(ValueError):
    """Raised when a word isn't in the lexicon AND can't be fingerspelled
    (contains a character with no dactyl handshape)."""


def _strip_punctuation(word: str) -> str:
    return _TRAILING_PUNCTUATION.sub("", word)


def _build_reverse_index() -> dict[str, str]:
    """Surface Ukrainian word -> gloss token, built once from the shared
    lexicon: every pronoun, every PRESENT-tense conjugated verb form,
    every declined noun form (including "nominative" -- a noun can appear
    either as an object, already covered here, or as a subject, e.g.
    "Машина їхала..."; both directions read the same case name), and
    every invariant time adverb. Multiple grammatical forms of the same
    word (e.g. "води"/"воду" for WATER) all resolve to the same gloss --
    text_to_gloss only needs to recover *which word*, not which case it
    was inflected for. PAST-tense verb forms are handled separately (see
    `_build_past_verb_index()`) since a single surface word like "їхала"
    must expand into *two* gloss tokens (PAST + the verb), not one."""
    index: dict[str, str] = {}
    for gloss, entry in PRONOUNS.items():
        index[entry.lemma.lower()] = gloss
    for gloss, entry in VERBS.items():
        for surface_form in entry.conjugation.values():
            index[surface_form] = gloss
    for gloss, entry in NOUNS.items():
        for surface_form in entry.cases.values():
            index[surface_form] = gloss
    for gloss, entry in ADVERBS.items():
        index[entry.text] = gloss
    return index


def _build_past_verb_index() -> dict[str, str]:
    """Past-tense surface form -> verb gloss (e.g. "їхала" -> "RIDE").
    Kept separate from `_REVERSE_INDEX` because recognizing one of these
    words must emit the TENSE_PAST_GLOSS marker before the verb gloss --
    see lexicon.py's module docstring for why past tense is a marker
    gloss rather than a distinct verb-form gloss."""
    index: dict[str, str] = {}
    for gloss, entry in VERBS.items():
        for surface_form in entry.past.values():
            index[surface_form] = gloss
    return index


_REVERSE_INDEX = _build_reverse_index()
_PAST_VERB_INDEX = _build_past_verb_index()


def parse_gloss_sequence(text: str) -> list[str]:
    """Ukrainian text -> gloss sequence. See module docstring for coverage;
    an unrecognized word raises UnrecognizedWordError naming it."""
    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("text must not be empty")

    # A standalone phrase (interjection) is matched against the whole
    # normalized text first, since "будь ласка" is two words but one gloss.
    stripped_whole = _strip_punctuation(normalized)
    for gloss, entry in STANDALONE.items():
        if stripped_whole == entry.text:
            return [gloss]

    words = [_strip_punctuation(word) for word in normalized.split()]
    words = [word for word in words if word]
    if not words:
        raise ValueError("text must not be empty")

    gloss_sequence: list[str] = []
    for word in words:
        if word == NEGATION_PARTICLE:
            gloss_sequence.append(NEGATION_GLOSS)
            continue

        past_verb_gloss = _PAST_VERB_INDEX.get(word)
        if past_verb_gloss is not None:
            gloss_sequence.append(TENSE_PAST_GLOSS)
            gloss_sequence.append(past_verb_gloss)
            continue

        gloss = _REVERSE_INDEX.get(word)
        if gloss is not None:
            gloss_sequence.append(gloss)
            continue

        try:
            letters = spell_word(word)
        except UnspellableCharacterError as exc:
            raise UnrecognizedWordError(
                f"Unrecognized word {word!r} in {text!r} -- not in the Phase 12/14 "
                f"lexicon, and can't be fingerspelled either: {exc}"
            ) from exc
        # Two fingerspelled words in a row (both outside the lexicon) need a
        # real separator, or a consumer collapsing a run of FS_ letters back
        # into a word would glue them into one -- see FINGERSPELL_WORD_BOUNDARY's
        # own docstring.
        if gloss_sequence and is_fingerspell_gloss(gloss_sequence[-1]):
            gloss_sequence.append(FINGERSPELL_WORD_BOUNDARY)
        gloss_sequence.extend(letters)

    return gloss_sequence
