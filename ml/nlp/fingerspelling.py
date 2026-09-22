"""
fingerspelling — Ukrainian dactyl alphabet (дактилологія), Phase 16.

Real sign languages don't just fail on an out-of-vocabulary word: a signer
spells it letter-by-letter using a standardized one-handshape-per-letter
alphabet (dactylology). This module is the letter<->gloss half of that --
`ml/nlp/text_to_gloss.py` and `ml/nlp/gloss_to_text.py` use it as a fallback
for any word not in the (intentionally small, Phase 8-blocked) hand-authored
lexicon, instead of refusing outright.

What this project has vs. doesn't:
- The letter<->gloss token mapping is genuine, unambiguous data (the 33
  letters of the Ukrainian alphabet), not a guess.
- Spelling is caseless/formless: it records exactly the letters typed, not a
  grammatically "corrected" citation form -- so composing it back reproduces
  the same letters, nothing more.
- What's NOT here: an actual dactyl handshape for the 3D avatar. The
  avatar's puppet (frontend/components/Avatar/puppet.ts) has no finger
  geometry, so it can't render 33 visually distinct handshapes -- inventing
  fake ones would violate this project's "no fake AI" rule just as much as
  a fabricated ML prediction would. FS_ glosses are real gloss tokens with
  no defined pose (poseForGloss() returns None for them, same as any other
  unmapped gloss), so the avatar honestly holds neutral rather than faking
  a handshape.
"""
from __future__ import annotations

# The 33 letters of the modern Ukrainian alphabet. Apostrophe (') and
# hyphen (-) are punctuation/orthography marks, not letters with their own
# dactyl handshape, so they're deliberately excluded -- a word containing
# them can't be fingerspelled by this module (see spell_word).
UKRAINIAN_ALPHABET: frozenset[str] = frozenset(
    "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
)

FINGERSPELL_PREFIX = "FS_"

# A flat gloss_sequence has no structure beyond the array itself -- two
# regular words are already distinguishable (each is its own token), but
# two DIFFERENT fingerspelled words sitting next to each other (e.g. "тобі
# допомогти", neither in the lexicon) would otherwise both just be runs of
# FS_<letter> tokens with nothing between them, so a consumer that collapses
# a consecutive FS_ run back into one word (ml/nlp/gloss_to_text.py's
# _group_units, lib/glossDisplay.ts, groupGlossesForVideo.ts) would wrongly
# glue them into a single word ("тобідопомогти") instead of two. Inserted
# only when that ambiguity is real (i.e. the previous emitted token was
# also a fingerspelling letter) -- see text_to_gloss.py and useUtterance.ts,
# the two producers of a flat gloss_sequence. Never itself a valid
# fingerspell letter (is_fingerspell_gloss below only matches a single
# letter after the prefix), so a consumer can tell them apart on sight.
FINGERSPELL_WORD_BOUNDARY = "FS_BOUNDARY"


class UnspellableCharacterError(ValueError):
    """Raised when a word contains a character with no dactyl handshape."""


def spell_word(word: str) -> list[str]:
    """A word -> one FS_<LETTER> gloss token per character, in order.
    Raises UnspellableCharacterError naming the exact character if any
    character isn't a Ukrainian dactyl-alphabet letter (digits, Latin
    script, apostrophes, ...)."""
    if not word:
        raise ValueError("word must not be empty")

    letters = word.lower()
    for char in letters:
        if char not in UKRAINIAN_ALPHABET:
            raise UnspellableCharacterError(
                f"Cannot fingerspell {char!r} in {word!r} -- not a letter of "
                "the Ukrainian dactyl alphabet."
            )
    return [f"{FINGERSPELL_PREFIX}{char.upper()}" for char in letters]


def is_fingerspell_gloss(gloss: str) -> bool:
    """True for a well-formed single-letter fingerspelling gloss token."""
    if not gloss.startswith(FINGERSPELL_PREFIX):
        return False
    letter = gloss[len(FINGERSPELL_PREFIX) :]
    return len(letter) == 1 and letter.lower() in UKRAINIAN_ALPHABET


def despell(glosses: list[str]) -> str:
    """The reverse of spell_word: a run of FS_ gloss tokens -> the lowercase
    word they spell. Every gloss must be a fingerspelling token; raises
    ValueError naming the first one that isn't."""
    if not glosses:
        raise ValueError("glosses must not be empty")

    letters = []
    for gloss in glosses:
        if not is_fingerspell_gloss(gloss):
            raise ValueError(f"{gloss!r} is not a fingerspelling gloss token.")
        letters.append(gloss[len(FINGERSPELL_PREFIX) :].lower())
    return "".join(letters)
