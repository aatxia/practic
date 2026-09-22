"""
vocabulary — the single source of truth for which words the CAMERA-based
dynamic-sign recognizer (ml/training/train.py -> models/checkpoints/) is
trained to recognize now, and which additional words are proposed next.

This is a narrower, distinct concept from ml/nlp/lexicon.py: lexicon.py is
the full grammar/inflection vocabulary used for TEXT<->gloss composition
(~50 words, hand-authored, no camera model needed for that direction at
all). This module is only about the much smaller set of glosses a trained
LSTM checkpoint actually outputs from a live video window -- adding a word
here does nothing by itself; it still has to be recorded, trained, and
land in a real checkpoint's own label_to_index. The checkpoint file is
always the final authority on what it can predict (see
ml/inference/recognizer.py's `labels`) -- this module exists so that "what
should be trained next" and "what is trained now" are defined once, not
re-typed independently in the recording script, the training report, and
the startup compatibility check below.
"""
from __future__ import annotations

from ml.nlp.lexicon import (
    ADVERBS,
    NEGATION_GLOSS,
    NOUNS,
    PRONOUNS,
    STANDALONE,
    TENSE_PAST_GLOSS,
    VERBS,
)

# The exact 7 classes models/checkpoints/real_v1/latest.pt was trained on
# (PROJECT_STATUS.md, "Перший реальний word-level класифікатор"). Update
# this the moment a differently-scoped checkpoint becomes the one
# MODEL_CHECKPOINT_PATH points at in production, so it always describes
# the checkpoint actually configured -- never the aspirational target.
TRAINED_DYNAMIC_WORDS: tuple[str, ...] = (
    "HAVE", "I", "NOT", "PAST", "WANT", "WATER", "WE",
)

# Next 8 dynamic classes to record and train -- NOT yet recorded or
# trained. Reuses gloss codes ml/nlp/lexicon.py already defines
# (YOU/EAT/DRINK/GO) wherever one already existed; NEED/HELP/FOOD/HOME
# were added to the lexicon alongside this list specifically so the
# recording tool (scripts/collect_dynamic_signs.py) can show a real
# Ukrainian prompt for every one of them, not a raw gloss code fallback.
PROPOSED_NEW_DYNAMIC_WORDS: tuple[str, ...] = (
    "YOU",   # ти
    "NEED",  # потребувати / need
    "HELP",  # допомагати / help
    "FOOD",  # їжа / food
    "EAT",   # їсти / eat
    "DRINK", # пити / drink
    "GO",    # іти / go
    "HOME",  # додому / home
)

# What real_v2 should be trained on once PROPOSED_NEW_DYNAMIC_WORDS has
# real recordings behind it. Deliberately a plain concatenation, not a
# set/sorted union -- order here is display order only (train.py's own
# build_label_vocabulary() re-sorts for the actual label_to_index), and
# duplicating a word between the two tuples above is a real bug worth an
# obvious error, not something to silently deduplicate.
TARGET_DYNAMIC_VOCABULARY: tuple[str, ...] = TRAINED_DYNAMIC_WORDS + PROPOSED_NEW_DYNAMIC_WORDS

# Every gloss code any part of this app currently knows how to handle --
# the grammar lexicon's full vocabulary, plus every past/future dynamic
# class above, plus the negation/tense grammar markers. Used only for the
# startup compatibility check below: a checkpoint class outside this set
# is not a smaller/older/dev vocabulary (those are all still full lexicon
# words and stay valid), it is a class nothing downstream -- NLP
# composition, ml/nlp/gloss_to_text.py, frontend glossLabels.ts -- was
# ever told to expect.
KNOWN_GLOSSES: frozenset[str] = frozenset(
    set(PRONOUNS) | set(VERBS) | set(NOUNS) | set(ADVERBS) | set(STANDALONE)
    | {NEGATION_GLOSS, TENSE_PAST_GLOSS}
    | set(TARGET_DYNAMIC_VOCABULARY)
)


class VocabularyMismatchError(RuntimeError):
    """A loaded checkpoint predicts at least one class nothing else in the
    app was told to expect. Raised instead of silently trusting whichever
    labels happen to be in the checkpoint file -- see inference_provider.py,
    which turns this into the same honest "not ready" behavior already
    used for a missing checkpoint file, rather than crashing the process
    or serving predictions that would show up as a raw, unmapped gloss
    code in the NLP/frontend layers."""


def assert_labels_are_known(checkpoint_labels: frozenset[str] | set[str]) -> None:
    """Real-value check performed at inference-service construction time:
    every class the checkpoint can predict must be a word the rest of the
    app (lexicon.py's grammar, or a currently-trained/proposed dynamic
    class) actually knows about. Deliberately NOT an exact-equality check
    against TRAINED_DYNAMIC_WORDS -- a checkpoint trained on a real subset
    of known words (e.g. a small dev/test checkpoint) is still an honest,
    usable model for the words it does support; only a genuinely unknown
    label (a typo, or a class from an unrelated experiment) is a real
    mismatch worth failing on."""
    unknown = frozenset(checkpoint_labels) - KNOWN_GLOSSES
    if unknown:
        raise VocabularyMismatchError(
            f"Loaded checkpoint predicts unrecognized class(es) {sorted(unknown)} -- "
            "not in ml/nlp/lexicon.py or ml/nlp/vocabulary.py's dynamic word lists. "
            "Add the word to the lexicon/vocabulary first, or point "
            "MODEL_CHECKPOINT_PATH at the intended checkpoint."
        )
