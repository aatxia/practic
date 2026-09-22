"""
gloss_to_text — rule-based УЖМ gloss sequence -> natural Ukrainian sentence
(section 14/17). Framework-agnostic (no FastAPI import); wrapped by
backend/app/services/translation_service.py::RuleBasedTranslationService.

This is a genuine (if narrow) grammar engine, not `' '.join(gloss_sequence)`:
it conjugates verbs by subject person, declines nouns into whichever case
the verb governs, handles preverbal negation ("не" before the verb, the
correct position in Ukrainian -- not wherever NOT happened to appear in the
gloss order), and composes standard Ukrainian word order from sign order
(which doesn't have to match it).

Coverage is intentionally small: no public annotated УЖМ dataset exists yet
(Phase 8), so there is no real gloss vocabulary to build a broad lexicon
from -- see ml/nlp/lexicon.py, shared with ml/nlp/text_to_gloss.py (Phase 14)
so the two directions never disagree about vocabulary. An unrecognized
gloss, or a recognized-glosses-but-unrecognized-pattern combination, raises
a clear, typed error rather than guessing a plausible-looking but possibly
ungrammatical sentence -- the same "no fake AI" rule the rest of this
project follows for CV/ML.

Phase 16: a run of consecutive FS_ fingerspelling gloss tokens (see
ml/nlp/fingerspelling.py) composes into the word they spell -- standalone,
or filling the object slot of the SVO pattern. Spelling is caseless/formless
(exactly the letters that were spelled, capitalized), never Ukrainian-
declined into the verb's governed case the way a lexicon NOUN is -- there's
no real declension data for arbitrary fingerspelled words to draw on.

Phase 17: `is_question` swaps the trailing "." for "?" -- sign language
questions are marked non-manually (raised eyebrows for yes/no, furrowed for
wh-, section 19), never by a separate manual gloss, so there's no "?" gloss
token for a caller to include. See ml/features/facial_grammar.py, which
detects this from face landmarks; the caller decides whether it applies.

Subject position accepts a pronoun OR a noun (e.g. ["CAR", "PAST", "RIDE"]
-> "Машина їхала.") -- not only pronoun-subject sentences. An optional
TENSE_PAST_GLOSS marker right before the verb switches it to past tense
(gender/number agreement with the subject, per lexicon.py's module
docstring); zero or more trailing ADVERBS entries (time words) can follow
the verb/object. None of this reorders sign order into a different spoken
order -- these are still exactly the tokens gloss_to_text.py was given,
composed left to right.

The subject is itself optional, but only alongside an explicit PAST
marker (e.g. ["HAVE", "PAST", "WATER"] -> "Мав воду.") -- a real,
elliptical Ukrainian construction for when the doer is contextually
understood, defaulting to the masculine past form (the unmarked default
Ukrainian dictionaries/grammars use when gender isn't otherwise
specified -- a documented rule, not a guess). Present-tense verb+object
with no subject at all remains unsupported -- too ambiguous even
elliptically (a command? a label?) to have one obvious reading.
"""
from __future__ import annotations

from ml.nlp.fingerspelling import FINGERSPELL_WORD_BOUNDARY, despell, is_fingerspell_gloss
from ml.nlp.lexicon import (
    ADVERBS,
    NEGATION_GLOSS,
    NOUNS,
    PRONOUNS,
    STANDALONE,
    TENSE_PAST_GLOSS,
    VERBS,
)


class UnknownGlossError(ValueError):
    """Raised when a gloss token isn't in the lexicon at all."""


class UnsupportedPatternError(ValueError):
    """Raised when every gloss is individually recognized, but their
    combination doesn't match any implemented composition rule."""


def _classify(gloss: str) -> str:
    if gloss in PRONOUNS:
        return "pronoun"
    if gloss in VERBS:
        return "verb"
    if gloss in NOUNS:
        return "noun"
    if gloss in ADVERBS:
        return "adverb"
    if gloss in STANDALONE:
        return "standalone"
    if gloss == NEGATION_GLOSS:
        return "negation"
    if gloss == TENSE_PAST_GLOSS:
        return "tense_past"
    if gloss == FINGERSPELL_WORD_BOUNDARY:
        return "fingerspell_boundary"
    if is_fingerspell_gloss(gloss):
        return "fingerspell"
    raise UnknownGlossError(f"Unknown gloss {gloss!r} -- not in the Phase 12 lexicon yet.")


def _group_units(gloss_sequence: list[str]) -> list[tuple[str, str]]:
    """gloss_sequence -> [(pose, value)], collapsing each consecutive run of
    fingerspelling glosses into one ("fingerspell", <word>) unit. `value` is
    the original gloss for every other pose. Raises UnknownGlossError first,
    same as classifying gloss-by-gloss would."""
    poses = [_classify(gloss) for gloss in gloss_sequence]

    units: list[tuple[str, str]] = []
    i = 0
    while i < len(gloss_sequence):
        if poses[i] == "fingerspell_boundary":
            # A pure separator between two distinct fingerspelled words --
            # contributes no unit of its own (same treatment as tense_past
            # in gloss_display_labels below).
            i += 1
            continue
        if poses[i] != "fingerspell":
            units.append((poses[i], gloss_sequence[i]))
            i += 1
            continue
        j = i
        while j < len(gloss_sequence) and poses[j] == "fingerspell":
            j += 1
        units.append(("fingerspell", despell(gloss_sequence[i:j])))
        i = j
    return units


def gloss_display_labels(gloss_sequence: list[str]) -> list[tuple[str, bool]]:
    """gloss_sequence -> a list of (Ukrainian word/phrase, is_fingerspell)
    pairs, for showing a gloss sequence to a person without leaking the
    internal English gloss identifiers (e.g. "WANT", "CAR") into the UI --
    those are code, not Ukrainian, and were never meant to be read by an
    end user. A pronoun becomes its lemma, a verb its infinitive
    (dictionary citation form, not any one conjugated form), a noun its
    nominative form, an adverb/standalone its own text, negation "не", and
    a fingerspelled run the word it spells (is_fingerspell=True, so a
    caller can still show its own "spelled, not from the dictionary"
    indicator -- the same distinction lib/glossDisplay.ts's
    groupGlossesForDisplay() already draws on the frontend). TENSE_PAST_GLOSS
    has no Ukrainian surface form of its own (see lexicon.py's module
    docstring -- it's a marker, not a word), so it contributes no label at
    all rather than an invented placeholder. Raises UnknownGlossError
    first, same as compose_sentence."""
    labels: list[tuple[str, bool]] = []
    for pose, value in _group_units(gloss_sequence):
        if pose == "pronoun":
            labels.append((PRONOUNS[value].lemma, False))
        elif pose == "verb":
            labels.append((VERBS[value].infinitive, False))
        elif pose == "noun":
            labels.append((NOUNS[value].cases["nominative"], False))
        elif pose == "adverb":
            labels.append((ADVERBS[value].text, False))
        elif pose == "standalone":
            labels.append((STANDALONE[value].text, False))
        elif pose == "negation":
            labels.append(("не", False))
        elif pose == "fingerspell":
            labels.append((value.capitalize(), True))
        elif pose == "tense_past":
            continue
    return labels


def compose_sentence(gloss_sequence: list[str], *, is_question: bool = False) -> str:
    """УЖМ gloss sequence -> natural Ukrainian sentence. See module
    docstring for exactly which glosses/patterns are supported; anything
    else raises UnknownGlossError or UnsupportedPatternError. `is_question`
    (Phase 17) ends the sentence with "?" instead of "." -- pass it when a
    non-manual question marker was detected alongside these glosses."""
    if not gloss_sequence:
        raise ValueError("gloss_sequence must not be empty")

    terminator = "?" if is_question else "."
    units = _group_units(gloss_sequence)  # raises UnknownGlossError first
    poses = [pose for pose, _ in units]

    # Pattern: a single standalone word (interjection/particle, or a fully
    # fingerspelled word on its own -- e.g. someone spelling just a name).
    if len(units) == 1 and poses[0] == "standalone":
        return STANDALONE[units[0][1]].text.capitalize() + terminator
    if len(units) == 1 and poses[0] == "fingerspell":
        return units[0][1].capitalize() + terminator

    # General pattern: [SUBJECT?, (NOT)?, (PAST)?, VERB, (PAST)?, (OBJECT)?,
    # (ADVERB)*] -- PAST may appear either right before OR right after the
    # verb (not both), since a signer marking tense after the sign it
    # modifies rather than before is an easy, real mistake, not a
    # different grammatical construction. SUBJECT is a pronoun or a noun
    # (nominative-case subject, e.g. "Машина їхала..." -- "the car was
    # driving...") and OBJECT is a noun (declined into whatever case the
    # verb governs) or a fingerspelled word. See lexicon.py's module
    # docstring for why PAST is a separate marker gloss rather than a
    # distinct verb-form gloss, and for exactly which subjects can take it
    # (pronoun past tense is only unambiguous for HE/SHE/WE/YOU_PL/THEY --
    # I/YOU depend on a gender this app has no way to know from a gloss
    # sequence alone).
    #
    # SUBJECT itself is optional, but only for a PAST-tense clause (e.g.
    # signing just "мати"/PAST/"вода" -- "had water" -- with no stated
    # "who"): a real elliptical construction, common when the doer is
    # contextually understood, not signed just present-tense verb+object
    # on its own (too ambiguous even elliptically -- could be a command,
    # a label, ...). Defaults to the masculine past form -- the same
    # unmarked default Ukrainian dictionaries/grammars use when gender
    # isn't otherwise specified, a documented rule (not a guess) -- see
    # test_gloss_to_text.py for the exact expected output.
    has_subject = poses[0] in ("pronoun", "noun")
    subject_pose = subject_gloss = None
    i = 0
    if has_subject:
        subject_pose, subject_gloss = units[0]
        i = 1

    negated = i < len(poses) and poses[i] == "negation"
    if negated:
        i += 1
    # PAST is accepted either right before OR right after the verb --
    # signing a tense marker after the sign it modifies (rather than
    # before, the primary documented order) is an easy mistake for a
    # signer who hasn't memorized this app's convention, so it composes
    # the same way ("мати"/"минулий час"/"вода", VERB then PAST, still
    # yields "Мав воду." like the PAST-before-VERB order does) instead of
    # being refused outright.
    past = i < len(poses) and poses[i] == "tense_past"
    if past:
        i += 1

    if i >= len(poses) or poses[i] != "verb":
        raise UnsupportedPatternError(
            f"No composition rule matches gloss sequence {gloss_sequence!r} "
            "(expected a verb after the subject)."
        )
    verb_gloss = units[i][1]
    verb = VERBS[verb_gloss]
    i += 1

    if not past and i < len(poses) and poses[i] == "tense_past":
        past = True
        i += 1

    if not has_subject and not past:
        raise UnsupportedPatternError(
            f"No composition rule matches gloss sequence {gloss_sequence!r} "
            "(expected a pronoun or noun in subject position, or an explicit "
            "PAST marker for a subject-less past-tense clause)."
        )

    subject_lemma: str | None
    if not has_subject:
        subject_lemma = None
        predicate = verb.past["masc"]
    elif subject_pose == "pronoun":
        pronoun = PRONOUNS[subject_gloss]
        subject_lemma = pronoun.lemma
        if past:
            if pronoun.past_gender is None:
                raise UnsupportedPatternError(
                    f"Past tense is ambiguous for {subject_gloss!r} -- it depends on "
                    "the speaker's/addressee's gender, which a gloss sequence alone "
                    "doesn't encode."
                )
            predicate = verb.past[pronoun.past_gender]
        else:
            if pronoun.person_key not in verb.conjugation:
                raise UnsupportedPatternError(
                    f"Verb {verb_gloss!r} has no known conjugation for person {pronoun.person_key!r}."
                )
            predicate = verb.conjugation[pronoun.person_key]
    else:
        noun_subject = NOUNS[subject_gloss]
        subject_lemma = noun_subject.cases["nominative"].capitalize()
        if past:
            past_key = "plural" if noun_subject.gender == "plural_tantum" else noun_subject.gender
            predicate = verb.past[past_key]
        else:
            if "3sg" not in verb.conjugation:
                raise UnsupportedPatternError(
                    f"Verb {verb_gloss!r} has no known 3rd-person conjugation for a noun subject."
                )
            predicate = verb.conjugation["3sg"]

    if negated:
        predicate = f"не {predicate}"

    if subject_lemma is None:
        # No subject to carry the sentence-initial capital -- the
        # predicate (or "не" ahead of it, if negated) is now the first word.
        predicate = predicate[:1].upper() + predicate[1:]

    object_form: str | None = None
    if i < len(poses) and poses[i] in ("noun", "fingerspell"):
        if poses[i] == "fingerspell":
            object_form = units[i][1].capitalize()
        else:
            object_noun = NOUNS[units[i][1]]
            if verb.governs_case not in object_noun.cases:
                raise UnsupportedPatternError(
                    f"Noun {units[i][1]!r} has no {verb.governs_case!r} form needed "
                    f"by verb {verb_gloss!r}."
                )
            object_form = object_noun.cases[verb.governs_case]
        i += 1

    adverb_forms: list[str] = []
    while i < len(poses) and poses[i] == "adverb":
        adverb_forms.append(ADVERBS[units[i][1]].text)
        i += 1

    if i != len(poses):
        raise UnsupportedPatternError(
            f"No composition rule matches gloss sequence {gloss_sequence!r} "
            "(unexpected gloss(es) after the recognized subject-verb-object-adverbs pattern)."
        )

    parts = [predicate] if subject_lemma is None else [subject_lemma, predicate]
    if object_form is not None:
        parts.append(object_form)
    parts.extend(adverb_forms)
    return " ".join(parts) + terminator
