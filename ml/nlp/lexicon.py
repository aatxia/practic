"""
lexicon — the shared, hand-authored gloss<->Ukrainian vocabulary used by
both directions of rule-based translation: ml/nlp/gloss_to_text.py (Phase 12,
gloss -> text) and ml/nlp/text_to_gloss.py (Phase 14, text -> gloss). Defined
once so the two directions can never silently drift apart -- a word this
project can compose into text is also a word it can parse back out of text.

Coverage is intentionally hand-authored, not scraped or guessed: no public
annotated УЖМ dataset exists yet (Phase 8), so there's no real gloss
vocabulary to build a lexicon from -- every entry here is written and
grammatically checked by hand, not automatically derived. ~50 words across
7 pronouns x 16 verbs x 24 nouns (matched by the case each verb genuinely
governs) lets compose_sentence() build hundreds of distinct, grammatically
real sentences, not just single-word output. See PROJECT_STATUS.md.

Ukrainian present tense doesn't distinguish grammatical gender (він/вона
"хоче", identical) -- so "Він"/"Вона" share one "3sg" conjugation slot
rather than needing separate verb forms each.

Case coverage is deliberately asymmetric and this is a grammar decision,
not an oversight: "genitive_partitive" (some of a substance -- "хочу
води") only makes idiomatic sense for mass/substance nouns (water, tea,
coffee, ...), so only those get that case filled in. Count nouns (book,
phone, house, ...) only get "accusative" -- pairing one with a
genitive_partitive-governing verb correctly raises UnsupportedPatternError
(ml/nlp/gloss_to_text.py) rather than composing an odd-sounding sentence.

PAST TENSE (added alongside NOUN-as-subject sentences and trailing time
adverbs, so a sentence like "Машина їхала вчора ввечері." -- "the car was
driving yesterday evening" -- becomes composable, not just pronoun-subject
present-tense sentences): Ukrainian past tense doesn't conjugate by PERSON
the way present tense does -- it conjugates by the grammatical GENDER/NUMBER
of the subject (masc/fem/neut singular, or one shared plural form). For a
NOUN subject, that gender is a fixed lexical property of the noun itself
("машина" is always feminine), so NounEntry.gender is unconditionally safe
to use. For a PRONOUN subject it's only unambiguous for HE (masc)/SHE
(fem)/WE, YOU_PL, THEY (plural -- Ukrainian plural past has no gender
distinction at all, so this needs no gender info). It is genuinely
ambiguous for I and YOU (2sg): "я їхав" vs "я їхала" depends on the
SPEAKER's gender, which this app has no way to know from a gloss sequence
alone -- PronounEntry.past_gender is explicitly `None` for those two, and
gloss_to_text.py raises UnsupportedPatternError rather than guessing one.
Sign languages typically don't conjugate a verb sign for tense at all --
time is carried by a separate sign/context instead -- so representing
"past" as its own marker gloss (like negation's "NOT") rather than folding
it into a hypothetical verb-form gloss is the linguistically apt choice
here, not an arbitrary implementation shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PronounEntry:
    lemma: str  # capitalized surface form, e.g. "Я"
    person_key: str  # selects the matching VerbEntry.conjugation entry
    # "masc" | "fem" | "plural" | None (None = past tense is genuinely
    # ambiguous for this pronoun -- see module docstring).
    past_gender: str | None


@dataclass(frozen=True)
class VerbEntry:
    infinitive: str  # dictionary-citation form, e.g. "хотіти" -- used for display labels
    conjugation: dict[str, str]  # person_key -> conjugated surface form
    governs_case: str  # which NounEntry.cases key a direct object takes
    past: dict[str, str]  # "masc" | "fem" | "neut" | "plural" -> past form


@dataclass(frozen=True)
class NounEntry:
    cases: dict[str, str]  # case name -> surface form
    # "masc" | "fem" | "neut" | "plural_tantum" (a noun with no singular
    # at all, e.g. "гроші" -- past tense agreement uses the plural form
    # for these regardless of real-world number).
    gender: str


@dataclass(frozen=True)
class AdverbEntry:
    text: str  # invariant surface form; only ever appears trailing a clause


@dataclass(frozen=True)
class StandaloneEntry:
    text: str  # lowercase surface form; gloss_to_text capitalizes it


PRONOUNS: dict[str, PronounEntry] = {
    "I": PronounEntry("Я", "1sg", past_gender=None),
    "YOU": PronounEntry("Ти", "2sg", past_gender=None),
    "HE": PronounEntry("Він", "3sg", past_gender="masc"),
    "SHE": PronounEntry("Вона", "3sg", past_gender="fem"),
    "WE": PronounEntry("Ми", "1pl", past_gender="plural"),
    "YOU_PL": PronounEntry("Ви", "2pl", past_gender="plural"),
    "THEY": PronounEntry("Вони", "3pl", past_gender="plural"),
}

VERBS: dict[str, VerbEntry] = {
    "WANT": VerbEntry(
        infinitive="хотіти",
        conjugation={
            "1sg": "хочу", "2sg": "хочеш", "3sg": "хоче",
            "1pl": "хочемо", "2pl": "хочете", "3pl": "хочуть",
        },
        governs_case="genitive_partitive",
        past={"masc": "хотів", "fem": "хотіла", "neut": "хотіло", "plural": "хотіли"},
    ),
    "LIKE": VerbEntry(
        infinitive="любити",
        conjugation={
            "1sg": "люблю", "2sg": "любиш", "3sg": "любить",
            "1pl": "любимо", "2pl": "любите", "3pl": "люблять",
        },
        governs_case="accusative",
        past={"masc": "любив", "fem": "любила", "neut": "любило", "plural": "любили"},
    ),
    "HAVE": VerbEntry(
        infinitive="мати",
        conjugation={
            "1sg": "маю", "2sg": "маєш", "3sg": "має",
            "1pl": "маємо", "2pl": "маєте", "3pl": "мають",
        },
        governs_case="accusative",
        past={"masc": "мав", "fem": "мала", "neut": "мало", "plural": "мали"},
    ),
    "KNOW": VerbEntry(
        infinitive="знати",
        conjugation={
            "1sg": "знаю", "2sg": "знаєш", "3sg": "знає",
            "1pl": "знаємо", "2pl": "знаєте", "3pl": "знають",
        },
        governs_case="accusative",
        past={"masc": "знав", "fem": "знала", "neut": "знало", "plural": "знали"},
    ),
    "UNDERSTAND": VerbEntry(
        infinitive="розуміти",
        conjugation={
            "1sg": "розумію", "2sg": "розумієш", "3sg": "розуміє",
            "1pl": "розуміємо", "2pl": "розумієте", "3pl": "розуміють",
        },
        governs_case="accusative",
        past={"masc": "розумів", "fem": "розуміла", "neut": "розуміло", "plural": "розуміли"},
    ),
    "SEE": VerbEntry(
        infinitive="бачити",
        conjugation={
            "1sg": "бачу", "2sg": "бачиш", "3sg": "бачить",
            "1pl": "бачимо", "2pl": "бачите", "3pl": "бачать",
        },
        governs_case="accusative",
        past={"masc": "бачив", "fem": "бачила", "neut": "бачило", "plural": "бачили"},
    ),
    "READ": VerbEntry(
        infinitive="читати",
        conjugation={
            "1sg": "читаю", "2sg": "читаєш", "3sg": "читає",
            "1pl": "читаємо", "2pl": "читаєте", "3pl": "читають",
        },
        governs_case="accusative",
        past={"masc": "читав", "fem": "читала", "neut": "читало", "plural": "читали"},
    ),
    "WRITE": VerbEntry(
        infinitive="писати",
        conjugation={
            "1sg": "пишу", "2sg": "пишеш", "3sg": "пише",
            "1pl": "пишемо", "2pl": "пишете", "3pl": "пишуть",
        },
        governs_case="accusative",
        past={"masc": "писав", "fem": "писала", "neut": "писало", "plural": "писали"},
    ),
    "EAT": VerbEntry(
        infinitive="їсти",
        conjugation={
            "1sg": "їм", "2sg": "їси", "3sg": "їсть",
            "1pl": "їмо", "2pl": "їсте", "3pl": "їдять",
        },
        governs_case="accusative",
        past={"masc": "їв", "fem": "їла", "neut": "їло", "plural": "їли"},
    ),
    "DRINK": VerbEntry(
        infinitive="пити",
        conjugation={
            "1sg": "п'ю", "2sg": "п'єш", "3sg": "п'є",
            "1pl": "п'ємо", "2pl": "п'єте", "3pl": "п'ють",
        },
        governs_case="genitive_partitive",
        past={"masc": "пив", "fem": "пила", "neut": "пило", "plural": "пили"},
    ),
    # The remaining verbs are only ever used without a direct object
    # (the [PRONOUN, VERB] pattern, e.g. "Я йду." -- see gloss_to_text.py) --
    # governs_case is still filled in with the case each would genuinely
    # take if it ever gained an object (none of the current NOUNS pair with
    # them), not a placeholder.
    "GO": VerbEntry(
        infinitive="іти",
        conjugation={
            "1sg": "іду", "2sg": "ідеш", "3sg": "іде",
            "1pl": "ідемо", "2pl": "ідете", "3pl": "ідуть",
        },
        governs_case="accusative",
        past={"masc": "йшов", "fem": "йшла", "neut": "йшло", "plural": "йшли"},
    ),
    "WORK": VerbEntry(
        infinitive="працювати",
        conjugation={
            "1sg": "працюю", "2sg": "працюєш", "3sg": "працює",
            "1pl": "працюємо", "2pl": "працюєте", "3pl": "працюють",
        },
        governs_case="accusative",
        past={"masc": "працював", "fem": "працювала", "neut": "працювало", "plural": "працювали"},
    ),
    "LIVE": VerbEntry(
        infinitive="жити",
        conjugation={
            "1sg": "живу", "2sg": "живеш", "3sg": "живе",
            "1pl": "живемо", "2pl": "живете", "3pl": "живуть",
        },
        governs_case="accusative",
        past={"masc": "жив", "fem": "жила", "neut": "жило", "plural": "жили"},
    ),
    "SLEEP": VerbEntry(
        infinitive="спати",
        conjugation={
            "1sg": "сплю", "2sg": "спиш", "3sg": "спить",
            "1pl": "спимо", "2pl": "спите", "3pl": "сплять",
        },
        governs_case="accusative",
        past={"masc": "спав", "fem": "спала", "neut": "спало", "plural": "спали"},
    ),
    "SPEAK": VerbEntry(
        infinitive="говорити",
        conjugation={
            "1sg": "говорю", "2sg": "говориш", "3sg": "говорить",
            "1pl": "говоримо", "2pl": "говорите", "3pl": "говорять",
        },
        governs_case="accusative",
        past={"masc": "говорив", "fem": "говорила", "neut": "говорило", "plural": "говорили"},
    ),
    # "Їхати" (travel by vehicle) vs. "GO"/"іти" (travel on foot) -- a real
    # distinction Ukrainian makes that English "go" doesn't. Like the other
    # object-less verbs above, its genuine object case would be
    # instrumental ("їхати машиною" -- "by car"), which this lexicon
    # doesn't model at all (no NOUNS entry has an instrumental form) --
    # governs_case here is never actually read, since no [SUBJECT, RIDE,
    # NOUN] pattern is supported.
    "RIDE": VerbEntry(
        infinitive="їхати",
        conjugation={
            "1sg": "їду", "2sg": "їдеш", "3sg": "їде",
            "1pl": "їдемо", "2pl": "їдете", "3pl": "їдуть",
        },
        governs_case="accusative",
        past={"masc": "їхав", "fem": "їхала", "neut": "їхало", "plural": "їхали"},
    ),
    # "Потребувати" governs genitive case, same as WANT/DRINK -- "потребую
    # води"/"потребую їжі" reuse the exact same NounEntry.cases key those
    # two already use, no new case needed.
    "NEED": VerbEntry(
        infinitive="потребувати",
        conjugation={
            "1sg": "потребую", "2sg": "потребуєш", "3sg": "потребує",
            "1pl": "потребуємо", "2pl": "потребуєте", "3pl": "потребують",
        },
        governs_case="genitive_partitive",
        past={"masc": "потребував", "fem": "потребувала", "neut": "потребувало", "plural": "потребували"},
    ),
    # "Допомагати" genuinely governs dative case ("допомагаю другу"), which
    # this lexicon doesn't model (only genitive_partitive/accusative exist
    # on NounEntry) -- like GO/WORK/LIVE/SLEEP/SPEAK/RIDE above, only the
    # object-less [PRONOUN, VERB] pattern is supported; governs_case is
    # filled in with the real value it doesn't currently need to be
    # readable but is fixed as a real fact if dative is ever added.
    "HELP": VerbEntry(
        infinitive="допомагати",
        conjugation={
            "1sg": "допомагаю", "2sg": "допомагаєш", "3sg": "допомагає",
            "1pl": "допомагаємо", "2pl": "допомагаєте", "3pl": "допомагають",
        },
        governs_case="dative",
        past={"masc": "допомагав", "fem": "допомагала", "neut": "допомагало", "plural": "допомагали"},
    ),
}

NOUNS: dict[str, NounEntry] = {
    # Mass/substance nouns -- genuinely take both "some of X" (genitive
    # partitive, pairs with WANT/DRINK) and a definite object (accusative,
    # pairs with LIKE/HAVE/KNOW/...).
    "WATER": NounEntry({"nominative": "вода", "genitive_partitive": "води", "accusative": "воду"}, gender="fem"),
    "BREAD": NounEntry({"nominative": "хліб", "genitive_partitive": "хліба", "accusative": "хліб"}, gender="masc"),
    "TEA": NounEntry({"nominative": "чай", "genitive_partitive": "чаю", "accusative": "чай"}, gender="masc"),
    "COFFEE": NounEntry({"nominative": "кава", "genitive_partitive": "кави", "accusative": "каву"}, gender="fem"),
    "MILK": NounEntry({"nominative": "молоко", "genitive_partitive": "молока", "accusative": "молоко"}, gender="neut"),
    "JUICE": NounEntry({"nominative": "сік", "genitive_partitive": "соку", "accusative": "сік"}, gender="masc"),
    "SOUP": NounEntry({"nominative": "суп", "genitive_partitive": "супу", "accusative": "суп"}, gender="masc"),
    "MEAT": NounEntry({"nominative": "м'ясо", "genitive_partitive": "м'яса", "accusative": "м'ясо"}, gender="neut"),
    "SUGAR": NounEntry({"nominative": "цукор", "genitive_partitive": "цукру", "accusative": "цукор"}, gender="masc"),
    "FOOD": NounEntry({"nominative": "їжа", "genitive_partitive": "їжі", "accusative": "їжу"}, gender="fem"),
    # Count nouns -- only accusative: "хочу книги" is not idiomatic partitive
    # the way "хочу води" is, so WANT/DRINK + a count noun correctly raises
    # UnsupportedPatternError rather than composing a dubious sentence.
    "BOOK": NounEntry({"nominative": "книга", "accusative": "книгу"}, gender="fem"),
    "FLOWER": NounEntry({"nominative": "квітка", "accusative": "квітку"}, gender="fem"),
    "LETTER": NounEntry({"nominative": "лист", "accusative": "лист"}, gender="masc"),
    "PHONE": NounEntry({"nominative": "телефон", "accusative": "телефон"}, gender="masc"),
    "HOUSE": NounEntry({"nominative": "будинок", "accusative": "будинок"}, gender="masc"),
    "CAR": NounEntry({"nominative": "машина", "accusative": "машину"}, gender="fem"),
    "WORK_N": NounEntry({"nominative": "робота", "accusative": "роботу"}, gender="fem"),
    "SCHOOL": NounEntry({"nominative": "школа", "accusative": "школу"}, gender="fem"),
    "CITY": NounEntry({"nominative": "місто", "accusative": "місто"}, gender="neut"),
    # Animate: accusative singular equals genitive singular, not nominative
    # (a real Ukrainian rule -- "бачу друга", not "*бачу друг").
    "FRIEND": NounEntry({"nominative": "друг", "accusative": "друга"}, gender="masc"),
    "NAME": NounEntry({"nominative": "ім'я", "accusative": "ім'я"}, gender="neut"),
    "TIME": NounEntry({"nominative": "час", "accusative": "час"}, gender="masc"),
    "DAY": NounEntry({"nominative": "день", "accusative": "день"}, gender="masc"),
    "COUNTRY": NounEntry({"nominative": "країна", "accusative": "країну"}, gender="fem"),
    "FAMILY": NounEntry({"nominative": "сім'я", "accusative": "сім'ю"}, gender="fem"),
    # Plurale tantum -- "гроші" has no singular form at all, so past tense
    # agreement uses the plural verb form regardless of grammatical gender.
    "MONEY": NounEntry({"nominative": "гроші", "accusative": "гроші"}, gender="plural_tantum"),
}

# Trailing time adverbs -- invariant (never declined/conjugated), always
# appear after the verb/object, e.g. "Машина їхала вчора ввечері."
# ("The car was driving yesterday evening.") composes as
# [CAR, PAST, RIDE, YESTERDAY, EVENING].
ADVERBS: dict[str, AdverbEntry] = {
    "YESTERDAY": AdverbEntry("вчора"),
    "TODAY": AdverbEntry("сьогодні"),
    "TOMORROW": AdverbEntry("завтра"),
    "MORNING": AdverbEntry("вранці"),
    "EVENING": AdverbEntry("ввечері"),
    "NIGHT": AdverbEntry("вночі"),
    "NOW": AdverbEntry("зараз"),
    # Directional ("(to) home"), distinct from locative "вдома" ("at home")
    # -- Ukrainian genuinely distinguishes these and this lexicon only
    # models the directional form, the one that pairs with GO ("йде
    # додому" -- "goes home"), not a stative "is at home" sentence.
    "HOME": AdverbEntry("додому"),
}

# ml/datasets/synthetic.py's DEMO_GLOSSES, mapped to their real meaning,
# plus a few more common standalone words/phrases.
STANDALONE: dict[str, StandaloneEntry] = {
    "PRIVIT": StandaloneEntry("привіт"),
    "DYAKUYU": StandaloneEntry("дякую"),
    "TAK": StandaloneEntry("так"),
    "NI": StandaloneEntry("ні"),
    "BUD_LASKA": StandaloneEntry("будь ласка"),
    "DOBRANICH": StandaloneEntry("добраніч"),
    "VYBACHTE": StandaloneEntry("вибачте"),
    "DO_POBACHENNYA": StandaloneEntry("до побачення"),
}

NEGATION_GLOSS = "NOT"  # e.g. ["I", "NOT", "WANT", "WATER"] <-> "Я не хочу води."
NEGATION_PARTICLE = "не"

# Marker gloss, not a Ukrainian word -- see module docstring's "PAST TENSE"
# section for why tense is represented this way rather than as a distinct
# verb-form gloss.
TENSE_PAST_GLOSS = "PAST"
