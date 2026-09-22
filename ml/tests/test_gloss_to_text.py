import pytest

from ml.nlp.gloss_to_text import (
    UnknownGlossError,
    UnsupportedPatternError,
    compose_sentence,
    gloss_display_labels,
)


@pytest.mark.parametrize(
    ("gloss_sequence", "expected"),
    [
        (["PRIVIT"], "Привіт."),
        (["DYAKUYU"], "Дякую."),
        (["TAK"], "Так."),
        (["NI"], "Ні."),
        (["BUD_LASKA"], "Будь ласка."),
    ],
)
def test_standalone_gloss_composes_a_capitalized_sentence(gloss_sequence, expected):
    assert compose_sentence(gloss_sequence) == expected


def test_pronoun_verb_noun_conjugates_and_declines_not_a_naive_join():
    # The exact example from TranslationService.gloss_to_text()'s docstring.
    assert compose_sentence(["I", "WANT", "WATER"]) == "Я хочу води."


def test_different_pronoun_selects_different_conjugation():
    assert compose_sentence(["YOU", "WANT", "WATER"]) == "Ти хочеш води."
    assert compose_sentence(["WE", "WANT", "WATER"]) == "Ми хочемо води."


def test_verb_governs_case_correctly_per_verb():
    # LIKE/HAVE govern accusative, not genitive_partitive like WANT.
    assert compose_sentence(["I", "LIKE", "BREAD"]) == "Я люблю хліб."
    assert compose_sentence(["I", "HAVE", "TEA"]) == "Я маю чай."


def test_pronoun_verb_without_object():
    assert compose_sentence(["I", "HAVE"]) == "Я маю."


def test_preverbal_negation_is_inserted_before_the_verb_not_where_not_appeared():
    assert compose_sentence(["I", "NOT", "WANT", "WATER"]) == "Я не хочу води."
    assert compose_sentence(["I", "NOT", "HAVE"]) == "Я не маю."


def test_unknown_gloss_raises_a_clear_error_not_a_guess():
    with pytest.raises(UnknownGlossError, match="WATERMELON"):
        compose_sentence(["I", "WANT", "WATERMELON"])


def test_recognized_glosses_in_an_unsupported_pattern_raise_clearly():
    # Two standalone words in a row isn't a pattern this engine composes.
    with pytest.raises(UnsupportedPatternError):
        compose_sentence(["TAK", "NI"])


def test_noun_then_verb_then_pronoun_is_not_the_supported_order():
    with pytest.raises(UnsupportedPatternError):
        compose_sentence(["WATER", "WANT", "I"])


def test_empty_sequence_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        compose_sentence([])


def test_verb_missing_a_conjugation_for_the_subject_raises_unsupported_pattern(monkeypatch):
    from ml.nlp import gloss_to_text
    from ml.nlp.lexicon import VerbEntry

    incomplete_verb = VerbEntry(
        infinitive="хотіти", conjugation={"2sg": "хочеш"}, governs_case="accusative", past={}
    )
    monkeypatch.setitem(gloss_to_text.VERBS, "WANT", incomplete_verb)

    with pytest.raises(UnsupportedPatternError, match="1sg"):
        compose_sentence(["I", "WANT"])


def test_verb_governing_a_case_the_noun_lacks_raises_unsupported_pattern(monkeypatch):
    from ml.nlp import gloss_to_text
    from ml.nlp.lexicon import NounEntry

    incomplete_noun = NounEntry(cases={"accusative": "хліб"}, gender="masc")
    monkeypatch.setitem(gloss_to_text.NOUNS, "BREAD", incomplete_noun)

    with pytest.raises(UnsupportedPatternError, match="genitive_partitive"):
        compose_sentence(["I", "WANT", "BREAD"])


def test_fingerspelled_glosses_compose_into_the_spelled_word_standalone():
    gloss_sequence = ["FS_О", "FS_К", "FS_С", "FS_А", "FS_Н", "FS_А"]
    assert compose_sentence(gloss_sequence) == "Оксана."


def test_noun_subject_present_tense_uses_third_person_conjugation():
    # "The car drives" (present tense doesn't distinguish grammatical
    # gender in Ukrainian, so any noun subject just takes the 3sg form).
    assert compose_sentence(["CAR", "RIDE"]) == "Машина їде."


def test_noun_subject_with_object():
    assert compose_sentence(["FRIEND", "HAVE", "PHONE"]) == "Друг має телефон."


def test_past_tense_marker_selects_the_correct_gendered_form():
    assert compose_sentence(["HE", "PAST", "RIDE"]) == "Він їхав."
    assert compose_sentence(["SHE", "PAST", "RIDE"]) == "Вона їхала."
    assert compose_sentence(["THEY", "PAST", "RIDE"]) == "Вони їхали."
    assert compose_sentence(["WE", "PAST", "RIDE"]) == "Ми їхали."
    assert compose_sentence(["YOU_PL", "PAST", "RIDE"]) == "Ви їхали."


def test_past_tense_for_a_noun_subject_uses_the_nouns_own_gender():
    assert compose_sentence(["CAR", "PAST", "RIDE"]) == "Машина їхала."  # fem
    assert compose_sentence(["FRIEND", "PAST", "HAVE", "PHONE"]) == "Друг мав телефон."  # masc


def test_past_tense_for_a_plurale_tantum_noun_uses_the_plural_form():
    assert compose_sentence(["MONEY", "PAST", "HAVE"]) == "Гроші мали."


def test_past_tense_is_ambiguous_and_refused_for_first_and_second_person_singular():
    with pytest.raises(UnsupportedPatternError, match="ambiguous"):
        compose_sentence(["I", "PAST", "WANT"])
    with pytest.raises(UnsupportedPatternError, match="ambiguous"):
        compose_sentence(["YOU", "PAST", "WANT"])


def test_past_tense_combines_with_negation_in_the_documented_order():
    assert compose_sentence(["HE", "NOT", "PAST", "RIDE"]) == "Він не їхав."


def test_past_tense_marker_is_also_accepted_right_after_the_verb_with_a_subject():
    # Same real mistake as the subject-less case below, but with a subject
    # present -- marking tense after the verb instead of before it.
    assert compose_sentence(["HE", "RIDE", "PAST"]) == "Він їхав."
    assert compose_sentence(["FRIEND", "HAVE", "PAST", "PHONE"]) == "Друг мав телефон."


def test_subjectless_past_tense_defaults_to_the_masculine_form():
    # Signing just "мати"/PAST/"вода" (no stated "who") composes "Мав
    # воду.", not a refusal, per gloss_to_text.py's documented
    # elliptical-construction rule.
    assert compose_sentence(["HAVE", "PAST", "WATER"]) == "Мав воду."


def test_subjectless_past_tense_with_no_object():
    assert compose_sentence(["PAST", "RIDE"]) == "Їхав."


def test_subjectless_past_tense_combines_with_negation_and_capitalizes_the_negation():
    assert compose_sentence(["NOT", "PAST", "HAVE", "WATER"]) == "Не мав воду."


def test_subjectless_past_tense_combines_with_trailing_adverbs():
    assert compose_sentence(["PAST", "HAVE", "WATER", "YESTERDAY"]) == "Мав воду вчора."


def test_subjectless_present_tense_remains_unsupported_too_ambiguous_even_elliptically():
    with pytest.raises(UnsupportedPatternError):
        compose_sentence(["HAVE", "WATER"])


def test_the_users_example_sentence_car_rode_yesterday_evening():
    assert compose_sentence(["CAR", "PAST", "RIDE", "YESTERDAY", "EVENING"]) == "Машина їхала вчора ввечері."


def test_trailing_adverbs_compose_after_the_object():
    assert compose_sentence(["I", "WANT", "WATER", "TODAY"]) == "Я хочу води сьогодні."


def test_gloss_starting_with_neither_pronoun_nor_noun_is_unsupported():
    with pytest.raises(UnsupportedPatternError):
        compose_sentence(["WANT", "I"])


def test_gloss_with_trailing_junk_after_a_valid_pattern_is_unsupported():
    with pytest.raises(UnsupportedPatternError):
        compose_sentence(["I", "WANT", "WATER", "TAK"])


def test_gloss_display_labels_are_ukrainian_words_not_the_internal_gloss_codes():
    # The whole point: a UI must never show "I", "WANT", "WATER" -- those
    # are code identifiers, not Ukrainian.
    assert gloss_display_labels(["I", "WANT", "WATER"]) == [("Я", False), ("хотіти", False), ("вода", False)]


def test_gloss_display_labels_use_the_verbs_infinitive_not_a_conjugated_form():
    assert gloss_display_labels(["HE", "RIDE"]) == [("Він", False), ("їхати", False)]


def test_gloss_display_labels_omit_the_past_tense_marker_entirely():
    # PAST has no Ukrainian surface form of its own -- see lexicon.py.
    assert gloss_display_labels(["CAR", "PAST", "RIDE"]) == [("машина", False), ("їхати", False)]


def test_gloss_display_labels_include_negation_and_adverbs():
    assert gloss_display_labels(["I", "NOT", "WANT", "WATER", "TODAY"]) == [
        ("Я", False),
        ("не", False),
        ("хотіти", False),
        ("вода", False),
        ("сьогодні", False),
    ]


def test_gloss_display_labels_spell_out_a_fingerspelled_run_as_one_word_and_flag_it():
    assert gloss_display_labels(["FS_О", "FS_К", "FS_С", "FS_А", "FS_Н", "FS_А"]) == [("Оксана", True)]


def test_fingerspelled_glosses_fill_the_object_slot_of_the_svo_pattern():
    gloss_sequence = ["I", "LIKE", "FS_О", "FS_К", "FS_С", "FS_А", "FS_Н", "FS_У"]
    assert compose_sentence(gloss_sequence) == "Я люблю Оксану."


def test_fingerspelled_object_is_not_declined_into_the_verbs_governed_case():
    # WANT governs genitive_partitive for a lexicon NOUN ("WATER" -> "води"),
    # but a fingerspelled word has no case data -- it's inserted as spelled,
    # capitalized, not force-declined into a case that doesn't exist for it.
    gloss_sequence = ["I", "WANT", "FS_К", "FS_А", "FS_В", "FS_У", "FS_Н"]
    assert compose_sentence(gloss_sequence) == "Я хочу Кавун."


def test_is_question_ends_standalone_sentence_with_a_question_mark():
    assert compose_sentence(["TAK"], is_question=True) == "Так?"


def test_is_question_ends_a_fingerspelled_standalone_word_with_a_question_mark():
    gloss_sequence = ["FS_О", "FS_К", "FS_С", "FS_А", "FS_Н", "FS_А"]
    assert compose_sentence(gloss_sequence, is_question=True) == "Оксана?"


def test_fingerspell_word_boundary_keeps_two_adjacent_spelled_words_separate():
    # Without FINGERSPELL_WORD_BOUNDARY, "тобі допомогти" (both outside the
    # lexicon) would flatten into one indistinguishable run of FS_ letters
    # and despell back as the single wrong word "Тобідопомогти" --
    # gloss_display_labels must report two separate words.
    gloss_sequence = ["FS_Т", "FS_О", "FS_Б", "FS_І", "FS_BOUNDARY", "FS_Д", "FS_О", "FS_П", "FS_О", "FS_М", "FS_О", "FS_Г", "FS_Т", "FS_И"]
    assert gloss_display_labels(gloss_sequence) == [("Тобі", True), ("Допомогти", True)]


def test_fingerspell_word_boundary_contributes_no_label_of_its_own():
    gloss_sequence = ["I", "FS_А", "FS_BOUNDARY", "FS_Б"]
    assert gloss_display_labels(gloss_sequence) == [("Я", False), ("А", True), ("Б", True)]


def test_fingerspell_word_boundary_between_two_spelled_words_is_an_unsupported_pattern_not_a_guess():
    # Two consecutive OOV words after the verb don't match the single-object
    # SVO pattern -- must be refused, never silently glued into one word.
    gloss_sequence = ["I", "WANT", "FS_Т", "FS_О", "FS_Б", "FS_І", "FS_BOUNDARY", "FS_Д", "FS_О", "FS_П"]
    with pytest.raises(UnsupportedPatternError):
        compose_sentence(gloss_sequence)


def test_is_question_ends_pronoun_verb_with_a_question_mark():
    assert compose_sentence(["I", "HAVE"], is_question=True) == "Я маю?"


def test_is_question_ends_svo_sentence_with_a_question_mark():
    assert compose_sentence(["I", "WANT", "WATER"], is_question=True) == "Я хочу води?"


def test_is_question_defaults_to_false():
    assert compose_sentence(["TAK"]) == "Так."
