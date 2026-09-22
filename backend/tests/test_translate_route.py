from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_text_to_gloss_returns_the_parsed_sequence():
    response = client.post("/translate/text-to-gloss", json={"text": "Я хочу води."})

    assert response.status_code == 200
    body = response.json()
    assert body["gloss_sequence"] == ["I", "WANT", "WATER"]
    # The internal English gloss codes above are never what a person should
    # read -- gloss_labels/composed_text carry the actual Ukrainian.
    assert body["gloss_labels"] == [
        {"text": "Я", "is_fingerspell": False},
        {"text": "хотіти", "is_fingerspell": False},
        {"text": "вода", "is_fingerspell": False},
    ]
    assert body["composed_text"] == "Я хочу води."


def test_text_to_gloss_standalone_phrase():
    response = client.post("/translate/text-to-gloss", json={"text": "Привіт."})

    assert response.status_code == 200
    body = response.json()
    assert body["gloss_sequence"] == ["PRIVIT"]
    assert body["gloss_labels"] == [{"text": "привіт", "is_fingerspell": False}]
    assert body["composed_text"] == "Привіт."


def test_text_to_gloss_fingerspells_a_word_outside_the_lexicon():
    # Phase 16: no lexicon noun for "кавун", but it's spellable letter-by-
    # letter, so this succeeds instead of a 422.
    response = client.post("/translate/text-to-gloss", json={"text": "Я хочу кавун."})

    assert response.status_code == 200
    body = response.json()
    assert body["gloss_sequence"] == ["I", "WANT", "FS_К", "FS_А", "FS_В", "FS_У", "FS_Н"]
    assert body["gloss_labels"] == [
        {"text": "Я", "is_fingerspell": False},
        {"text": "хотіти", "is_fingerspell": False},
        {"text": "Кавун", "is_fingerspell": True},
    ]
    assert body["composed_text"] == "Я хочу Кавун."


def test_text_to_gloss_returns_422_with_a_clear_message_for_unspellable_words():
    response = client.post("/translate/text-to-gloss", json={"text": "Я хочу pizza."})

    assert response.status_code == 422
    assert "pizza" in response.json()["detail"]


def test_text_to_gloss_rejects_empty_text():
    response = client.post("/translate/text-to-gloss", json={"text": ""})

    assert response.status_code == 422


def test_gloss_to_text_composes_a_full_sentence_from_an_accumulated_sequence():
    # Same sequence a signer would build up over several live-camera
    # confirmations before pressing "Зупинити" -- see websocket/handler.py.
    response = client.post("/translate/gloss-to-text", json={"gloss_sequence": ["I", "WANT", "WATER"]})

    assert response.status_code == 200
    body = response.json()
    assert body["gloss_labels"] == [
        {"text": "Я", "is_fingerspell": False},
        {"text": "хотіти", "is_fingerspell": False},
        {"text": "вода", "is_fingerspell": False},
    ]
    assert body["composed_text"] == "Я хочу води."
    # Grammar-guaranteed within ml/nlp/gloss_to_text.py's coverage -- never
    # to be confused with an unverified AI-composed guess.
    assert body["composed_text_source"] == "rule_based"


def test_gloss_to_text_handles_negation_and_past_tense_together():
    response = client.post(
        "/translate/gloss-to-text", json={"gloss_sequence": ["WE", "NOT", "PAST", "WANT", "WATER"]}
    )

    assert response.status_code == 200
    assert response.json()["composed_text"] == "Ми не хотіли води."


def test_gloss_to_text_returns_null_composed_text_for_an_unsupported_sequence_not_a_422():
    # An arbitrary/garbled accumulated sequence still gets a 200 with
    # composed_text=None -- gloss_labels lets the user see what was
    # understood, same honesty rule as text-to-gloss's composition step.
    # No GEMINI_API_KEY is set in the test environment, so the AI fallback
    # is a guaranteed no-op here too (see test_ai_sentence_composer.py for
    # that boundary's own tests) -- composed_text really does stay null.
    response = client.post("/translate/gloss-to-text", json={"gloss_sequence": ["WATER", "I"]})

    assert response.status_code == 200
    body = response.json()
    assert body["composed_text"] is None
    assert body["composed_text_source"] is None
    assert body["gloss_labels"] == [
        {"text": "вода", "is_fingerspell": False},
        {"text": "Я", "is_fingerspell": False},
    ]


def test_gloss_to_text_rejects_an_empty_sequence():
    response = client.post("/translate/gloss-to-text", json={"gloss_sequence": []})

    assert response.status_code == 422


def test_gloss_to_text_uses_the_ai_fallback_only_when_the_rule_based_composer_could_not(monkeypatch):
    # The AI fallback (app/services/ai_sentence_composer.py) is exercised
    # against a real, injectable httpx transport in its own test file --
    # here we only verify the ROUTE's wiring: it's consulted exactly when
    # rule-based composition yields None, and its result is labeled
    # "ai_fallback", never conflated with a grammar-guaranteed one.
    monkeypatch.setattr(
        "app.api.routes.translate.compose_with_ai",
        lambda words, settings, client=None: "Вода Я (AI-складене речення).",
    )

    response = client.post("/translate/gloss-to-text", json={"gloss_sequence": ["WATER", "I"]})

    assert response.status_code == 200
    body = response.json()
    assert body["composed_text"] == "Вода Я (AI-складене речення)."
    assert body["composed_text_source"] == "ai_fallback"


def test_gloss_to_text_never_calls_the_ai_fallback_when_rule_based_composition_already_succeeded(monkeypatch):
    def fail_if_called(words, settings, client=None):
        raise AssertionError("the AI fallback must not run when rule-based composition already succeeded")

    monkeypatch.setattr("app.api.routes.translate.compose_with_ai", fail_if_called)

    response = client.post("/translate/gloss-to-text", json={"gloss_sequence": ["I", "WANT", "WATER"]})

    assert response.status_code == 200
    body = response.json()
    assert body["composed_text"] == "Я хочу води."
    assert body["composed_text_source"] == "rule_based"
