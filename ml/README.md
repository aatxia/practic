# ML Pipeline — UKSL Translator

## Setup

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## MediaPipe models (external dependency)

Hand/pose/face landmark detection uses Google's pretrained MediaPipe Tasks
model bundles (Apache 2.0) — this project doesn't train these itself. Download
them once:

```bash
../scripts/download_mediapipe_models.sh
```

This fetches `hand_landmarker.task`, `pose_landmarker.task`, and
`face_landmarker.task` into `models/mediapipe/` (gitignored — ~13 MB total,
binary, doesn't belong in git). If `storage.googleapis.com` is blocked in
your network, hand/face automatically fall back to a GitHub mirror; pose
currently has no known mirror (get it manually — see the script's header
comment for the link).

## Preprocessing pipeline (Phase 6)

```
ml/preprocessing/
├── video_reader.py     # base64 JPEG (from WebSocket) or video file -> BGR numpy frames
├── landmarks.py         # MediaPipe Tasks: hands/pose/face detection (LandmarkExtractor)
├── normalization.py    # position/scale-invariant landmark normalization
└── augmentation.py     # training-time sequence augmentation (rotate/scale/noise/dropout)
```

Quick example:

```python
from ml.preprocessing.landmarks import LandmarkExtractor, FeatureToggles
from ml.preprocessing.normalization import normalize_frame
from ml.preprocessing.video_reader import decode_base64_frame

frame = decode_base64_frame(base64_jpeg_string)

with LandmarkExtractor("models/mediapipe", FeatureToggles(hands=True, pose=True, face=False)) as ex:
    raw = ex.extract(frame)
    normalized = normalize_frame(raw)  # fixed-shape, zero-filled when a modality is absent
```

`normalize_frame` never fabricates a detection: if MediaPipe doesn't find a
hand/pose/face in a frame, the corresponding array is zero-filled AND
`normalized.present[...]` is `False`, so downstream code (Phase 7 feature
builder, Phase 9 model) can tell "detected at the origin" apart from "not
detected".

## Feature vectors (Phase 7)

```
ml/features/
├── hands.py            # left+right hand -> (126,) vector
├── pose.py              # 8 selected upper-body landmarks -> (24,) vector
├── face.py              # 24 selected landmarks (brows/eyes/mouth) -> (72,) vector
└── feature_vector.py    # combines enabled modalities per FeatureConfig (section 9 experiments)
```

```python
from ml.features.feature_vector import FeatureConfig, build_feature_vector

config = FeatureConfig(hands=True, pose=True, face=False)  # hands+pose experiment
vector = build_feature_vector(normalized, config)  # shape driven entirely by config
```

This is exactly what `backend/websocket/handler.py` calls today for every incoming
frame — the WebSocket already reports real detection status and feature vector
size, it just has no trained model yet to turn that vector into a prediction.

## Running tests

```bash
cd ml
pytest -v
```

Tests that need a real model file (`test_landmarks.py`) skip automatically
with a clear reason if you haven't run the download script yet — they are
not faked or hardcoded to pass.

## Dataset pipeline (Phase 8)

```
ml/datasets/
├── annotation.py        # SampleAnnotation schema + honest JSONL load/write
├── split.py              # signer-independent train/val/test split (section 11)
├── synthetic.py          # DEMO MODE ONLY: synthetic dataset generator
└── dataset.py             # loads a sample's feature-vector sequence off disk
```

No public isolated-sign УЖМ (Ukrainian Sign Language) dataset with
`signer_id` labeling could be found (see `PROJECT_STATUS.md`, Phase 8, for
what was checked). Until a real one is available, `scripts/generate_demo_dataset.py`
produces a small synthetic dataset (`source="demo_synthetic"`, never to be
mistaken for real data) so the rest of the pipeline can be exercised
end-to-end:

```bash
python scripts/generate_demo_dataset.py --output-dir data --seed 42
python scripts/create_dataset_split.py \
  --annotations data/annotations/demo_annotations.jsonl \
  --output-dir data/splits
```

Full format details: `docs/dataset_format.md`.

## Training (Phase 9)

```
ml/models/
└── lstm.py              # LSTMSignClassifier: stacked LSTM + linear classifier head

ml/training/
├── config.py             # loads configs/model.yaml -> TrainingConfig
├── dataset.py             # SignSequenceDataset: SampleAnnotation -> fixed-length tensor
└── train.py               # CLI: split -> train -> evaluate -> save checkpoint
```

Install torch first (see `ml/requirements-training.txt` — **read its header
before running pip install in Google Colab**, since Colab already has a
GPU-matched torch preinstalled). Then, from the repo root:

```bash
pip install -r ml/requirements-training.txt   # local only, see note above for Colab
python scripts/generate_demo_dataset.py --output-dir data --seed 42   # if you haven't already
python -m ml.training.train --annotations data/annotations/demo_annotations.jsonl
```

Runs identically locally and in Colab (`!python -m ml.training.train ...`
after cloning the repo there) — no Colab-specific paths or hacks. Hyperparameters
(hidden size, epochs, batch size, learning rate) default to `configs/model.yaml`'s
`training:` section and can be overridden per-run with CLI flags (`--epochs`,
`--hidden-size`, ...) without editing the file.

Training does a signer-independent split (`ml/datasets/split.py`, section 11)
of whatever annotation file you point it at, trains on `train`, evaluates on
`val` each epoch, and saves the best checkpoint to
`models/checkpoints/<experiment-name>/latest.pt` (gitignored — a checkpoint is
a build artifact, not source). The checkpoint carries everything Phase 10
inference needs: model weights, architecture config, feature config, sequence
length, and the gloss↔index label mapping — plus `source_tags` and `demo_mode`,
so a checkpoint trained only on `demo_synthetic` data can never be silently
mistaken for one that recognizes real Ukrainian Sign Language.

**Running on the demo dataset is a pipeline sanity check, not a real model**:
since the synthetic classes are trivially separable by construction, val
accuracy reaches 100% in a few epochs — that only proves training/checkpoint
plumbing works end-to-end, it says nothing about real-world recognition.
Real training needs a real annotated УЖМ dataset (still not available as of
Phase 9, see `PROJECT_STATUS.md`).

## Inference (Phase 10)

```
ml/inference/
└── recognizer.py         # SignRecognizer: loads a checkpoint, predicts from a full window
```

Framework-agnostic (no FastAPI import) so it's testable standalone — the same
class powers `backend/app/services/lstm_inference_service.py`, which just
adapts its output to the app's `InferenceService`/`SignPrediction` contract.
The WebSocket handler (`backend/websocket/handler.py`) buffers each
connection's incoming feature vectors into a sliding window sized to the
checkpoint's `sequence_length`; once full, every subsequent frame runs a real
prediction (see Phase 11 below for how those raw per-frame predictions become
`"final_prediction"` events). A `demo_mode` checkpoint's predicted text is
prefixed `"[DEMO] "` so it can never be mistaken for real УЖМ recognition,
and `/health`'s `ml_pipeline_status` reports `"demo_mode"` (vs
`"not_implemented"` with no checkpoint, or `"ready"` once trained on real
data).

```python
from ml.inference.recognizer import SignRecognizer

recognizer = SignRecognizer("models/checkpoints/baseline/latest.pt", device="cpu")
result = recognizer.predict(feature_sequence)  # exactly recognizer.sequence_length frames
print(result.gloss, result.confidence, result.is_demo_mode)
```

## Gloss-sequence aggregation (Phase 11)

```
ml/inference/
└── aggregator.py         # GlossSequenceAggregator: debounces raw predictions into a gloss sequence
```

Phase 10's sliding window produces one prediction *per frame*, so a held sign
gets predicted dozens of times in a row. `GlossSequenceAggregator` is a
deterministic debounce heuristic (**not** real sign-boundary/linguistic
segmentation — no movement/hold-phase detection): a gloss must be predicted
`stability_frames` times in a row, above `confidence_threshold`, before it's
"confirmed" and appended to `.sequence`; it won't be re-confirmed while the
same sign keeps being held. The WebSocket handler feeds it every prediction
and only sends `"final_prediction"` (`is_final=true`) at the moment of
confirmation — everything else stays `"prediction"` (`is_final=false`),
configurable via `WS_GLOSS_STABILITY_FRAMES` / `WS_GLOSS_CONFIDENCE_THRESHOLD`
(.env).

```python
from ml.inference.aggregator import GlossSequenceAggregator

agg = GlossSequenceAggregator(stability_frames=5, confidence_threshold=0.5)
for gloss, confidence in predictions:
    if agg.update(gloss, confidence):
        print("confirmed:", agg.sequence[-1])
```

The accumulated `agg.sequence` (e.g. `["I", "WANT", "WATER"]`) is exactly
what `TranslationService.gloss_to_text()` (Phase 12, below) needs.

## Gloss<->text NLP (Phase 12 + Phase 14)

```
ml/nlp/
├── lexicon.py             # shared gloss<->Ukrainian vocabulary (both directions read this)
├── gloss_to_text.py       # compose_sentence(): gloss sequence -> Ukrainian sentence (Phase 12)
└── text_to_gloss.py       # parse_gloss_sequence(): Ukrainian text -> gloss sequence (Phase 14)
```

Two genuine (if narrow) rule-based engines sharing one lexicon, so they can
never silently disagree about vocabulary — anything one direction can
produce, the other can parse back. `compose_sentence()` conjugates verbs by
subject person, declines nouns into whichever case the verb governs, and
inserts negation in the correct preverbal position — **not**
`' '.join(gloss_sequence)`. `parse_gloss_sequence()` is its reverse: it
maps recognized Ukrainian surface forms (any conjugated/declined form) back
to their gloss token, in the order they appeared. Every gloss/word and
pattern either direction accepts is listed explicitly in its module;
anything else raises a clear `ValueError` subclass rather than guessing.

**Vocabulary** (`lexicon.py`) is hand-authored and grammar-checked, not
scraped or guessed: no public annotated УЖМ dataset exists yet (Phase 8),
so there's no real gloss vocabulary to build a broader lexicon from. It
currently covers 7 pronouns (1st/2nd/3rd person, singular and plural), 15
verbs, and 24 nouns — ~2,400 distinct grammatically-valid pronoun/verb/noun
sentences (not just single-word output), all round-trip tested
(`ml/tests/test_lexicon_expansion.py`). Case coverage is a deliberate
grammar decision, not an oversight: mass/substance nouns (water, tea,
coffee, ...) get both `genitive_partitive` ("хочу води" — some water) and
`accusative` ("п'ю воду" — the water) forms, since both are genuinely
idiomatic; count nouns (book, phone, house, ...) only get `accusative`,
so pairing one with a genitive_partitive-governing verb (e.g. `WANT` +
`MONEY`) correctly raises `UnsupportedPatternError` rather than composing
a dubious sentence.

```python
from ml.nlp.gloss_to_text import compose_sentence
from ml.nlp.text_to_gloss import parse_gloss_sequence

compose_sentence(["I", "WANT", "WATER"])  # -> "Я хочу води."
compose_sentence(["I", "NOT", "WANT", "WATER"])  # -> "Я не хочу води."
compose_sentence(["TAK"])  # -> "Так." (the demo-dataset glosses are all standalone words)

parse_gloss_sequence("Я хочу води.")  # -> ["I", "WANT", "WATER"]
parse_gloss_sequence("Привіт.")  # -> ["PRIVIT"]
```

`backend/app/services/translation_service.py::RuleBasedTranslationService`
wraps both for the `TranslationService` interface. Two live integration
points:

- **Recognition side** (Phase 10-11 WebSocket stream): on every
  just-**confirmed** gloss (Phase 11's `is_final=true` moment), the handler
  tries `gloss_to_text([gloss])`; if the lexicon covers it (all 5
  demo-dataset glosses do), the confirmed prediction's `text` becomes the
  composed Ukrainian sentence instead of the raw label — e.g. a confirmed
  `PRIVIT` becomes `[DEMO] Привіт.` rather than `[DEMO] PRIVIT`. Translating
  the full accumulated multi-gloss sequence isn't wired in yet -- that needs
  real multi-word gloss sequences (no dataset yet) and a way to know when a
  *sentence*, not just one sign, is complete.
- **Generation side** (Phase 13-14 UI, `POST /translate/text-to-gloss`): the
  "Текст / Голос → Жести" panel sends whatever text is typed or dictated
  (Phase 13) to this endpoint, displays the returned gloss sequence
  (`components/Transcript`), and drives the placeholder 3D avatar
  (`frontend/components/Avatar/`, Phase 15) -- a procedural Three.js puppet
  with hand-authored demo poses, **not** real УЖМ signs (no motion-capture
  data exists yet, see `PROJECT_STATUS.md` Phase 15). The avatar is also
  driven by recognition-side confirmed signs (Phase 11's WebSocket
  `final_prediction.gloss`), accumulated client-side -- whichever source
  (camera or text/voice) most recently produced a sequence animates it.

## Fingerspelling / дактилологія (Phase 16)

```
ml/nlp/
└── fingerspelling.py    # Ukrainian dactyl alphabet <-> FS_<letter> gloss tokens
```

Real signers don't just give up on an out-of-vocabulary word -- they spell
it letter-by-letter using a standardized dactyl alphabet. `text_to_gloss.py`
now falls back to this for any word not in the (still tiny, Phase 8-blocked)
lexicon, instead of refusing outright; `gloss_to_text.py` composes a run of
`FS_` gloss tokens back into the word they spell, standalone or filling the
object slot of the SVO pattern.

```python
from ml.nlp.fingerspelling import spell_word, despell
from ml.nlp.text_to_gloss import parse_gloss_sequence
from ml.nlp.gloss_to_text import compose_sentence

spell_word("Оксана")  # -> ["FS_О", "FS_К", "FS_С", "FS_А", "FS_Н", "FS_А"]
parse_gloss_sequence("Я хочу кавун.")  # -> ["I", "WANT", "FS_К", "FS_А", "FS_В", "FS_У", "FS_Н"]
compose_sentence(["FS_О", "FS_К", "FS_С", "FS_А", "FS_Н", "FS_А"])  # -> "Оксана."
```

What this does and doesn't claim:
- The letter↔gloss mapping is genuine, unambiguous data (all 33 Ukrainian
  alphabet letters) -- not a guess. A word with a character that has no
  dactyl handshape (Latin script, digits, apostrophe) still raises
  `UnrecognizedWordError`/`UnknownGlossError` rather than being silently
  dropped or faked.
- Spelling is caseless/formless: it records exactly the letters spelled, not
  a grammatically-declined citation form, so `compose_sentence` can't put a
  fingerspelled object into the verb's governed case the way a lexicon NOUN
  is (documented limitation, tested in `ml/tests/test_gloss_to_text.py`).
- **No avatar animation**: the 3D puppet (Phase 15) has no finger geometry,
  so it can't render 33 visually distinct handshapes -- `FS_` glosses have
  no defined pose (same as any other unmapped gloss) and the avatar honestly
  holds neutral rather than faking a handshape. The frontend still shows the
  spelled word as readable text (`frontend/lib/glossDisplay.ts` groups
  consecutive `FS_` chips into one "🔤 word" chip in `components/Transcript`
  and `components/Avatar`'s "no animation for" line).

## Facial grammar / немануальні маркери (Phase 17)

```
ml/features/
└── facial_grammar.py    # eyebrow-position non-manual grammar marker
```

Sign-language grammar isn't only manual: cross-linguistically (УЖМ
included), a yes/no question is marked by raised eyebrows and a wh-question
by furrowed/lowered eyebrows, with **no separate manual sign for "?" at
all**. Without this, a recognized question would come out of Phase 12's
`gloss_to_text` looking exactly like a statement.

`facial_grammar.py` measures one real, well-documented signal -- eyebrow
height relative to the eyes -- from the MediaPipe FaceMesh landmarks
`ml/features/face.py` already selects. `BaselineCalibrator` learns each
signer's own neutral-face gap over their first N frames with a detected
face (proportions vary by person/camera framing, so one fixed threshold for
everyone would be meaningless), then classifies later frames against it:

```python
from ml.features.facial_grammar import BaselineCalibrator, FacialGrammarMarker

calibrator = BaselineCalibrator(calibration_frames=30)
for face in normalized_face_landmarks_per_frame:
    marker = calibrator.update(face)  # NONE while calibrating / no face / neutral
    if marker == FacialGrammarMarker.EYEBROWS_RAISED:
        ...  # yes/no question
```

This is a **deterministic geometric heuristic, not a trained classifier** --
no annotated facial-grammar dataset exists to train or validate one against
(Phase 8 still hasn't found a real УЖМ dataset at all). The ratio thresholds
(`WS_FACIAL_RAISED_RATIO`/`WS_FACIAL_FURROWED_RATIO`, `.env`) are a
documented starting point, not empirically calibrated.

**Live integration** (`backend/websocket/handler.py`): every frame with a
detected face feeds a per-connection `BaselineCalibrator`, independent of
whether sign inference has a trained checkpoint. The resulting marker rides
on every `PredictionMessage` (`"facial_grammar"`), and when a gloss is
just-confirmed (Phase 11) while a marker is active, `gloss_to_text()`'s new
`is_question=True` swaps the composed sentence's `.` for `?` -- e.g. a
confirmed `TAK` with raised eyebrows becomes `[DEMO] Так?` instead of
`[DEMO] Так.`. `text_to_gloss` (Phase 14, text→gloss) needs no change: it
already strips `?`/`.`/etc. as trailing punctuation, since a *typed*
question has no non-manual channel to detect in the first place.

Frontend: `components/Translator/TranslatorView.tsx` shows a small badge
("🤨 Брови підняті...") next to the live translation whenever the latest
WebSocket message carries a non-`"NONE"` marker.
