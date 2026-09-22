# Архітектура — UKSL Translator

## 1. Огляд системи

Двонапрямковий перекладач української жестової мови (УЖМ):

```
FLOW A (жести → текст → мова):
Camera → WebRTC → WebSocket → Backend → CV (MediaPipe) →
Landmarks (hands+pose+face) → Temporal Model → Gloss sequence →
Ukrainian NLP → Natural Ukrainian text → TTS

FLOW B (текст/голос → жести):
Text/Speech → STT (якщо голос) → Ukrainian NLP → Gloss sequence →
Animation mapping → 3D Avatar (Three.js)
```

## 2. Модулі та відповідальність

| Модуль      | Технології                          | Відповідальність                                      |
|-------------|--------------------------------------|--------------------------------------------------------|
| `frontend/` | Next.js, TypeScript, Tailwind        | UI, камера, мікрофон, WebSocket клієнт, 3D avatar render |
| `backend/`  | FastAPI, Pydantic, WebSocket         | API, WebSocket protocol, оркестрація сервісів           |
| `ml/`       | PyTorch, MediaPipe, HF Transformers  | CV pipeline, temporal model, NLP, fingerspelling, eval  |
| `avatar/`   | Three.js assets                       | 3D модель, анімації, gloss→animation mapping            |
| `data/`     | —                                     | raw/processed відео, annotations, signer-independent splits |
| `models/`   | —                                     | checkpoints (не в git, `.gitignore`)                     |
| `configs/`  | YAML                                  | feature toggles, model config, camera config             |
| `experiments/` | YAML + CSV                        | результати експериментів (hands/pose/face, LSTM/Transformer) |

## 3. Принцип взаємозамінності (для наукової частини)

Кожен ML-компонент реалізований через абстрактний інтерфейс, щоб можна було
міняти реалізацію без зміни решти системи:

```python
class SignRecognizer:        # baseline (MLP+BiLSTM) <-> Transformer <-> Video-JEPA
class SignToTextTranslator:  # rule-based <-> HuggingFace NLP model
class TextToSignTranslator:  # rule-based <-> ML model
class TextToSpeech:          # browser SpeechSynthesis <-> інший TTS provider
class SpeechRecognizer:      # browser API <-> Whisper/інший STT
```

Вибір реалізації конфігурується через `.env` / `configs/model.yaml`, а не хардкодиться.

## 4. WebSocket protocol (детально в Phase 2/5)

```jsonc
// Client -> Server
{ "type": "frame", "timestamp": 123456789, "data": "<base64 jpeg>" }

// Server -> Client
{ "type": "prediction", "text": "...", "confidence": 0.92, "is_final": false }
{ "type": "final_prediction", "text": "...", "confidence": 0.95 }
{ "type": "error", "message": "..." }
{ "type": "connection", "status": "ok" }
```

## 5. Дані та приватність

- Сирі відеокадри за замовчуванням НЕ зберігаються на диск і НЕ логуються.
- Обробка кадрів — in-memory, кадр викидається одразу після inference.
- Жодні дані не відправляються стороннім API без явного `.env` налаштування.

## 6. ML: baseline → просунуті моделі (науковий план)

1. **Baseline A**: MediaPipe landmarks → normalize → MLP → BiLSTM/GRU → classifier.
2. **Baseline B**: MediaPipe landmarks → feature embedding → Transformer Encoder → classifier.
3. **Experimental**: Video-JEPA / інший self-supervised video encoder → temporal representation → classifier
   (окремий adapter в `ml/models/video_jepa/`, не ламає baseline).
4. **Transfer learning**: LSF (French Sign Language) pretrained representation → fine-tuning на UKSL dataset
   (документується в `docs/transfer_learning.md`, буде створено у відповідній фазі).

Порівняння відбувається через `ml/evaluation/` за єдиним інтерфейсом і `configs/*.yaml`.

## 7. Порядок реалізації (Phases)

Дивись `PROJECT_STATUS.md` для поточного статусу та `README.md` розділ "Roadmap".
