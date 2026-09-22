# UKSL Translator

Вебзастосунок для двостороннього перекладу **української жестової мови (УЖМ)** у реальному часі:
жести → текст → озвучка, і текст/голос → жести → 3D avatar.

Детальна архітектура: [`docs/architecture.md`](docs/architecture.md)

## Технологічний стек

- **Frontend**: Next.js, TypeScript, Tailwind CSS, WebSocket, WebRTC (camera), Three.js (avatar)
- **Backend**: Python, FastAPI, WebSocket, Pydantic
- **ML/CV**: MediaPipe (hands/pose/face landmarks), PyTorch (BiLSTM для розпізнавання слів,
  окремий класифікатор для дактильної абетки)
- **NLP**: власний rule-based gloss↔текст рушій (відмінювання, дієвідмінювання, заперечення,
  минулий час, словотвір), з опційним AI-резервом (Gemini) для речень, які rule-based рушій
  не може скласти

## Можливості

- Розпізнавання жестів з камери в реальному часі (WebSocket-стрімінг, MediaPipe landmarks)
- Розпізнавання дактильної абетки (побуквене відображення слів, яких немає у словнику)
- Складання граматично коректних українських речень із розпізнаних жестів (відмінки, час,
  заперечення) — не просто конкатенація слів
- Підтвердження кожного розпізнаного слова окремо, з можливістю виправити його з реальних
  ранжованих альтернатив моделі
- Зворотний напрямок: текст/голос → послідовність жестів → відео
- Історія перекладів, озвучення (TTS), копіювання результату
- Індикатори якості розпізнавання (чи видно руки/позу в кадрі) прямо на відео

## Структура репозиторію

```
frontend/    Next.js застосунок (камера, WS клієнт, UI, 3D avatar)
backend/     FastAPI сервер, WebSocket protocol, оркестрація сервісів
ml/          CV preprocessing, temporal models, NLP, fingerspelling, evaluation
data/        raw/processed відео, annotations, signer-independent splits (не в git)
models/      checkpoints (не в git)
scripts/     допоміжні скрипти (запис датасету, збір фото дактилю тощо)
docs/        технічна документація
docker/      Dockerfile'и для кожного сервісу
configs/     YAML конфігурації (feature toggles, model, camera)
```

## Встановлення

```bash
git clone <репозиторій>
cd uksl-translator
cp .env.example .env
```

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend буде доступний на `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend буде доступний на `http://localhost:3000`.
