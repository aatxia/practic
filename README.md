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
- **Avatar**: Three.js, rigged 3D-модель з реальним ретаргетингом жестів
- **Infra**: Docker / docker-compose (локальна розробка); деплой — Vercel (frontend) +
  Render (backend, через `docker/backend.Dockerfile`), див. розділ "Деплой" нижче

## Можливості

- Розпізнавання жестів з камери в реальному часі (WebSocket-стрімінг, MediaPipe landmarks)
- Розпізнавання дактильної абетки (побуквене відображення слів, яких немає у словнику)
- Складання граматично коректних українських речень із розпізнаних жестів (відмінки, час,
  заперечення) — не просто конкатенація слів
- Підтвердження кожного розпізнаного слова окремо, з можливістю виправити його з реальних
  ранжованих альтернатив моделі
- Зворотний напрямок: текст/голос → послідовність жестів → анімація 3D-аватара
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

### ML — dataset, тренування, inference

Повний пайплайн (preprocessing, dataset, тренування, real-time inference, gloss aggregation,
NLP, дактильна абетка) реалізований end-to-end — див. [`ml/README.md`](ml/README.md) для
налаштування та використання. Тренування моделей передбачається на Google Colab (GPU),
checkpoint переноситься в `models/checkpoints/` і використовується локально.

## Деплой

Frontend і backend деплояться окремо, на різних платформах — backend's CV/ML-залежності
(torch + mediapipe + opencv, разом кілька гігабайт) і постійне WebSocket-з'єднання для
стрімінгу камери не влазять у ліміти типового serverless-хостингу.

- **Frontend → Vercel**: звичайний Next.js-проєкт із `Root Directory = frontend`
  (framework detection — автоматичний). Env vars (`NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_WS_URL`)
  вказують на задеплоєний backend.
- **Backend → Render** (або будь-який Docker-хостинг із підтримкою WebSocket): `render.yaml`
  у корені репозиторію — Render-blueprint, що будує `docker/backend.Dockerfile`. Перед першим
  реальним використанням постав `CORS_ORIGINS` (у Render dashboard) на фактичний Vercel-домен
  фронтенду.

## Опційний AI-резерв для складання речень

Rule-based NLP-рушій (`ml/nlp/gloss_to_text.py`) — граматично гарантований у межах свого
покриття, але покриття навмисно вузьке (немає публічного анотованого датасету УЖМ для
побудови ширшого словника). Коли він не може скласти речення з розпізнаних слів, можна
підключити безкоштовний Gemini API як резерв — див. `.env.example` (`GEMINI_API_KEY`).
Без ключа функція вимкнена, жодних мережевих запитів не робиться. Речення, складене ШІ,
завжди позначається в інтерфейсі окремо від граматично гарантованого результату.
