# Формат датасету (Phase 8)

Цей документ описує, як анотується один семпл (кліп) для тренування моделі
розпізнавання УЖМ, і як датасет ділиться на train/val/test.

## Анотації

Формат: **JSON Lines** — один `SampleAnnotation` (`ml/datasets/annotation.py`)
на рядок, у файлі на кшталт `data/annotations/<dataset_name>.jsonl`.

```json
{"sample_id": "uksl_000123", "clip_path": "raw/uksl_000123.mp4", "signer_id": "signer_07", "gloss": "ПРИВІТ", "start_frame": 40, "end_frame": 85, "fps": 30.0, "source": "uksl_real"}
```

| Поле          | Тип     | Значення                                                                                                                                                                    |
| ------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_id`   | string  | Унікальний ідентифікатор семпла (перевіряється на дублікати при завантаженні).                                                                                          |
| `clip_path`   | string  | Шлях відносно кореня датасету. Для реального датасету — відео-файл (`start_frame`/`end_frame` вирізають сегмент жесту з нього). Для `source="demo_synthetic"` — вже готовий `.npy` feature-вектор (немає реального відео за ним). |
| `signer_id`   | string  | Ідентифікатор конкретної людини, яка жестикулює. **Критично** для signer-independent split (розділ 11) — див. нижче.                                                    |
| `gloss`       | string  | Мітка класу (слово/жест).                                                                                                                                                |
| `start_frame` | int     | Початок сегменту (включно). Має бути `< end_frame`.                                                                                                                      |
| `end_frame`   | int     | Кінець сегменту (виключно).                                                                                                                                              |
| `fps`         | float   | FPS джерела, потрібен щоб інтерпретувати `start_frame`/`end_frame` як час. Має бути `> 0`.                                                                               |
| `source`      | string  | Звідки взявся семпл — ніколи не вгадується. Наприклад `"demo_synthetic"` (`ml/datasets/synthetic.py`) або назва реального датасету (з'явиться у Phase 9+, коли реальний датасет буде підключено). |

`ml/datasets/annotation.py::load_annotations()` кидає чітку помилку
(з номером файлу і рядка) на некоректний JSON, дублікат `sample_id`, або
порушення інваріантів (`end_frame <= start_frame`, `fps <= 0`) — жоден
поганий рядок не пропускається мовчки.

## Signer-independent split (розділ 11)

**Суворе правило**: один `signer_id` не може одночасно бути в train і в
test (і в val). Якщо змішати — модель може навчитися впізнавати конкретну
людину (її одяг/фон/манеру рухів) замість самого жесту, і результат на
test буде оманливо високим.

`ml/datasets/split.py::signer_independent_split()` реалізує це, призначаючи
**цілого сигнера**, а не окремий семпл, в один зі спліттів (greedy
group-balancing за заданими ratio, детерміновано за `seed`). Якщо унікальних
сигнерів менше, ніж потрібно спліттів — кидає явну помилку замість того, щоб
мовчки залишити спліт порожнім.

CLI:

```bash
python scripts/create_dataset_split.py \
  --annotations data/annotations/demo_annotations.jsonl \
  --output-dir data/splits \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

Результат: `data/splits/{train,val,test}.txt` (по одному `sample_id` на
рядок) і `data/splits/split_manifest.json` (ratio, seed, який сигнер
куди потрапив — для аудиту).

## DEMO MODE: синтетичний датасет

На момент Phase 8 **не існує публічно доступного isolated-sign датасету
УЖМ з розміткою `signer_id`** (перевірено: жоден із основних реєстрів
sign-language датасетів — `sign-language-processing/datasets`,
PyPI `sign-language-datasets`, `SignLanguage-Dataset-Hub` — не містить
української мови жестів; єдиний знайдений реальний ресурс, **UkrSL**
("Towards a Ukrainian Continuous Sign Language Dataset", UNLP 2026), — це
continuous signing з ~6 сигнерів (переклад дикторського тексту), а не
ізольовані жести з gloss-міткою, і посилання на завантаження не вдалось
перевірити з цього sandbox — `aclanthology.org` заблоковано мережевою
політикою середовища).

Тому `ml/datasets/synthetic.py` генерує невеликий **повністю штучний**
датасет — псевдовипадкові feature-послідовності з класо-залежним зсувом
(НЕ з реального відео чи MediaPipe), єдина мета якого — перевірити, що
pipeline (анотації → split → завантаження послідовностей → майбутній
тренувальний цикл) працює end-to-end. Кожен семпл позначений
`source="demo_synthetic"`, і `ml/datasets/dataset.py::load_feature_sequence()`
явно кидає `NotImplementedError` для будь-якого іншого `source` — реальні
відео поки що нема як завантажити (Phase 9 підключить
`ml/preprocessing` + `ml/features` до цього шляху, коли з'явиться реальний
датасет).

```bash
python scripts/generate_demo_dataset.py --output-dir data --seed 42
```

**Це не замінник реального датасету УЖМ і ніколи не повинно видаватись за
нього** — модель, натренована лише на цих даних, не розпізнає жодного
реального жесту, вона лише доводить, що код тренування працює.
