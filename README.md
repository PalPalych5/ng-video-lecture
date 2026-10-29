# Символьная модель GPT на стихах А. С. Пушкина: сравнение длины контекста (64 vs 128)

Учебный исследовательский проект по исследованию архитектуры Transformer (nanoGPT А. Карпати). Основная задача — сравнительное исследование качества языкового моделирования, динамики обучения и вычислительной эффективности символьных GPT с длиной контекста 64 и 128 символов на корпусе русской поэзии А. С. Пушкина.

## Структура репозитория

```text
ng-video-lecture/
├── gpt.py                  # Единый источник истины: архитектура GPT, токенизатор, обучение, чекпоинты
├── context_comparison.ipynb # Главный рабочий файл: обучение, измерения, графики, генерация, выводы
├── input.txt               # Обучающий корпус (стихотворения А. С. Пушкина в UTF-8)
├── pyproject.toml          # Зависимости проекта (Python 3.12, PyTorch с поддержкой CUDA)
├── uv.lock                 # Зафиксированные версии пакетов uv
├── README.md               # Данная инструкция
├── .gitignore              # Исключения Git (чекпоинты, виртуальное окружение, кэши)
├── results/                # Результаты экспериментов, сводные таблицы и графики
├── bigram.py               # Материал лекции (базовая биграммная модель)
└── more.txt                # Материал лекции
```

## Установка окружения

1. **Клонирование репозитория:**
```bash
git clone https://github.com/PalPalych5/ng-video-lecture.git
cd ng-video-lecture
```

2. **Установка зависимостей:**
Проект управляется пакетным менеджером `uv`:

```bash
# Синхронизация виртуального окружения и установка зависимостей
uv sync
```

Если `uv` не установлен, можно использовать стандартный `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # на Linux/macOS
# или .venv\Scripts\activate на Windows
pip install torch matplotlib pandas notebook ipykernel nbformat nbclient
```

## Запуск экспериментов

### 1. Интерактивная работа в Jupyter Notebook (основной сценарий)
Основной рабочий артефакт — [`context_comparison.ipynb`](context_comparison.ipynb).

Запустите Jupyter или откройте ноутбук в VS Code / Google Colab:
```bash
uv run jupyter lab
# или
uv run jupyter notebook
```

- Ноутбук выполняется строго сверху вниз.
- Флаг `FORCE_RETRAIN = False` по умолчанию загружает готовые сохраненные чекпоинты и результаты из папки `results/pushkin_seeds/`, что позволяет мгновенно экспериментировать с генерацией текста и графиками без ожидания повторного обучения.
- При установке `FORCE_RETRAIN = True` запускается полный цикл 6 независимых сессий обучения по 3 сидам (`42`, `1337`, `2026`).

### 2. Автономный запуск базовой модели из консоли
Файл [`gpt.py`](gpt.py) содержит класс модели `GPTLanguageModel` и точку входа для автономного обучения базовой модели:
```bash
uv run python gpt.py
```

## Местонахождение результатов

Все результаты, графики и веса моделей сохраняются в каталоге `results/`:
- `results/pushkin_seeds/summary_all_runs.csv` — сводная таблица по всем 6 независимым запускам.
- `results/pushkin_seeds/loss_trajectories.csv` — полная пошаговая история потерь (Train NLL, Val NLL, сопоставимый NLL).
- `results/pushkin_seeds/plot1_val_comparison.png` — график 1: сравнение валидационной ошибки по трем сидам.
- `results/pushkin_seeds/plot2_train_val_dynamics.png` — график 2: динамика обучения и расхождение Train/Val NLL.
- `results/pushkin_seeds/microbenchmark.json` — результаты аппаратного ABBA-микробенчмарка на GPU.
- `results/pushkin_seeds/fixed_target_indices.json` — фиксированные индексы позиций для расчета NLL.
- `results/pushkin_seeds/checkpoint_{64,128}_seed{s}_final.pt` — финальные веса моделей на шаге 5000 (включены в репозиторий).
- `results/pushkin_seeds/checkpoint_{64,128}_seed{s}_best.pt` — промежуточные лучшие чекпоинты (локальные).
