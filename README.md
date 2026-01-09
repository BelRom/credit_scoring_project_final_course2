# Credit Score Project

Учебный **ML / MLOps проект** для предсказания кредитного дефолта.
Проект демонстрирует полный жизненный цикл модели машинного обучения: валидацию данных, обучение, трекинг экспериментов, тестирование и развёртывание с использованием Docker.

---

## Цель проекта

Разработка и автоматизация **скоринговой модели предсказания дефолта (PD-модель)** с акцентом на:

* воспроизводимое обучение моделей
* трекинг экспериментов с помощью MLflow
* контроль качества данных
* автоматическое тестирование
* контейнеризацию и развёртывание

---

## Стек технологий

* **Python 3.10+**
* **scikit-learn**
* **MLflow**
* **Great Expectations**
* **pytest**
* **Docker / Docker Compose**
* **FastAPI**
* **Streamlit**
* **Black / Flake8**

---

## Структура проекта

```
credit_scoring_project/
├── data/
│   ├── raw/                 # Исходные данные
│   └── processed/           # Подготовленные данные
│
├── gx/                      # Great Expectations (datasources, suites, checkpoints)
│
├── mlruns/                  # MLflow experiments и артефакты
│
├── models/                  # Сохранённые модели и метрики
│
├── references/              # Полезные материалы и ссылки
│
├── reports/                 # Отчёты, графики, результаты экспериментов
│
├── scripts/
│   ├── build.sh             # Сборка Docker-образов
│   └── run.sh               # Запуск сервисов
│
├── src/
│   ├── api/                 # FastAPI backend
│   ├── data/                # Подготовка и валидация данных
│   ├── feature/             # Feature engineering
│   ├── frontend/            # Streamlit frontend
│   ├── models/              # Обучение и инференс моделей
│   └── make_sample.py       # Генерация примеров данных
│
├── tests/                   # Unit и интеграционные тесты
│
├── .dvcignore
├── .flake8
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── Makefile
├── pyproject.toml
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## Датасет

* **Default of Credit Card Clients Dataset**
* Источник: Kaggle (UCI Machine Learning Repository)
* Целевая переменная: `target` (дефолт / не дефолт)

---

## Валидация данных (Great Expectations)

В проекте используется **Great Expectations** для контроля качества данных.

### Инициализация

```bash
export GX_BASE_DIR="$(pwd)"
```

### Создание источника данных

```bash
great_expectations datasource new
```

### Создание набора ожиданий (suite)

```bash
great_expectations suite new
```

### Создание checkpoint

```bash
great_expectations checkpoint new credit_default_checkpoint
```

### Запуск проверки

```bash
great_expectations checkpoint run credit_default_checkpoint
```

---

## Обучение модели

Обучение реализовано в виде CLI-модуля.

### Пример: логистическая регрессия

```bash
python -m src.models.train \
  --data data/processed/credit_default.csv \
  --target target \
  --model logreg \
  --n-iter 30 \
  --cv 5 \
  --experiment Credit_Default_Prediction5
```

### Во время обучения выполняется:

* разбиение на train / validation
* пайплайн предобработки
* подбор гиперпараметров
* расчёт метрик (ROC-AUC, Precision, Recall, F1)
* сохранение модели
* логирование эксперимента в MLflow

---

## Трекинг экспериментов (MLflow)

Запуск интерфейса MLflow:

```bash
mlflow ui
```

Интерфейс доступен по адресу:

```
http://localhost:5000
```

Логируются:

* параметры модели
* метрики качества
* артефакты
* версии обученных моделей

---

## Тестирование

### Запуск всех тестов

```bash
pytest -q
```

### Линтинг и форматирование

```bash
black .
flake8 src tests
pytest
```

---

## API сервис

Для инференса используется **FastAPI**.

* Swagger-документация:

```
http://localhost:8000/docs
```

---

## Frontend

Простой интерфейс на **Streamlit** для отправки признаков и получения предсказаний.

```
http://localhost:8501
```

---

## Docker

### Сборка образов

```bash
./scripts/build.sh
```

### Запуск сервисов

```bash
./scripts/run.sh
```

В составе запускаются:

* API
* Frontend
* MLflow

---

## Команды, используемые в CI и локально

```bash
black .
flake8 src tests
pytest
```

```bash
great_expectations checkpoint run credit_default_checkpoint
```

```bash
python -m src.models.train ...
```
---

## CI/CD (GitHub Actions)
Проект поддерживает CI/CD пайплайн на GitHub Actions, ориентированный на задачи MLOps и качество ML-кода.
Триггеры пайплайна
CI запускается при:
push в ветки main и develop
pull request в main
