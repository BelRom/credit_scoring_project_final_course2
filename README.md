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


# Этап 1. Подготовка модели к промышленной эксплуатации

source .venv/bin/activate
pip install -r requirements.txt

Обучение модели NN
python -m src.models_nn.train_nn \
  --data data/processed/credit_default.csv \
  --target target \
  --epochs 30

Сначала экспорт:
python -m src.models_nn.export_nn_to_onnx
Проверка корректности:
python -m src.models_nn.validate_nn_onnx
Бенчмарк CPU:
python -m src.models_nn.benchmark_nn_cpu


### экспорт ONNX
python -m src.models_nn.export_nn_to_onnx

#### quantization
python -m src.models_nn.optimization.quantize_torch_dynamic
python -m src.models_nn.optimization.quantize_onnx_int8

### метрики
python -m src.models_nn.optimization.evaluate_nn_variants

### benchmark CPU/GPU
python -m src.models_nn.optimization.benchmark_nn_variants

### отчет
python -m src.models_nn.optimization.report_benchmarks

## Оптимизация модели и оценка производительности

### Метод оптимизации

С целью повышения эффективности инференса была применена **посттренировочная квантизация INT8** нейронной сети.

Были рассмотрены следующие варианты:

* **Dynamic INT8-квантизация в PyTorch** для полносвязных (`Linear`) слоёв (поддерживается только на CPU).
* **INT8-квантизация в ONNX Runtime**, ориентированная на высокопроизводительный инференс на CPU.

Исходная модель в формате FP32 использовалась в качестве базовой для сравнения.

---

### Оценка качества модели

Качество модели оценивалось на отложенной тестовой выборке с использованием стандартных метрик классификации: ROC-AUC, Precision, Recall и F1-score.

**Результат:**
В результате INT8-квантизации размер модели был уменьшен в 3–3.5 раза по сравнению с FP32-версией без ухудшения качества предсказаний, что снижает требования к памяти и повышает эффективность инференса на CPU.

---

### Бенчмарк инференса

Производительность инференса измерялась при размере батча 1024 на CPU и GPU.

#### Результаты на CPU

| Вариант модели       | Задержка (мс / батч) | Пропускная способность (samples/s) |
| -------------------- | -------------------: | ---------------------------------: |
| Torch FP32           |                0.353 |                              2.90M |
| Torch INT8 (dynamic) |                0.345 |                              2.97M |
| ONNX FP32            |                0.431 |                              2.38M |
| **ONNX INT8**        |            **0.155** |                          **6.59M** |

INT8-модель в формате ONNX обеспечила **ускорение инференса более чем в 2.3 раза** по сравнению с базовой моделью Torch FP32.

#### Результаты на GPU

| Вариант модели | Задержка (мс / батч) |
| -------------- | -------------------: |
| Torch FP32     |                0.070 |
| ONNX FP32      |                0.083 |
| ONNX INT8      |                0.565 |

Использование INT8-квантизации на GPU не привело к ускорению из-за накладных расходов на передачу данных и особенностей выполнения.

---

### Выводы

* Квантизация INT8 существенно повышает производительность инференса на CPU без потери качества.
* **ONNX Runtime с INT8-квантизацией на CPU** является наиболее эффективной конфигурацией.
* Использование GPU целесообразно только для моделей в формате FP32; применение INT8 на GPU в данной задаче неэффективно.

**Рекомендуемая конфигурация для продакшена:**
 **CPU-инференс с использованием ONNX Runtime и INT8-квантизации**.


 ## Этап 2. Cloud Infrastructure as Code

 Команды: удалить Kubernetes-ресурсы (внутри кластера)
Удалить демо-приложение
kubectl delete namespace demo

Удалить Ingress NGINX (и namespace)
helm uninstall ingress-nginx -n ingress-nginx
kubectl delete namespace ingress-nginx

Проверка, что всё чисто
kubectl get ns
kubectl get all -A | head


Сначала узнай имя/ID:
yc managed-kubernetes node-group list

Удалить node group по имени:
yc managed-kubernetes node-group delete --name mlops-staging-cpu-ng

Проверка:
yc managed-kubernetes node-group list
kubectl get nodes   # после удаления нод будет пусто/недоступно


Полное удаление всего, что описано в конфиге
cd infrastructure/environments/staging
terraform destroy


Если node group удаляли — создать заново через Terraform
В каталоге окружения:

cd infrastructure/environments/staging
terraform init -backend-config=backend.hcl -reconfigure
terraform apply

Перезапустить деплоймент
kubectl rollout restart deployment/hello -n demo
kubectl rollout status deployment/hello -n demo


docker build \
  --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -t credit_api -f Dockerfile.api .



