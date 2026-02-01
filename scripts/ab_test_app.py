"""
Простое FastAPI-приложение для A/B-тестирования двух версий модели.

Назначение
----------
Данный сервис повторяет схему основного API из ``src/api/app.py``, но добавляет
endpoint ``/predict_ab``, который для каждого запроса случайным образом
маршрутизирует инференс на одну из двух моделей (версия A или версия B).

Это позволяет:
- развернуть одновременно две версии кредитного скоринга,
- собирать обратную связь и сравнивать, какая версия работает лучше в продакшене
  (по качеству, стабильности, метрикам бизнеса и т.д.).

Логика распределения трафика
----------------------------
По умолчанию используется равномерное распределение 50/50.
При необходимости можно сделать взвешенное распределение (например 90/10),
изменив вероятность в коде.

Хранение моделей
----------------
Обе модели должны быть сохранены в формате joblib.

Пути к моделям задаются через переменные окружения:
- ``MODEL_A_PATH`` — путь к модели версии A,
- ``MODEL_B_PATH`` — путь к модели версии B.

Если переменные окружения не заданы, используются значения по умолчанию:
- ``models/model_a.joblib``
- ``models/model_b.joblib``

Пример запуска
--------------
    export MODEL_A_PATH=models/best_model.joblib
    export MODEL_B_PATH=models/best_model_v2.joblib
    uvicorn src.api.ab_test_app:app --host 0.0.0.0 --port 8001

Использование endpoint
----------------------
Далее можно отправлять POST-запросы на ``/predict_ab`` с тем же JSON-форматом,
что используется в основном endpoint ``/predict``.
Ответ содержит:
- предсказанный класс (0/1),
- вероятность дефолта (positive class probability).

Дополнительно (опционально) endpoint можно расширить:
- возвращать в ответе, какая версия модели была использована,
- или логировать назначения (A/B) и результаты в постоянное хранилище
  (файл/БД/лог-систему) для последующего анализа.
"""

from __future__ import annotations

import json
import os
import random
from typing import Tuple
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

from src.api.schemas import PredictRequest, PredictResponse
from src.data.prepare_data import primary_cleaning, feature_engineering, finalize_dtypes


def load_models() -> Tuple[object, object]:
    """
    Загружает две версии модели по путям из переменных окружения.

    Возвращает
    ----------
    Tuple[object, object]
        Две sklearn-совместимые модели, которые реализуют:
        - ``predict_proba`` (предпочтительно),
        - либо ``decision_function`` (если predict_proba недоступен).

    Примечание
    ----------
    Если файлы моделей не найдены, будет выброшено исключение FileNotFoundError.
    """
    model_a_path = os.getenv("MODEL_A_PATH", "models/model_a.joblib")
    model_b_path = os.getenv("MODEL_B_PATH", "models/model_b.joblib")
    model_a = joblib.load(model_a_path)
    model_b = joblib.load(model_b_path)
    return model_a, model_b


def transform_features(features: dict) -> pd.DataFrame:
    """
    Применяет тот же пайплайн предобработки, что и при обучении модели.

    Последовательно выполняются этапы:
    - первичная очистка (primary_cleaning),
    - feature engineering,
    - приведение типов (finalize_dtypes).

    Возвращаются только признаки, необходимые модели.
    Если в данных присутствует target-колонка, она удаляется.

    Параметры
    ----------
    features : dict
        Словарь с исходными (сырыми) признаками заемщика.

    Возвращает
    ----------
    pd.DataFrame
        Подготовленный DataFrame, готовый для инференса модели.
    """
    df = pd.DataFrame([features])
    df = primary_cleaning(df)
    df = feature_engineering(df)
    df = finalize_dtypes(df)

    # На всякий случай удаляем target/label, если они присутствуют
    return df.drop(
        columns=[c for c in df.columns if c in {"target", "default"}],
        errors="ignore",
    )


def get_prediction(model, X: pd.DataFrame) -> Tuple[int, float]:
    """
    Вычисляет предсказание и вероятность для заданной модели и признаков.

    Логика:
    - если модель поддерживает ``predict_proba`` -> используем вероятность класса 1,
    - иначе используем ``decision_function`` и приводим скор к (0..1) через сигмоиду.

    Параметры
    ----------
    model : sklearn-совместимая модель
        Модель, реализующая ``predict_proba`` или ``decision_function``.
    X : pd.DataFrame
        Признаки после предобработки.

    Возвращает
    ----------
    Tuple[int, float]
        (pred, proba), где:
        - pred — класс 0/1,
        - proba — вероятность класса 1 (default).
    """
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[:, 1][0])
    else:
        score = float(model.decision_function(X)[0])
        proba = float(1.0 / (1.0 + np.exp(-score)))

    pred = int(proba >= 0.5)
    return pred, proba


# Создаём FastAPI-приложение
app = FastAPI(title="Credit Scoring A/B Test API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    """
    Загружает обе модели при старте приложения.

    Это позволяет:
    - не загружать модели при каждом запросе,
    - уменьшить задержки и избежать лишних операций I/O.
    """
    global model_a, model_b
    model_a, model_b = load_models()


@app.post("/predict_ab", response_model=PredictResponse)
async def predict_ab(req: PredictRequest) -> PredictResponse:
    """
    Делает предсказание, используя одну из двух версий модели (A или B) случайно.

    Логика работы:
    1) Принимает входные данные в формате, идентичном основному endpoint ``/predict``.
    2) С вероятностью 0.5 выбирает модель A, иначе — модель B.
    3) Выполняет предобработку признаков.
    4) Возвращает предсказанный класс и вероятность.

    Важно
    -----
    Сейчас в ответе не указывается, какая модель была выбрана.
    Для анализа результатов A/B эксперимента рекомендуется логировать:
    - назначенную версию модели (A/B),
    - входные признаки,
    - предсказание и вероятность,
    - (при наличии) фактический исход, когда он станет доступен.

    Параметры
    ----------
    req : PredictRequest
        JSON-пayload с признаками заемщика.

    Возвращает
    ----------
    PredictResponse
        Объект ответа (prediction, probability).
    """
    features = req.model_dump()
    X = transform_features(features)

    # A/B assignment: 50/50
    use_a = random.random() < 0.5
    model = model_a if use_a else model_b

    pred, proba = get_prediction(model, X)

    # Опционально: сохраняем назначения и результаты для последующего анализа A/B-теста
    log_path = os.getenv("AB_LOG_PATH", "logs/ab_test_assignments.jsonl")
    try:
        assignment = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "model_version": "A" if use_a else "B",
            "features": features,
            "prediction": pred,
            "probability": proba,
        }
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(assignment) + "\n")
    except Exception:
        # Ошибка логирования не должна ломать инференс
        pass

    return PredictResponse(prediction=pred, probability=proba)
