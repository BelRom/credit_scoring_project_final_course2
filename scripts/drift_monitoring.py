"""
Модуль мониторинга дрейфа (data drift / concept drift) и деградации качества модели
с использованием библиотеки Evidently.

Назначение
----------
Скрипт предназначен для регулярной (батчевой) проверки стабильности данных и
качества ML-модели в условиях эксплуатации. Поддерживаются три ключевых аспекта:

1) Data drift (дрейф данных)
   - Проверяется изменение распределений входных признаков между:
     * reference (эталонной) выборкой — обычно данные обучения/валидации,
     * current (текущей) выборкой — данные продакшена/нового периода.
   - Используется пресет: ``DataDriftPreset``.

2) Concept drift (концептуальный дрейф)
   - Под concept drift понимается изменение зависимости между признаками X и
     целевой переменной y (в реальной среде меняется процесс генерации данных).
   - В практической реализации concept drift оценивается через:
     * дрейф целевой переменной (если метки доступны),
     * дрейф предсказаний модели (как прокси при отсутствии меток),
     * дополнительные статистики, предоставляемые Evidently.
   - Используется пресет: ``TargetDriftPreset``.

3) Model performance decay (деградация качества модели)
   - При наличии истинных меток на текущих данных оцениваются метрики качества
     (accuracy/precision/recall/F1/ROC-AUC и др.) и их изменения относительно эталона.
   - Используется пресет: ``ClassificationPerformancePreset``.

Результаты
----------
Отчёт Evidently сохраняется:
- в HTML (для визуального анализа и прикрепления в отчёт/артефакты проекта),
- в JSON (для автоматической обработки в CI/CD, Airflow или алертинге).

Пример запуска
--------------
    python src/monitoring/drift_monitoring.py \
        --reference-data data/processed/credit_default.csv \
        --current-data data/processed/new_batch.csv \
        --model-path models/best_model.joblib \
        --target-column default \
        --report-html reports/drift_report.html \
        --report-json reports/drift_report.json

Ожидаемые данные
----------------
- reference и current должны содержать одинаковый набор признаковых колонок.
- Целевая колонка (target) в current может отсутствовать:
  * тогда performance-метрики будут ограничены тем, что возможно вычислить без меток,
  * зато сохраняется мониторинг data drift и prediction/target drift (как прокси).
- Скрипт добавляет колонку ``prediction`` (вероятность/скор модели) в обе таблицы,
  чтобы Evidently мог выполнять анализ prediction/target drift и качества модели.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    ClassificationPerformancePreset,
    TargetDriftPreset,
)


def run_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    model,
    target_column: str = "default",
    report_html_path: Optional[str] = None,
    report_json_path: Optional[str] = None,
) -> dict:
    """
    Формирует единый Evidently-отчёт для мониторинга:
    - data drift (дрейф входных данных),
    - concept/target/prediction drift (признаки изменения зависимостей),
    - деградации качества модели во времени.

    Параметры
    ---------
    reference_data : pd.DataFrame
        Эталонный датасет (baseline), обычно:
        - обучающая/валидационная выборка,
        - либо исторический период, зафиксированный как “нормальный”.
    current_data : pd.DataFrame
        Текущий датасет (production/batch), который сравнивается с эталоном.
        Должен содержать те же признаковые колонки, что и reference_data.
    model : object
        Обученная модель (sklearn-compatible), поддерживающая:
        - predict_proba(X) (предпочтительно, для вероятностей),
        - либо decision_function(X) (если predict_proba отсутствует).
    target_column : str
        Название целевой колонки (label). В reference_data обычно присутствует,
        в current_data может отсутствовать (например, если метки приходят с задержкой).
    report_html_path : Optional[str]
        Путь для сохранения HTML-версии отчёта (для отчётности и визуального контроля).
    report_json_path : Optional[str]
        Путь для сохранения JSON-версии отчёта (для автоматизации и алертинга).

    Возвращает
    ----------
    dict
        Отчёт Evidently в виде python-словаря (as_dict), пригодный для парсинга.
    """
    # 1) Безопасно копируем DataFrame, чтобы не модифицировать входные аргументы
    ref = reference_data.copy()
    curr = current_data.copy()

    # 2) Определяем список признаков: всё кроме целевой переменной
    #    (важно, чтобы набор признаков совпадал между reference и current)
    feature_columns = [col for col in ref.columns if col != target_column]

    # 3) Получаем прогнозы модели на эталонных данных
    #    - если доступен predict_proba -> берём вероятность положительного класса
    #    - иначе используем decision_function как “скор” (подойдёт для drift по prediction)
    X_ref = ref[feature_columns]
    try:
        ref_pred = model.predict_proba(X_ref)[:, 1]
    except AttributeError:
        ref_pred = model.decision_function(X_ref)
    ref["prediction"] = ref_pred

    # 4) Получаем прогнозы модели на текущих данных
    X_curr = curr[feature_columns]
    try:
        curr_pred = model.predict_proba(X_curr)[:, 1]
    except AttributeError:
        curr_pred = model.decision_function(X_curr)
    curr["prediction"] = curr_pred

    # 5) Собираем Evidently-отчёт из пресетов:
    #    - DataDriftPreset: статистический дрейф по входным признакам
    #    - TargetDriftPreset: дрейф target/prediction как сигнал concept drift
    #    - ClassificationPerformancePreset: метрики качества (при наличии меток)
    report = Report(
        metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            ClassificationPerformancePreset(),
        ]
    )

    # 6) Запускаем расчёт отчёта на паре (reference, current)
    #    Column mapping явно не задаём: Evidently по умолчанию использует
    #    колонку "prediction" как предсказание, а target_column — как target (если есть).
    report.run(
        reference_data=ref,
        current_data=curr,
    )

    # 7) Сохраняем отчёт в HTML для визуальной проверки (дашборд)
    if report_html_path:
        html_path = Path(report_html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(html_path))

    # 8) Сохраняем отчёт в JSON для автоматического анализа/алертинга
    if report_json_path:
        json_path = Path(report_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_json(str(json_path))

    # 9) Возвращаем отчёт как dict (удобно для пайплайнов и unit-тестов)
    return report.as_dict()


def main() -> None:
    """
    CLI-обвязка для запуска мониторинга из консоли/пайплайна.

    Типовой сценарий в MLOps:
    - по расписанию (cron/Airflow) или по событию (новый батч данных),
    - генерируем отчёт,
    - сохраняем артефакты (HTML/JSON),
    - при превышении порогов (share_drifted_features, падение метрик) запускаем ретрейн.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Запуск анализа data drift, concept drift и деградации качества модели "
            "с использованием Evidently (HTML/JSON отчёт)."
        )
    )
    parser.add_argument(
        "--reference-data",
        required=True,
        help=(
            "Путь к CSV/Parquet с эталонными данными (baseline), например train/valid "
            "или исторический период без аномалий."
        ),
    )
    parser.add_argument(
        "--current-data",
        required=True,
        help=(
            "Путь к CSV/Parquet с текущими данными (production/batch) для сравнения "
            "с эталоном."
        ),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Путь к обученной модели (joblib).",
    )
    parser.add_argument(
        "--target-column",
        default="default",
        help="Название целевой колонки (label/target).",
    )
    parser.add_argument(
        "--report-html",
        default="reports/drift_report.html",
        help="Путь для сохранения HTML-отчёта Evidently (дашборд).",
    )
    parser.add_argument(
        "--report-json",
        default="reports/drift_report.json",
        help="Путь для сохранения JSON-отчёта Evidently (для автоматизации).",
    )
    args = parser.parse_args()

    # Загрузка данных (поддерживаем CSV и Parquet)
    def load_data(path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        if p.suffix == ".csv":
            return pd.read_csv(p)
        raise ValueError(f"Неподдерживаемый тип файла: {p.suffix}")

    reference_df = load_data(args.reference_data)
    current_df = load_data(args.current_data)

    # Загрузка модели (предполагается, что модель уже обучена и сериализована)
    model = joblib.load(args.model_path)

    # Генерация отчёта Evidently
    report_dict = run_drift_report(
        reference_data=reference_df,
        current_data=current_df,
        model=model,
        target_column=args.target_column,
        report_html_path=args.report_html,
        report_json_path=args.report_json,
    )

    # Краткая сводка для логов пайплайна:
    # обычно удобно выводить агрегаты data drift (сколько признаков “поплыло”)
    data_drift = report_dict.get("metrics", {}).get("data_drift", {}).get("data", {})
    summary = {
        "n_features": data_drift.get("n_features"),
        "n_drifted_features": data_drift.get("n_drifted_features"),
        "share_drifted_features": data_drift.get("share_drifted_features"),
        "dataset_drift": data_drift.get("dataset_drift"),
    }
    print("=== Drift Summary (Evidently) ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
