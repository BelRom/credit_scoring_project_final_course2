from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
from pathlib import Path

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay,
)

from src.models.pipeline import create_pipeline, get_search_space


def evaluate_on_test(best_estimator, X_test: pd.DataFrame, y_test: pd.Series, roc_path: str) -> dict:
    """Считает метрики на тесте и строит ROC-кривую + возвращает метрики dict."""
    if hasattr(best_estimator, "predict_proba"):
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
    else:
        y_proba = best_estimator.decision_function(X_test)

    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "test_auc": float(roc_auc_score(y_test, y_proba)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print("\n=== Test metrics ===")
    print(f"ROC-AUC   : {metrics['test_auc']:.4f}")
    print(f"Precision : {metrics['test_precision']:.4f}")
    print(f"Recall    : {metrics['test_recall']:.4f}")
    print(f"F1-Score  : {metrics['test_f1']:.4f}")

    # ROC curve artifact
    Path(roc_path).parent.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC curve (test)")
    plt.grid(True, alpha=0.3)
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nROC curve saved to: {roc_path}")

    return metrics


def safe_log_params(params: dict, prefix: str = "") -> None:
    """MLflow params должны быть простыми типами/строками."""
    for k, v in params.items():
        key = f"{prefix}{k}"
        try:
            mlflow.log_param(key, v if isinstance(v, (str, int, float, bool)) else str(v))
        except Exception:
            # на всякий случай не падаем на экзотике
            mlflow.log_param(key, str(v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed dataset (csv).")
    parser.add_argument("--target", default="default", help="Target column name.")
    parser.add_argument("--model", choices=["logreg", "gboost", "rf", "hgb"], default="logreg")

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=50)

    parser.add_argument("--n-iter", type=int, default=30, help="RandomizedSearchCV iterations.")
    parser.add_argument("--cv", type=int, default=5)

    parser.add_argument("--out-model", default="models/best_model.joblib")
    parser.add_argument("--out-roc", default="models/roc_curve.png")

    # MLflow
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="Credit_Default_Prediction")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--register-name", default="CreditDefaultModel")  # можно "", если не нужен registry
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset columns.")

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)

    # 1) split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # 2) pipeline
    pipe = create_pipeline(args.model, X_train, random_state=args.random_state)

    # 3) search
    param_dist = get_search_space(args.model)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="roc_auc",
        cv=args.cv,
        n_jobs=-1,
        random_state=args.random_state,
        verbose=1,
        refit=True,
    )

    # ---- MLflow setup
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or f"{args.model}_rs{args.random_state}_iter{args.n_iter}_cv{args.cv}"

    with mlflow.start_run(run_name=run_name):
        # базовые параметры
        mlflow.set_tag("model_type", args.model)
        safe_log_params(
            {
                "data_path": args.data,
                "target": args.target,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_iter": args.n_iter,
                "cv": args.cv,
            }
        )

        # обучение + подбор
        search.fit(X_train, y_train)

        print("\n=== Best params ===")
        print(search.best_params_)
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

        # логируем best params + CV metric
        safe_log_params(search.best_params_, prefix="best__")
        mlflow.log_metric("cv_best_auc", float(search.best_score_))

        best_model = search.best_estimator_

        # тестовые метрики + ROC artifact
        test_metrics = evaluate_on_test(best_model, X_test, y_test, args.out_roc)
        for k, v in test_metrics.items():
            mlflow.log_metric(k, v)

        # ROC как artifact
        mlflow.log_artifact(args.out_roc)

        # модель в MLflow
        register_name = args.register_name.strip()
        if register_name:
            mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name=register_name)
        else:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

        # параллельно сохраним joblib (удобно для локального predict.py)
        Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, args.out_model)
        mlflow.log_artifact(args.out_model)

        print(f"\nModel saved to: {args.out_model}")


if __name__ == "__main__":
    main()
