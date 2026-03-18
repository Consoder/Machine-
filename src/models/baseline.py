import pandas as pd
import numpy as np
import joblib
import mlflow
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, fbeta_score,
    classification_report, confusion_matrix
)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODEL_DIR, DATA_SPLITS_DIR, RANDOM_STATE


def load_splits():
    X_train = pd.read_parquet(DATA_SPLITS_DIR / "X_train.parquet")
    X_val   = pd.read_parquet(DATA_SPLITS_DIR / "X_val.parquet")
    X_test  = pd.read_parquet(DATA_SPLITS_DIR / "X_test.parquet")
    y_train = X_train.pop("target")
    y_val   = X_val.pop("target")
    y_test  = X_test.pop("target")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    f2  = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    return {"pr_auc": pr_auc, "f1": f1, "f2": f2}


def train_baseline():
    print("=" * 50)
    print("  BASELINE: Logistic Regression")
    print("=" * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    mlflow.set_experiment("hospital-readmission")
    with mlflow.start_run(run_name="baseline_logistic_regression"):

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=0.1,
        )
        model.fit(X_train, y_train)

        # Evaluate on val
        val_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, val_proba)

        # Evaluate on test
        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_proba)

        # Log to MLflow
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", 0.1)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        print(f"\n  Val  PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"  Val  F2:     {val_metrics['f2']:.4f}")
        print(f"  Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"  Test F2:     {test_metrics['f2']:.4f}")

        # Save model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "baseline_model.pkl")
        mlflow.sklearn.log_model(model, "baseline_model")
        print("\n  Baseline saved.")

    return val_metrics, test_metrics
