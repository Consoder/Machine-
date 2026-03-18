import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, fbeta_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODEL_DIR, DATA_SPLITS_DIR, RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f2_scores = []
    for p, r in zip(precision, recall):
        if (4 * p + r) > 0:
            f2 = (5 * p * r) / (4 * p + r)
        else:
            f2 = 0
        f2_scores.append(f2)
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"  Best threshold: {best_threshold:.3f} (F2={f2_scores[best_idx]:.4f})")
    return best_threshold


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 5, 15),
        "random_state":      RANDOM_STATE,
        "eval_metric":       "aucpr",
        "use_label_encoder": False,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, val_proba)
    return auc(recall, precision)


def train_xgboost(n_trials=50):
    print("=" * 50)
    print("  XGBoost + SMOTE + Optuna + MLflow")
    print("=" * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # Apply SMOTE to training data only
    print(f"\n  Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE:  {pd.Series(y_train_sm).value_counts().to_dict()}")

    # Optuna hyperparameter search
    print(f"\n  Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train_sm, y_train_sm, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = study.best_params
    best_params["random_state"] = RANDOM_STATE
    best_params["eval_metric"]  = "aucpr"
    print(f"\n  Best params: {best_params}")
    print(f"  Best val PR-AUC: {study.best_value:.4f}")

    # Train final model with best params
    print("\n  Training final model...")
    mlflow.set_experiment("hospital-readmission")
    with mlflow.start_run(run_name="xgboost_optuna"):

        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(
            X_train_sm, y_train_sm,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Find best threshold on validation set
        val_proba = final_model.predict_proba(X_val)[:, 1]
        best_threshold = find_best_threshold(y_val, val_proba)

        # Evaluate on validation
        val_metrics = compute_metrics(y_val, val_proba, best_threshold)

        # Evaluate on test
        test_proba = final_model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_proba, best_threshold)

        # Confusion matrix
        y_test_pred = (test_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\n  Confusion Matrix (Test):")
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_test_pred))

        # Log everything to MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("threshold", best_threshold)
        mlflow.log_param("smote", True)
        mlflow.log_param("n_optuna_trials", n_trials)
        mlflow.log_metrics({f"val_{k}":  v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.xgboost.log_model(final_model, "xgboost_model")

        print(f"\n  Val  PR-AUC : {val_metrics['pr_auc']:.4f}")
        print(f"  Val  F2      : {val_metrics['f2']:.4f}")
        print(f"  Test PR-AUC  : {test_metrics['pr_auc']:.4f}")
        print(f"  Test F2      : {test_metrics['f2']:.4f}")

        # Save model + threshold
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model,   MODEL_DIR / "xgboost_model.pkl")
        joblib.dump(best_threshold, MODEL_DIR / "threshold.pkl")
        joblib.dump(best_params,   MODEL_DIR / "best_params.pkl")
        print("\n  XGBoost model saved.")

    return final_model, best_threshold, val_metrics, test_metrics
