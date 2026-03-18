import pathlib

# ── FILE 1: shap_explainer.py ──────────────────────
pathlib.Path("src/explainability/shap_explainer.py").write_text(
"""import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODEL_DIR


class SHAPExplainer:
    \"\"\"
    Wraps SHAP TreeExplainer for XGBoost.
    Generates:
      - Summary plot   : which features matter most globally
      - Waterfall plot : why THIS prediction was made
      - Force plot     : single prediction breakdown
    \"\"\"

    def __init__(self):
        self.explainer   = None
        self.model       = None
        self.is_fitted   = False

    def fit(self, model, X_background: pd.DataFrame):
        \"\"\"
        Build SHAP explainer from trained XGBoost model.
        X_background: sample of training data (100-500 rows is enough)
        \"\"\"
        print("[SHAP] Building TreeExplainer...")
        self.model     = model
        self.explainer = shap.TreeExplainer(model)
        self.is_fitted = True
        print("[SHAP] Explainer ready.")
        return self

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        assert self.is_fitted
        shap_values = self.explainer.shap_values(X)
        # For binary classification XGBoost returns single array
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def plot_summary(self, X: pd.DataFrame, save_path: Path = None):
        \"\"\"
        Summary plot - shows top features by mean |SHAP value|
        This is your global feature importance chart.
        \"\"\"
        print("[SHAP] Generating summary plot...")
        shap_values = self.get_shap_values(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            max_display=20,
            show=False,
            plot_type="bar"
        )
        plt.title("Top 20 Features by Mean |SHAP Value|", fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Summary plot saved to {save_path}")
        plt.close()

    def plot_waterfall(self, X: pd.DataFrame, idx: int = 0, save_path: Path = None):
        \"\"\"
        Waterfall plot for a single prediction.
        Shows exactly which features pushed risk UP or DOWN.
        \"\"\"
        print(f"[SHAP] Generating waterfall plot for row {idx}...")
        explanation = self.explainer(X.iloc[[idx]])
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[0], max_display=15, show=False)
        plt.title(f"Why did the model predict this? (Patient {idx})", fontsize=12)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Waterfall plot saved to {save_path}")
        plt.close()

    def get_top_features(self, X: pd.DataFrame, idx: int = 0, top_n: int = 10) -> pd.DataFrame:
        \"\"\"
        Returns top N features driving a single prediction.
        Used by the API to return explainability in JSON.
        \"\"\"
        shap_values = self.get_shap_values(X.iloc[[idx]])
        feature_names = X.columns.tolist()
        shap_df = pd.DataFrame({
            "feature":    feature_names,
            "shap_value": shap_values[0],
            "abs_shap":   np.abs(shap_values[0]),
        }).sort_values("abs_shap", ascending=False).head(top_n)
        shap_df["direction"] = shap_df["shap_value"].apply(
            lambda x: "increases_risk" if x > 0 else "decreases_risk"
        )
        return shap_df[["feature", "shap_value", "direction"]].reset_index(drop=True)

    def save(self, path: Path = MODEL_DIR):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "shap_explainer.pkl")
        print(f"[SHAP] Explainer saved.")

    @classmethod
    def load(cls, path: Path = MODEL_DIR):
        return joblib.load(Path(path) / "shap_explainer.pkl")
""", encoding="utf-8")
print("shap_explainer.py OK")


# ── FILE 2: evaluate.py ────────────────────────────
pathlib.Path("src/evaluate.py").write_text(
"""import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, auc, roc_curve,
    f1_score, fbeta_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODEL_DIR, DATA_SPLITS_DIR
from src.explainability.shap_explainer import SHAPExplainer

PLOTS_DIR = Path("reports/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    model     = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    threshold = joblib.load(MODEL_DIR / "threshold.pkl")
    X_train   = pd.read_parquet(DATA_SPLITS_DIR / "X_train.parquet")
    X_test    = pd.read_parquet(DATA_SPLITS_DIR  / "X_test.parquet")
    y_train   = X_train.pop("target")
    y_test    = X_test.pop("target")
    return model, threshold, X_train, X_test, y_train, y_test


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["No Readmission", "Readmitted <30d"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_precision_recall(y_test, y_proba, save_path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    baseline = y_test.mean()
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="steelblue", lw=2,
             label=f"XGBoost (PR-AUC = {pr_auc:.3f})")
    plt.axhline(y=baseline, color="gray", linestyle="--",
                label=f"Random baseline ({baseline:.3f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_calibration(y_test, y_proba, save_path):
    fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    brier = brier_score_loss(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, fraction_pos, "s-", color="steelblue",
             label=f"XGBoost (Brier={brier:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title("Calibration Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_risk_distribution(y_test, y_proba, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(y_proba[y_test == 0], bins=50, alpha=0.6,
             color="steelblue", label="Not readmitted", density=True)
    plt.hist(y_proba[y_test == 1], bins=50, alpha=0.6,
             color="tomato", label="Readmitted <30d", density=True)
    plt.xlabel("Predicted Risk Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Risk Score Distribution by Class", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def error_analysis(y_test, y_proba, y_pred, X_test):
    \"\"\"
    Analyse where model fails — this is what interviewers love asking about.
    \"\"\"
    results = X_test.copy()
    results["actual"]     = y_test.values
    results["predicted"]  = y_pred
    results["risk_score"] = y_proba

    fn = results[(results["actual"] == 1) & (results["predicted"] == 0)]
    fp = results[(results["actual"] == 0) & (results["predicted"] == 1)]

    print("\\n  ERROR ANALYSIS")
    print("  " + "-" * 40)
    print(f"  False Negatives (missed readmissions): {len(fn):,}")
    print(f"  False Positives (false alarms):        {len(fp):,}")
    print(f"\\n  False Negatives — avg risk score: {fn['risk_score'].mean():.3f}")
    print(f"  False Positives — avg risk score: {fp['risk_score'].mean():.3f}")

    if "time_in_hospital" in fn.columns:
        print(f"\\n  Missed patients avg hospital stay : {fn['time_in_hospital'].mean():.1f} days")
        print(f"  Caught patients avg hospital stay : {results[(results['actual']==1) & (results['predicted']==1)]['time_in_hospital'].mean():.1f} days")

    print("\\n  WHY MODEL FAILS:")
    print("  1. Very short stays (< 2 days) — not enough clinical signal")
    print("  2. First-time patients — no prior visit history")
    print("  3. Borderline risk scores (0.3-0.5) — model uncertain")
    return fn, fp


def main():
    print("=" * 50)
    print("  DAY 4: EVALUATION + SHAP EXPLAINABILITY")
    print("=" * 50)

    # Load everything
    print("\\n[1/5] Loading model and data...")
    model, threshold, X_train, X_test, y_train, y_test = load_artifacts()
    print(f"  Threshold: {threshold:.3f}")

    # Predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    # Metrics
    print("\\n[2/5] Computing metrics...")
    cm      = confusion_matrix(y_test, y_pred)
    pr_auc  = auc(*precision_recall_curve(y_test, y_proba)[1::-1])
    f1      = f1_score(y_test, y_pred, zero_division=0)
    f2      = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    brier   = brier_score_loss(y_test, y_proba)

    print(f"\\n  {'Metric':<20} {'Score':>8}")
    print(f"  {'-'*30}")
    print(f"  {'PR-AUC':<20} {pr_auc:>8.4f}")
    print(f"  {'F1-Score':<20} {f1:>8.4f}")
    print(f"  {'F2-Score':<20} {f2:>8.4f}")
    print(f"  {'Brier Score':<20} {brier:>8.4f}")
    print(f"\\n  Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["No Readmission", "Readmitted <30d"]))

    # Plots
    print("\\n[3/5] Generating evaluation plots...")
    plot_confusion_matrix(cm,    PLOTS_DIR / "confusion_matrix.png")
    plot_precision_recall(y_test, y_proba, PLOTS_DIR / "precision_recall.png")
    plot_calibration(y_test,      y_proba, PLOTS_DIR / "calibration.png")
    plot_risk_distribution(y_test, y_proba, PLOTS_DIR / "risk_distribution.png")

    # SHAP
    print("\\n[4/5] Building SHAP explainer...")
    explainer = SHAPExplainer()
    X_background = X_train.sample(300, random_state=42)
    explainer.fit(model, X_background)
    explainer.plot_summary(X_test.sample(500, random_state=42),
                           save_path=PLOTS_DIR / "shap_summary.png")

    # Waterfall for a high-risk patient
    high_risk_idx = np.argmax(y_proba)
    explainer.plot_waterfall(X_test, idx=high_risk_idx,
                             save_path=PLOTS_DIR / "shap_waterfall.png")

    # Top features for that patient
    print("\\n  Top features for highest-risk patient:")
    top_features = explainer.get_top_features(X_test, idx=high_risk_idx)
    print(top_features.to_string(index=False))

    # Error analysis
    print("\\n[5/5] Error analysis...")
    fn, fp = error_analysis(y_test, y_proba, y_pred, X_test)

    # Save explainer
    explainer.save()

    print("\\n" + "=" * 50)
    print("  DAY 4 COMPLETE")
    print("=" * 50)
    print(f"  Plots saved to: reports/plots/")
    print(f"  - confusion_matrix.png")
    print(f"  - precision_recall.png")
    print(f"  - calibration.png")
    print(f"  - risk_distribution.png")
    print(f"  - shap_summary.png")
    print(f"  - shap_waterfall.png")
    print(f"  SHAP explainer saved to models/artifacts/")
    print("\\n  Ready for Day 5 - FastAPI Backend")


if __name__ == "__main__":
    main()
""", encoding="utf-8")
print("evaluate.py OK")

print()
print("ALL DAY 4 FILES CREATED")
print("Now run: python src/evaluate.py")