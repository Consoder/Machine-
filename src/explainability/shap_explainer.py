import shap
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
    """
    Wraps SHAP TreeExplainer for XGBoost.
    Generates:
      - Summary plot   : which features matter most globally
      - Waterfall plot : why THIS prediction was made
      - Force plot     : single prediction breakdown
    """

    def __init__(self):
        self.explainer   = None
        self.model       = None
        self.is_fitted   = False

    def fit(self, model, X_background: pd.DataFrame):
        """
        Build SHAP explainer from trained XGBoost model.
        X_background: sample of training data (100-500 rows is enough)
        """
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
        """
        Summary plot - shows top features by mean |SHAP value|
        This is your global feature importance chart.
        """
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
        """
        Waterfall plot for a single prediction.
        Shows exactly which features pushed risk UP or DOWN.
        """
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
        """
        Returns top N features driving a single prediction.
        Used by the API to return explainability in JSON.
        """
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
