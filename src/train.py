import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.baseline    import train_baseline
from src.models.xgboost_model import train_xgboost


def main():
    print("\n STEP 1: Baseline Model")
    print("-" * 50)
    val_b, test_b = train_baseline()

    print("\n STEP 2: XGBoost Model")
    print("-" * 50)
    model, threshold, val_x, test_x = train_xgboost(n_trials=50)

    print("\n" + "=" * 50)
    print("  FINAL COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<20} {'Baseline':>10} {'XGBoost':>10}")
    print("-" * 42)
    print(f"{'Val PR-AUC':<20} {val_b['pr_auc']:>10.4f} {val_x['pr_auc']:>10.4f}")
    print(f"{'Val F2':<20} {val_b['f2']:>10.4f} {val_x['f2']:>10.4f}")
    print(f"{'Test PR-AUC':<20} {test_b['pr_auc']:>10.4f} {test_x['pr_auc']:>10.4f}")
    print(f"{'Test F2':<20} {test_b['f2']:>10.4f} {test_x['f2']:>10.4f}")
    print("\n  Models saved to models/artifacts/")
    print("  MLflow runs saved to mlruns/")
    print("\n  Ready for Day 4 - SHAP Explainability")


if __name__ == "__main__":
    main()
