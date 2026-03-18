import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    TARGET_COLUMN, PATIENT_ID_COLUMN, POSITIVE_LABEL,
    DROP_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    MEDICATION_FEATURES, DIAGNOSIS_FEATURES, ICD9_GROUPS, MODEL_DIR
)

def map_icd9_to_group(code):
    if pd.isna(code) or str(code).strip() in ("?", ""):
        return "unknown"
    code = str(code).strip()
    if code.startswith("V"):
        return "supplementary"
    if code.startswith("E"):
        return "external_injury"
    try:
        num = float(code.split(".")[0])
        for group_name, (low, high) in ICD9_GROUPS.items():
            if low <= num <= high:
                return group_name
        return "other"
    except ValueError:
        return "unknown"


class HospitalDataPreprocessor:
    def __init__(self):
        self._medians  = {}
        self._modes    = {}
        self.is_fitted = False
        self._age_map  = {
            "[0-10)": 5,   "[10-20)": 15, "[20-30)": 25,
            "[30-40)": 35, "[40-50)": 45, "[50-60)": 55,
            "[60-70)": 65, "[70-80)": 75, "[80-90)": 85,
            "[90-100)": 95,
        }
        self._med_map = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}

    def fit(self, df):
        print(f"[Preprocessor] Fitting on {len(df):,} rows...")
        df = df.replace("?", np.nan)
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                self._medians[col] = df[col].median()
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                mode_vals = df[col].mode()
                self._modes[col] = mode_vals[0] if len(mode_vals) > 0 else "unknown"
        self.is_fitted = True
        print(f"[Preprocessor] Fitted.")
        return self

    def transform(self, df):
        assert self.is_fitted, "Call fit() first!"
        df = df.copy()
        print(f"[Preprocessor] Transforming {len(df):,} rows...")
        df.replace("?", np.nan, inplace=True)
        cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"  Dropped {len(cols_to_drop)} columns")
        if PATIENT_ID_COLUMN in df.columns:
            before = len(df)
            df.drop_duplicates(subset=[PATIENT_ID_COLUMN], keep="first", inplace=True)
            df.drop(columns=[PATIENT_ID_COLUMN], inplace=True)
            print(f"  Deduplicated: {before:,} to {len(df):,} rows")
        if TARGET_COLUMN in df.columns:
            df["target"] = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
            df.drop(columns=[TARGET_COLUMN], inplace=True)
            print(f"  Positive rate: {df['target'].mean()*100:.1f}%")
        if "age" in df.columns:
            df["age"] = df["age"].map(self._age_map)
        for diag_col in DIAGNOSIS_FEATURES:
            if diag_col in df.columns:
                df[f"{diag_col}_group"] = df[diag_col].apply(map_icd9_to_group)
                df.drop(columns=[diag_col], inplace=True)
        med_cols = [m for m in MEDICATION_FEATURES if m in df.columns]
        for med in med_cols:
            df[med] = df[med].map(self._med_map).fillna(0).astype(int)
        df["total_meds_on"]      = (df[med_cols] > 0).sum(axis=1)
        df["total_meds_changed"] = (df[med_cols].isin([2, 3])).sum(axis=1)
        df["total_meds_up"]      = (df[med_cols] == 3).sum(axis=1)
        df["insulin_changed"]    = df.get("insulin", pd.Series(1, index=df.index)).isin([2, 3]).astype(int)
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col].fillna(self._medians.get(col, 0), inplace=True)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col].fillna(self._modes.get(col, "unknown"), inplace=True)
        df["prior_visits_total"] = (
            df.get("number_outpatient", 0) +
            df.get("number_emergency", 0) +
            df.get("number_inpatient", 0)
        )
        df["labs_per_day"] = (
            df.get("num_lab_procedures", 0) /
            df.get("time_in_hospital", pd.Series(1, index=df.index)).clip(lower=1)
        )
        df["a1c_not_measured"] = (df.get("A1Cresult", pd.Series("None", index=df.index)) == "None").astype(int)
        print(f"  Output shape: {df.shape}")
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def save(self, path=None):
        if path is None:
            path = MODEL_DIR
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "preprocessor.pkl")
        print(f"[Preprocessor] Saved.")

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = MODEL_DIR
        return joblib.load(Path(path) / "preprocessor.pkl")
