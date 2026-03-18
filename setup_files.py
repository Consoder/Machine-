import pathlib

# ── config.py ──────────────────────────────────────
pathlib.Path("src/config.py").write_text(
"""from pathlib import Path

ROOT_DIR           = Path(__file__).parent.parent
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_SPLITS_DIR    = ROOT_DIR / "data" / "splits"
MODEL_DIR          = ROOT_DIR / "models" / "artifacts"

RAW_FILE          = DATA_RAW_DIR / "dataset_diabetes" / "diabetic_data.csv"
TARGET_COLUMN     = "readmitted"
PATIENT_ID_COLUMN = "patient_nbr"
POSITIVE_LABEL    = "<30"

RANDOM_STATE = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.10

NUMERICAL_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "age", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed",
]

MEDICATION_FEATURES = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

DIAGNOSIS_FEATURES = ["diag_1", "diag_2", "diag_3"]

DROP_COLUMNS = [
    "encounter_id",
    "weight",
    "payer_code",
    "examide",
    "citoglipton",
]

ICD9_GROUPS = {
    "circulatory":     (390, 459),
    "respiratory":     (460, 519),
    "digestive":       (520, 579),
    "diabetes":        (250, 250),
    "injury":          (800, 999),
    "musculoskeletal": (710, 739),
    "genitourinary":   (580, 629),
    "neoplasms":       (140, 239),
    "mental":          (290, 319),
    "nervous":         (320, 389),
    "skin":            (680, 709),
    "endocrine":       (240, 279),
    "blood":           (280, 289),
    "infectious":      (1,   139),
    "perinatal":       (760, 779),
    "congenital":      (740, 759),
    "ill_defined":     (780, 799),
    "supplementary":   (0,     0),
}
""", encoding="utf-8")
print("config.py OK")


# ── preprocessor.py ────────────────────────────────
pathlib.Path("src/data/preprocessor.py").write_text(
"""import pandas as pd
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
""", encoding="utf-8")
print("preprocessor.py OK")


# ── feature_engineer.py ────────────────────────────
pathlib.Path("src/data/feature_engineer.py").write_text(
"""import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODEL_DIR, DATA_SPLITS_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE


class FeatureEngineer:
    def __init__(self):
        self.encoder       = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._cat_cols     = []
        self._num_cols     = []
        self.feature_names = []
        self.is_fitted     = False

    def fit(self, df, target_col="target"):
        features = df.drop(columns=[target_col], errors="ignore")
        self._cat_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
        self._num_cols = features.select_dtypes(include=["number"]).columns.tolist()
        if self._cat_cols:
            self.encoder.fit(features[self._cat_cols])
        self.feature_names = self._num_cols + self._cat_cols
        self.is_fitted = True
        print(f"[FeatureEngineer] Fitted: {len(self._num_cols)} numeric, {len(self._cat_cols)} categorical")
        return self

    def transform(self, df, target_col="target"):
        assert self.is_fitted
        y = df[target_col].copy() if target_col in df.columns else None
        features = df.drop(columns=[target_col], errors="ignore")
        X_num = features[self._num_cols].copy()
        if self._cat_cols:
            X_cat = pd.DataFrame(
                self.encoder.transform(features[self._cat_cols]),
                columns=self._cat_cols,
                index=features.index,
            )
            X = pd.concat([X_num, X_cat], axis=1)
        else:
            X = X_num
        return X, y

    def fit_transform(self, df, target_col="target"):
        return self.fit(df, target_col).transform(df, target_col)

    def save(self, path=None):
        if path is None:
            path = MODEL_DIR
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "feature_engineer.pkl")
        print(f"[FeatureEngineer] Saved.")

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = MODEL_DIR
        return joblib.load(Path(path) / "feature_engineer.pkl")


def create_splits(df, target_col="target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_adjusted, stratify=y_tv, random_state=RANDOM_STATE
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df   = pd.concat([X_val,   y_val],   axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)
    DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(DATA_SPLITS_DIR / "train.parquet", index=False)
    val_df.to_parquet(DATA_SPLITS_DIR   / "val.parquet",   index=False)
    test_df.to_parquet(DATA_SPLITS_DIR  / "test.parquet",  index=False)
    print(f"[Splits] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"[Splits] Positive rate: Train={y_train.mean():.3f} Val={y_val.mean():.3f} Test={y_test.mean():.3f}")
    return train_df, val_df, test_df
""", encoding="utf-8")
print("feature_engineer.py OK")


# ── run_pipeline.py ────────────────────────────────
pathlib.Path("src/run_pipeline.py").write_text(
"""import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_FILE, DATA_SPLITS_DIR, MODEL_DIR
from src.data.preprocessor import HospitalDataPreprocessor
from src.data.feature_engineer import FeatureEngineer, create_splits

def main():
    print("=" * 50)
    print("  HOSPITAL READMISSION - DATA PIPELINE")
    print("=" * 50)

    print("\\n[1/5] Loading raw data...")
    df = pd.read_csv(RAW_FILE)
    print(f"      Raw shape: {df.shape}")

    print("\\n[2/5] Preprocessing...")
    preprocessor = HospitalDataPreprocessor()
    df_clean = preprocessor.fit_transform(df)
    print(f"      Clean shape: {df_clean.shape}")

    print("\\n[3/5] Creating splits...")
    train_df, val_df, test_df = create_splits(df_clean)

    print("\\n[4/5] Feature engineering...")
    fe = FeatureEngineer()
    X_train, y_train = fe.fit_transform(train_df)
    X_val,   y_val   = fe.transform(val_df)
    X_test,  y_test  = fe.transform(test_df)
    print(f"      Features: {X_train.shape[1]}")

    print("\\n[5/5] Saving...")
    preprocessor.save()
    fe.save()
    DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    X_train.assign(target=y_train).to_parquet(DATA_SPLITS_DIR / "X_train.parquet", index=False)
    X_val.assign(target=y_val).to_parquet(DATA_SPLITS_DIR     / "X_val.parquet",   index=False)
    X_test.assign(target=y_test).to_parquet(DATA_SPLITS_DIR   / "X_test.parquet",  index=False)

    print("\\n" + "=" * 50)
    print("  PIPELINE COMPLETE")
    print("=" * 50)
    print(f"  Train: {len(X_train):,} rows")
    print(f"  Val:   {len(X_val):,} rows")
    print(f"  Test:  {len(X_test):,} rows")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Positive rate: {y_train.mean()*100:.1f}%")
    print("\\n  Ready for Day 3 - Model Training")

if __name__ == "__main__":
    main()
""", encoding="utf-8")
print("run_pipeline.py OK")

print()
print("ALL FILES CREATED SUCCESSFULLY")
print("Now run: python src/run_pipeline.py")