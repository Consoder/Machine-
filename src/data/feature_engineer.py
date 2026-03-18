import pandas as pd
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
