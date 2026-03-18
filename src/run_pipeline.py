import pandas as pd
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

    print("\n[1/5] Loading raw data...")
    df = pd.read_csv(RAW_FILE)
    print(f"      Raw shape: {df.shape}")

    print("\n[2/5] Preprocessing...")
    preprocessor = HospitalDataPreprocessor()
    df_clean = preprocessor.fit_transform(df)
    print(f"      Clean shape: {df_clean.shape}")

    print("\n[3/5] Creating splits...")
    train_df, val_df, test_df = create_splits(df_clean)

    print("\n[4/5] Feature engineering...")
    fe = FeatureEngineer()
    X_train, y_train = fe.fit_transform(train_df)
    X_val,   y_val   = fe.transform(val_df)
    X_test,  y_test  = fe.transform(test_df)
    print(f"      Features: {X_train.shape[1]}")

    print("\n[5/5] Saving...")
    preprocessor.save()
    fe.save()
    DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    X_train.assign(target=y_train).to_parquet(DATA_SPLITS_DIR / "X_train.parquet", index=False)
    X_val.assign(target=y_val).to_parquet(DATA_SPLITS_DIR     / "X_val.parquet",   index=False)
    X_test.assign(target=y_test).to_parquet(DATA_SPLITS_DIR   / "X_test.parquet",  index=False)

    print("\n" + "=" * 50)
    print("  PIPELINE COMPLETE")
    print("=" * 50)
    print(f"  Train: {len(X_train):,} rows")
    print(f"  Val:   {len(X_val):,} rows")
    print(f"  Test:  {len(X_test):,} rows")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Positive rate: {y_train.mean()*100:.1f}%")
    print("\n  Ready for Day 3 - Model Training")

if __name__ == "__main__":
    main()
