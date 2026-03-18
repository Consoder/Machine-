import pathlib

pathlib.Path("src/api/predictor.py").write_text(
"""import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODEL_DIR, MEDICATION_FEATURES
from src.api.schemas import PatientInput, PredictionResponse, FeatureImportance


class ReadmissionPredictor:
    def __init__(self):
        self.model     = None
        self.explainer = None
        self.fe        = None
        self.threshold = 0.5
        self.is_loaded = False

    def load(self):
        print("[Predictor] Loading model artifacts...")
        self.model     = joblib.load(MODEL_DIR / "xgboost_model.pkl")
        self.explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
        self.fe        = joblib.load(MODEL_DIR / "feature_engineer.pkl")
        self.threshold = float(joblib.load(MODEL_DIR / "threshold.pkl"))
        self.is_loaded = True
        print(f"[Predictor] Loaded. Threshold={self.threshold:.3f}")

    def _input_to_dataframe(self, patient: PatientInput) -> pd.DataFrame:
        med_map = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}

        # All 23 medication columns — default to 0 (not prescribed)
        med_data = {med: 0 for med in MEDICATION_FEATURES}
        # Override the two we collect from user
        med_data["metformin"] = med_map.get(patient.metformin, 0)
        med_data["insulin"]   = med_map.get(patient.insulin, 0)

        insulin_val   = med_data["insulin"]
        metformin_val = med_data["metformin"]

        base_data = {
            "time_in_hospital":         patient.time_in_hospital,
            "num_lab_procedures":       patient.num_lab_procedures,
            "num_procedures":           patient.num_procedures,
            "num_medications":          patient.num_medications,
            "number_outpatient":        patient.number_outpatient,
            "number_emergency":         patient.number_emergency,
            "number_inpatient":         patient.number_inpatient,
            "number_diagnoses":         patient.number_diagnoses,
            "age":                      patient.age,
            "gender":                   patient.gender,
            "race":                     patient.race,
            "admission_type_id":        patient.admission_type_id,
            "discharge_disposition_id": patient.discharge_disposition_id,
            "admission_source_id":      patient.admission_source_id,
            "medical_specialty":        patient.medical_specialty,
            "max_glu_serum":            patient.max_glu_serum,
            "A1Cresult":                patient.A1Cresult,
            "change":                   patient.change,
            "diabetesMed":              patient.diabetesMed,
            "diag_1_group":             patient.diag_1_group,
            "diag_2_group":             patient.diag_2_group,
            "diag_3_group":             patient.diag_3_group,
            # Engineered features
            "total_meds_on":            patient.num_medications,
            "total_meds_changed":       1 if patient.change == "Ch" else 0,
            "total_meds_up":            1 if patient.insulin == "Up" else 0,
            "insulin_changed":          1 if patient.insulin in ["Up", "Down"] else 0,
            "prior_visits_total":       patient.number_outpatient + patient.number_emergency + patient.number_inpatient,
            "labs_per_day":             patient.num_lab_procedures / max(patient.time_in_hospital, 1),
            "a1c_not_measured":         1 if patient.A1Cresult == "None" else 0,
        }

        # Merge base + all medications
        data = {**base_data, **med_data}
        df = pd.DataFrame([data])

        # Use the same FeatureEngineer from training to encode categoricals
        X, _ = self.fe.transform(df.assign(target=0))
        return X

    def predict_single(self, patient: PatientInput) -> PredictionResponse:
        assert self.is_loaded
        X = self._input_to_dataframe(patient)
        risk_score = float(self.model.predict_proba(X)[:, 1][0])
        prediction = int(risk_score >= self.threshold)

        if risk_score >= 0.6:
            risk_label = "HIGH"
        elif risk_score >= 0.3:
            risk_label = "MEDIUM"
        else:
            risk_label = "LOW"

        try:
            top_df = self.explainer.get_top_features(X, idx=0, top_n=10)
            top_features = [
                FeatureImportance(
                    feature=row["feature"],
                    shap_value=round(float(row["shap_value"]), 4),
                    direction=row["direction"],
                )
                for _, row in top_df.iterrows()
            ]
        except Exception as e:
            print(f"SHAP error: {e}")
            top_features = []

        return PredictionResponse(
            risk_score=round(risk_score, 4),
            risk_label=risk_label,
            risk_percent=f"{risk_score * 100:.1f}%",
            prediction=prediction,
            threshold=round(self.threshold, 3),
            top_features=top_features,
        )

    def predict_batch(self, patients: List[PatientInput]) -> List[PredictionResponse]:
        return [self.predict_single(p) for p in patients]


predictor = ReadmissionPredictor()
""", encoding="utf-8")
print("predictor.py fixed OK")