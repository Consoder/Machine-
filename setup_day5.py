import pathlib

# ── FILE 1: schemas.py ─────────────────────────────
pathlib.Path("src/api/schemas.py").write_text(
"""from pydantic import BaseModel, Field
from typing import List, Optional


class PatientInput(BaseModel):
    time_in_hospital:        int   = Field(..., ge=1, le=14,  description="Days in hospital")
    num_lab_procedures:      int   = Field(..., ge=0, le=132, description="Number of lab procedures")
    num_procedures:          int   = Field(..., ge=0, le=6,   description="Number of procedures")
    num_medications:         int   = Field(..., ge=0, le=81,  description="Number of medications")
    number_outpatient:       int   = Field(..., ge=0,         description="Prior outpatient visits")
    number_emergency:        int   = Field(..., ge=0,         description="Prior emergency visits")
    number_inpatient:        int   = Field(..., ge=0,         description="Prior inpatient visits")
    number_diagnoses:        int   = Field(..., ge=0, le=16,  description="Number of diagnoses")
    age:                     int   = Field(...,               description="Age midpoint (e.g. 55)")
    gender:                  str   = Field(...,               description="Male or Female")
    race:                    str   = Field("Caucasian",       description="Patient race")
    admission_type_id:       int   = Field(1,                 description="1=Emergency, 2=Urgent, 3=Elective")
    discharge_disposition_id:int   = Field(1,                 description="Discharge destination")
    admission_source_id:     int   = Field(7,                 description="Admission source")
    medical_specialty:       str   = Field("InternalMedicine",description="Physician specialty")
    max_glu_serum:           str   = Field("None",            description="Max glucose serum result")
    A1Cresult:               str   = Field("None",            description="HbA1c test result")
    change:                  str   = Field("No",              description="Change in diabetes meds")
    diabetesMed:             str   = Field("Yes",             description="On diabetes medication")
    metformin:               str   = Field("No",              description="Metformin dosage change")
    insulin:                 str   = Field("No",              description="Insulin dosage change")
    diag_1_group:            str   = Field("circulatory",     description="Primary diagnosis group")
    diag_2_group:            str   = Field("other",           description="Secondary diagnosis group")
    diag_3_group:            str   = Field("other",           description="Tertiary diagnosis group")

    class Config:
        json_schema_extra = {
            "example": {
                "time_in_hospital": 5,
                "num_lab_procedures": 41,
                "num_procedures": 1,
                "num_medications": 12,
                "number_outpatient": 0,
                "number_emergency": 1,
                "number_inpatient": 2,
                "number_diagnoses": 7,
                "age": 65,
                "gender": "Female",
                "race": "Caucasian",
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "medical_specialty": "InternalMedicine",
                "max_glu_serum": "None",
                "A1Cresult": "None",
                "change": "Ch",
                "diabetesMed": "Yes",
                "metformin": "Steady",
                "insulin": "Up",
                "diag_1_group": "circulatory",
                "diag_2_group": "diabetes",
                "diag_3_group": "respiratory"
            }
        }


class FeatureImportance(BaseModel):
    feature:    str
    shap_value: float
    direction:  str


class PredictionResponse(BaseModel):
    risk_score:        float
    risk_label:        str
    risk_percent:      str
    prediction:        int
    threshold:         float
    top_features:      List[FeatureImportance]
    model_version:     str = "xgboost_v1"


class BatchPredictionResponse(BaseModel):
    total:             int
    high_risk_count:   int
    predictions:       List[PredictionResponse]


class HealthResponse(BaseModel):
    status:            str
    model_loaded:      bool
    version:           str = "1.0.0"
""", encoding="utf-8")
print("schemas.py OK")


# ── FILE 2: predictor.py ───────────────────────────
pathlib.Path("src/api/predictor.py").write_text(
"""import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODEL_DIR
from src.api.schemas import PatientInput, PredictionResponse, FeatureImportance


class ReadmissionPredictor:
    \"\"\"
    Loads trained model + explainer and runs inference.
    Loaded once at startup — not on every request.
    \"\"\"

    def __init__(self):
        self.model      = None
        self.explainer  = None
        self.threshold  = 0.5
        self.is_loaded  = False

    def load(self):
        print("[Predictor] Loading model artifacts...")
        self.model     = joblib.load(MODEL_DIR / "xgboost_model.pkl")
        self.explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
        self.threshold = float(joblib.load(MODEL_DIR / "threshold.pkl"))
        self.is_loaded = True
        print(f"[Predictor] Loaded. Threshold={self.threshold:.3f}")

    def _input_to_dataframe(self, patient: PatientInput) -> pd.DataFrame:
        \"\"\"Convert Pydantic model to DataFrame the model expects.\"\"\"
        med_map = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
        data = {
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
            "metformin":                med_map.get(patient.metformin, 0),
            "insulin":                  med_map.get(patient.insulin, 0),
            "diag_1_group":             patient.diag_1_group,
            "diag_2_group":             patient.diag_2_group,
            "diag_3_group":             patient.diag_3_group,
            # Engineered features
            "total_meds_on":      patient.num_medications,
            "total_meds_changed": 1 if patient.change == "Ch" else 0,
            "total_meds_up":      1 if patient.insulin == "Up" else 0,
            "insulin_changed":    1 if patient.insulin in ["Up", "Down"] else 0,
            "prior_visits_total": patient.number_outpatient + patient.number_emergency + patient.number_inpatient,
            "labs_per_day":       patient.num_lab_procedures / max(patient.time_in_hospital, 1),
            "a1c_not_measured":   1 if patient.A1Cresult == "None" else 0,
        }
        return pd.DataFrame([data])

    def predict_single(self, patient: PatientInput) -> PredictionResponse:
        assert self.is_loaded, "Call load() first"

        df = self._input_to_dataframe(patient)

        # Get risk score
        risk_score = float(self.model.predict_proba(df)[:, 1][0])
        prediction = int(risk_score >= self.threshold)

        # Risk label
        if risk_score >= 0.6:
            risk_label = "HIGH"
        elif risk_score >= 0.3:
            risk_label = "MEDIUM"
        else:
            risk_label = "LOW"

        # SHAP top features
        try:
            top_df = self.explainer.get_top_features(df, idx=0, top_n=10)
            top_features = [
                FeatureImportance(
                    feature=row["feature"],
                    shap_value=round(float(row["shap_value"]), 4),
                    direction=row["direction"],
                )
                for _, row in top_df.iterrows()
            ]
        except Exception:
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


# Global singleton — loaded once at startup
predictor = ReadmissionPredictor()
""", encoding="utf-8")
print("predictor.py OK")


# ── FILE 3: main.py ────────────────────────────────
pathlib.Path("src/api/main.py").write_text(
"""import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    PatientInput, PredictionResponse,
    BatchPredictionResponse, HealthResponse
)
from src.api.predictor import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    logger.info("Loading model artifacts...")
    predictor.load()
    logger.info("Model loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Predicts 30-day readmission risk for diabetic patients with SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration}ms)"
    )
    return response


# ── Routes ────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    \"\"\"Check if API and model are running.\"\"\"
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientInput):
    \"\"\"
    Predict 30-day readmission risk for a single patient.
    Returns risk score, label, and top SHAP features explaining the prediction.
    \"\"\"
    try:
        result = predictor.predict_single(patient)
        logger.info(f"Prediction: risk={result.risk_score} label={result.risk_label}")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(patients: List[PatientInput]):
    \"\"\"
    Predict readmission risk for multiple patients at once.
    Maximum 100 patients per request.
    \"\"\"
    if len(patients) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 patients per batch request."
        )
    try:
        predictions = predictor.predict_batch(patients)
        high_risk   = sum(1 for p in predictions if p.prediction == 1)
        logger.info(f"Batch: {len(patients)} patients, {high_risk} high risk")
        return BatchPredictionResponse(
            total=len(predictions),
            high_risk_count=high_risk,
            predictions=predictions,
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Hospital Readmission Prediction API",
        "docs":    "/docs",
        "health":  "/health",
    }
""", encoding="utf-8")
print("main.py OK")

print()
print("ALL DAY 5 FILES CREATED")
print("Now run: uvicorn src.api.main:app --reload --port 8000")