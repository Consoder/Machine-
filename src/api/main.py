import time
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
    """Check if API and model are running."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Predict 30-day readmission risk for a single patient.
    Returns risk score, label, and top SHAP features explaining the prediction.
    """
    try:
        result = predictor.predict_single(patient)
        logger.info(f"Prediction: risk={result.risk_score} label={result.risk_label}")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(patients: List[PatientInput]):
    """
    Predict readmission risk for multiple patients at once.
    Maximum 100 patients per request.
    """
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
