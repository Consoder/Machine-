import pathlib

pathlib.Path("app.py").write_text(
"""from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from src.config import MODEL_DIR, MEDICATION_FEATURES

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ─────────────────────
print("[Flask] Loading model artifacts...")
model     = joblib.load(MODEL_DIR / "xgboost_model.pkl")
explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
fe        = joblib.load(MODEL_DIR / "feature_engineer.pkl")
threshold = float(joblib.load(MODEL_DIR / "threshold.pkl"))
print(f"[Flask] Loaded. Threshold={threshold:.3f}")


def input_to_dataframe(data: dict) -> pd.DataFrame:
    med_map = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}

    # All 23 medication columns default to 0
    med_data = {med: 0 for med in MEDICATION_FEATURES}
    med_data["metformin"] = med_map.get(data.get("metformin", "No"), 0)
    med_data["insulin"]   = med_map.get(data.get("insulin", "No"), 0)

    insulin_val = med_data["insulin"]

    base_data = {
        "time_in_hospital":         int(data.get("time_in_hospital", 5)),
        "num_lab_procedures":       int(data.get("num_lab_procedures", 41)),
        "num_procedures":           int(data.get("num_procedures", 1)),
        "num_medications":          int(data.get("num_medications", 12)),
        "number_outpatient":        int(data.get("number_outpatient", 0)),
        "number_emergency":         int(data.get("number_emergency", 0)),
        "number_inpatient":         int(data.get("number_inpatient", 0)),
        "number_diagnoses":         int(data.get("number_diagnoses", 7)),
        "age":                      int(data.get("age", 65)),
        "gender":                   data.get("gender", "Female"),
        "race":                     data.get("race", "Caucasian"),
        "admission_type_id":        int(data.get("admission_type_id", 1)),
        "discharge_disposition_id": int(data.get("discharge_disposition_id", 1)),
        "admission_source_id":      int(data.get("admission_source_id", 7)),
        "medical_specialty":        data.get("medical_specialty", "InternalMedicine"),
        "max_glu_serum":            data.get("max_glu_serum", "None"),
        "A1Cresult":                data.get("A1Cresult", "None"),
        "change":                   data.get("change", "No"),
        "diabetesMed":              data.get("diabetesMed", "Yes"),
        "diag_1_group":             data.get("diag_1_group", "circulatory"),
        "diag_2_group":             data.get("diag_2_group", "other"),
        "diag_3_group":             data.get("diag_3_group", "other"),
        # Engineered features
        "total_meds_on":            int(data.get("num_medications", 12)),
        "total_meds_changed":       1 if data.get("change") == "Ch" else 0,
        "total_meds_up":            1 if data.get("insulin") == "Up" else 0,
        "insulin_changed":          1 if data.get("insulin") in ["Up", "Down"] else 0,
        "prior_visits_total":       int(data.get("number_outpatient", 0)) + int(data.get("number_emergency", 0)) + int(data.get("number_inpatient", 0)),
        "labs_per_day":             int(data.get("num_lab_procedures", 41)) / max(int(data.get("time_in_hospital", 5)), 1),
        "a1c_not_measured":         1 if data.get("A1Cresult") == "None" else 0,
    }

    merged = {**base_data, **med_data}
    df = pd.DataFrame([merged])
    X, _ = fe.transform(df.assign(target=0))
    return X


# ── Routes ────────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Hospital Readmission Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/batch-predict"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "threshold": threshold,
        "version": "1.0.0"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        X = input_to_dataframe(data)
        risk_score = float(model.predict_proba(X)[:, 1][0])
        prediction = int(risk_score >= threshold)

        if risk_score >= 0.6:
            risk_label = "HIGH"
        elif risk_score >= 0.3:
            risk_label = "MEDIUM"
        else:
            risk_label = "LOW"

        # SHAP explanation
        try:
            top_df = explainer.get_top_features(X, idx=0, top_n=10)
            top_features = [
                {
                    "feature":    row["feature"],
                    "shap_value": round(float(row["shap_value"]), 4),
                    "direction":  row["direction"],
                }
                for _, row in top_df.iterrows()
            ]
        except Exception as e:
            print(f"SHAP error: {e}")
            top_features = []

        return jsonify({
            "risk_score":    round(risk_score, 4),
            "risk_label":    risk_label,
            "risk_percent":  f"{risk_score * 100:.1f}%",
            "prediction":    prediction,
            "threshold":     round(threshold, 3),
            "top_features":  top_features,
            "model_version": "xgboost_v1"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    try:
        data = request.get_json()
        if not data or "patients" not in data:
            return jsonify({"error": "Send JSON with 'patients' list"}), 400

        patients = data["patients"]
        if len(patients) > 100:
            return jsonify({"error": "Max 100 patients per request"}), 400

        predictions = []
        for patient in patients:
            X = input_to_dataframe(patient)
            risk_score = float(model.predict_proba(X)[:, 1][0])
            prediction = int(risk_score >= threshold)
            risk_label = "HIGH" if risk_score >= 0.6 else "MEDIUM" if risk_score >= 0.3 else "LOW"
            predictions.append({
                "risk_score":   round(risk_score, 4),
                "risk_label":   risk_label,
                "risk_percent": f"{risk_score * 100:.1f}%",
                "prediction":   prediction,
            })

        high_risk = sum(1 for p in predictions if p["prediction"] == 1)
        return jsonify({
            "total":           len(predictions),
            "high_risk_count": high_risk,
            "predictions":     predictions,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
""", encoding="utf-8")
print("app.py OK")


# ── requirements for PythonAnywhere ───────────────
pathlib.Path("requirements_flask.txt").write_text(
"""flask==3.0.0
flask-cors==4.0.0
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
xgboost==2.0.0
shap==0.43.0
imbalanced-learn==0.11.0
joblib==1.3.2
pyarrow==13.0.0
""", encoding="utf-8")
print("requirements_flask.txt OK")

print()
print("ALL FLASK FILES CREATED")
print("Test locally: python app.py")