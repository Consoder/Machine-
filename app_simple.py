from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '/home/Kartik095/Machine-')

app = Flask(__name__)
CORS(app)

BASE = Path('/home/Kartik095/Machine-')
model     = joblib.load(BASE / 'models/artifacts/xgboost_model.pkl')
fe        = joblib.load(BASE / 'models/artifacts/feature_engineer.pkl')
threshold = float(joblib.load(BASE / 'models/artifacts/threshold.pkl'))

MEDS = [
    "metformin","repaglinide","nateglinide","chlorpropamide",
    "glimepiride","acetohexamide","glipizide","glyburide",
    "tolbutamide","pioglitazone","rosiglitazone","acarbose",
    "miglitol","troglitazone","tolazamide","examide",
    "citoglipton","insulin","glyburide-metformin",
    "glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone",
]

def to_df(d):
    mm = {"No":0,"Steady":1,"Down":2,"Up":3}
    md = {m:0 for m in MEDS}
    md["metformin"] = mm.get(d.get("metformin","No"),0)
    md["insulin"]   = mm.get(d.get("insulin","No"),0)
    b = {
        "time_in_hospital":         int(d.get("time_in_hospital",5)),
        "num_lab_procedures":       int(d.get("num_lab_procedures",41)),
        "num_procedures":           int(d.get("num_procedures",1)),
        "num_medications":          int(d.get("num_medications",12)),
        "number_outpatient":        int(d.get("number_outpatient",0)),
        "number_emergency":         int(d.get("number_emergency",0)),
        "number_inpatient":         int(d.get("number_inpatient",0)),
        "number_diagnoses":         int(d.get("number_diagnoses",7)),
        "age":                      int(d.get("age",65)),
        "gender":                   d.get("gender","Female"),
        "race":                     d.get("race","Caucasian"),
        "admission_type_id":        int(d.get("admission_type_id",1)),
        "discharge_disposition_id": int(d.get("discharge_disposition_id",1)),
        "admission_source_id":      int(d.get("admission_source_id",7)),
        "medical_specialty":        d.get("medical_specialty","InternalMedicine"),
        "max_glu_serum":            d.get("max_glu_serum","None"),
        "A1Cresult":                d.get("A1Cresult","None"),
        "change":                   d.get("change","No"),
        "diabetesMed":              d.get("diabetesMed","Yes"),
        "diag_1_group":             d.get("diag_1_group","circulatory"),
        "diag_2_group":             d.get("diag_2_group","other"),
        "diag_3_group":             d.get("diag_3_group","other"),
        "total_meds_on":            int(d.get("num_medications",12)),
        "total_meds_changed":       1 if d.get("change")=="Ch" else 0,
        "total_meds_up":            1 if d.get("insulin")=="Up" else 0,
        "insulin_changed":          1 if d.get("insulin") in ["Up","Down"] else 0,
        "prior_visits_total":       int(d.get("number_outpatient",0))+int(d.get("number_emergency",0))+int(d.get("number_inpatient",0)),
        "labs_per_day":             int(d.get("num_lab_procedures",41))/max(int(d.get("time_in_hospital",5)),1),
        "a1c_not_measured":         1 if d.get("A1Cresult")=="None" else 0,
    }
    df = pd.DataFrame([{**b,**md}])
    X,_ = fe.transform(df.assign(target=0))
    return X

@app.route("/")
def root():
    return jsonify({"message":"Hospital Readmission API","status":"live"})

@app.route("/health")
def health():
    return jsonify({"status":"healthy","model_loaded":True,"threshold":threshold})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = to_df(data)
        score = float(model.predict_proba(X)[:,1][0])
        pred  = int(score >= threshold)
        label = "HIGH" if score>=0.6 else "MEDIUM" if score>=0.3 else "LOW"
        return jsonify({
            "risk_score":   round(score,4),
            "risk_label":   label,
            "risk_percent": f"{score*100:.1f}%",
            "prediction":   pred,
            "threshold":    round(threshold,3),
            "top_features": [],
            "model_version":"xgboost_v1"
        })
    except Exception as e:
        return jsonify({"error":str(e)}),500

if __name__ == "__main__":
    app.run(debug=True)