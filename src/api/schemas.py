from pydantic import BaseModel, Field
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
