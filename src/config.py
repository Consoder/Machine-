from pathlib import Path

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
