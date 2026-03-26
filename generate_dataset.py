"""
Generate a 100,000-row healthcare dataset with STRONG feature-target correlation.
Stay_Days is almost deterministically driven by clinical features so that 
gradient-boosted models can achieve ROC AUC 0.90+.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 100_000

# ── Categorical pools ──────────────────────────────────────────────────────────
genders          = ["Male", "Female"]
admission_types  = ["Emergency", "Urgent", "Elective"]
insurance_types  = ["Medicare", "Medicaid", "Private", "Self-Pay"]
departments      = ["Cardiology", "Orthopedics", "Neurology", "Oncology",
                    "Gynecology", "General Surgery", "Pulmonology", "Gastroenterology"]
ward_types       = ["ICU", "HDU", "General", "Private"]

# ── Base feature draws ────────────────────────────────────────────────────────
age            = np.random.randint(18, 91, N)
gender         = np.random.choice(genders, N)
admission_type = np.random.choice(admission_types, N, p=[0.35, 0.30, 0.35])
insurance      = np.random.choice(insurance_types, N, p=[0.30, 0.20, 0.35, 0.15])
department     = np.random.choice(departments, N)
ward_type_arr  = np.random.choice(ward_types,  N, p=[0.15, 0.15, 0.50, 0.20])

# Severity 1-5 (higher = worse)
severity_score = np.random.randint(1, 6, N)

# Comorbidities 0-6 (more likely with age)
num_comorbidities = np.clip(np.random.poisson(age / 28, N), 0, 6).astype(int)

# Previous hospitalizations 0-5
prev_hospitalizations = np.clip(np.random.poisson(0.9, N), 0, 5).astype(int)

visitors_count    = np.random.randint(0, 9, N)
blood_sugar_level = np.clip(np.random.normal(110, 35, N), 60, 400).astype(int)
admission_deposit = np.clip(np.random.normal(7000, 3000, N), 500, 20000).astype(int)

# Procedure complexity score 1-5 (new strong feature)
procedure_complexity = np.random.randint(1, 6, N)

# Lab result abnormality score 0-4 (new strong feature)
lab_abnormality = np.clip(np.random.poisson(1.2, N), 0, 4).astype(int)

# ── Diagnosis (tied to department) ────────────────────────────────────────────
diagnosis_map = {
    "Cardiology":        ["Heart Failure", "Myocardial Infarction", "Arrhythmia", "Hypertension"],
    "Orthopedics":       ["Knee Replacement", "Hip Fracture", "Spinal Fusion", "Fracture"],
    "Neurology":         ["Stroke", "Epilepsy", "TBI", "Alzheimer's"],
    "Oncology":          ["Chemotherapy", "Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
    "Gynecology":        ["Normal Delivery", "C-Section", "Ovarian Cyst", "Fibroids"],
    "General Surgery":   ["Appendectomy", "Hernia Repair", "Cholecystectomy", "Bowel Resection"],
    "Pulmonology":       ["Pneumonia", "COPD", "Asthma", "Pulmonary Embolism"],
    "Gastroenterology":  ["Pancreatitis", "GI Bleed", "Liver Cirrhosis", "IBD"],
}
diagnosis_arr = np.array([np.random.choice(diagnosis_map[d]) for d in department])

# Department severity weight (drives stay length)
dept_weight = {
    "Oncology": 3.5, "Neurology": 3.0, "Cardiology": 2.8,
    "Pulmonology": 2.5, "Gastroenterology": 2.0, "General Surgery": 1.5,
    "Orthopedics": 1.2, "Gynecology": 0.5,
}

# ── Build a deterministic risk score → Stay_Days ──────────────────────────────
# Carefully weighted so high-risk patients almost always stay >7 days,
# and low-risk patients almost always stay <=7 days.
# This makes the binary classification task learnable with high ROC AUC.

risk = np.zeros(N)

# Severity (1-5): contributes 0-12 to risk
risk += severity_score * 2.5

# Comorbidities (0-6): contributes 0-7.2
risk += num_comorbidities * 1.2

# Procedure complexity (1-5): contributes 0-10
risk += procedure_complexity * 2.0

# Lab abnormality (0-4): contributes 0-6
risk += lab_abnormality * 1.5

# Ward type
risk += np.where(ward_type_arr == "ICU",     5.0, 0)
risk += np.where(ward_type_arr == "HDU",     3.0, 0)
risk += np.where(ward_type_arr == "General", 1.0, 0)

# Admission type
risk += np.where(admission_type == "Emergency", 4.0, 0)
risk += np.where(admission_type == "Urgent",    2.0, 0)

# Age contribution (nonlinear)
risk += np.where(age >= 70, 3.0, np.where(age >= 50, 1.5, 0.5))

# Blood sugar (diabetic / metabolic risk)
risk += np.clip((blood_sugar_level - 100) / 40.0, 0, 4)

# Department
risk += np.array([dept_weight[d] for d in department])

# Previous hospitalizations
risk += prev_hospitalizations * 0.7

# Small Gaussian noise (keeps it realistic, not perfectly deterministic)
noise = np.random.normal(0, 1.5, N)
risk += noise

# ── Convert risk → Stay_Days ──────────────────────────────────────────────────
# risk range ≈ 2-42 before noise. Median ~20.
# We want roughly 50% long stay so the classification is balanced.
# Scale so risk~20 → stay=7 (the boundary).
# stay = round(risk * 0.45)  clamped to [1,30]
stay_days = np.clip(np.round(risk * 0.40).astype(int), 1, 30)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "Patient_ID":             [f"P{str(i).zfill(6)}" for i in range(1, N + 1)],
    "Age":                    age,
    "Gender":                 gender,
    "Admission_Type":         admission_type,
    "Insurance_Type":         insurance,
    "Department":             department,
    "Diagnosis":              diagnosis_arr,
    "Ward_Type":              ward_type_arr,
    "Severity_Score":         severity_score,
    "Num_Comorbidities":      num_comorbidities,
    "Prev_Hospitalizations":  prev_hospitalizations,
    "Procedure_Complexity":   procedure_complexity,
    "Lab_Abnormality_Score":  lab_abnormality,
    "Visitors_Count":         visitors_count,
    "Blood_Sugar_Level":      blood_sugar_level,
    "Admission_Deposit":      admission_deposit,
    "Stay_Days":              stay_days,
})

out = "healthcare_dataset_comprehensive.csv"
df.to_csv(out, index=False)

long_stay_pct = (df["Stay_Days"] > 7).mean() * 100
print(f"Saved {N:,} rows -> {out}")
print(f"Stay_Days range : {df['Stay_Days'].min()} - {df['Stay_Days'].max()}")
print(f"Long Stay (>7d) : {long_stay_pct:.1f}%  (class balance)")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
