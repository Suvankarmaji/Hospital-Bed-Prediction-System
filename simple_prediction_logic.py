import pandas as pd
import joblib
import numpy as np
import sys

# 1. Load the Model (The Brain)
# ------------------------------------------
try:
    print("1. Loading AI Model...")
    data = joblib.load("best_hospital_stay_model_comprehensive.pkl")
    model = data["model"]
    print("   [OK] Model Loaded Successfully!")
    print(f"   INFO: Model Type: {data['metadata'].get('model_type', 'Ensemble')}")
    print(f"   INFO: Trained AUC: {data['metadata'].get('auc_score', 0):.4f}")
except Exception as e:
    print(f"   [ERROR] Model not found: {e}")
    print("   Please run 'train_comprehensive_model.py' first.")
    sys.exit(1)

# 2. Define the Prediction Function
# ------------------------------------------
def ai_predict_stay(patient_data):
    """
    Takes raw patient data, processes it, and calculates the risk factor.
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame([patient_data])
    
    # Get Probability from Model (0.0 to 1.0) for Class 1 (Long Stay)
    probability = model.predict_proba(df)[0][1] 
    
    # Threshold Logic
    is_long_stay = probability > 0.5
    
    return is_long_stay, probability

# 3. Simulation: Admit Multiple Patients (Batch Process)
# ------------------------------------------
print("\n2. Simulating New Batched Admissions...")

patients = [
    {
        "Name": "Rahul (Elderly, High Complexity)",
        "Data": {
            "Age": 78, "Gender": "Male", "Admission_Type": "Emergency", "Insurance_Type": "Medicare",
            "Num_Comorbidities": 4, "Prev_Hospitalizations": 3, "Procedure_Complexity": 5, 
            "Lab_Abnormality_Score": 4, "Visitors_Count": 1, "Blood_Sugar_Level": 180, 
            "Admission_Deposit": 4000, "Department": "Cardiology", "Diagnosis": "Heart Failure", 
            "Severity_Score": 5, "Ward_Type": "ICU"
        }
    },
    {
        "Name": "Lola (Young, Routine)",
        "Data": {
            "Age": 24, "Gender": "Female", "Admission_Type": "Elective", "Insurance_Type": "Private",
            "Num_Comorbidities": 0, "Prev_Hospitalizations": 0, "Procedure_Complexity": 1, 
            "Lab_Abnormality_Score": 0, "Visitors_Count": 4, "Blood_Sugar_Level": 90, 
            "Admission_Deposit": 8000, "Department": "Gynecology", "Diagnosis": "Normal Delivery", 
            "Severity_Score": 1, "Ward_Type": "Private"
        }
    },
    {
        "Name": "Dinesh (Middle Age, Moderate)",
        "Data": {
            "Age": 45, "Gender": "Male", "Admission_Type": "Urgent", "Insurance_Type": "Private",
            "Num_Comorbidities": 1, "Prev_Hospitalizations": 1, "Procedure_Complexity": 3, 
            "Lab_Abnormality_Score": 2, "Visitors_Count": 2, "Blood_Sugar_Level": 110, 
            "Admission_Deposit": 5000, "Department": "Orthopedics", "Diagnosis": "Knee Replacement", 
            "Severity_Score": 2, "Ward_Type": "General"
        }
    }
]

for p in patients:
    print(f"\n" + "="*56)
    print(f" CASE STUDY: {p['Name']}")
    print("="*56)
    
    # Show Patient Details
    print("   PATIENT PROFILE:")
    data = p['Data']
    print(f"      - Age: {data['Age']} yrs | Gender: {data['Gender']}")
    print(f"      - Condition: {data['Diagnosis']} ({data['Department']})")
    print(f"      - Severity: {data['Severity_Score']}/5 | Complexity: {data['Procedure_Complexity']}/5")
    print(f"      - Comorbidities: {data['Num_Comorbidities']} | Prev Admits: {data['Prev_Hospitalizations']}")
    
    # 4. Run the Prediction
    is_long, prob = ai_predict_stay(data)

    # 5. Show Results
    print(f"\n   AI ANALYSIS:")
    print(f"      - Risk Probability: {prob*100:.2f}%")

    if is_long:
        print("      - RESULT: LONG STAY PREDICTION (>7 Days)")
    else:
        print("      - RESULT: SHORT STAY PREDICTION (<=7 Days)")
        
    # 6. Explanation Logic (Simple Rules-based for Demo)
    print(f"\n   KEY DRIVERS (Top factors detected):")
    drivers = []
    
    # Long Stay Drivers
    if data['Severity_Score'] >= 4: drivers.append("Critical Severity (Score 4-5)")
    if data['Procedure_Complexity'] >= 4: drivers.append("High Procedure Complexity")
    if data['Lab_Abnormality_Score'] >= 3: drivers.append("Abnormal Lab Indicators")
    if data['Age'] > 70: drivers.append("Advanced Age (>70)")
    if data['Num_Comorbidities'] >= 3: drivers.append("Multiple Comorbidities")
    if data['Ward_Type'] == "ICU": drivers.append("ICU Admission")
    if data['Admission_Type'] == "Emergency": drivers.append("Emergency Admission")
         
    # Short Stay Drivers
    if data['Severity_Score'] <= 2: drivers.append("Low Severity (Score 1-2)")
    if data['Procedure_Complexity'] <= 2: drivers.append("Low Procedure Complexity")
    if data['Age'] < 40: drivers.append("Young Patient (<40)")
    if data['Admission_Type'] == "Elective": drivers.append("Elective Procedure")
    if data['Num_Comorbidities'] == 0: drivers.append("No Comorbidities")

    # Print relevant drivers
    displayed = 0
    for driver in drivers:
        if is_long:
            if any(key in driver for key in ["Critical", "High", "Abnormal", "Advanced", "Multiple", "ICU", "Emergency"]):
                print(f"      - {driver}")
                displayed += 1
        else:
            if any(key in driver for key in ["Low", "Young", "Elective", "No"]):
                print(f"      - {driver}")
                displayed += 1
                
    if displayed == 0:
        print("      - Complex interactions of multiple minor clinical factors.")

    print("\n   RECOMMENDED ACTION:")
    if is_long:
        print("      -> Reserve Bed for 7+ days")
        print("      -> Schedule Specialist Review & Discharge Planning")
    else:
        print("      -> standard Stay Pathway (Estimated Discharge < 7 days)")
    print("-" * 56)
