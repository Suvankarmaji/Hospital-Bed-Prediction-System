import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend  saves PNGs instead of popup windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               HistGradientBoostingClassifier,
                               VotingClassifier)
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              accuracy_score, precision_score,
                              f1_score, confusion_matrix, roc_curve)
from datetime import datetime
import os

#  Output folder for plots 
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# =============================
# HELPER: SAVE CONFUSION MATRIX
# =============================
def save_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Short Stay', 'Long Stay'],
                yticklabels=['Short Stay', 'Long Stay'])
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, f"cm_{title.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"   [PLOT] Confusion matrix saved -> {fname}")

# =============================
# HELPER: SAVE ROC CURVE
# =============================
def save_roc_curve(y_test, y_proba, title, auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve  {title}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, f"roc_{title.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"   [PLOT] ROC curve saved -> {fname}")

# =============================
# HELPER: SAVE COMPARISON BAR CHART
# =============================
def save_comparison_charts(results):
    models  = list(results.keys())
    metrics = ['ROC AUC', 'Accuracy', 'Precision', 'F1 Score']
    data    = {m: [results[mdl][m] for mdl in models] for m in metrics}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(' Final Model Comparison', fontsize=16)
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    for idx, metric in enumerate(metrics):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        bars = ax.bar(models, data[metric], color=colors[idx])
        ax.set_title(f'{metric} Comparison')
        lo = max(0.5, min(data[metric]) - 0.05)
        ax.set_ylim(lo, 1.0)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "model_comparison.png")
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"   [PLOT] Comparison chart saved -> {fname}")

# ==========================================
#   TRAINING & COMPARING ALL MODELS
# ==========================================
print("===========================================")
print("   TRAINING & COMPARING ALL MODELS         ")
print("===========================================")

print("1. Loading dataset...")
df = pd.read_csv("healthcare_dataset_comprehensive.csv")

# Create Target
df['Long_Stay'] = (df['Stay_Days'] > 7).astype(int)
X = df.drop(columns=['Stay_Days', 'Long_Stay', 'Patient_ID'])
y = df['Long_Stay']

print(f"   Data Shape: {df.shape}")
print(f"   Long Stay %: {y.mean()*100:.1f}%")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train Size: {X_train.shape[0]:,}, Test Size: {X_test.shape[0]:,}")

# Preprocessing
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# ---------------------------------------------------------
# DEFINE CONTESTANTS
# ---------------------------------------------------------
models = {
    "Logistic Regression": {
        "model":  LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=42),
        "params": {}
    },
    "Random Forest": {
        "model":  RandomForestClassifier(random_state=42,
                                          class_weight="balanced", n_jobs=-1),
        "params": {
            "classifier__n_estimators": [200],
            "classifier__max_depth":    [20],
            "classifier__min_samples_leaf": [5],
        }
    },
    "XGBoost": {
        "model":  XGBClassifier(eval_metric='logloss', random_state=42,
                                 n_jobs=-1),
        "params": {
            "classifier__n_estimators":  [300],
            "classifier__max_depth":     [8],
            "classifier__learning_rate": [0.05],
            "classifier__subsample":     [0.8],
            "classifier__colsample_bytree": [0.8],
        }
    },
    "HistGradientBoosting": {
        "model":  HistGradientBoostingClassifier(random_state=42,
                                                  early_stopping=True,
                                                  n_iter_no_change=10),
        "params": {
            "classifier__max_depth":     [15],
            "classifier__learning_rate": [0.05],
            "classifier__max_iter":      [500],
        }
    },
}

best_overall_model = None
best_overall_score = 0
best_overall_name  = ""
all_results        = {}

# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
for name, config in models.items():
    print(f"\n{'='*50}")
    print(f"  Training: {name}")
    print(f"{'='*50}")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier',   config["model"])
    ])

    if config["params"]:
        grid = GridSearchCV(pipeline, config["params"],
                            cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"   Best CV AUC: {grid.best_score_:.4f}")
    else:
        pipeline.fit(X_train, y_train)
        model = pipeline

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    auc  = roc_auc_score(y_test, y_proba)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n   [OK] Results for {name}:")
    print(f"   - ROC AUC:   {auc:.4f}")
    print(f"   - Accuracy:  {acc:.4f}")
    print(f"   - Precision: {prec:.4f}")
    print(f"   - F1 Score:  {f1:.4f}")

    all_results[name] = {
        'ROC AUC': auc, 'Accuracy': acc, 'Precision': prec, 'F1 Score': f1
    }

    save_confusion_matrix(y_test, y_pred, name)
    save_roc_curve(y_test, y_proba, name, auc)

    if auc > best_overall_score:
        best_overall_score = auc
        best_overall_model = model
        best_overall_name  = name

# ---------------------------------------------------------
# VOTING ENSEMBLE (Final Production Model)
# ---------------------------------------------------------
print(f"\n{'='*50}")
print("  Building Voting Ensemble (Top 3 Models)...")
print(f"{'='*50}")

hgb = HistGradientBoostingClassifier(max_depth=15, learning_rate=0.05,
                                      max_iter=500, random_state=42,
                                      early_stopping=True, n_iter_no_change=10)
rf  = RandomForestClassifier(n_estimators=200, max_depth=20,
                              min_samples_leaf=5, n_jobs=-1,
                              random_state=42, class_weight='balanced')
xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     eval_metric='logloss', random_state=42, n_jobs=-1)

ensemble = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   VotingClassifier(
        estimators=[('hgb', hgb), ('rf', rf), ('xgb', xgb)],
        voting='soft'
    ))
])

print("  Training Voting Ensemble...")
ensemble.fit(X_train, y_train)

y_proba_ens = ensemble.predict_proba(X_test)[:, 1]
y_pred_ens  = ensemble.predict(X_test)

auc_ens  = roc_auc_score(y_test, y_proba_ens)
acc_ens  = accuracy_score(y_test, y_pred_ens)
prec_ens = precision_score(y_test, y_pred_ens, zero_division=0)
f1_ens   = f1_score(y_test, y_pred_ens, zero_division=0)

print(f"\n   [OK] Voting Ensemble Results:")
print(f"   - ROC AUC:   {auc_ens:.4f}")
print(f"   - Accuracy:  {acc_ens:.4f}")
print(f"   - Precision: {prec_ens:.4f}")
print(f"   - F1 Score:  {f1_ens:.4f}")

all_results["Voting Ensemble"] = {
    'ROC AUC': auc_ens, 'Accuracy': acc_ens,
    'Precision': prec_ens, 'F1 Score': f1_ens
}

save_confusion_matrix(y_test, y_pred_ens, "Voting_Ensemble")
save_roc_curve(y_test, y_proba_ens, "Voting_Ensemble", auc_ens)
save_comparison_charts(all_results)

# ---------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------
print("\n" + "="*60)
print("  FINAL LEADERBOARD")
print("="*60)
print(f"{'Model':<26} {'ROC AUC':>9} {'Accuracy':>9} {'Precision':>10} {'F1':>8}")
print("-"*60)
for mdl, mets in all_results.items():
    print(f"{mdl:<26} {mets['ROC AUC']:>9.4f} {mets['Accuracy']:>9.4f} "
          f"{mets['Precision']:>10.4f} {mets['F1 Score']:>8.4f}")
print("="*60)

# Always save the Voting Ensemble as production model
best_overall_score = auc_ens
best_overall_model = ensemble
best_overall_name  = "Voting Ensemble (Top 3 Combined)"

print(f"\n*** Production Model: {best_overall_name}")
print(f"   Final: AUC={auc_ens:.4f}, Acc={acc_ens:.4f}, F1={f1_ens:.4f}")

model_data = {
    "model": best_overall_model,
    "metadata": {
        "auc_score":     auc_ens,
        "accuracy":      acc_ens,
        "precision":     prec_ens,
        "f1_score":      f1_ens,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type":    best_overall_name,
        "feature_stats": X_train.describe().to_dict()
    }
}
joblib.dump(model_data, "best_hospital_stay_model_comprehensive.pkl")
print("[OK] Model saved -> best_hospital_stay_model_comprehensive.pkl")
print(f"[OK] All plots  -> {PLOT_DIR}/")
