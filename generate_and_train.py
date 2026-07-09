"""
Generate synthetic medical dataset and train ML models.
Saves trained models and encoders for use by Flask app.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
import pickle
import json
import os

np.random.seed(42)

# ─── Disease → Symptom definitions ───────────────────────────────────────────
DISEASE_SYMPTOMS = {
    "Flu": {
        "core": ["fever", "cough", "fatigue", "body_aches", "headache"],
        "secondary": ["sore_throat", "runny_nose", "chills", "sweating", "loss_of_appetite"],
    },
    "Common Cold": {
        "core": ["runny_nose", "sore_throat", "sneezing", "cough"],
        "secondary": ["mild_fever", "fatigue", "headache", "nasal_congestion", "watery_eyes"],
    },
    "Pneumonia": {
        "core": ["high_fever", "chest_pain", "difficulty_breathing", "cough", "fatigue"],
        "secondary": ["chills", "sweating", "rapid_breathing", "confusion", "bluish_lips"],
    },
    "Diabetes": {
        "core": ["frequent_urination", "excessive_thirst", "unexplained_weight_loss", "fatigue"],
        "secondary": ["blurred_vision", "slow_healing", "tingling_hands_feet", "frequent_infections", "headache"],
    },
    "Hypertension": {
        "core": ["headache", "dizziness", "chest_pain", "shortness_of_breath"],
        "secondary": ["nausea", "blurred_vision", "palpitations", "fatigue", "nosebleeds"],
    },
    "Malaria": {
        "core": ["high_fever", "chills", "sweating", "headache", "body_aches"],
        "secondary": ["nausea", "vomiting", "fatigue", "diarrhea", "jaundice"],
    },
    "Dengue": {
        "core": ["high_fever", "severe_headache", "joint_pain", "rash", "eye_pain"],
        "secondary": ["nausea", "vomiting", "fatigue", "bleeding_gums", "bruising"],
    },
    "Typhoid": {
        "core": ["prolonged_fever", "abdominal_pain", "weakness", "headache"],
        "secondary": ["loss_of_appetite", "constipation", "diarrhea", "rash", "sweating"],
    },
    "Asthma": {
        "core": ["wheezing", "shortness_of_breath", "chest_tightness", "cough"],
        "secondary": ["difficulty_sleeping", "fatigue", "rapid_breathing", "anxiety", "sweating"],
    },
    "Migraine": {
        "core": ["severe_headache", "nausea", "sensitivity_to_light", "sensitivity_to_sound"],
        "secondary": ["vomiting", "blurred_vision", "dizziness", "fatigue", "aura"],
    },
    "Gastroenteritis": {
        "core": ["nausea", "vomiting", "diarrhea", "abdominal_cramps"],
        "secondary": ["fever", "headache", "fatigue", "loss_of_appetite", "dehydration"],
    },
    "Anemia": {
        "core": ["fatigue", "weakness", "pale_skin", "dizziness", "shortness_of_breath"],
        "secondary": ["headache", "cold_hands_feet", "chest_pain", "irregular_heartbeat", "brittle_nails"],
    },
}

ALL_SYMPTOMS = sorted(set(
    s for d in DISEASE_SYMPTOMS.values()
    for lst in d.values()
    for s in lst
))

print(f"Total unique symptoms: {len(ALL_SYMPTOMS)}")
print(f"Diseases: {list(DISEASE_SYMPTOMS.keys())}")


def generate_record(disease, noise=0.08):
    info = DISEASE_SYMPTOMS[disease]
    row = {s: 0 for s in ALL_SYMPTOMS}
    # Core symptoms almost always present
    for s in info["core"]:
        row[s] = 1 if np.random.random() > 0.10 else 0
    # Secondary symptoms 40-70 % present
    for s in info["secondary"]:
        row[s] = 1 if np.random.random() > 0.45 else 0
    # Random noise symptoms
    for s in ALL_SYMPTOMS:
        if row[s] == 0 and np.random.random() < noise:
            row[s] = 1
    row["disease"] = disease
    return row


# Generate 5 000 records (≈ 416 per disease)
records = []
per_disease = 5000 // len(DISEASE_SYMPTOMS)
for disease in DISEASE_SYMPTOMS:
    for _ in range(per_disease):
        records.append(generate_record(disease))
# top up to exactly 5 000
for i in range(5000 - len(records)):
    records.append(generate_record(list(DISEASE_SYMPTOMS.keys())[i % len(DISEASE_SYMPTOMS)]))

df = shuffle(pd.DataFrame(records), random_state=42).reset_index(drop=True)
df.to_csv("/home/claude/disease_prediction/data/medical_dataset.csv", index=False)
print(f"Dataset saved: {df.shape}")

# ─── Preprocessing ────────────────────────────────────────────────────────────
X = df[ALL_SYMPTOMS]
y = df["disease"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ─── Random Forest ────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_preds, target_names=le.classes_))

# ─── Logistic Regression ──────────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, C=1.5)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# ─── Save artefacts ───────────────────────────────────────────────────────────
os.makedirs("/home/claude/disease_prediction/models", exist_ok=True)

with open("/home/claude/disease_prediction/models/random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("/home/claude/disease_prediction/models/logistic_regression.pkl", "wb") as f:
    pickle.dump(lr, f)
with open("/home/claude/disease_prediction/models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

metadata = {
    "symptoms": ALL_SYMPTOMS,
    "diseases": list(le.classes_),
    "rf_accuracy": round(rf_acc, 4),
    "lr_accuracy": round(lr_acc, 4),
    "dataset_size": len(df),
    "disease_symptoms": DISEASE_SYMPTOMS,
}
with open("/home/claude/disease_prediction/models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nAll models and metadata saved.")
print(f"RF Accuracy: {rf_acc:.1%}  |  LR Accuracy: {lr_acc:.1%}")
