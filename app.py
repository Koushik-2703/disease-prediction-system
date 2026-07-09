"""
Flask web application for Disease Prediction System.
Endpoints:
  GET  /              → main UI
  POST /predict       → run prediction, return JSON
  GET  /api/symptoms  → list all symptoms
  GET  /api/diseases  → list all diseases + their symptoms
  GET  /api/metrics   → model evaluation metrics
"""
from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models")

# ─── Load artefacts ────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
    RF_MODEL = pickle.load(f)

with open(os.path.join(MODEL_DIR, "logistic_regression.pkl"), "rb") as f:
    LR_MODEL = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    LABEL_ENC = pickle.load(f)

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    META = json.load(f)

SYMPTOMS = META["symptoms"]
DISEASES = META["diseases"]
DISEASE_SYMPTOMS = META["disease_symptoms"]


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", symptoms=SYMPTOMS, diseases=DISEASES)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    selected = data.get("symptoms", [])
    model_choice = data.get("model", "random_forest")

    if not selected:
        return jsonify({"error": "No symptoms provided"}), 400

    # Build feature vector
    feat = np.array([[1 if s in selected else 0 for s in SYMPTOMS]])

    model = RF_MODEL if model_choice == "random_forest" else LR_MODEL

    pred_idx = model.predict(feat)[0]
    pred_disease = LABEL_ENC.inverse_transform([pred_idx])[0]
    probas = model.predict_proba(feat)[0]

    # Top-3 predictions
    top3_idx = np.argsort(probas)[::-1][:3]
    top3 = [
        {
            "disease": LABEL_ENC.inverse_transform([i])[0],
            "confidence": round(float(probas[i]) * 100, 1),
        }
        for i in top3_idx
    ]

    # Matched symptoms for predicted disease
    dis_syms = (
        DISEASE_SYMPTOMS.get(pred_disease, {}).get("core", [])
        + DISEASE_SYMPTOMS.get(pred_disease, {}).get("secondary", [])
    )
    matched = [s for s in selected if s in dis_syms]
    unmatched = [s for s in selected if s not in dis_syms]

    return jsonify(
        {
            "prediction": pred_disease,
            "confidence": round(float(probas[pred_idx]) * 100, 1),
            "top3": top3,
            "matched_symptoms": matched,
            "unmatched_symptoms": unmatched,
            "model_used": model_choice,
            "symptoms_provided": len(selected),
        }
    )


@app.route("/api/symptoms")
def api_symptoms():
    return jsonify({"symptoms": SYMPTOMS, "count": len(SYMPTOMS)})


@app.route("/api/diseases")
def api_diseases():
    return jsonify({"diseases": DISEASES, "disease_symptoms": DISEASE_SYMPTOMS})


@app.route("/api/metrics")
def api_metrics():
    return jsonify(
        {
            "random_forest": {
                "accuracy": META["rf_accuracy"],
                "description": "Ensemble of 200 decision trees with balanced class weights",
            },
            "logistic_regression": {
                "accuracy": META["lr_accuracy"],
                "description": "Multinomial logistic regression, C=1.5, balanced weights",
            },
            "dataset_size": META["dataset_size"],
            "num_symptoms": len(SYMPTOMS),
            "num_diseases": len(DISEASES),
        }
    )


if __name__ == "__main__":
    print("\n Disease Prediction System")
    print("=" * 40)
    print(f"  Symptoms : {len(SYMPTOMS)}")
    print(f"  Diseases : {len(DISEASES)}")
    print(f"  RF Acc   : {META['rf_accuracy']:.1%}")
    print(f"  LR Acc   : {META['lr_accuracy']:.1%}")
    print("=" * 40)
    print("  Running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
