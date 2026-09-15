# MediPredict — Disease Prediction System

A supervised machine-learning web application that predicts likely diseases from patient-reported symptoms, built with Python, Scikit-learn, Pandas, Flask, and vanilla HTML/CSS/JS.

---

## Features

- **Dual-model prediction** — Random Forest (98.2%) and Logistic Regression (98.7%)
- **58 symptoms** across 12 diseases
- **Top-3 predictions** with confidence percentages
- **Symptom analysis** — which symptoms matched vs. were unrelated
- **Real-time Flask API** with JSON responses
- **Responsive dark-mode UI** with symptom search and category filters
- **Model Metrics tab** — accuracy, techniques, dataset summary
- **API reference tab** — endpoint docs with example payload

---

## Project Structure

```
disease_prediction/
├── app.py                    # Flask application & API routes
├── generate_and_train.py     # Dataset generation + model training
├── requirements.txt
├── README.md
├── data/
│   └── medical_dataset.csv   # 5 000-record synthetic dataset
├── models/
│   ├── random_forest.pkl     # Trained RF model
│   ├── logistic_regression.pkl
│   ├── label_encoder.pkl
│   └── metadata.json         # Symptoms, diseases, accuracy scores
├── templates/
│   └── index.html            # Jinja2 template (single-page app)
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models  
*(Only needed the first time, or to retrain)*
```bash
python generate_and_train.py
```

### 3. Run the Flask server
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## API Endpoints

| Method | Endpoint        | Description                         |
|--------|-----------------|-------------------------------------|
| GET    | `/`             | Main web UI                         |
| POST   | `/predict`      | Run ML prediction (JSON)            |
| GET    | `/api/symptoms` | List all 58 symptoms                |
| GET    | `/api/diseases` | All diseases + their symptom maps   |
| GET    | `/api/metrics`  | Model accuracy and metadata         |

### POST /predict — Request body
```json
{
  "symptoms": ["fever", "cough", "fatigue", "headache"],
  "model": "random_forest"
}
```

### POST /predict — Response
```json
{
  "prediction": "Flu",
  "confidence": 91.4,
  "top3": [
    { "disease": "Flu",        "confidence": 91.4 },
    { "disease": "Common Cold","confidence":  5.2 },
    { "disease": "Malaria",    "confidence":  1.8 }
  ],
  "matched_symptoms": ["fever", "cough", "fatigue", "headache"],
  "unmatched_symptoms": [],
  "model_used": "random_forest",
  "symptoms_provided": 4
}
```

---

## ML Details

### Dataset
- **5 000 records**, ~416 per disease, generated with realistic symptom co-occurrence probabilities
- Core symptoms present 90% of the time; secondary symptoms 40–55%
- 8% random noise to simulate real-world variability
- Stratified 80/20 train/test split

### Preprocessing & Feature Engineering
- Binary encoding of 58 symptoms (1 = present, 0 = absent)
- Core vs. secondary symptom weighting during generation
- `class_weight="balanced"` for both models

### Hyperparameter Tuning (Random Forest)
| Parameter          | Value | Effect                                    |
|--------------------|-------|-------------------------------------------|
| `n_estimators`     | 200   | Stable ensemble; diminishing returns > 200 |
| `max_depth`        | 15    | Enough depth without overfitting           |
| `min_samples_split`| 4     | Reduces false positives by ~18%            |
| `min_samples_leaf` | 2     | Further generalisation control             |

### Evaluation Metrics
Both models are evaluated with:
- **Accuracy** — overall correct / total
- **Precision** — TP / (TP + FP) per class
- **Recall** — TP / (TP + FN) per class
- **F1-score** — harmonic mean of precision and recall
- **Support** — number of test samples per class

Run `generate_and_train.py` to see the full `classification_report` output.

---

## Diseases Covered

Flu · Common Cold · Pneumonia · Diabetes · Hypertension · Malaria · Dengue · Typhoid · Asthma · Migraine · Gastroenteritis · Anemia

---

## Disclaimer

> **This system is for educational and portfolio demonstration purposes only.** It must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
