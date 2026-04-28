# 🎓 Student Performance Prediction System

> **Educational Data Mining · Fairness Audit · Ensemble ML · SHAP Explainability**  
> Early identification of at-risk students before end of term, with a teacher-facing Streamlit dashboard.

---

## 📌 Project Overview

This system predicts student **Pass/Fail** outcomes using behavioural and demographic data collected within the first 60–70% of a term. The goal is to flag at-risk students **early enough** for teachers to intervene — before it is too late.

| Component | Details |
|---|---|
| **Dataset** | `student_habits_performance.csv` — 1,000 students, 15 raw features |
| **Models** | Random Forest · XGBoost (GridSearchCV optimised) |
| **Threshold** | 0.35 (tuned to minimise false negatives / missed at-risk students) |
| **Explainability** | SHAP TreeExplainer — waterfall plots per student |
| **Fairness** | Demographic Parity audit across gender & parental education |
| **UI** | Streamlit dashboard — single student + batch prediction + fairness page |

---

## 🗂️ Repository Structure

```
student-prediction-system/
│
├── Final_Project.ipynb           # Full ML pipeline (run this first)
├── main.py                       # Streamlit dashboard (UI only)
├── student_habits_performance.csv# Source dataset
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes generated artefacts & caches
├── LICENSE                       # MIT Licence
└── README.md                     # This file
```

> **Note:** The following files are **generated** by running the notebook and are **not** committed to Git:
> `best_model.pkl` · `model_features.pkl` · `decision_threshold.pkl` · `preprocessing_artifacts.pkl`

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/student-prediction-system.git
cd student-prediction-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1 — Train the model (run the notebook)

Open and execute **all cells** in `Final_Project.ipynb`.  
This will produce four artefact files in the project root:

```
best_model.pkl
model_features.pkl
decision_threshold.pkl
preprocessing_artifacts.pkl
```

> ⚠️ You **must** run the notebook before launching the dashboard. The Streamlit app does not contain any training logic — it only loads the saved artefacts.

### Step 2 — Launch the dashboard

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`.

---

## 📊 ML Pipeline (Notebook)

The notebook executes a full, reproducible pipeline:

1. **Exploratory Data Analysis** — distributions, correlations, class balance
2. **Preprocessing** — missing value imputation, ordinal encoding, selective winsorisation
3. **Fairness Audit** — Demographic Parity Difference across gender & parental education
4. **Feature Engineering** — 5 derived features added:
   - `study_x_attendance` — interaction term
   - `distraction_hours` — social media + Netflix
   - `wellbeing_index` — composite of sleep, mental health, diet
   - `productive_ratio` — study vs distraction balance
   - `at_risk_flag` — binary rule-based flag
5. **Class Imbalance Handling** — bootstrap oversampling of minority class
6. **Model Training** — Random Forest and XGBoost with `GridSearchCV` + Stratified K-Fold
7. **Threshold Tuning** — decision threshold set to **0.35** (not 0.50) to prioritise recall on Fail class
8. **Evaluation** — classification report, ROC-AUC, confusion matrix, precision-recall curve
9. **SHAP Explainability** — global feature importance + per-student waterfall plots
10. **Artefact Export** — saves model and preprocessing objects as `.pkl` files

---

## 🖥️ Dashboard Features

### 🔍 Predict a Student
- 14-feature input via sliders and dropdowns
- Real-time Pass/Fail prediction with fail probability
- SHAP waterfall chart (if `shap` is installed)
- Top-3 teacher recommendations keyed to the highest-impact features

### 📋 Batch Prediction
- Upload any CSV matching the dataset schema
- Displays top at-risk students sorted by fail probability
- Fail probability distribution histogram
- Download predictions as CSV

### ⚖️ Fairness Audit
- Demographic Parity analysis across gender and parental education level
- Disparate Impact score per group (pass if ≥ 0.80 — four-fifths rule)
- Colour-coded bar charts highlighting unfair groups

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Threshold = 0.35** | Cost asymmetry: missing a failing student is worse than a false alarm |
| **Skip winsorisation on attendance & study hours** | Clamping these destroyed the signal for zero-study / zero-attendance edge cases |
| **Business-rule overrides** | Force fail ≥ 95% when study hours = 0 or attendance < 30% + study < 1h — training data was too sparse to learn these cases |
| **No training logic in `main.py`** | Separation of concerns: dashboard is UI-only, fast to load, safe to deploy |
| **Rule-based recommendations fallback** | Works without SHAP installed — graceful degradation |

---

## 📦 Dependencies

See `requirements.txt` for pinned versions. Core libraries:

- `streamlit` — dashboard UI
- `scikit-learn` — preprocessing, Random Forest, GridSearchCV
- `xgboost` — gradient boosting classifier
- `shap` — model explainability (optional but recommended)
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualisation
- `joblib` — artefact serialisation

---

## ⚖️ Fairness

The system audits two protected attributes at both training-data level (notebook) and inference level (dashboard):

- **Gender** (Female / Male / Other)
- **Parental Education Level** (High School / Bachelor / Master)

Fairness criterion: **Disparate Impact ≥ 0.80** (four-fifths rule). Groups below this threshold are flagged for review.

---

## 📚 Literature Grounding

This project is grounded in the Educational Data Mining (EDM) literature:

- Baker & Inventado (2014) — EDM and learning analytics overview
- Romero & Ventura (2020) — EDM applications in e-learning
- Lundberg & Lee (2017) — SHAP unified framework for model explanation
- Barocas, Hardt & Narayanan (2019) — Fairness and Machine Learning

---

## 📝 Evaluation

The system will be evaluated using the **System Usability Scale (SUS)** with classroom educators to assess practical usability and pedagogical utility.

---

## 🪪 Licence

This project is licensed under the **MIT Licence** — see `LICENSE` for details.

