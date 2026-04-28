"""
Student Early-Warning Dashboard
================================
Run AFTER executing Final_Project.ipynb which saves:
  best_model.pkl
  model_features.pkl
  decision_threshold.pkl
  preprocessing_artifacts.pkl

Launch:  streamlit run main.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ── Load artefacts saved by the notebook ─────────────────────────
model     = joblib.load("best_model.pkl")
FEATURES  = joblib.load("model_features.pkl")
THRESHOLD = joblib.load("decision_threshold.pkl")
prep      = joblib.load("preprocessing_artifacts.pkl")
ORDINAL_MAPS     = prep["ordinal_maps"]
WINSORISE_BOUNDS = prep["winsorise_bounds"]

PASS_SCORE = 40   # exam_score cut-off used in the notebook

# Features that should NEVER be winsorised because clamping hides
# extreme values the model must see (e.g. attendance=0 gets clamped to 58.5)
SKIP_WINSORISE = {"attendance_percentage", "study_hours_per_day"}


# ── Preprocessing helpers (encode + engineer, no training) ───────

def preprocess_row(raw: dict) -> pd.DataFrame:
    """Encode and feature-engineer a single student's raw inputs."""
    df = pd.DataFrame([raw])

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Winsorise — but skip attendance & study hours
    for col, (lo, hi) in WINSORISE_BOUNDS.items():
        if col in df.columns and col not in SKIP_WINSORISE:
            df[col] = df[col].clip(lo, hi)

    df["study_x_attendance"] = df["study_hours_per_day"] * df["attendance_percentage"] / 100.0
    df["distraction_hours"]  = df["social_media_hours"] + df["netflix_hours"]
    df["wellbeing_index"]    = (
        df["sleep_hours"] / 8.0 * 0.4
        + df["mental_health_rating"] / 10.0 * 0.4
        + df["diet_quality"] / 2.0 * 0.2
    )
    df["productive_ratio"] = df["study_hours_per_day"] / (df["distraction_hours"] + 1)
    df["at_risk_flag"]     = (
        (df["attendance_percentage"] < 60) | (df["study_hours_per_day"] < 1.5)
    ).astype(int)

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
    return df[FEATURES]


def predict_single(raw: dict):
    """
    Predict with business-rule override for extreme edge cases.

    The training data contains only 13 students with study_hours=0,
    and 7 of them passed (scores 42-56). This makes the model think
    zero-study students have a ~50% chance of passing, which is
    unrealistic. The override catches these cases.

    Returns: (label, fail_prob, preprocessed_row, override_applied, override_reason)
    """
    row = preprocess_row(raw)
    proba = model.predict_proba(row)[0]
    fail_prob = float(proba[0])

    override = False
    reason = ""

    # Business rule: 0 study hours should always be FAIL
    if raw["study_hours_per_day"] == 0:
        fail_prob = max(fail_prob, 0.95)
        override = True
        reason = "Zero study hours — overridden to Fail (model has too few 0-study examples to learn this correctly)"

    # Business rule: near-zero engagement should always be FAIL
    if raw["attendance_percentage"] < 30 and raw["study_hours_per_day"] < 1.0:
        fail_prob = max(fail_prob, 0.95)
        override = True
        reason = "Near-zero attendance AND study hours — overridden to Fail"

    label = "FAIL" if fail_prob >= THRESHOLD else "PASS"
    return label, fail_prob, row, override, reason


def preprocess_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Same pipeline applied to a full DataFrame."""
    df = df_raw.copy()
    df.drop(columns=["student_id", "exam_score"], errors="ignore", inplace=True)

    if "parental_education_level" in df.columns:
        mode_val = df["parental_education_level"].mode()
        if len(mode_val) > 0:
            df["parental_education_level"] = df["parental_education_level"].fillna(mode_val[0])

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    for col, (lo, hi) in WINSORISE_BOUNDS.items():
        if col in df.columns and col not in SKIP_WINSORISE:
            df[col] = df[col].clip(lo, hi)

    df["study_x_attendance"] = df["study_hours_per_day"] * df["attendance_percentage"] / 100.0
    df["distraction_hours"]  = df["social_media_hours"] + df["netflix_hours"]
    df["wellbeing_index"]    = (
        df["sleep_hours"] / 8.0 * 0.4
        + df["mental_health_rating"] / 10.0 * 0.4
        + df["diet_quality"] / 2.0 * 0.2
    )
    df["productive_ratio"] = df["study_hours_per_day"] / (df["distraction_hours"] + 1)
    df["at_risk_flag"]     = (
        (df["attendance_percentage"] < 60) | (df["study_hours_per_day"] < 1.5)
    ).astype(int)

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
    return df[FEATURES]


def batch_predict_with_overrides(df_raw: pd.DataFrame):
    """Batch prediction with business-rule overrides."""
    X_batch = preprocess_batch(df_raw)
    probs = model.predict_proba(X_batch)[:, 0].copy()  # fail probabilities

    # Apply overrides
    overrides = np.zeros(len(probs), dtype=bool)

    if "study_hours_per_day" in df_raw.columns:
        zero_study = (df_raw["study_hours_per_day"] == 0).values
        probs = np.where(zero_study, np.maximum(probs, 0.95), probs)
        overrides = overrides | zero_study

    if "attendance_percentage" in df_raw.columns and "study_hours_per_day" in df_raw.columns:
        extreme = ((df_raw["attendance_percentage"] < 30) & (df_raw["study_hours_per_day"] < 1.0)).values
        probs = np.where(extreme, np.maximum(probs, 0.95), probs)
        overrides = overrides | extreme

    preds = (probs >= THRESHOLD).astype(int)
    return probs, preds, overrides


# ── Teacher recommendations keyed by feature name ────────────────

RECS = {
    "study_x_attendance":    ("📚 Study × Attendance", "Both study time and attendance are low — address them jointly."),
    "study_hours_per_day":   ("📚 Study Time",          "Encourage ≥ 2 h focused study per day. Try Pomodoro sessions."),
    "attendance_percentage": ("🏫 Attendance",           "Below 75% attendance strongly predicts failure. Discuss barriers."),
    "at_risk_flag":          ("⚠️ At-Risk Flag",         "Trigger an early-intervention meeting — student meets risk criteria."),
    "productive_ratio":      ("⚡ Productivity Ratio",   "High distraction vs study ratio. Help plan a digital-detox schedule."),
    "distraction_hours":     ("📱 Screen Time",          "Reduce social-media / Netflix to < 2 h/day."),
    "mental_health_rating":  ("🧠 Mental Wellbeing",     "Low mental health score. Refer to counselling or pastoral care."),
    "wellbeing_index":       ("❤️ Wellbeing Index",      "Composite wellbeing is low — discuss sleep, diet, and stress."),
    "sleep_hours":           ("😴 Sleep",                "Under 6 h of sleep impairs memory. Promote healthy sleep habits."),
    "part_time_job":         ("💼 Part-Time Job",        "Working may crowd out study time. Discuss workload balance."),
}


def top_recommendations(shap_vals, n=3):
    pairs = sorted(zip(shap_vals, FEATURES), key=lambda x: -x[0])
    out = []
    for sv, feat in pairs:
        if sv > 0 and feat in RECS:
            out.append((RECS[feat][0], RECS[feat][1], sv))
            if len(out) == n:
                break
    return out


# ════════════════════════════════════════════════════════════════
# Streamlit layout
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Student Early-Warning System",
    page_icon="🎓",
    layout="wide",
)

st.sidebar.title("🎓 Student Early-Warning")
st.sidebar.caption(f"Decision threshold: **{THRESHOLD:.2f}**")

page = st.sidebar.radio("Go to", [
    "🔍 Predict a Student",
    "📋 Batch Prediction",
    "⚖️ Fairness Audit",
])


# ════════════════ PAGE 1 — Single prediction ════════════════════

if page == "🔍 Predict a Student":

    st.title("🔍 Predict a Single Student")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Academic")
        study    = st.slider("Study hours / day",      0.0, 9.0, 3.5, 0.1)
        attend   = st.slider("Attendance %",            0.0, 100.0, 84.0, 0.5)
        mhealth  = st.slider("Mental health (1–10)",   1, 10, 5)

    with c2:
        st.subheader("Lifestyle")
        sleep    = st.slider("Sleep hours / night",    3.0, 12.0, 6.5, 0.1)
        social   = st.slider("Social media hrs / day", 0.0, 8.0, 2.5, 0.1)
        netflix  = st.slider("Streaming hrs / day",    0.0, 6.0, 1.8, 0.1)
        exercise = st.slider("Exercise days / week",   0, 7, 3)

    with c3:
        st.subheader("Background")
        age      = st.slider("Age", 17, 25, 20)
        gender   = st.selectbox("Gender",           ["Female", "Male", "Other"])
        ptjob    = st.selectbox("Part-time job",    ["No", "Yes"])
        diet     = st.selectbox("Diet quality",     ["Poor", "Fair", "Good"])
        internet = st.selectbox("Internet quality", ["Poor", "Average", "Good"])
        paredu   = st.selectbox("Parent education", ["High School", "Bachelor", "Master"])
        extra    = st.selectbox("Extra-curricular", ["No", "Yes"])

    if st.button("🎯 Predict", type="primary", use_container_width=True):

        raw = {
            "age": age, "gender": gender,
            "study_hours_per_day": study, "social_media_hours": social,
            "netflix_hours": netflix, "part_time_job": ptjob,
            "attendance_percentage": attend, "sleep_hours": sleep,
            "diet_quality": diet, "exercise_frequency": exercise,
            "parental_education_level": paredu, "internet_quality": internet,
            "mental_health_rating": mhealth,
            "extracurricular_participation": extra,
        }

        label, fail_prob, row, override, override_reason = predict_single(raw)

        # Result banner + mini bar chart
        left, right = st.columns([3, 1])
        with left:
            if label == "FAIL":
                st.error(f"## ❌ FAIL  —  Fail probability: **{fail_prob*100:.1f}%**")
                if override:
                    st.warning(f"🔒 **Business rule override applied:** {override_reason}")
                else:
                    st.warning("This student is at risk. See recommendations below.")
            else:
                st.success(f"## ✅ PASS  —  Pass probability: **{(1-fail_prob)*100:.1f}%**")

        with right:
            fig, ax = plt.subplots(figsize=(3, 2))
            is_fail = label == "FAIL"
            clr = "#ef5350" if is_fail else "#66bb6a"
            ax.barh(["Fail", "Pass"],
                    [fail_prob * 100, (1 - fail_prob) * 100],
                    color=[clr if is_fail else "#cfd8dc",
                           "#cfd8dc" if is_fail else clr])
            ax.set_xlim(0, 100)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter())
            plt.tight_layout()
            st.pyplot(fig)

        # SHAP waterfall + teacher recommendations
        if HAS_SHAP:
            st.subheader("🔍 SHAP Explanation")
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(row)
            sv_fail = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]

            base_val = (
                explainer.expected_value[0]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
            sv_obj = shap.Explanation(
                values=sv_fail,
                base_values=base_val,
                data=row.values[0],
                feature_names=FEATURES,
            )
            shap.plots.waterfall(sv_obj, max_display=12, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")

            if label == "FAIL":
                recs = top_recommendations(sv_fail)
                if recs:
                    st.subheader("📌 Teacher Recommendations")
                    for title, advice, sv in recs:
                        with st.expander(f"{title}  (SHAP +{sv:.3f})"):
                            st.write(advice)
        else:
            st.info("Install `shap` to enable SHAP waterfall explanations.")

        # Always show recommendations for FAIL, even without SHAP
        if label == "FAIL" and not HAS_SHAP:
            st.subheader("📌 Teacher Recommendations")
            if raw["study_hours_per_day"] < 1.5:
                st.warning("📚 **Study Time** — Student studies less than 1.5 hours/day. "
                           "Encourage structured study with ≥ 2h daily.")
            if raw["attendance_percentage"] < 60:
                st.warning("🏫 **Attendance** — Below 60%. Investigate barriers to attendance.")
            if raw["social_media_hours"] + raw["netflix_hours"] > 5:
                st.warning("📱 **Screen Time** — Over 5 hours of digital distraction. "
                           "Discuss digital wellness strategies.")
            if raw["mental_health_rating"] <= 3:
                st.warning("🧠 **Mental Wellbeing** — Rating ≤ 3. Refer to counselling.")


# ════════════════ PAGE 2 — Batch prediction ══════════════════════

elif page == "📋 Batch Prediction":

    st.title("📋 Batch Prediction")
    st.write("Upload a CSV file with student data and the model will predict Pass/Fail for every row.")

    with st.expander("ℹ️ Expected columns"):
        st.write("""
        Your CSV should include these columns (same as the training dataset):
        `age`, `gender`, `study_hours_per_day`, `social_media_hours`, `netflix_hours`,
        `part_time_job`, `attendance_percentage`, `sleep_hours`, `diet_quality`,
        `exercise_frequency`, `parental_education_level`, `internet_quality`,
        `mental_health_rating`, `extracurricular_participation`

        `student_id` and `exam_score` are optional.
        """)

    uploaded = st.file_uploader("Upload your CSV", type="csv")
    if uploaded is None:
        st.info("👆 Upload a CSV file to get predictions.")
        st.stop()

    df_raw = pd.read_csv(uploaded)
    st.success(f"✅ Loaded **{len(df_raw)} students** from `{uploaded.name}`")

    probs, preds, overrides = batch_predict_with_overrides(df_raw)

    out = pd.DataFrame()
    if "student_id" in df_raw.columns:
        out["student_id"] = df_raw["student_id"].values
    out["Fail_Probability_%"] = (probs * 100).round(1)
    out["Prediction"]         = np.where(preds == 1, "❌ Fail", "✅ Pass")
    out["Override"]           = np.where(overrides, "🔒 Yes", "")
    if "exam_score" in df_raw.columns:
        out["Actual"] = np.where(df_raw["exam_score"] >= PASS_SCORE, "Pass", "Fail")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students", len(preds))
    m2.metric("⚠️ Predicted Fail", int(preds.sum()),
              delta=f"{preds.mean()*100:.1f}%", delta_color="inverse")
    m3.metric("✅ Predicted Pass", int((preds == 0).sum()))
    m4.metric("🔒 Overrides", int(overrides.sum()))

    st.subheader("🚨 Top At-Risk Students")
    st.dataframe(
        out[out["Prediction"] == "❌ Fail"]
           .sort_values("Fail_Probability_%", ascending=False)
           .head(20),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(probs[preds == 0] * 100, bins=30, color="#66bb6a", alpha=0.7, label="Predicted Pass")
    ax.hist(probs[preds == 1] * 100, bins=30, color="#ef5350", alpha=0.7, label="Predicted Fail")
    ax.axvline(THRESHOLD * 100, color="black", linestyle="--", linewidth=1.2,
               label=f"Threshold {THRESHOLD*100:.0f}%")
    ax.set_xlabel("Fail Probability (%)")
    ax.set_ylabel("Students")
    ax.legend()
    ax.set_title("Fail Probability Distribution")
    plt.tight_layout()
    st.pyplot(fig)

    st.download_button("⬇️ Download Predictions CSV", out.to_csv(index=False),
                       "predictions.csv", "text/csv", use_container_width=True)


# ════════════════ PAGE 3 — Fairness audit ════════════════════════

elif page == "⚖️ Fairness Audit":

    st.title("⚖️ Fairness Audit — Demographic Parity")
    st.write("Disparate Impact ≥ 0.80 satisfies the four-fifths rule.")

    df_raw = pd.read_csv("student_habits_performance.csv")
    df_raw["pass_fail"] = (df_raw["exam_score"] >= PASS_SCORE).astype(int)

    for col, label in [("gender", "Gender"),
                       ("parental_education_level", "Parental Education")]:
        st.subheader(label)
        grp = df_raw.groupby(col)["pass_fail"].agg(["mean", "count"])
        grp.columns = ["Pass Rate", "Count"]
        grp["Pass Rate %"]      = (grp["Pass Rate"] * 100).round(1)
        grp["Disparate Impact"] = (grp["Pass Rate"] / grp["Pass Rate"].max()).round(3)
        grp["Fair (≥0.80)?"]    = grp["Disparate Impact"].apply(
            lambda x: "✅ Yes" if x >= 0.80 else "❌ No"
        )
        st.dataframe(grp, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ["#66bb6a" if v >= 0.80 else "#ef5350" for v in grp["Disparate Impact"]]
        bars = ax.bar(grp.index, grp["Pass Rate %"], color=colors, edgecolor="white")
        ax.axhline(80, color="black", linestyle="--", linewidth=1.2, label="80% floor")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Pass Rate (%)")
        ax.set_title(f"Pass Rate by {label}")
        ax.legend()
        for bar, val in zip(bars, grp["Pass Rate %"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.divider()
