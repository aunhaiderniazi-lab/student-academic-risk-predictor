"""
Student Academic Risk Predictor - Single-File Version
-----------------------------------------------------
Runs the whole Data Science pipeline (synthetic data, cleaning, EDA,
model comparison) and launches a Streamlit dashboard in one file.

Author: Aun Haider Niazi
Course: Introduction to Data Science Lab (IET 3082), FUSST
"""

import numpy as np
import pandas as pd
import streamlit as st

# Render Matplotlib in memory (prevents GUI crashes on Windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             accuracy_score, confusion_matrix, make_scorer)

plt.style.use("dark_background")
BG = "#0e1117"

# ==========================================
# 1. DATA + MODELS (cached, runs once)
# ==========================================
st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="wide")


@st.cache_data
def get_data():
    """Generates the synthetic dataset. Missing values are left in on purpose:
    they are filled inside the model Pipeline, so no test information leaks."""
    np.random.seed(42)
    N = 500

    study_hours = np.clip(np.random.normal(12, 5, N), 0, 40)
    attendance = np.clip(np.random.normal(78, 12, N), 40, 100)
    previous_gpa = np.clip(np.random.normal(2.8, 0.6, N), 0, 4.0)
    assignments = np.random.randint(0, 11, N)
    sleep_hours = np.clip(np.random.normal(6.5, 1.3, N), 3, 10)
    extracurricular = np.random.choice([0, 1], N, p=[0.6, 0.4])

    # Weighted score + noise decides pass (1) / fail (0)
    score = (0.20 * (study_hours / 40) + 0.25 * (attendance / 100) +
             0.30 * (previous_gpa / 4) + 0.15 * (assignments / 10) +
             0.10 * (sleep_hours / 10) + 0.05 * extracurricular +
             np.random.normal(0, 0.06, N))
    passed = (score > 0.55).astype(int)

    df = pd.DataFrame({
        "study_hours_per_week": study_hours.round(1),
        "attendance_percent": attendance.round(1),
        "previous_gpa": previous_gpa.round(2),
        "assignments_completed": assignments,
        "sleep_hours": sleep_hours.round(1),
        "extracurricular": extracurricular,
        "passed": passed,
    })

    # Make 3% of two columns missing (realistic messy data)
    for col in ["attendance_percent", "sleep_hours"]:
        idx = np.random.choice(df.index, size=int(N * 0.03), replace=False)
        df.loc[idx, col] = np.nan
    return df


@st.cache_resource
def train_models(_df):
    """Compares LR and RF using cross-validation on the TRAIN set only,
    then reports the winner once on the untouched TEST set.
    The target of interest is the AT-RISK class (passed = 0)."""
    X = _df.drop(columns=["passed"])
    y = _df["passed"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=42))]),
    }

    risk_f1 = make_scorer(f1_score, pos_label=0, zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {name: cross_val_score(m, X_tr, y_tr, cv=cv, scoring=risk_f1).mean()
                 for name, m in candidates.items()}

    best_name = max(cv_scores, key=cv_scores.get)
    best_model = candidates[best_name].fit(X_tr, y_tr)

    pred = best_model.predict(X_te)
    metrics = {
        "Winner": best_name,
        "CV F1 (at-risk)": {k: round(v, 3) for k, v in cv_scores.items()},
        "Test F1 (at-risk)": round(f1_score(y_te, pred, pos_label=0, zero_division=0), 3),
        "Test Recall (at-risk)": round(recall_score(y_te, pred, pos_label=0, zero_division=0), 3),
        "Test Precision (at-risk)": round(precision_score(y_te, pred, pos_label=0, zero_division=0), 3),
        "Test Accuracy": round(accuracy_score(y_te, pred), 3),
        "Confusion": confusion_matrix(y_te, pred, labels=[0, 1]),
        "Test size": len(y_te),
        "Features": list(X.columns),
    }
    return best_model, metrics


def dark_fig(w=6, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


df = get_data()
model, metrics = train_models(df)

# ==========================================
# 2. STREAMLIT UI (Dark Mode)
# ==========================================
st.markdown("""
    <style>
      .stApp { background-color: #0e1117; color: #fafafa; }
      .stButton>button { background-color: #238636; color: white; border: none; }
      .stButton>button:hover { background-color: #2ea043; color: white; }
      .author-footer { color: #8b949e; font-size: 12px; text-align: center; margin-top: 40px; }
      .author-footer b { color: #c9d1d9; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Student Academic Risk Predictor")
st.caption(f"Identify at-risk students early  ·  Data Science Lab (FUSST)  ·  "
           f"Model: *{metrics['Winner']}*  ·  Synthetic data (500 students)")

tab1, tab2 = st.tabs(["🎯 Risk Predictor", "📊 EDA & Model Data"])

with tab1:
    st.subheader("Student Inputs")
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Study hours / week", 0.0, 40.0, 12.0, 0.5)
        attendance = st.slider("Attendance %", 40.0, 100.0, 78.0, 0.5)
        prev_gpa = st.slider("Previous GPA", 0.0, 4.0, 2.8, 0.01)
    with col2:
        assignments = st.slider("Assignments completed (of 10)", 0, 10, 7)
        sleep_hours = st.slider("Sleep hours / night", 3.0, 10.0, 6.5, 0.1)
        extracurric = st.selectbox("Extracurricular", ["No", "Yes"])

    if st.button("Predict Risk", use_container_width=True):
        input_df = pd.DataFrame([{
            "study_hours_per_week": study_hours,
            "attendance_percent": attendance,
            "previous_gpa": prev_gpa,
            "assignments_completed": assignments,
            "sleep_hours": sleep_hours,
            "extracurricular": 1 if extracurric == "Yes" else 0,
        }])[metrics["Features"]]

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]   # [P(at risk), P(pass)]

        st.divider()
        if pred == 1:
            st.success(f"✅ **Likely to Pass**  ·  Confidence: {proba[1]:.1%}")
        else:
            st.error(f"⚠️ **At Risk of Failing**  ·  Risk: {proba[0]:.1%}")
            st.info("Suggested action: schedule a faculty advisor meeting and tutoring support.")

with tab2:
    st.subheader("Model Evaluation (target: at-risk students)")
    cv_lr = metrics["CV F1 (at-risk)"]["Logistic Regression"]
    cv_rf = metrics["CV F1 (at-risk)"]["Random Forest"]
    st.write(f"**5-fold cross-validation, F1 for at-risk class:** "
             f"Logistic Regression = `{cv_lr}` | Random Forest = `{cv_rf}`  "
             f"→ winner: **{metrics['Winner']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test F1 (at-risk)", metrics["Test F1 (at-risk)"])
    m2.metric("Test Recall (at-risk)", metrics["Test Recall (at-risk)"])
    m3.metric("Test Precision (at-risk)", metrics["Test Precision (at-risk)"])
    m4.metric("Test Accuracy", metrics["Test Accuracy"])
    st.caption(f"Scores are on a held-out test set of {metrics['Test size']} students. "
               "Recall = share of truly at-risk students the model catches.")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig0, ax0 = dark_fig()
        sns.heatmap(metrics["Confusion"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred At-Risk", "Pred Pass"],
                    yticklabels=["Actual At-Risk", "Actual Pass"], ax=ax0)
        ax0.set_title("Confusion Matrix (test set)")
        st.pyplot(fig0)
    with c2:
        fig1, ax1 = dark_fig()
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax1)
        ax1.set_title("Feature Correlation Heatmap")
        st.pyplot(fig1)
    with c3:
        fig2, ax2 = dark_fig()
        sns.boxplot(data=df, x="passed", y="previous_gpa", hue="passed",
                    palette="Set2", legend=False, ax=ax2)
        ax2.set_title("Previous GPA vs Pass/Fail")
        ax2.set_xlabel("Passed (0 = Fail, 1 = Pass)")
        st.pyplot(fig2)

    plt.close("all")  # prevents memory leaks across reruns

st.markdown(
    '<div class="author-footer">Built by <b>Aun Haider Niazi</b><br>'
    'Introduction to Data Science Lab - FUSST</div>',
    unsafe_allow_html=True
)
