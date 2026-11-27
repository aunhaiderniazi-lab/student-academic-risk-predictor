"""
Student Academic Risk Predictor — Single-File Version
-----------------------------------------------------
Runs data generation, missing-value cleaning, EDA, and ML model training
(Logistic Regression vs. Random Forest) inside a dark-mode Streamlit app.

Author: Aun Haider Niazi
Course: Introduction to Data Science Lab (IET 3082), FUSST
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="wide")
plt.style.use("dark_background")


@st.cache_data
def get_data():
    """Generates and cleans the synthetic student dataset."""
    np.random.seed(42)
    n = 500

    study_hours = np.clip(np.random.normal(12, 5, n), 0, 40)
    attendance = np.clip(np.random.normal(78, 12, n), 40, 100)
    previous_gpa = np.clip(np.random.normal(2.8, 0.6, n), 0, 4.0)
    assignments = np.random.randint(0, 11, n)
    sleep_hours = np.clip(np.random.normal(6.5, 1.3, n), 3, 10)
    extracurricular = np.random.choice([0, 1], n, p=[0.6, 0.4])

    score = (
        0.20 * (study_hours / 40)
        + 0.25 * (attendance / 100)
        + 0.30 * (previous_gpa / 4)
        + 0.15 * (assignments / 10)
        + 0.10 * (sleep_hours / 10)
        + 0.05 * extracurricular
        + np.random.normal(0, 0.06, n)
    )
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

    for col in ["attendance_percent", "sleep_hours"]:
        drop_idx = np.random.choice(df.index, size=int(n * 0.03), replace=False)
        df.loc[drop_idx, col] = np.nan

    return df.fillna(df.median(numeric_only=True))


@st.cache_resource
def train_models(_df):
    """Trains Logistic Regression and Random Forest and selects the higher F1 model."""
    X = _df.drop(columns=["passed"])
    y = _df["passed"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)

    lr_metrics = {
        "acc": round(accuracy_score(y_te, lr_pred), 3),
        "rec": round(recall_score(y_te, lr_pred, zero_division=0), 3),
        "f1": round(f1_score(y_te, lr_pred, zero_division=0), 3),
    }
    rf_metrics = {
        "acc": round(accuracy_score(y_te, rf_pred), 3),
        "rec": round(recall_score(y_te, rf_pred, zero_division=0), 3),
        "f1": round(f1_score(y_te, rf_pred, zero_division=0), 3),
    }

    if lr_metrics["f1"] >= rf_metrics["f1"]:
        best_model, best_name = lr, "Logistic Regression"
    else:
        best_model, best_name = rf, "Random Forest"

    return best_model, {
        "lr": lr_metrics,
        "rf": rf_metrics,
        "winner": best_name,
        "features": list(X.columns),
    }


df = get_data()
model, metrics = train_models(df)

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
st.caption(
    f"Early academic risk screening · Data Science Lab (IET 3082, FUSST) · Model: *{metrics['winner']}*"
)

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
        }])[metrics["features"]]

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.divider()
        if pred == 1:
            st.success(f"✅ **Likely to Pass** · Confidence: {proba[1]:.1%}")
        else:
            st.error(f"⚠️ **At Risk of Failing** · Risk: {proba[0]:.1%}")
            st.info("Suggested action: schedule a faculty advisor meeting and tutoring support.")

with tab2:
    st.subheader("Model Evaluation & Exploratory Data Analysis")
    lr_m, rf_m = metrics["lr"], metrics["rf"]
    st.write(
        f"**Logistic Regression:** Acc `{lr_m['acc']}` | Recall `{lr_m['rec']}` | F1 `{lr_m['f1']}`  ·  "
        f"**Random Forest:** Acc `{rf_m['acc']}` | Recall `{rf_m['rec']}` | F1 `{rf_m['f1']}`"
    )

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax1)
        ax1.set_title("Feature Correlation Heatmap")
        st.pyplot(fig1)
        plt.close(fig1)

    with col_chart2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.boxplot(
            data=df, x="passed", y="previous_gpa", hue="passed",
            palette="Set2", legend=False, ax=ax2
        )
        ax2.set_title("Previous GPA vs Pass/Fail")
        ax2.set_xlabel("Passed (0 = Fail, 1 = Pass)")
        st.pyplot(fig2)
        plt.close(fig2)

st.markdown(
    '<div class="author-footer">Built by <b>Aun Haider Niazi</b><br>'
    'Introduction to Data Science Lab (IET 3082) — FUSST</div>',
    unsafe_allow_html=True,
)
