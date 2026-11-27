"""
Student Academic Risk Predictor
Made by Aun Haider Niazi - Intro to Data Science Lab (IET 3082), FUSST

What it does: makes 500 fake students, cleans them, compares two models
and shows the result in a small Streamlit dashboard.
Run it with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # draw plots in memory, avoids crashes on Windows
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

st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="wide")


@st.cache_data
def make_data():
    """Build the fake student data. Some values are left empty on purpose,
    the pipeline fills them later so the test set stays untouched."""
    np.random.seed(42)
    n = 500

    study = np.clip(np.random.normal(12, 5, n), 0, 40)
    attend = np.clip(np.random.normal(78, 12, n), 40, 100)
    gpa = np.clip(np.random.normal(2.8, 0.6, n), 0, 4.0)
    tasks = np.random.randint(0, 11, n)
    sleep = np.clip(np.random.normal(6.5, 1.3, n), 3, 10)
    extra = np.random.choice([0, 1], n, p=[0.6, 0.4])

    # a student passes if this weighted score is high enough (plus some luck)
    score = (0.20 * study / 40 + 0.25 * attend / 100 + 0.30 * gpa / 4 +
             0.15 * tasks / 10 + 0.10 * sleep / 10 + 0.05 * extra +
             np.random.normal(0, 0.06, n))

    df = pd.DataFrame({
        "study_hours_per_week": study.round(1),
        "attendance_percent": attend.round(1),
        "previous_gpa": gpa.round(2),
        "assignments_completed": tasks,
        "sleep_hours": sleep.round(1),
        "extracurricular": extra,
        "passed": (score > 0.55).astype(int),
    })

    # blank out 3% of two columns, real data is never perfect
    for col in ["attendance_percent", "sleep_hours"]:
        rows = np.random.choice(df.index, size=int(n * 0.03), replace=False)
        df.loc[rows, col] = np.nan
    return df


@st.cache_resource
def train(_df):
    """Compare both models with cross-validation (training data only),
    then test the winner once. We care about at-risk students (passed = 0)."""
    X, y = _df.drop(columns=["passed"]), _df["passed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    fill = ("fill", SimpleImputer(strategy="median"))
    models = {
        "Logistic Regression": Pipeline([fill, ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": Pipeline([fill,
            ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                             random_state=42))]),
    }

    risk_f1 = make_scorer(f1_score, pos_label=0, zero_division=0)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = {name: cross_val_score(m, X_train, y_train, cv=folds, scoring=risk_f1).mean()
          for name, m in models.items()}

    winner = max(cv, key=cv.get)
    best = models[winner].fit(X_train, y_train)
    pred = best.predict(X_test)

    info = {
        "winner": winner,
        "cv": {k: round(float(v), 3) for k, v in cv.items()},
        "f1": round(f1_score(y_test, pred, pos_label=0, zero_division=0), 3),
        "recall": round(recall_score(y_test, pred, pos_label=0, zero_division=0), 3),
        "precision": round(precision_score(y_test, pred, pos_label=0, zero_division=0), 3),
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "matrix": confusion_matrix(y_test, pred, labels=[0, 1]),
        "test_size": len(y_test),
        "columns": list(X.columns),
    }
    return best, info


def new_plot():
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


df = make_data()
model, info = train(df)

st.markdown("""
    <style>
      .stApp { background-color: #0e1117; color: #fafafa; }
      .stButton>button { background-color: #238636; color: white; border: none; }
      .stButton>button:hover { background-color: #2ea043; color: white; }
      .footer { color: #8b949e; font-size: 12px; text-align: center; margin-top: 40px; }
      .footer b { color: #c9d1d9; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Student Academic Risk Predictor")
st.caption(f"By Aun Haider Niazi  ·  Data Science Lab, FUSST  ·  "
           f"Model: {info['winner']}  ·  Synthetic data (500 students)")

tab1, tab2 = st.tabs(["🎯 Risk Predictor", "📊 EDA & Model Data"])

with tab1:
    st.subheader("Student details")
    left, right = st.columns(2)
    with left:
        study_h = st.slider("Study hours / week", 0.0, 40.0, 12.0, 0.5)
        attend_p = st.slider("Attendance %", 40.0, 100.0, 78.0, 0.5)
        last_gpa = st.slider("Previous GPA", 0.0, 4.0, 2.8, 0.01)
    with right:
        done = st.slider("Assignments completed (of 10)", 0, 10, 7)
        sleep_h = st.slider("Sleep hours / night", 3.0, 10.0, 6.5, 0.1)
        extra_yes = st.selectbox("Extracurricular", ["No", "Yes"])

    if st.button("Predict Risk"):
        student = pd.DataFrame([{
            "study_hours_per_week": study_h,
            "attendance_percent": attend_p,
            "previous_gpa": last_gpa,
            "assignments_completed": done,
            "sleep_hours": sleep_h,
            "extracurricular": 1 if extra_yes == "Yes" else 0,
        }])[info["columns"]]

        result = model.predict(student)[0]
        chance = model.predict_proba(student)[0]  # [at risk, pass]

        st.divider()
        if result == 1:
            st.success(f"✅ **Likely to pass**  ·  Confidence: {chance[1]:.1%}")
        else:
            st.error(f"⚠️ **At risk of failing**  ·  Risk: {chance[0]:.1%}")
            st.info("Suggestion: book a meeting with a faculty advisor and arrange tutoring.")

with tab2:
    st.subheader("How well does the model do? (at-risk students)")
    st.write(f"**Cross-validation F1:** Logistic Regression = `{info['cv']['Logistic Regression']}` "
             f"| Random Forest = `{info['cv']['Random Forest']}`  →  winner: **{info['winner']}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1", info["f1"])
    c2.metric("Recall", info["recall"])
    c3.metric("Precision", info["precision"])
    c4.metric("Accuracy", info["accuracy"])
    st.caption(f"Tested on {info['test_size']} students the model never saw. "
               "Recall = how many truly at-risk students it catches.")

    a, b, c = st.columns(3)
    with a:
        fig, ax = new_plot()
        sns.heatmap(info["matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred at-risk", "Pred pass"],
                    yticklabels=["Real at-risk", "Real pass"], ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    with b:
        fig, ax = new_plot()
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation")
        st.pyplot(fig)
    with c:
        fig, ax = new_plot()
        sns.boxplot(data=df, x="passed", y="previous_gpa", hue="passed",
                    palette="Set2", legend=False, ax=ax)
        ax.set_title("Previous GPA vs Pass/Fail")
        ax.set_xlabel("Passed (0 = fail, 1 = pass)")
        st.pyplot(fig)

    plt.close("all")  # free memory after each rerun

st.markdown('<div class="footer">Built by <b>Aun Haider Niazi</b><br>'
            'Introduction to Data Science Lab - FUSST</div>', unsafe_allow_html=True)
