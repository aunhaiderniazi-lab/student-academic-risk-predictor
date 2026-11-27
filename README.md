# Student Academic Risk Predictor

Single-file Streamlit application and machine learning pipeline built for **Introduction to Data Science Lab (IET 3082)** at **Foundation University School of Science and Technology (FUSST)** by **Aun Haider Niazi**.

## Overview
Screens students for academic pass/fail risk using synthetic academic and lifestyle features (`study_hours_per_week`, `attendance_percent`, `previous_gpa`, `assignments_completed`, `sleep_hours`, `extracurricular`), compares **Logistic Regression** (with `StandardScaler`) and **Random Forest** on F1-score, and presents interactive predictions alongside EDA visuals in a dark-mode UI.

## Fixes & Features
* **Used Evaluation Metrics:** Computes and displays Accuracy, Recall, and F1-score for both models (`classification_report` unused import removed).
* **Clean Matplotlib Lifecycle:** Applies `plt.style.use("dark_background")` and closes figures (`plt.close(fig)`) after rendering in Streamlit to prevent memory leaks.
* **Cached Pipeline:** Uses `@st.cache_data` for dataset creation/cleaning and `@st.cache_resource` for model training.

## Run Locally
```bash
git clone https://github.com/aunhaiderniazi-lab/student-academic-risk-predictor.git
cd student-academic-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```
