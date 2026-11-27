# 🎓 Student Academic Risk Predictor

**Author:** Aun Haider Niazi  
**Course:** Introduction to Data Science Lab (IET 3082) — FUSST  

## 📌 Problem Statement
Faculty often identify struggling students too late. This project uses machine learning to flag students at risk of failing **early**, enabling targeted tutoring and advisor interventions based on academic indicators.

## ⚙️ Architecture & Pipeline
This project is built as a single-file Streamlit application (`app.py`) that runs the entire data science pipeline in memory:
1. **Data Generation:** Simulates 500 student records to protect privacy.
2. **Preprocessing:** Handles missing values using `SimpleImputer` inside the pipeline to prevent data leakage.
3. **Modeling:** Compares Logistic Regression vs. Random Forest using 5-fold Cross-Validation. It optimizes for the **F1-Score of the at-risk class**, because catching a failing student (recall) is more important than a false alarm.
4. **UI:** A dark-mode Streamlit dashboard containing a Risk Predictor tab and an EDA/Model Metrics tab.

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
