# Student Academic Risk Predictor

Single-file Streamlit app and machine learning pipeline built for **Introduction to Data Science Lab (IET 3082)** at **Foundation University School of Science and Technology (FUSST)** by **Aun Haider Niazi**.

## Overview
Predicts whether a student is likely to pass or is **at risk of failing**, so advisors can step in early. The dashboard has two tabs:
- **Risk Predictor:** sliders for six student features give a prediction with a risk percentage and a suggested action.
- **EDA & Model Data:** cross-validation scores, test metrics, a confusion matrix, a correlation heatmap and a GPA boxplot.

## Pipeline
1. **Synthetic data:** 500 generated students, to protect real student privacy. Pass/fail comes from a weighted score plus noise.
2. **Missing values:** 3% of attendance and sleep values are missing. They are filled with the **median inside the model Pipeline** (`SimpleImputer`), so the test set never influences training.
3. **Split:** 80/20 stratified train/test split.
4. **Model comparison:** Logistic Regression (scaled) vs Random Forest, both with `class_weight="balanced"`, compared with **5-fold cross-validation on the training set only**.
5. **Metric:** F1-score for the **at-risk class** (`pos_label=0`), because missing an at-risk student is worse than a false alarm.
6. **Final check:** the winner is scored once on the untouched test set.

## Results (test set, 100 students)
| Metric (at-risk class) | Value |
|---|---|
| Cross-validation F1, Logistic Regression | 0.669 |
| Cross-validation F1, Random Forest | 0.551 |
| Test F1 | 0.679 |
| Test Recall | 0.783 |
| Test Precision | 0.600 |
| Test Accuracy | 0.830 |

Logistic Regression wins. It catches 18 of 23 at-risk students and gives 12 false alarms.

## Limitations
- The data is **synthetic**, and labels come from a formula, so the model mostly relearns that formula. Results show the pipeline works, not how well it predicts real students.
- The test set is small (100 students), so scores vary between random seeds.
- Future work: real university data, threshold tuning and fairness checks.

## Run Locally
```bash
git clone https://github.com/aunhaiderniazi-lab/student-academic-risk-predictor.git
cd student-academic-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Author
Aun Haider Niazi - BS Information Engineering Technology, Foundation University Islamabad
