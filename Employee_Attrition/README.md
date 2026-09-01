# Employee Attrition Prediction & Workforce Analytics

## Project Overview
An end-to-end machine learning system that predicts which employees are at
risk of resigning, so HR teams can act before losing valuable talent. The
project spans data analysis, SQL, Excel, Power BI, a full ML pipeline with
explainability, and a production-style deployment (Streamlit + Docker).

## Business Problem
Employee attrition is costly — recruiting, onboarding, and lost productivity
can cost 50–200% of an employee's annual salary. This project builds a
classifier that flags at-risk employees early, paired with SHAP-based
explanations so HR knows *why* an employee is at risk and what to do about it.

## Dataset
IBM HR Analytics Employee Attrition Dataset (1,470 employees, 35 columns).
Place the CSV at `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

## Tech Stack
| Layer | Tools |
|---|---|
| Data analysis | pandas, numpy, matplotlib, seaborn |
| Database | MySQL |
| BI | Excel, Power BI |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| App | Streamlit |



## Future Improvement
- Add CI/CD (GitHub Actions) to retrain and redeploy on new data.
- Add a monitoring layer for prediction drift over time.

