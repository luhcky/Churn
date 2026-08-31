# 📉 Customer Churn Prediction

Predicting which telecom customers are likely to churn using historical usage and service data.

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Pipeline Architecture](#pipeline-architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Overview

A supervised machine learning pipeline that predicts customer churn in a telecom dataset. Three models were trained and compared —Random Forest,Gradient Boosting and XGBoost — with the final model selected based on **AUC-ROC** and the business cost of false negatives (i.e. missing a customer who was about to churn).

## Problem Statement

Customer acquisition costs far more than retention. Telecom providers need to identify at-risk customers *before* they cancel, so retention teams can intervene with targeted offers. This project frames churn as a binary classification problem and prioritises **recall on the churn class** without sacrificing overall model usefulness.

## Dataset

- Historical telecom customer records: demographics, contract type, tenure, monthly/total charges, service usage, and support interactions.
- Target variable: `Churn` (Yes/No).
- Class imbalance present (churners are the minority class).

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Modelling | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn |

## Pipeline Architecture

```
Raw CSV
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Feature Engineering
   │
   ▼
Stratified Train / Test Split
   │
   ▼
Model Training  (Logistic Regression → Random Forest → XGBoost)
   │
   ▼
Evaluation  (AUC-ROC, Precision/Recall, Confusion Matrix)
   │
   ▼
SHAP Explainability
   │
   ▼
Stakeholder Report
```

## Methodology

- **Class imbalance** handled using `scale_pos_weight="ratio"` rather than naive resampling, to preserve the natural distribution of the evaluation set.
- **Feature engineering** centred on call duration, contract type, tenure, and service usage patterns.
- **Model selection** compared Gradient Boosting, Random Forest, and XGBoost (final model) using stratified k-fold cross-validation.
- **Explainability**: SHAP summary and dependence plots used to translate model output into a business narrative, not just a leaderboard metric.

## Results

| Metric | Value |
|---|---|
| Best Model | XGBoost |
| AUC-ROC | **0.85** |

**Top churn drivers identified:**
1. Short tenure (newer customers churn more)
2. Month-to-month contract type (vs. annual/2-year contracts)

## Future Improvements
- Add a monitoring layer to track model/data drift over time.
- Experiment with cost-sensitive learning tied to actual retention-offer economics.

