# 💳 Credit Card Fraud Detection

Detecting fraudulent credit card transactions in a highly imbalanced dataset of 284,807 transactions — with a focus on realistic, financial-grade evaluation rather than accuracy alone.


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

A binary classification project built on the well-known UCI credit card fraud dataset. The core challenge is the **extreme class imbalance**: only ~0.17% of transactions are fraudulent. This project focuses on techniques that make a model *useful in production*, not just accurate on paper.

## Problem Statement

A model that predicts "not fraud" for every transaction would be 99.8% accurate — and completely useless. Financial fraud detection needs a model tuned for the **Precision-Recall trade-off**: catching as much fraud as possible while keeping false positives low enough that customers aren't constantly declined for legitimate purchases.

## Dataset

- 284,807 anonymised transactions (PCA-transformed features `V1`–`V28`, plus `Time` and `Amount`).
- Target variable: `Class` (1 = fraud, 0 = legitimate).
- Fraud cases: ~0.17% of all transactions — a textbook severe class imbalance problem.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Modelling | Scikit-learn, XGBoost |
| Visualisation | Seaborn, Matplotlib |
| Evaluation | Precision-Recall curves, SHAP |

## Pipeline Architecture

```
Raw Transaction Data
        │
        ▼
EDA (imbalance analysis, distribution checks)
        │
        ▼
Preprocessing (scaling, train/test split)
        │
        ▼
SMOTE Oversampling (training set only)
        │
        ▼
Model Training  (XGBoost)
        │
        ▼
Threshold Optimisation  (Precision-Recall trade-off)
        │
        ▼
Evaluation  (Confusion Matrix, SHAP)
```

## Methodology

- **SMOTE** (Synthetic Minority Oversampling Technique) applied *only* to the training data — never to the test set — to avoid data leakage and inflated metrics.
- Combined with `class_weight` tuning for additional imbalance correction.
- **Decision threshold** was deliberately optimised away from the default 0.5 cutoff, using the Precision-Recall curve to select a threshold appropriate for a financial fraud context (where false negatives are costlier than false positives, up to a point).
- Model interpretability delivered via SHAP to explain *why* a transaction was flagged — important for compliance and manual review teams.

## Results

| Metric | Value |
|---|---|
| AUC-ROC | **0.98** |

Precision-Recall curves, a full confusion matrix, and SHAP feature-importance analysis are included in the evaluation notebook to support the threshold and model choices with evidence, not just a single headline number.


## Future Improvements

- Explore anomaly-detection approaches (Isolation Forest, Autoencoders) as a complement to supervised classification.
- Add real-time scoring simulation to demonstrate latency-aware inference.
- Cost-sensitive threshold tuning based on an assumed average fraud loss per transaction.

