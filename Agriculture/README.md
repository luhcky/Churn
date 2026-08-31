# 🌾 Agricultural Yield Prediction

Predicting crop yield per hectare across Kenyan counties using rainfall, area,temperature, and fertiliser-use data — built for agri-finance, insurance, and policy planning use cases.


## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Pipeline Architecture](#pipeline-architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

A regression-based machine learning model that predicts agricultural yield (tonnes/hectare) across Kenyan counties. The project is aimed at use cases in **agri-finance, crop insurance underwriting, and county-level policy planning**.

## Problem Statement

Yield variability across counties — driven by rainfall, soil quality, and input use — creates real risk for lenders, insurers, and farmers. A data-driven yield estimate helps de-risk agricultural lending and insurance pricing, and gives policymakers a county-level view of where agricultural interventions would have the most impact.

## Dataset

- Synthetic data generated using KNBS statistical data.
- Features: rainfall (mm), mean temperature, soil type, fertiliser quantity, crop type.
- Granularity: county-level, aggregated across growing seasons.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas |
| Modelling | Scikit-learn (Random Forest Regressor) |
| Validation | K-Fold Cross-Validation |
| Visualisation | Matplotlib, Seaborn |

## Pipeline Architecture

```
 Data Ingestion
          │
          ▼
   Pandas Cleaning
          │
          ▼
EDA (rainfall vs. yield correlations)
          │
          ▼
   Feature Selection
          │
          ▼
Random Forest Regression (K-Fold CV)
          │
          ▼
   R² Evaluation
          │
          ▼
County-Level Prediction Maps
```

## Methodology

- **Target variable:** yield (tonnes/hectare).
- **Features:** rainfall (mm), mean temperature, soil type , fertiliser quantity, crop type.
- **Validation:** K-Fold cross-validation used to get a robust estimate of generalisation performance given the relatively small county-level sample size.
- **Output:** a county-level prediction map was produced to make the results usable in a stakeholder presentation, not just a table of numbers.

## Results

| Metric | Value |
|---|---|
| Model | Random Forest Regression |
| R² (test set) | **0.80** |

**Top predictors identified:** rainfall and fertiliser dose — consistent with agronomic expectations, which helped validate that the model was learning genuine signal rather than noise.


## Future Improvements

- Incorporate satellite-derived vegetation indices (e.g. NDVI) as additional features.
- Extend to sub-county granularity where data permits.
