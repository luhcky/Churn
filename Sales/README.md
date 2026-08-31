# 📊 Sales & BI Dashboard

An end-to-end sales analytics solution — from raw CSV to a 5-page interactive Power BI dashboard covering revenue trends, product performance, and regional breakdown.

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Pipeline Architecture](#pipeline-architecture)
- [Future Improvements](#future-improvement)

---

## Overview

A complete business intelligence solution that transforms raw sales CSV exports into a multi-page interactive Power BI dashboard, complete with drill-through, slicers, and executive KPI cards. Built to demonstrate the **full BI workflow** — not just dashboard design, but the data engineering that makes a trustworthy dashboard possible.

## Problem Statement

Raw transactional sales data isn't decision-ready. Stakeholders need a single source of truth that answers: *How is revenue trending? Which products and regions are driving (or dragging) performance? Where should the sales team focus next?*

## Dataset

- Raw sales transaction CSV exports (order-level detail: product, region, date, revenue, quantity).

## Tech Stack

| Category | Tools |
|---|---|
| Data Cleaning | Python (Pandas) |
| Staging | SQL |
| Modelling & Visualisation | Power BI, DAX |
| Source Prep | Excel |

## Pipeline Architecture

```
Raw Sales CSV
      │
      ▼
Python Cleaning (Pandas)
      │
      ▼
SQL Staging Database
      │
      ▼
Power BI Data Model (Star Schema)
      │
      ▼
DAX Measures
      │
      ▼
Multi-Page Interactive Dashboard
```

## Future Improvements

- Automate the refresh pipeline with a scheduled SQL job.
- Add a forecasting visual using a proper time-series model


