
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List
import numpy as np
import pandas as pd
import joblib
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LOADING ARTIFACTS
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model    = joblib.load(f"{MODEL_DIR}/churn_model.pk1")
scaler   = joblib.load(f"{MODEL_DIR}/churn_scaler.pk1")
FEATURES = joblib.load(f"{MODEL_DIR}/churn_feature_names.pk1")
try:
    import shap
    EXPLAINER    = shap.TreeExplainer(model)
    SHAP_AVAILABLE = True
    ev = EXPLAINER.expected_value
    if hasattr(ev,"__len__") and len(ev) > 1:
        EXPECTED_VALUE = float(ev[1]) 
    elif hasattr(ev,"__len__"):
        EXPECTED_VALUE = float (ev[0])
    else:
        EXPECTED_VALUE = float(ev)
    logger.info(f"SHAP loaded. Base value={EXPECTED_VALUE:.4f}")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed — pip install shap")

logger.info(f"Churn model loaded. Features={len(FEATURES)}")

# APP 
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "XGBoost churn model with SHAP explainability.\n\n"
        "**Endpoints:**\n"
        "- `POST /predict` — churn score + risk tier + signals\n"
        "- `POST /predict/explain` — full SHAP values per feature\n"
        "- `POST /predict/batch` — score up to 500 customers\n"
        "- `GET /health` — model status\n"
        "- `GET /model/info` — model metadata"
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

#  INPUT SCHEMA
class CustomerInput(BaseModel):
    tenure           : int   = Field(..., ge=0,   le=72,   example=2)
    MonthlyCharges   : float = Field(..., ge=0,   le=200,  example=75.5)
    TotalCharges     : float = Field(..., ge=0,   le=10000,example=150.0)
    Contract         : str   = Field(...,                  example="Month-to-month")
    PaymentMethod    : str   = Field(...,                  example="Electronic check")
    InternetService  : str   = Field("Fiber optic",        example="Fiber optic")
    OnlineSecurity   : str   = Field("No",                 example="No")
    TechSupport      : str   = Field("No",                 example="No")
    PaperlessBilling : str   = Field("Yes",                example="Yes")
    SeniorCitizen    : int   = Field(0, ge=0, le=1,        example=0)
    Partner          : str   = Field("No",                 example="No")
    Dependents       : str   = Field("No",                 example="No")
    PhoneService     : str   = Field("Yes",                example="Yes")
    MultipleLines    : str   = Field("No",                 example="No")
    OnlineBackup     : str   = Field("No",                 example="No")
    DeviceProtection : str   = Field("No",                 example="No")
    StreamingTV      : str   = Field("No",                 example="No")
    StreamingMovies  : str   = Field("No",                 example="No")

    @validator("Contract")
    def v_contract(cls, v):
        valid = ("Month-to-month", "One year", "Two year")
        if v not in valid: raise ValueError(f"Contract must be one of {valid}")
        return v

    @validator("PaymentMethod")
    def v_payment(cls, v):
        valid = ("Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)")
        if v not in valid: raise ValueError("Invalid PaymentMethod")
        return v

    @validator("InternetService")
    def v_internet(cls, v):
        if v not in ("DSL", "Fiber optic", "No"):
            raise ValueError("InternetService must be DSL, Fiber optic, or No")
        return v

    class Config:
        schema_extra = {"example": {
            "tenure": 2, "MonthlyCharges": 85.5, "TotalCharges": 171.0,
            "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "TechSupport": "No", "PaperlessBilling": "Yes", "SeniorCitizen": 0,
        }}

# FEATURE BUILDER 
def build_features(c: CustomerInput) -> pd.DataFrame:
    yn  = lambda v: 1 if v == "Yes" else 0
    row = {f: 0 for f in FEATURES}
    row.update({
        "tenure"           : c.tenure,
        "MonthlyCharges"   : c.MonthlyCharges,
        "TotalCharges"     : c.TotalCharges,
        "SeniorCitizen"    : c.SeniorCitizen,
        "Partner"          : yn(c.Partner),
        "Dependents"       : yn(c.Dependents),
        "PhoneService"     : yn(c.PhoneService),
        "PaperlessBilling" : yn(c.PaperlessBilling),
        "OnlineSecurity"   : yn(c.OnlineSecurity),
        "OnlineBackup"     : yn(c.OnlineBackup),
        "DeviceProtection" : yn(c.DeviceProtection),
        "TechSupport"      : yn(c.TechSupport),
        "StreamingTV"      : yn(c.StreamingTV),
        "StreamingMovies"  : yn(c.StreamingMovies),
        "MultipleLines"    : yn(c.MultipleLines),
    })
    for col in [f"Contract_{c.Contract}",
                f"PaymentMethod_{c.PaymentMethod}",
                f"InternetService_{c.InternetService}"]:
        if col in row: row[col] = 1
    return pd.DataFrame([row])[FEATURES]

def get_churn_signals(c: CustomerInput) -> List[str]:
    signals = []
    if c.Contract == "Month-to-month":
        signals.append("Month-to-month contract — 42.7% churn rate")
    if c.PaymentMethod == "Electronic check":
        signals.append("Electronic check payment — 45.3% churn rate")
    if c.tenure <= 6:
        signals.append(f"New customer ({c.tenure} months) — 61.4% churn rate")
    if c.InternetService == "Fiber optic" and c.OnlineSecurity == "No":
        signals.append("Fiber optic + no online security — 2.1x average churn")
    if c.MonthlyCharges > 70:
        signals.append(f"High monthly charges (${c.MonthlyCharges}) — above average risk")
    if c.SeniorCitizen == 1:
        signals.append("Senior citizen — elevated churn segment")
    return signals or ["No strong churn signals detected"]

def get_protective_signals(c: CustomerInput) -> List[str]:
    protective = []
    if c.Contract == "Two year":
        protective.append("Two-year contract — strongest retention signal")
    if c.Contract == "One year":
        protective.append("One-year contract — moderate retention")
    if c.tenure > 24:
        protective.append(f"Long tenure ({c.tenure} months) — high loyalty")
    if c.PaymentMethod == "Credit card (automatic)":
        protective.append("Credit card (auto) — lowest churn payment method (15.2%)")
    if c.OnlineSecurity == "Yes":
        protective.append("Online security enabled — protective factor")
    return protective or ["No strong protective factors detected"]

#  ENDPOINTS 
@app.get("/", tags=["Info"])
def root():
    return {
        "api"      : "Customer Churn Prediction API",
        "version"  : "1.0.0",
        "shap"     : SHAP_AVAILABLE,
        "docs"     : "/docs",
        "endpoints": {
            "predict"        : "POST /predict",
            "predict_explain": "POST /predict/explain",
            "predict_batch"  : "POST /predict/batch",
            "health"         : "GET /health",
            "model_info"     : "GET /model/info",
        }
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status"         : "healthy",
        "model"          : "XGBClassifier",
        "auc"            : 0.941,
        "recall_pct"     : 78.4,
        "feature_count"  : len(FEATURES),
        "shap_available" : SHAP_AVAILABLE,
        "shap_base_value": EXPECTED_VALUE,
    }

@app.get("/model/info", tags=["Info"])
def model_info():
    return {
        "algorithm"     : "XGBClassifier",
        "dataset"       : "IBM Telco Customer Churn",
        "records"       : 7043,
        "churn_rate_pct": 26.5,
        "auc_roc"       : 0.941,
        "recall"        : 0.784,
        "top_drivers"   : ["tenure", "Contract_Month-to-month",
                           "MonthlyCharges", "PaymentMethod_Electronic check",
                           "InternetService_Fiber optic"],
        "shap_explainer": "TreeExplainer" if SHAP_AVAILABLE else "not available",
    }

@app.post("/predict", tags=["Prediction"])
def predict(customer: CustomerInput):
    """Score one customer — returns churn probability, tier, signals."""
    start = time.time()
    try:
        X    = build_features(customer)
        X_s  = scaler.transform(X)
        prob = float(model.predict_proba(X_s)[0][1])
        tier = "HIGH" if prob >= 0.5 else "MEDIUM" if prob >= 0.3 else "LOW"
        return {
            "churn_probability"  : round(prob, 4),
            "churn_pct"          : round(prob * 100, 2),
            "risk_tier"          : tier,
            "will_churn"         : prob >= 0.5,
            "churn_signals"      : get_churn_signals(customer),
            "protective_signals" : get_protective_signals(customer),
            "recommendation"     : (
                "Immediate retention offer — discount or contract upgrade"
                if tier == "HIGH" else
                "Monitor closely — proactive check-in recommended"
                if tier == "MEDIUM" else
                "No action needed — customer appears stable"
            ),
            "processing_ms"      : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/explain", tags=["Prediction"])
def predict_explain(customer: CustomerInput):
    if not SHAP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="SHAP not installed. Run: pip install shap"
        )
    start = time.time()
    try:
        X    = build_features(customer)
        X_s  = scaler.transform(X)
        X_df = pd.DataFrame(X_s, columns=FEATURES)

        # Prediction
        prob = float(model.predict_proba(X_s)[0][1])
        tier = "HIGH" if prob >= 0.5 else "MEDIUM" if prob >= 0.3 else "LOW"

        # SHAP values
        shap_vals = EXPLAINER.shap_values(X_df)
        sv        = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        sv_row    = sv[0]

        # Full SHAP dict — every feature
        shap_dict = {
            feat: round(float(val), 6)
            for feat, val in zip(FEATURES, sv_row)
        }

        # Top 10 drivers for this customer
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_drivers = [
            {
                "feature"    : feat,
                "shap_value" : val,
                "direction"  : "increases churn risk" if val > 0 else "decreases churn risk",
                "magnitude"  : "high" if abs(val) > 0.1 else "medium" if abs(val) > 0.03 else "low",
            }
            for feat, val in sorted_shap[:10]
        ]

        #explanation
        top3 = sorted_shap[:3]
        parts = []
        for feat, val in top3:
            direction = "increased" if val > 0 else "decreased"
            parts.append(
                f"{feat} {direction} churn probability "
                f"by {abs(val)*100:.1f} percentage points"
            )
        explanation = ". ".join(parts) + "."

        return {
            # Standard prediction
            "churn_probability"  : round(prob, 4),
            "churn_pct"          : round(prob * 100, 2),
            "risk_tier"          : tier,
            "will_churn"         : prob >= 0.5,
            "churn_signals"      : get_churn_signals(customer),
            "protective_signals" : get_protective_signals(customer),
            "recommendation"     : (
                "Immediate retention offer"  if tier == "HIGH" else
                "Proactive check-in"         if tier == "MEDIUM" else
                "No action needed"
            ),
            # SHAP explanation
            "shap_base_value"    : round(EXPECTED_VALUE, 4),
            "shap_values"        : shap_dict,
            "shap_top_drivers"   : top_drivers,
            "shap_explanation"   : explanation,
            "shap_note"          : (
                f"Base churn rate across all customers: "
                f"{EXPECTED_VALUE*100:.1f}%. "
                f"SHAP values show how each feature moved this customer's "
                f"probability from that base to {prob*100:.1f}%."
            ),
            "processing_ms"      : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: List[CustomerInput]):
    """Score 1–500 customers. Returns all predictions + summary."""
    if len(customers) > 500:
        raise HTTPException(status_code=422, detail="Max 500 customers per batch")
    start   = time.time()
    results = []
    flagged = 0
    for c in customers:
        X_s  = scaler.transform(build_features(c))
        prob = float(model.predict_proba(X_s)[0][1])
        will = prob >= 0.5
        if will: flagged += 1
        results.append({
            "churn_probability": round(prob, 4),
            "risk_tier"        : "HIGH" if prob>=0.5 else "MEDIUM" if prob>=0.3 else "LOW",
            "will_churn"       : will,
        })
    return {
        "total_customers"   : len(results),
        "flagged_count"     : flagged,
        "churn_rate_pct"    : round(flagged / len(results) * 100, 1),
        "tier_breakdown"    : {
            t: sum(1 for r in results if r["risk_tier"] == t)
            for t in ["LOW", "MEDIUM", "HIGH"]
        },
        "predictions"       : results,
        "processing_ms"     : round((time.time() - start) * 1000, 2),
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"status":"error","detail":str(exc)})
