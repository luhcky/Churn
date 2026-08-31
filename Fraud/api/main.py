from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import numpy as np 
import pandas as pd
import joblib
import os 
import time 

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
model = joblib.load(f'{MODEL_DIR}/fraud_model.pkl')
scaler = joblib.load(f'{MODEL_DIR}/fraud_scaler.pkl')
FEATURES = joblib.load(f'{MODEL_DIR}/fraud_feature_names.pkl')

app = FastAPI(title='Credit Card Fraud Detection API', version ='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'],allow_headers=['*'])

class TransactionInput(BaseModel):
    Time : float =Field(... ,description='Seconds from first transaction')
    Amount: float =Field(...,ge=0,description='Transaction amount in USD')
    V1 : float=Field(0.0)
    V2 : float=Field(0.0)
    V3 : float=Field(0.0)
    V4 : float=Field(0.0)
    V5 : float=Field(0.0)
    V6 : float=Field(0.0)
    V7 : float=Field(0.0)
    V8 : float=Field(0.0)
    V9 : float=Field(0.0)
    V10 : float=Field(0.0)
    V11: float=Field(0.0)
    V12 : float=Field(0.0)
    V13 : float=Field(0.0)
    V14 : float=Field(0.0)
    V15 : float=Field(0.0)
    V16 : float=Field(0.0)
    V17 : float=Field(0.0)
    V18 : float=Field(0.0)
    V19 : float=Field(0.0)
    V20 : float=Field(0.0)
    V21 : float=Field(0.0)
    V22 : float=Field(0.0)
    V23 : float=Field(0.0)
    V24 : float=Field(0.0)
    V25 : float=Field(0.0)
    V26 : float=Field(0.0)
    V27 : float=Field(0.0)
    V28 : float=Field(0.0)
    
@app.get('/')
def root():
    return {'api':'Credit Card Fraud Detection'}
@app.get('/health')
def health():
    return{'status':'OK','feature_count':len(FEATURES)}
@app.post('/predict')
def predict(tx:TransactionInput):
    start = time.time()
    try:
        row = {f:getattr(tx, f, 0.0) for f in FEATURES}
        X = pd.DataFrame([row])[FEATURES]
        X_s = scaler.transform(X)
        prob = float(model.predict_proba(X_s)[0][1])
        fraud = prob >= THRESHOLD
        
        return{
            'Fraud probability': round(prob, 4),
            'is_fraud': fraud,
            'threshold_used':THRESHOLD,
            'alert_level':'BLOCK' if prob>= 0.7 else 'REVIEW' if fraud else 'CLEAR',
            'processing_ms': round((time.time()-start)*1000,2) , 
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/predict/batch')
def predict_batch():
    if len(transactions) > 1000:
        raise HTTPException(status_code = 422, detail= 'Max 1000 transactions')
    results, flagged = [],0
    for tx in transactions:
        row = {f:getattr(tx, f,0.0) for f in FEATURES}
        X_s = scaler.transform(pd.DataFrame([row])[FEATURES])
        is_f = prob >= THRESHOLD
        if is_f: flagged += 1
        results.append({'fraud_probability':round(prob,4), 'is_fraud':is_f})
    return {'total':len(results),'fraud_count':flagged,'fraud_rate pct': round(flagged/len(results)*100,4),'predictions':results}