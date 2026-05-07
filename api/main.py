"""
api/main.py — FastAPI REST API for Customer Segmentation & Churn Prediction
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path(os.getenv("MODEL_DIR", "../models"))

CLUSTER_LABELS = {
    0: "Active Low-Value",
    1: "High-Value Loyal",
    2: "At-Risk High-Value",
    3: "Inactive / Churned",
}

# ── Load artifacts at startup ─────────────────────────────────────────────────
def load_models():
    try:
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        kmeans = joblib.load(MODEL_DIR / "kmeans.pkl")
        model  = joblib.load(MODEL_DIR / "churn_model.pkl")
        return scaler, kmeans, model
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model artifacts not found in {MODEL_DIR}. "
            "Run training/train.py first."
        ) from e

scaler, kmeans, churn_model = load_models()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title        = "Customer Segmentation & Churn Prediction API",
    description  = "RFM-based customer segmentation with churn probability scoring",
    version      = "1.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    customer_id : str  = Field(...,  example="12345",  description="Unique customer identifier")
    recency     : int  = Field(...,  ge=0, le=730,     description="Days since last purchase")
    frequency   : int  = Field(...,  ge=1,             description="Number of unique invoices")
    monetary    : float = Field(..., gt=0,             description="Total spend in GBP")

    @validator("monetary")
    def monetary_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("monetary must be greater than 0")
        return round(v, 2)


class CustomerBatch(BaseModel):
    customers: List[CustomerInput] = Field(..., min_items=1, max_items=500)


class PredictionResult(BaseModel):
    customer_id       : str
    segment           : str
    cluster_id        : int
    churn_probability : float
    churn_risk        : str
    retention_action  : str
    rfm_summary       : dict


# ── Helpers ───────────────────────────────────────────────────────────────────
CHURN_DAYS = 90

def _predict_one(c: CustomerInput) -> PredictionResult:
    rfm_raw    = np.array([[c.recency, c.frequency, c.monetary]])
    rfm_log    = np.log1p(rfm_raw)
    rfm_scaled = scaler.transform(rfm_log)

    cluster_id = int(kmeans.predict(rfm_scaled)[0])
    segment    = CLUSTER_LABELS.get(cluster_id, "Unknown")

    churn_prob = float(churn_model.predict_proba(rfm_raw)[0, 1])

    if churn_prob >= 0.7:     churn_risk = "High"
    elif churn_prob >= 0.4:   churn_risk = "Medium"
    else:                     churn_risk = "Low"

    # Retention logic (mirrors notebook)
    if segment == "High-Value Loyal"   and churn_prob > 0.5:
        action = "Immediate Retention (VIP Offer)"
    elif segment == "At-Risk High-Value" and churn_prob > 0.4:
        action = "Targeted Win-Back Campaign"
    elif segment == "Active Low-Value"   and churn_prob > 0.5:
        action = "Upsell / Engagement Offer"
    elif segment == "Inactive / Churned":
        action = "No Action / Low-Cost Re-engagement"
    else:
        action = "Monitor"

    return PredictionResult(
        customer_id       = c.customer_id,
        segment           = segment,
        cluster_id        = cluster_id,
        churn_probability = round(churn_prob, 4),
        churn_risk        = churn_risk,
        retention_action  = action,
        rfm_summary       = {
            "recency"  : c.recency,
            "frequency": c.frequency,
            "monetary" : c.monetary,
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Customer Segmentation & Churn Prediction API",
        "version": "1.0.0",
        "status" : "running",
        "docs"   : "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict_single(customer: CustomerInput):
    """
    Predict segment and churn probability for a **single customer**.

    - **recency**: days since last purchase (lower = more active)
    - **frequency**: number of unique invoices
    - **monetary**: total £ spent
    """
    try:
        return _predict_one(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResult], tags=["Prediction"])
def predict_batch(payload: CustomerBatch):
    """
    Predict segment and churn for a **batch of customers** (up to 500).
    """
    try:
        return [_predict_one(c) for c in payload.customers]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments", tags=["Info"])
def list_segments():
    """Returns all customer segment definitions."""
    return {
        "segments": [
            {
                "id"         : k,
                "name"       : v,
                "description": {
                    0: "Recent buyers, lower spend — nurture for growth",
                    1: "Your best customers — protect at all costs",
                    2: "High spenders going quiet — act fast",
                    3: "Long-inactive customers — low-cost re-engagement",
                }[k],
            }
            for k, v in CLUSTER_LABELS.items()
        ]
    }


@app.get("/model/info", tags=["Info"])
def model_info():
    """Returns metadata about the deployed models."""
    return {
        "churn_model"      : "Logistic Regression",
        "segmentation"     : f"KMeans (k={kmeans.n_clusters})",
        "feature_set"      : ["Recency", "Frequency", "Monetary"],
        "churn_threshold"  : f"{CHURN_DAYS} days inactivity",
        "preprocessing"    : "log1p + StandardScaler",
    }
