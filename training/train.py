"""
train.py — Reproducible training script
Reads online_retail_II.xlsx, builds the full pipeline,
and saves 3 artifacts: scaler.pkl, kmeans.pkl, churn_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH     = os.getenv("DATA_PATH", "online_retail_II.xlsx")
MODEL_DIR     = os.getenv("MODEL_DIR", "../models")
N_CLUSTERS    = 4
CHURN_DAYS    = 90
RANDOM_STATE  = 42

CLUSTER_LABELS = {
    0: "Active Low-Value",
    1: "High-Value Loyal",
    2: "At-Risk High-Value",
    3: "Inactive / Churned",
}


# ── 1. Load & Clean ───────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading data from {path} ...")
    df = pd.read_excel(path)

    df = df[df["Customer ID"].notna()]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[~df["Description"].str.contains("POSTAGE", na=False)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"]  = df["Quantity"] * df["Price"]

    print(f"    Cleaned shape: {df.shape}")
    return df


# ── 2. RFM Aggregation ────────────────────────────────────────────────────────
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Building RFM table ...")
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg(
            Recency   = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency = ("Invoice",      "nunique"),
            Monetary  = ("TotalPrice",   "sum"),
        )
        .reset_index()
        .rename(columns={"Customer ID": "CustomerID"})
    )
    print(f"    Customers: {len(rfm)}")
    return rfm


# ── 3. Feature Engineering & Scaling ─────────────────────────────────────────
def scale_features(rfm: pd.DataFrame):
    print("[3/5] Log-transforming & scaling ...")
    rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(rfm_log)
    return scaled, scaler


# ── 4. Clustering ─────────────────────────────────────────────────────────────
def cluster_customers(rfm: pd.DataFrame, scaled):
    print("[4/5] Running KMeans clustering ...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    rfm    = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(scaled)
    rfm["Segment"] = rfm["Cluster"].map(CLUSTER_LABELS)
    return rfm, kmeans


# ── 5. Churn Model ────────────────────────────────────────────────────────────
def train_churn_model(rfm: pd.DataFrame):
    print("[5/5] Training churn model ...")
    rfm = rfm.copy()
    rfm["Churn"] = (rfm["Recency"] > CHURN_DAYS).astype(int)

    X = rfm[["Recency", "Frequency", "Monetary"]]
    y = rfm["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("\n── Churn Model Evaluation ──────────────────────────")
    print(classification_report(y_test, model.predict(X_test)))

    rfm["Churn_Probability"] = model.predict_proba(X)[:, 1]
    return rfm, model


# ── Retention Action ──────────────────────────────────────────────────────────
def assign_retention_action(row: dict) -> str:
    seg   = row["Segment"]
    prob  = row["Churn_Probability"]

    if seg == "High-Value Loyal"   and prob > 0.5:  return "Immediate Retention (VIP Offer)"
    if seg == "At-Risk High-Value" and prob > 0.4:  return "Targeted Win-Back Campaign"
    if seg == "Active Low-Value"   and prob > 0.5:  return "Upsell / Engagement Offer"
    if seg == "Inactive / Churned":                 return "No Action / Low-Cost Re-engagement"
    return "Monitor"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df          = load_and_clean(DATA_PATH)
    rfm         = build_rfm(df)
    scaled, scaler = scale_features(rfm)
    rfm, kmeans = cluster_customers(rfm, scaled)
    rfm, model  = train_churn_model(rfm)

    rfm["Retention_Action"] = rfm.apply(assign_retention_action, axis=1)

    # Save all 3 artifacts
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(kmeans, f"{MODEL_DIR}/kmeans.pkl")
    joblib.dump(model,  f"{MODEL_DIR}/churn_model.pkl")

    print(f"\n✅ Artifacts saved to {MODEL_DIR}/")
    print("   scaler.pkl | kmeans.pkl | churn_model.pkl")

    # Save a sample output for reference
    rfm.to_csv(f"{MODEL_DIR}/rfm_output_sample.csv", index=False)
    print("   rfm_output_sample.csv\n")


if __name__ == "__main__":
    main()
