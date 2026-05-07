"""
frontend/app.py — Streamlit Dashboard
Customer Segmentation & Churn Prediction
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://api:8000"   # Docker service name; change to localhost for local dev

st.set_page_config(
    page_title = "Customer Intelligence Platform",
    page_icon  = "🎯",
    layout     = "wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .high-risk  { color: #e53935; font-weight: bold; }
    .med-risk   { color: #fb8c00; font-weight: bold; }
    .low-risk   { color: #43a047; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/customer-insight.png", width=72)
    st.title("Customer Intelligence")
    st.caption("Powered by RFM + ML")

    st.divider()
    api_url_override = st.text_input("API URL", value=API_URL)
    API_URL = api_url_override

    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except Exception:
        st.warning("⚠️ API Unreachable — using demo mode")

    st.divider()
    st.caption("Built by Diya Mathew ")
    st.caption("Stack: FastAPI · Streamlit · Docker · sklearn")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Customer", "📦 Batch Prediction", "📊 Segment Overview"])


# ── TAB 1: Single Prediction ──────────────────────────────────────────────────
with tab1:
    st.header("Single Customer Prediction")
    st.markdown("Enter a customer's RFM values to get their segment and churn risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        customer_id = st.text_input("Customer ID", value="C12345")
        recency     = st.slider("Recency (days since last purchase)", 0, 365, 30,
                                help="Lower = more recently active")

    with col2:
        frequency = st.slider("Frequency (unique invoices)", 1, 200, 10,
                              help="Higher = more loyal")
        monetary  = st.number_input("Monetary (total £ spent)", min_value=1.0,
                                    max_value=100000.0, value=500.0, step=50.0)

    if st.button("🚀 Predict", use_container_width=True, type="primary"):
        payload = {
            "customer_id": customer_id,
            "recency"    : recency,
            "frequency"  : frequency,
            "monetary"   : monetary,
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            st.divider()
            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.metric("Segment", result["segment"])
            with col_b:
                prob = result["churn_probability"]
                st.metric("Churn Probability", f"{prob:.0%}")
            with col_c:
                risk = result["churn_risk"]
                colour = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                st.metric("Risk Level", f"{colour[risk]} {risk}")
            with col_d:
                st.metric("Retention Action", result["retention_action"])

            # Churn gauge
            fig = go.Figure(go.Indicator(
                mode   = "gauge+number",
                value  = round(prob * 100, 1),
                title  = {"text": "Churn Risk Score"},
                number = {"suffix": "%"},
                gauge  = {
                    "axis" : {"range": [0, 100]},
                    "bar"  : {"color": "#e53935" if prob > 0.7 else "#fb8c00" if prob > 0.4 else "#43a047"},
                    "steps": [
                        {"range": [0,  40],  "color": "#e8f5e9"},
                        {"range": [40, 70],  "color": "#fff3e0"},
                        {"range": [70, 100], "color": "#ffebee"},
                    ],
                },
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("Could not reach the API. Make sure it's running.")
        except Exception as e:
            st.error(f"Error: {e}")


# ── TAB 2: Batch Prediction ───────────────────────────────────────────────────
with tab2:
    st.header("Batch Customer Prediction")
    st.markdown("Upload a CSV with columns: `customer_id, recency, frequency, monetary`")

    sample_csv = """customer_id,recency,frequency,monetary
C001,10,25,1500.00
C002,120,3,200.00
C003,5,80,8500.00
C004,300,1,50.00
C005,45,12,750.00"""

    st.download_button("⬇️ Download Sample CSV", sample_csv, "sample_customers.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            customers = df.to_dict(orient="records")
            payload   = {"customers": customers}

            with st.spinner("Predicting..."):
                try:
                    resp = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=30)
                    resp.raise_for_status()
                    results = resp.json()
                    results_df = pd.DataFrame(results)

                    st.success(f"✅ Predicted {len(results_df)} customers")

                    # Segment breakdown
                    col1, col2 = st.columns(2)
                    with col1:
                        seg_counts = results_df["segment"].value_counts().reset_index()
                        seg_counts.columns = ["Segment", "Count"]
                        fig = px.pie(seg_counts, values="Count", names="Segment",
                                     title="Segment Distribution",
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig = px.histogram(results_df, x="churn_probability", nbins=20,
                                           title="Churn Probability Distribution",
                                           color_discrete_sequence=["#e53935"])
                        st.plotly_chart(fig, use_container_width=True)

                    # Full results table
                    st.dataframe(
                        results_df[["customer_id","segment","churn_probability","churn_risk","retention_action"]],
                        use_container_width=True,
                    )

                    # Download
                    csv_out = results_df.to_csv(index=False)
                    st.download_button("⬇️ Download Results", csv_out, "predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"Error: {e}")


# ── TAB 3: Segment Overview ───────────────────────────────────────────────────
with tab3:
    st.header("Customer Segment Guide")
    st.markdown("Understanding the four segments and recommended actions.")

    segments = [
        {
            "name"   : "🏆 High-Value Loyal",
            "colour" : "#4CAF50",
            "rfm"    : "Low Recency · High Frequency · High Monetary",
            "desc"   : "Your most profitable customers. They buy often and spend a lot.",
            "action" : "VIP programmes, early access, loyalty rewards",
            "risk"   : "Low normally — act immediately if churn probability spikes",
        },
        {
            "name"   : "⚠️ At-Risk High-Value",
            "colour" : "#FF9800",
            "rfm"    : "High Recency · High Monetary",
            "desc"   : "Big spenders who have gone quiet recently.",
            "action" : "Personalised win-back campaigns, exclusive discounts",
            "risk"   : "Medium-High — every day counts",
        },
        {
            "name"   : "📈 Active Low-Value",
            "colour" : "#2196F3",
            "rfm"    : "Low Recency · Low Monetary",
            "desc"   : "Frequent buyers with smaller baskets — growth opportunity.",
            "action" : "Upsell bundles, cross-sell recommendations",
            "risk"   : "Low — focus on increasing basket size",
        },
        {
            "name"   : "💤 Inactive / Churned",
            "colour" : "#9E9E9E",
            "rfm"    : "Very High Recency · Low Frequency · Low Monetary",
            "desc"   : "Long-absent customers. Recovery rate is low.",
            "action" : "Low-cost re-engagement (email only), sunset if unresponsive",
            "risk"   : "Very High — already churned or near churned",
        },
    ]

    for seg in segments:
        with st.expander(f"{seg['name']}", expanded=True):
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                st.markdown(f"**RFM Profile**\n\n{seg['rfm']}")
                st.markdown(f"**Description**\n\n{seg['desc']}")
            with c2:
                st.markdown(f"**Recommended Action**\n\n{seg['action']}")
            with c3:
                st.markdown(f"**Churn Risk**\n\n{seg['risk']}")
