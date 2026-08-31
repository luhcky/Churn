import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)
st.markdown("""
            <style>
            .stApp{
                background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
                background-attachment:fixed;
            }
            [data-testid="stSidebar"]{
                background:rgba(15, 12, 41, 0.85) !important;
                border-right: 1px solid rgba(255,255,255,0.1);
            }
            [data-testid="stMetric],[data-testid="stExpander"],[data-testid="stDataFrame"]{
                background: rgba(255,255,255,0.06) !important;
                backdrop-filter:blur(12px);
                border: 1px solid rgba(255,255,255,0.1);
                border_radius: 15px;
            }
            h1,h2,h3,p,label, .stMarkdown{color:#FFFFFF !important;}
            .stButton>button{
                background: linear-gradient(90deg, #667eea,#764ba2);
                color: white; border-radius: 12px; border:none;
                font-weight:bold;
            }
            .stButton.button:hover{transform:scale(1.02);}
            </style>
            """, unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8001"

st.title("📉 TelcoNova Customer Churn Prediction System")

# API health check 
try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    shap_ok = h.get("shap_available", False)
    st.success(
        f"✅ API connected  "
        f"SHAP: {'✓ enabled' if shap_ok else '✗ install shap'}"
    )
except Exception:
    st.error(
        "⚠ API not running. Start it first:\n"
        "`uvicorn api.main:app --reload --port 8001`"
    )
    st.stop()

st.divider()

#  Sidebar inputs 
st.sidebar.header("Customer Profile")

tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 2)
monthly        = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 75.5)
total          = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 150.0)
contract       = st.sidebar.selectbox("Contract",
                    ["Month-to-month", "One year", "Two year"])
payment        = st.sidebar.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"])
internet       = st.sidebar.selectbox("Internet Service",
                    ["Fiber optic", "DSL", "No"])
online_sec     = st.sidebar.selectbox("Online Security", ["No", "Yes"])
tech_support   = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior         = st.sidebar.selectbox("Senior Citizen",
                    [0, 1], format_func=lambda x: "Yes" if x else "No")
partner        = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents     = st.sidebar.selectbox("Dependents", ["No", "Yes"])
online_backup  = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
device_prot    = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
streaming_tv   = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
streaming_mov  = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])

predict_btn = st.sidebar.button(
    "🔍 Predict + Explain", type="primary", use_container_width=True
)

# Predict + Explain
if predict_btn:
    payload = {
        "tenure"           : tenure,
        "MonthlyCharges"   : monthly,
        "TotalCharges"     : total,
        "Contract"         : contract,
        "PaymentMethod"    : payment,
        "InternetService"  : internet,
        "OnlineSecurity"   : online_sec,
        "TechSupport"      : tech_support,
        "PaperlessBilling" : paperless,
        "SeniorCitizen"    : senior,
        "Partner"          : partner,
        "Dependents"       : dependents,
        "OnlineBackup"     : online_backup,
        "DeviceProtection" : device_prot,
        "StreamingTV"      : streaming_tv,
        "StreamingMovies"  : streaming_mov,
        "MultipleLines"    : multiple_lines,
    }

    with st.spinner("Scoring and generating SHAP explanation..."):
        try:
            # Call /predict/explain — returns prediction + SHAP in one shot
            r = requests.post(
                f"{API_URL}/predict/explain",
                json=payload,
                timeout=15
            )
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.json()}")
                st.stop()

            data = r.json()

        except requests.exceptions.ConnectionError:
            st.error("Lost connection to API.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out — SHAP can take up to 15 seconds.")
            st.stop()

    prob  = data["churn_probability"]
    tier  = data["risk_tier"]
    churn = data["will_churn"]

    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Probability", f"{prob*100:.1f}%",
              delta=f"{(prob - 0.265)*100:+.1f}% vs baseline")
    c2.metric("Risk Tier",  tier)
    c3.metric("Decision",   "⚠️Will Churn" if churn else "✅ Will Stay")
    c4.metric("Response",   f"{data['processing_ms']}ms")

    # Alert banner
    if tier == "HIGH":
        st.error(f"⚠️ HIGH CHURN RISK — {data['recommendation']}")
    elif tier == "MEDIUM":
        st.warning(f"🟡 MEDIUM RISK — {data['recommendation']}")
    else:
        st.success(f"✅ LOW RISK — {data['recommendation']}")

    st.progress(min(prob, 1.0), text=f"Churn score: {prob*100:.1f}%")
    st.divider()

    # Two columns: SHAP chart + signals 
    col_shap, col_signals = st.columns([1.4, 1])

    with col_shap:
        st.subheader("🧠 SHAP Explanation")
        st.caption(data.get("shap_explanation", ""))

        # Build SHAP bar chart from API response
        top_drivers = data.get("shap_top_drivers", [])

        if top_drivers:
            features = [d["feature"] for d in top_drivers]
            values   = [d["shap_value"] for d in top_drivers]
            colors   = ["#EF4444" if v > 0 else "#3B82F6" for v in values]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(
                range(len(features)),
                values,
                color=colors,
                edgecolor="white",
                height=0.6,
            )
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP Value (impact on churn probability)", fontsize=9)
            ax.set_title("Top SHAP Drivers — This Customer", fontsize=11, fontweight="bold")
            ax.invert_yaxis()

            # Add value labels on bars
            for bar, val in zip(bars, values):
                xpos = val + 0.002 if val >= 0 else val - 0.002
                ax.text(xpos, bar.get_y() + bar.get_height()/2,
                        f"{val:+.3f}",
                        va="center", ha="left" if val >= 0 else "right",
                        fontsize=8, color="#1F2937")

            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.caption(
                "🔴 Red bars push towards **churn**. "
                "🔵 Blue bars push towards **staying**."
            )
            st.caption(data.get("shap_note", ""))
        else:
            st.info("SHAP values not available — check API logs.")

    with col_signals:
        st.subheader("⚡ Churn Signals")
        for signal in data.get("churn_signals", []):
            st.warning(f"▸ {signal}")

        st.subheader("🛡️ Protective Factors")
        for prot in data.get("protective_signals", []):
            st.success(f"▸ {prot}")

        st.divider()
        st.subheader("📋 Recommended Action")
        if tier == "HIGH":
            st.error(
                "**Immediate retention offer:**\n"
                "- Offer contract upgrade discount\n"
                "- Call within 48 hours\n"
                "- Escalate to retention team"
            )
        elif tier == "MEDIUM":
            st.warning(
                "**Proactive check-in:**\n"
                "- Schedule customer satisfaction call\n"
                "- Review billing and service issues\n"
                "- Consider loyalty incentive"
            )
        else:
            st.success(
                "**No immediate action needed.**\n"
                "- Include in standard NPS survey\n"
                "- Maintain normal contact cadence"
            )

else:
    # ── Default state ─────────────────────────────────────
    st.info("👈 Fill in the customer profile and click **Predict + Explain**.")
    