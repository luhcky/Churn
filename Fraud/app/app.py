import streamlit as st
import pandas as pd, numpy as np, joblib
page_bg ="""
<style>
[data-testid= "stAppViewContainer"]{
    background-color: blue;}
    [data-testid="stHeader"]{
        background-color: rgba(0,0,0,0);
        }
        [data-testid="stSidebar"]{
            background: linear-gradient(180deg, #0D1B2A 0%, #1B263B 100%);
            border-right: 1px solid #00B4D8;
        }
        h1, h2, h3,label,p,.stMarkdown {
            color:#FFFFFF
            }
            alert:#E63946
            accent:#00B4D8
        .stButton>button{
            background: linear-gradient(90deg, #0288D1, #26C6DA);
            color:white;
            border-radius:10px;
            font-weight:600;
        </style>
        """
st.set_page_config(page_title='Fraud Detector', page_icon='💳', layout='wide')
@st.cache_resource
def load_artifacts():
    return (joblib.load('models/fraud_model.pkl'),
            joblib.load('models/fraud_scaler.pkl'),
            joblib.load('models/fraud_feature_names.pkl'))
model, scaler, feature_names = load_artifacts()
st.title('💳SecurePay — Fraud Detection System')
st.markdown('Real-time transaction fraud scoring.')
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.subheader('Transaction Details')
    amount   = st.number_input('Transaction Amount (EUR)', 0.01, 25000.0, 50.0)
    hour     = st.slider('Hour of Day', 0, 23, 14)
    off_hours= int(hour <= 5)
    log_amount = np.log1p(amount)
with col2:
    st.subheader('PCA Features (V1-V5 sample)')
    st.caption('In production these come from the payment processor automatically.')
    v1 = st.number_input('V1', -10.0, 10.0, 0.0, step=0.1)
    v2 = st.number_input('V2', -10.0, 10.0, 0.0, step=0.1)
    v3 = st.number_input('V3', -10.0, 10.0, 0.0, step=0.1)
    v4 = st.number_input('V4', -5.0, 5.0, 0.0, step=0.1)
    v14= st.number_input('V14 (key feature)', -10.0, 10.0, 0.0, step=0.1)
threshold = st.sidebar.slider('Detection Threshold', 0.1, 0.9, 0.05, 0.05)
st.sidebar.caption('Lower = more sensitive (higher recall, more false positives)')
if st.button('Score Transaction', type='primary', use_container_width=True):

    row = {f: 0.0 for f in feature_names}
    row.update({
        'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V14': v14,
        'Log_Amount': log_amount, 'Hour': hour, 'OffHours': off_hours,
    })
    X = pd.DataFrame([row])[feature_names]
    prob = model.predict_proba(scaler.transform(X))[0][1]
    flagged = prob >= threshold
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric('Fraud Probability', f'{prob*100:.2f}%')
    c2.metric('Threshold', f'{threshold*100:.0f}%')
    c3.metric('Decision', 'BLOCK' if flagged else 'APPROVE')
    if flagged:
        st.error(f'FRAUD ALERT — Transaction flagged for review (score: {prob:.3f})')
        st.markdown('**Actions:** Block transaction, notify cardholder, escalate to fraud team.')
    else:
        st.success(f'Transaction approved (fraud score: {prob:.3f})')
    
    risk = '🚨Critical' if prob>0.08 else '⚠️High' if prob>0.06 else 'Medium' if prob>0.03 else '✅Low'
    st.info(f'Risk Level: {risk} | Off-Hours: {"Yes" if off_hours else "No"} | Amount: EUR{amount:.2f}')