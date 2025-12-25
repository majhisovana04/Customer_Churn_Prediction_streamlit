import streamlit as st
import joblib
import numpy as np
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------------- BACKGROUND + STYLES ----------------
def add_bg_and_style():
    # ---------------- CUSTOM CSS ----------------
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 1.5rem;
    }
    .card {
        background: #161b22;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 0 25px rgba(0,0,0,0.6);
    }
    h1, h2, h3, label {
        color: #ffffff !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        font-size: 16px;
        background: linear-gradient(90deg, #2563eb, #1e40af);
    }
    footer {
        visibility: hidden;
    }
    .footer-text {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

add_bg_and_style()




# Load model once
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()


# ---------------- HEADER IMAGE ----------------
st.image("assets/background.jpg", use_container_width=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#9ca3af;'>Predict whether a customer is likely to leave the service</p>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)


# ---------------- USER INPUTS ----------------
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Has Partner?", ["No", "Yes"])
Dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes"])

Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])


MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)




# ---------------- ENCODING ----------------
binary_map = {"No": 0, "Yes": 1}
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

SeniorCitizen = binary_map[SeniorCitizen]
Partner = binary_map[Partner]
Dependents = binary_map[Dependents]
OnlineSecurity = binary_map[OnlineSecurity]
TechSupport = binary_map[TechSupport]
PaperlessBilling = binary_map[PaperlessBilling]
Contract = contract_map[Contract]




# ---------------- MODEL INPUT ----------------
input_data = np.array([[
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    OnlineSecurity,
    TechSupport,
    Contract,
    PaperlessBilling,
    MonthlyCharges,
    TotalCharges
]])




# ---------------- PREDICTION ----------------
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is NOT likely to churn (Probability: {probability:.2f})")


st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer-text">
    Built with ❤️ using Streamlit & Machine Learning <br>
    End-to-End ML Deployment
</div>
""", unsafe_allow_html=True)
