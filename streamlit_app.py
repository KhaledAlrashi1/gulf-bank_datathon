import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ----------------- Load artifacts -----------------
ROOT_DIR = Path(__file__).resolve().parent
artifacts_path = ROOT_DIR / "models" / "credit_risk_model.pkl"

artifacts = joblib.load(artifacts_path)
model = artifacts["model"]
feature_names = artifacts["feature_names"]
cat_cols = artifacts["cat_cols"]

st.set_page_config(page_title="Credit Risk Demo", layout="centered")

st.title("📊 Credit Risk Prediction Demo")
st.markdown(
    """
    Enter a sample loan application and see the predicted **default risk**.
    This is a prototype – not for real credit decisions.
    """
)

# ----------------- UI: adjust field names to your actual columns -----------------
# Below I’m using the typical Kaggle credit_risk_dataset column names.
# If your processed file uses different names, change the keys in `input_data` accordingly.

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=90, value=35)
    person_income = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
    person_emp_length = st.number_input("Employment length (years)", min_value=0, max_value=40, value=3)

with col2:
    loan_amnt = st.number_input("Loan Amount", min_value=500, value=10000, step=500)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0)
    loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.2)

loan_intent = st.selectbox(
    "Loan intent",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
)

loan_grade = st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"])

person_home_ownership = st.selectbox(
    "Home ownership",
    ["OWN", "MORTGAGE", "RENT", "OTHER"],
)

cb_person_default_on_file = st.selectbox(
    "Previous default on file",
    ["Y", "N"],
)

cb_person_cred_hist_length = st.number_input(
    "Credit history length (years)", min_value=0, max_value=50, value=5
)

# ----------------- Build raw input row -----------------
input_data = pd.DataFrame(
    {
        "person_age": [person_age],
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "person_emp_length": [person_emp_length],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [cb_person_default_on_file],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
    }
)

# IMPORTANT:
# The keys above MUST match your original column names in df_baseline
# (before get_dummies). If your dataset uses different names, adjust here
# AND make sure they were the ones used when you trained df_model.


def preprocess_for_model(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing as in 02_modeling:
    - pd.get_dummies on cat_cols with drop_first=True
    - align columns to feature_names and fill missing with 0
    """
    df = df_raw.copy()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Ensure all expected feature columns exist
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Drop any unexpected columns
    df_encoded = df_encoded[feature_names]

    return df_encoded


# ----------------- Prediction button -----------------
if st.button("Predict Default Risk"):
    X_input = preprocess_for_model(input_data)

    proba_default = model.predict_proba(X_input)[0, 1]
    proba_repaid = 1 - proba_default

    if proba_default >= 0.7:
        risk_label = "High risk"
    elif proba_default >= 0.4:
        risk_label = "Medium risk"
    else:
        risk_label = "Low risk"

    st.markdown("---")
    st.subheader("Prediction")

    st.metric("Probability of Default", f"{proba_default*100:.1f}%")
    st.write(f"**Risk category:** {risk_label}")
    st.progress(float(proba_default))

    with st.expander("Details"):
        st.write(f"Probability of repaid: **{proba_repaid*100:.1f}%**")
        st.write("Prediction based on the trained XGBoost credit risk model.")