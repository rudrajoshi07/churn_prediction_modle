"""
Streamlit dashboard: live churn-risk scoring with SHAP explainability.
Run with:  streamlit run churn_dashboard_app.py
(Requires churn_model.pkl, encoders.pkl, feature_columns.pkl from
 churn_model_training.py to be in the same folder.)
"""

import streamlit as st          # builds the web UIpip install pandas numpy scikit-learn xgboost imbalanced-learn shap streamlit matplotlib joblib
import pandas as pd             # dataframe handling
import numpy as np              # numeric ops
import joblib                   # loads the saved model/encoders
import shap                     # per-customer explainability
import matplotlib.pyplot as plt # renders the SHAP plot

# ---------------------------------------------------------------------------
# 1. PAGE CONFIG & LOAD ARTIFACTS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Risk Dashboard", layout="wide")

model = joblib.load("churn_model.pkl")               # the trained XGBoost model
encoders = joblib.load("encoders.pkl")                 # dict of LabelEncoders per category column
feature_columns = joblib.load("feature_columns.pkl")   # exact column order the model expects

st.title("📉 Customer Churn Risk Dashboard")
st.caption("Enter a customer's details to get a live churn-risk score and see what's driving it.")


# ---------------------------------------------------------------------------
# 2. SIDEBAR INPUTS  (one widget per raw feature the model needs)
# ---------------------------------------------------------------------------
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
contract = st.sidebar.selectbox("Contract", encoders["Contract"].classes_)
internet_service = st.sidebar.selectbox("Internet Service", encoders["InternetService"].classes_)
tech_support = st.sidebar.selectbox("Tech Support", encoders["TechSupport"].classes_)
payment_method = st.sidebar.selectbox("Payment Method", encoders["PaymentMethod"].classes_)

total_charges = monthly_charges * (tenure + 1)          # same formula used at training time


# ---------------------------------------------------------------------------
# 3. REBUILD THE ENGINEERED FEATURES  (must match training exactly)
# ---------------------------------------------------------------------------
tenure_bucket_raw = pd.cut(
    [tenure], bins=[-1, 12, 24, 48, 72], labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"]
)[0]

row = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "PaymentMethod": payment_method,
    "TenureBucket": tenure_bucket_raw,
    "AvgMonthlySpendRatio": (total_charges / (tenure + 1)) / monthly_charges,
    "ContractRiskScore": {"Month-to-month": 2, "One year": 1, "Two year": 0}[contract],
}])

# Apply the SAME label encoders used during training
for col in ["Contract", "InternetService", "TechSupport", "PaymentMethod", "TenureBucket"]:
    row[col] = encoders[col].transform(row[col])

row = row[feature_columns]                               # enforce identical column order


# ---------------------------------------------------------------------------
# 4. PREDICT CHURN RISK
# ---------------------------------------------------------------------------
churn_prob = model.predict_proba(row)[0, 1]               # probability this customer churns

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Churn Risk", f"{churn_prob*100:.1f}%")
    if churn_prob > 0.5:
        st.error("⚠️ High risk — recommend retention outreach")
    else:
        st.success("✅ Low risk")


# ---------------------------------------------------------------------------
# 5. SHAP EXPLAINABILITY FOR THIS SPECIFIC CUSTOMER
# ---------------------------------------------------------------------------
with col2:
    st.subheader("What's driving this score?")
    explainer = shap.TreeExplainer(model)                  # same explainer type as training script
    shap_values = explainer.shap_values(row)                # per-feature contribution for this customer

    fig, ax = plt.subplots(figsize=(6, 3))
    shap.summary_plot(
        shap_values, row, plot_type="bar", show=False, plot_size=None
    )
    st.pyplot(fig)                                           # render the SHAP bar chart in the app


st.divider()
st.caption(
    "Model: tuned XGBoost classifier trained with SMOTE-balanced data. "
    "Retrain via churn_model_training.py on your own customer export to update this dashboard."
)
