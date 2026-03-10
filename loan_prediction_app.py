import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load Loan Prediction Pipeline
# -------------------------------
pipeline = joblib.load(r'C:\Users\Bhawna\OneDrive\Desktop\Loan\loan_pipeline.pkl')
model = pipeline["model"]
scaler = pipeline["scaler"]

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("💰 Loan Prediction App")
st.write("Enter applicant details to predict loan approval.")

# Input fields
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income", min_value=0, step=1000)
loan_amount = st.number_input("Loan Amount", min_value=0, step=1000)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 60, 120])
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0, step=1000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, step=1000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, step=1000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, step=1000)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Loan Approval"):
    # Manual mapping for categorical features
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0

    # Numeric features to scale
    numeric_features = np.array([[no_of_dependents, income_annum, loan_amount, loan_term,
                                  cibil_score, residential_assets_value, commercial_assets_value,
                                  luxury_assets_value, bank_asset_value]], dtype=float)

    # Scale only numeric features
    scaled_numeric = scaler.transform(numeric_features)

    # Concatenate categorical values back (reshape to 2D before hstack)
    categorical_features = np.array([[education_val, self_employed_val]], dtype=float)

    input_data = np.hstack([scaled_numeric, categorical_features])

    # Predict
    prediction = model.predict(input_data)

    # Output
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")