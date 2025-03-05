import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

models = {
   "logistic_model" : joblib.load('C:\\Users\\HP\\Desktop\\Project\\CODE TEC INTERNSHIP\\MACHINE LEARNING  task\\logistic_model.pkl'),
   "random_forest_model": joblib.load("C:\\Users\\HP\\Desktop\\Project\\CODE TEC INTERNSHIP\\MACHINE LEARNING  task\\random_forest_model.pkl"),
}

# Load the scaler used during training
scaler = joblib.load("C:\\Users\\HP\\Desktop\\Project\\CODE TEC INTERNSHIP\\MACHINE LEARNING  task\\scaler.pkl")


# Title
st.title("🏦 Loan Approval Prediction")

# Sidebar
st.sidebar.header("🔹 Enter Loan Details")

# Define feature names in the required order
sidebar_features = [
    'no_of_dependents', 'income_annum',
    'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
    'luxury_assets_value', 
]

# User input collection
user_input = {}

# Numerical and categorical inputs
user_input['no_of_dependents'] = st.sidebar.number_input("No. of Dependents", min_value=0, step=1)
user_input['income_annum'] = st.sidebar.number_input("Annual Income", min_value=0.0, step=1000.0)
user_input['loan_amount'] = st.sidebar.number_input("Loan Amount", min_value=0.0, step=1000.0)
user_input['loan_term'] = st.sidebar.number_input("Loan Term (Months)", min_value=0, step=1)
user_input['cibil_score'] = st.sidebar.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
user_input['residential_assets_value'] = st.sidebar.number_input("Residential Assets Value", min_value=0.0, step=1000.0)
user_input['luxury_assets_value'] = st.sidebar.number_input("Luxury Assets Value", min_value=0.0, step=1000.0)


# Convert to DataFrame
input_data = pd.DataFrame([user_input])

# Ensure correct feature order before scaling
correct_feature_order = list(scaler.feature_names_in_)
input_data = input_data[correct_feature_order]

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Convert scaled data back to DataFrame with correct feature names
input_data_scaled = pd.DataFrame(input_data_scaled, columns=correct_feature_order)

# Model selection
model_choice = st.sidebar.selectbox("🔍 Choose Model:", list(models.keys()))
model = models[model_choice]

# Predict loan approval
if st.button("🚀 Predict Loan Approval"):
    prediction = model.predict(input_data_scaled)[0]
    
    # Mapping: 0 -> Approved, 1 -> Not Approved
    result = "✅ Approved" if prediction == 0 else "❌ Not Approved"
    
    st.subheader(f"Loan Status: {result}")
