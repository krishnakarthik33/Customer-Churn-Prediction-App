import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# -------------------------
# Custom Dark Styling
# -------------------------
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1 {color: #00BFFF;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction App")
st.write("Fill the details below to predict churn probability.")

# -------------------------
# Input Section
# -------------------------
col1, col2 = st.columns(2)

countries = [
    "France", "Germany", "Spain",
    "India", "USA", "Canada",
    "Australia", "Brazil",
    "UK", "UAE", "Singapore"
]

with col1:
    geography = st.selectbox("🌍 Geography", countries)
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    age = st.slider("🎂 Age", 18, 92, 30)
    credit_score = st.number_input("💳 Credit Score", value=600)

with col2:
    balance = st.number_input("💰 Balance", value=0.0)
    estimated_salary = st.number_input("💼 Estimated Salary", value=50000.0)
    tenure = st.slider("⏳ Tenure", 0, 10, 5)
    num_products = st.slider("📦 Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
    is_active = st.selectbox("🟢 Is Active Member", [0, 1])

# -------------------------
# Prepare Data
# -------------------------
if st.button("🚀 Predict Churn"):

    gender = encoder.transform([gender])[0]

    input_dict = {
        "CreditScore": credit_score,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary
    }

    # Add geography columns dynamically
    for col in feature_columns:
        if col.startswith("Geography_"):
            input_dict[col] = 1 if col == f"Geography_{geography}" else 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # -------------------------
    # Gauge Chart
    # -------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
        }
    ))

    st.plotly_chart(fig)

    if prediction == 1:
        st.error("⚠ High Risk: Customer Likely to Churn")
    else:
        st.success("✅ Low Risk: Customer Likely to Stay")

    # -------------------------
    # Feature Importance
    # -------------------------
    st.subheader("📊 Feature Importance")

    importance = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(feature_imp_df.set_index("Feature"))
