import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ABC Ltd Credit Risk", layout="centered")
st.title("ABC Ltd – Client Credit Risk Checker")
st.write("Simple tool for managers to check default risk of clients.")

# Load the saved model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.sidebar.header("Enter Client Details")

# Create simple input boxes (one for each feature)
inputs = []
for feat in features:
    value = st.sidebar.number_input(f"{feat}", value=0.0)
    inputs.append(value)

if st.sidebar.button("Check Risk"):
    # Make prediction
    data = pd.DataFrame([inputs], columns=features)
    data_scaled = scaler.transform(data)
    
    prob = model.predict_proba(data_scaled)[0][1]
    pred = model.predict(data_scaled)[0]
    
    st.subheader("Result")
    st.metric("Default Probability", f"{prob:.1%}")
    
    if pred == 1:
        st.error("Higher risk – recommend careful review")
    else:
        st.success("Lower risk – standard terms possible")
    
    st.progress(min(float(prob), 1.0))
    st.caption("Basic logistic regression model. Always use with professional judgment.")

st.markdown("---")
st.write("Built for ABC Ltd managers • Personal finance / private wealth risk screening")