import streamlit as st
import json
import numpy as np
import joblib

# Load trained model
model = joblib.load("rfr_model.pkl")

# Load category mappings
with open("category_mappings.json", "r") as f:
    mappings = json.load(f)

# ===== MODEL METRICS (from training) =====
MODEL_R2 = 0.804
MODEL_MAE = 593149

# App Title
st.title("🚗 Used Car Price Prediction (Pakistan)")
st.write("Enter car details to predict price")

# Display model accuracy
st.subheader("📊 Model Performance")
st.write(f"**R² Score:** {MODEL_R2}")
st.write(f"**Mean Absolute Error:** Rs {MODEL_MAE:,.0f}")

st.divider()

# ===== USER INPUTS =====
year = st.number_input("Model Year", min_value=1990, max_value=2025, step=1)
mileage = st.number_input("Mileage (km)", min_value=0, step=1000)
engine = st.number_input("Engine Displacement (cc)", min_value=600, step=100)

make = st.selectbox("Car Make", list(mappings["make"].keys()))
model_name = st.selectbox("Car Model", list(mappings["model"].keys()))
city = st.selectbox("City", list(mappings["city"].keys()))

# Encode categorical inputs
make_encoded = mappings["make"][make]
model_encoded = mappings["model"][model_name]
city_encoded = mappings["city"][city]

# ===== PREDICTION =====
if st.button("Predict Price"):
    input_data = np.array([[year, mileage, engine,
                            make_encoded, model_encoded, city_encoded]])

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Price: Rs {prediction[0]:,.0f}")
