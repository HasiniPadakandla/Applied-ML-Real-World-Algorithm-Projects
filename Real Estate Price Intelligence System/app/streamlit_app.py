import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/linear_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Real Estate Price Intelligence System")

area = st.number_input("Living Area (sq ft)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
garage = st.number_input("Garage Cars")
quality = st.number_input("Overall Quality (1-10)")
year = st.number_input("Year Built")

if st.button("Predict Price"):

    data = np.array([[area, bedrooms, bathrooms, garage, quality, year]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")