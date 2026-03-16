import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

model = joblib.load("models/linear_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/features.pkl")

st.title("Real Estate Price Intelligence System")

area = st.number_input("Living Area (sq ft)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
garage = st.number_input("Garage Cars")
quality = st.slider("Overall Quality",1,10)
year = st.number_input("Year Built")

locality = st.selectbox(
    "Select Locality",
    ["NAmes","CollgCr","OldTown","Edwards","Somerst","NridgHt"]
)

currency = st.radio(
    "Select Currency",
    ["USD ($)", "INR (₹)"]
)

if st.button("Predict Price"):

    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0

    input_data["Gr Liv Area"] = area
    input_data["Bedroom AbvGr"] = bedrooms
    input_data["Full Bath"] = bathrooms
    input_data["Garage Cars"] = garage
    input_data["Overall Qual"] = quality
    input_data["Year Built"] = year

    col_name = f"Neighborhood_{locality}"
    if col_name in input_data.columns:
        input_data[col_name] = 1

    data_scaled = scaler.transform(input_data)

    prediction = model.predict(data_scaled)[0]

    if currency == "INR (₹)":
        prediction = prediction * 83
        st.success(f"Estimated Price: ₹ {prediction:,.0f}")
    else:
        st.success(f"Estimated Price: $ {prediction:,.0f}")

# Data Visualization

df = pd.read_csv("data/housing.csv")

st.subheader("Average House Price by Locality")

locality_price = df.groupby("Neighborhood")["SalePrice"].mean().reset_index()

fig = px.bar(
    locality_price,
    x="Neighborhood",
    y="SalePrice",
    title="Average House Price by Locality"
)

st.plotly_chart(fig)

#price distribution by locality

st.subheader("Price Distribution by Locality")

fig2 = px.box(
    df,
    x="Neighborhood",
    y="SalePrice",
    title="Price Distribution Across Localities"
)

st.plotly_chart(fig2)

# feature importance

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.coef_
})

importance_df = importance_df.sort_values(by="Importance")

# Increase figure size
fig, ax = plt.subplots(figsize=(10, 14))

ax.barh(importance_df["Feature"], importance_df["Importance"])

ax.set_title("Feature Importance for House Price Prediction")

ax.set_xlabel("Coefficient Value")

plt.tight_layout()

st.pyplot(fig)

# Geographical Visualization of Average Prices

locality_coords = {
    "NAmes": (42.0347, -93.6200),
    "CollgCr": (42.0308, -93.6319),
    "OldTown": (42.0253, -93.6135),
    "Edwards": (42.0205, -93.6422),
    "Somerst": (42.0451, -93.6412),
    "NridgHt": (42.0520, -93.6487)
}

df["lat"] = df["Neighborhood"].map(lambda x: locality_coords.get(x, (None, None))[0])
df["lon"] = df["Neighborhood"].map(lambda x: locality_coords.get(x, (None, None))[1])

# Heatmap of Average Prices by Locality

st.subheader("Real Estate Price Heatmap")

map_data = df.groupby("Neighborhood").agg({
    "SalePrice": "mean",
    "lat": "first",
    "lon": "first"
}).reset_index()

fig = px.scatter_mapbox(
    map_data,
    lat="lat",
    lon="lon",
    size="SalePrice",
    color="SalePrice",
    hover_name="Neighborhood",
    zoom=11,
    mapbox_style="carto-positron",
    title="Average House Price by Locality"
)

st.plotly_chart(fig)