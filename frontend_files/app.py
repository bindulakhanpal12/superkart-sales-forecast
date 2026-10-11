import os
import pandas as pd
import requests
import streamlit as st

# Forwarded URL of the backend (set when the container is started)
API_URL = os.getenv("API_URL", "http://localhost:5000")

st.title("SuperKart Sales Forecast")

# ---------- Single prediction ----------
st.subheader("Online Prediction")

product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.66)
product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_area = st.number_input("Product Allocated Area (ratio)", min_value=0.0,
                               max_value=1.0, value=0.027, step=0.001, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=1.0, max_value=500.0, value=117.08)
product_id_char = st.selectbox("Product Group", ["FD", "NC", "DR"])
product_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_city = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1",
                                         "Supermarket Type2", "Food Mart"])
store_age = st.number_input("Store Age (years)", min_value=0, max_value=100, value=16)

if st.button("Predict"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_city,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Product_Type_Category": product_category,
        "Store_Age_Years": store_age,
    }
    response = requests.post(f"{API_URL}/v1/predict", json=payload)
    if response.status_code == 200:
        st.success(f"Predicted Sales: {response.json()['Predicted_Sales']}")
    else:
        st.error(f"Error {response.status_code}: {response.text}")

# ---------- Batch prediction ----------
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None and st.button("Predict Batch"):
    response = requests.post(f"{API_URL}/v1/predictbatch",
                             files={"file": uploaded_file.getvalue()})
    if response.status_code == 200:
        result = pd.read_csv(uploaded_file)
        result["Predicted_Sales"] = response.json()["Predicted_Sales"]
        st.dataframe(result)
    else:
        st.error(f"Error {response.status_code}: {response.text}")
