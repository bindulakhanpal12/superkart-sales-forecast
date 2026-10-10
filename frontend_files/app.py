import os
import pandas as pd
import requests
import streamlit as st

# Forwarded URL of the backend (set when the container is started)
API_URL = os.getenv("API_URL", "http://localhost:5000")

st.title("SuperKart Sales Forecast")

# ---------- Single prediction ----------
st.subheader("Online Prediction")

product_group = st.selectbox("Product Group", ["Food", "Non-Consumable", "Drinks"])
product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.5)
product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_area = st.number_input("Product Allocated Area (ratio)", min_value=0.0,
                               max_value=1.0, value=0.05, step=0.01)
product_type = st.selectbox("Product Type", [
    "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", "Household",
    "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks", "Breads",
    "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood",
])
product_mrp = st.number_input("Product MRP", min_value=1.0, max_value=500.0, value=147.0)
store_id = st.selectbox("Store", ["OUT001", "OUT002", "OUT003", "OUT004"])

group_codes = {"Food": "FD", "Non-Consumable": "NC", "Drinks": "DR"}

if st.button("Predict"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Id": store_id,
        "Product_Id_char": group_codes[product_group],
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
