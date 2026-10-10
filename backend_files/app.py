import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask("SuperKart Sales Forecast")

# Load the serialized model pipeline once, at startup
model = joblib.load("superkart_sales_model_v1.joblib")

# Columns the model was trained on
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_Type", "Product_MRP", "Store_Id",
    "Product_Id_char", "Product_Type_Category",
]

PERISHABLES = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"]


def prepare_features(df):
    """Apply the same cleaning and feature engineering used in training."""
    df = df.copy()
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace("reg", "Regular")
    if "Product_Id_char" not in df.columns:
        df["Product_Id_char"] = df["Product_Id"].str[:2]
    df["Product_Type_Category"] = df["Product_Type"].apply(
        lambda x: "Perishables" if x in PERISHABLES else "Non Perishables"
    )
    return df[FEATURES]


@app.get("/")
def home():
    return "Welcome to the SuperKart Sales Forecast API!"


# Single prediction: JSON in, JSON out
@app.post("/v1/predict")
def predict_sales():
    data = request.get_json()
    input_df = prepare_features(pd.DataFrame([data]))
    prediction = model.predict(input_df)[0]
    return jsonify({"Predicted_Sales": round(float(prediction), 2)})


# Batch prediction: CSV file in, JSON list out
@app.post("/v1/predictbatch")
def predict_sales_batch():
    file = request.files["file"]
    raw_df = pd.read_csv(file)
    predictions = model.predict(prepare_features(raw_df))
    return jsonify({"Predicted_Sales": [round(float(p), 2) for p in predictions]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
