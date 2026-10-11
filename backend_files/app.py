import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask("SuperKart Sales Forecast")

# Load the serialized model pipeline once, at startup
model = joblib.load("superkart_sales_model_v1.joblib")

# Columns the model was trained on, in training order
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Product_Type_Category", "Store_Age_Years",
]


@app.get("/")
def home():
    return "Welcome to the SuperKart Sales Forecast API!"


# Single prediction: JSON in, JSON out
@app.post("/v1/predict")
def predict_sales():
    data = request.get_json()
    input_df = pd.DataFrame([data])[FEATURES]
    prediction = model.predict(input_df)[0]
    return jsonify({"Predicted_Sales": round(float(prediction), 2)})


# Batch prediction: CSV file in, JSON list out
@app.post("/v1/predictbatch")
def predict_sales_batch():
    file = request.files["file"]
    input_df = pd.read_csv(file)[FEATURES]
    predictions = model.predict(input_df)
    return jsonify({"Predicted_Sales": [round(float(p), 2) for p in predictions]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
