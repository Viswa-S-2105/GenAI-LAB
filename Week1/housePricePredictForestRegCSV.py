import pandas as pd
import numpy as np
import gradio as gr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load CSV
data = pd.read_csv("house_prices.csv")

# Features and Target
X = data[["Bedrooms", "Bathrooms", "SqFt", "Floors", "Age"]]
y = data["Price"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Gradio Prediction Function
def predict_price(bedrooms, bathrooms, sqft, floors, age):
    input_data = np.array([[bedrooms, bathrooms, sqft, floors, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return f"Predicted House Price: ${prediction[0]:,.2f}"

# Gradio UI
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(0, 10, step=1, label="Bedrooms"),
        gr.Slider(0, 10, step=1, label="Bathrooms"),
        gr.Slider(100, 10000, step=50, label="SqFt"),
        gr.Slider(1, 4, step=1, label="Floors"),
        gr.Slider(0, 100, step=1, label="Age"),
    ],
    outputs="text",
    title="Custom House Price Predictor - Random Forest",
    description="Predict house prices using Random Forest Regression on your custom dataset."
)

interface.launch()
