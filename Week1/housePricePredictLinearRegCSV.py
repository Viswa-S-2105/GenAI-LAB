import pandas as pd
import numpy as np
import gradio as gr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("house_prices.csv")

# Features and target
X = data[["Bedrooms", "Bathrooms", "SqFt", "Floors", "Age"]]
y = data["Price"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction function for Gradio
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
    title="Custom House Price Predictor",
    description="Predict house prices using a custom dataset and Linear Regression."
)

interface.launch()
