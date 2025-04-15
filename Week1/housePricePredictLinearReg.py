import gradio as gr
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Gradio interface function
def predict_house_price(*inputs):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    return f"Predicted House Price: ${prediction[0]*100000:.2f}"

# Inputs in the same order as feature_names
input_components = [
    gr.Slider(minimum=float(X[:, i].min()), maximum=float(X[:, i].max()), label=feature_names[i])
    for i in range(X.shape[1])
]

# Launch Gradio app
gr.Interface(
    fn=predict_house_price,
    inputs=input_components,
    outputs="text",
    title="California House Price Predictor",
    description="Predict house prices using Linear Regression on the California Housing dataset."
).launch()
