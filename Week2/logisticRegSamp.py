import gradio as gr
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Load sample dataset (Iris)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = (iris.target == 0).astype(int)  # Binary classification: Class 0 vs rest

# 2. Train a logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Define prediction function
def predict_logistic(sepal_length, sepal_width, petal_length, petal_width):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=iris.feature_names)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return f"Prediction: {'Iris-setosa' if prediction == 1 else 'Not Iris-setosa'} (Probability: {probability:.2f})"

# 4. Build Gradio Interface
interface = gr.Interface(
    fn=predict_logistic,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)"),
    ],
    outputs="text",
    title="Logistic Regression Classifier",
    description="Predict whether the Iris flower is Setosa using Logistic Regression."
)

# 5. Launch on local system
interface.launch()
