import gradio as gr
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Train decision tree classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 3. Prediction function
def predict_tree(sepal_length, sepal_width, petal_length, petal_width):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=iris.feature_names)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]
    class_name = iris.target_names[prediction]
    return f"Prediction: {class_name} (Confidence: {probability:.2f})"

# 4. Gradio UI
interface = gr.Interface(
    fn=predict_tree,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)"),
    ],
    outputs="text",
    title="Decision Tree Classifier",
    description="Predict Iris flower species using a Decision Tree model."
)

# 5. Launch
interface.launch()
