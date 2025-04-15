import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load Iris data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Simplify to binary for Logistic Regression decision boundary
binary_mask = (y == 0) | (y == 1)
X_binary = X[binary_mask]
y_binary = y[binary_mask]

# Train models
X_train_lr, _, y_train_lr, _ = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)
model_lr = LogisticRegression()
model_lr.fit(X_train_lr, y_train_lr)

X_train_dt, _, y_train_dt, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model_dt = DecisionTreeClassifier(max_depth=3)
model_dt.fit(X_train_dt, y_train_dt)

# Logistic regression prediction
def predict_and_plot(sepal_length, sepal_width, petal_length, petal_width):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Logistic Regression (binary)
    lr_pred = model_lr.predict(input_data[:1])[0]
    lr_proba = model_lr.predict_proba(input_data[:1])[0][lr_pred]
    lr_result = f"Logistic Regression → Class: {iris.target_names[lr_pred]} (Conf: {lr_proba:.2f})"

    # Decision Tree
    dt_pred = model_dt.predict(input_data)[0]
    dt_proba = model_dt.predict_proba(input_data)[0][dt_pred]
    dt_result = f"Decision Tree → Class: {iris.target_names[dt_pred]} (Conf: {dt_proba:.2f})"

    # Visual: Decision Tree plot
    fig_tree, ax1 = plt.subplots(figsize=(6, 4))
    plot_tree(model_dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, ax=ax1)
    plt.tight_layout()
    tree_path = "tree_plot.png"
    fig_tree.savefig(tree_path)
    plt.close(fig_tree)

    # Visual: Logistic Regression 2D decision boundary (using PCA for visualization)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_binary)
    model_vis = LogisticRegression()
    model_vis.fit(X_pca, y_binary)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig_lr, ax2 = plt.subplots(figsize=(6, 4))
    ax2.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_binary, palette="Set1", ax=ax2)
    plt.title("Logistic Regression Decision Boundary (PCA)")
    plt.tight_layout()
    lr_path = "lr_plot.png"
    fig_lr.savefig(lr_path)
    plt.close(fig_lr)

    return lr_result, dt_result, lr_path, tree_path

# Gradio UI
inputs = [
    gr.Number(label="Sepal Length (cm)", value=5.0),
    gr.Number(label="Sepal Width (cm)", value=3.5),
    gr.Number(label="Petal Length (cm)", value=1.5),
    gr.Number(label="Petal Width (cm)", value=0.2),
]

outputs = [
    gr.Text(label="Logistic Regression Prediction"),
    gr.Text(label="Decision Tree Prediction"),
    gr.Image(label="Logistic Regression Visualization"),
    gr.Image(label="Decision Tree Visualization")
]

demo = gr.Interface(
    fn=predict_and_plot,
    inputs=inputs,
    outputs=outputs,
    title="Logistic Regression vs Decision Tree Classifier",
    description="Make predictions and visualize model behavior for Iris dataset."
)

# Required for plots
import numpy as np

demo.launch()
