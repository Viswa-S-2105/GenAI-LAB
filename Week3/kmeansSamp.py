import gradio as gr
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load dataset options
def get_dataset(name):
    if name == "Iris":
        data = load_iris()
    elif name == "Wine":
        data = load_wine()
    elif name == "Digits":
        data = load_digits()
    return data.data, data.target

# Main function
def cluster_and_plot(dataset_name, n_clusters):
    X, y = get_dataset(dataset_name)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # Reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    # Plotting
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', s=50)
    centers = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.6, marker='X', label='Centroids')
    ax.set_title(f"{dataset_name} Clustering with k={n_clusters}")
    ax.legend()
    return fig

# Gradio interface
demo = gr.Interface(
    fn=cluster_and_plot,
    inputs=[
        gr.Dropdown(["Iris", "Wine", "Digits"], label="Select Dataset"),
        gr.Slider(2, 10, step=1, value=3, label="Number of Clusters (k)")
    ],
    outputs=gr.Plot(),
    title="KMeans Clustering Visualizer",
    description="Select a dataset and number of clusters to visualize KMeans clustering."
)

# Launch locally
demo.launch()
