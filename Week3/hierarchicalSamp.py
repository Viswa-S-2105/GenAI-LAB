import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset options
def get_dataset(name):
    if name == "Iris":
        data = load_iris()
    elif name == "Wine":
        data = load_wine()
    elif name == "Digits":
        data = load_digits()
    return data.data, data.target

# Plotting function
def hierarchical_cluster_plot(dataset_name, n_clusters, show_dendrogram):
    X, y = get_dataset(dataset_name)

    # Agglomerative clustering
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2 if show_dendrogram else 1, figsize=(12 if show_dendrogram else 6, 5))

    # Plot clusters in PCA space
    ax_scatter = axes[0] if show_dendrogram else axes
    scatter = ax_scatter.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    ax_scatter.set_title(f"{dataset_name} Clustering (k={n_clusters})")
    
    # Plot dendrogram
    if show_dendrogram:
        linked = linkage(X, 'ward')
        dendro_ax = axes[1]
        dendrogram(linked, ax=dendro_ax, truncate_mode='lastp', p=n_clusters, leaf_rotation=90., leaf_font_size=10.)
        dendro_ax.set_title("Dendrogram")

    plt.tight_layout()
    return fig

# Gradio Interface
demo = gr.Interface(
    fn=hierarchical_cluster_plot,
    inputs=[
        gr.Dropdown(["Iris", "Wine", "Digits"], label="Select Dataset"),
        gr.Slider(2, 10, step=1, value=3, label="Number of Clusters"),
        gr.Checkbox(label="Show Dendrogram")
    ],
    outputs=gr.Plot(),
    title="Hierarchical Clustering Visualizer",
    description="Explore Agglomerative Clustering with PCA and optional dendrogram"
)

# Launch the app
demo.launch()
