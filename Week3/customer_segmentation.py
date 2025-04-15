import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import gradio as gr

# Global storage
cluster_models = {}

# Load and scale data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df = df.rename(columns=lambda x: x.strip())
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    X_scaled = StandardScaler().fit_transform(X)
    return df, X, X_scaled

# Interpret cluster label based on centroid values
def interpret_clusters(centroids):
    labels = []
    for c in centroids:
        age, income, score = c
        if income > 1 and score > 1:
            labels.append("High Earners, High Spenders")
        elif income > 1 and score < 0:
            labels.append("High Earners, Low Spenders")
        elif income < 0 and score > 1:
            labels.append("Low Earners, High Spenders")
        else:
            labels.append("Others")
    return labels

# Main clustering function
def cluster_customers(n_clusters, method, dim_reduce):
    df, X, X_scaled = load_data()

    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)

    labels = model.fit_predict(X_scaled)
    df["Cluster"] = labels

    # Save model for prediction
    cluster_models["model"] = model
    cluster_models["scaler"] = StandardScaler().fit(X)
    cluster_models["labels"] = labels

    # Cluster centroids (for KMeans only)
    if method == "KMeans":
        centroids = model.cluster_centers_
        interpretations = interpret_clusters(centroids)
    else:
        centroids = None
        interpretations = ["Interpretation only for KMeans"]

    # Dimension reduction
    if dim_reduce == "PCA":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    reduced = reducer.fit_transform(X_scaled)

    # Plotting
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=50)
    ax.set_title(f"{method} Clustering ({dim_reduce}) - k={n_clusters}")
    ax.grid(True)

    csv_path = "clustered_customers.csv"
    df.to_csv(csv_path, index=False)
    return fig, interpretations, csv_path


# Plotly 3D plot
def plot_3d():
    df, _, X_scaled = load_data()
    labels = cluster_models.get("labels", [0] * len(df))
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    fig = px.scatter_3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
        color=labels,
        title="3D Cluster Visualization (PCA)",
        labels={"x": "PC1", "y": "PC2", "z": "PC3"}
    )
    return fig

# Predict cluster from user input
def predict_cluster(age, income, score):
    model = cluster_models.get("model", None)
    scaler = cluster_models.get("scaler", None)

    if model and scaler and hasattr(model, "predict"):
        X_input = scaler.transform([[age, income, score]])
        label = model.predict(X_input)[0]
        return f"Predicted Cluster: {label}"
    else:
        return "Prediction is only available for KMeans clustering."
# Dendrogram for hierarchical
def show_dendrogram():
    _, _, X_scaled = load_data()
    linked = linkage(X_scaled, method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, truncate_mode='lastp', p=10, leaf_rotation=90., leaf_font_size=10., ax=ax)
    ax.set_title("Hierarchical Dendrogram")
    return fig

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🛍️ Customer Segmentation Dashboard")

    with gr.Row():
        n_clusters = gr.Slider(2, 10, value=5, label="Number of Clusters")
        method = gr.Dropdown(["KMeans", "Hierarchical"], value="KMeans", label="Clustering Method")
        dim_reduce = gr.Radio(["PCA", "t-SNE"], value="PCA", label="Visualization Method")

    with gr.Row():
        cluster_btn = gr.Button("Run Clustering")
        dendro_btn = gr.Button("Show Dendrogram")
        plotly_btn = gr.Button("3D Plot")

    plot_out = gr.Plot(label="2D Plot")
    interp_out = gr.Textbox(label="Cluster Interpretation", lines=5)
    csv_out = gr.File(label="Download Clustered CSV")

    cluster_btn.click(cluster_customers, inputs=[n_clusters, method, dim_reduce], outputs=[plot_out, interp_out, csv_out])
    dendro_btn.click(show_dendrogram, outputs=plot_out)
    plotly_btn.click(plot_3d, outputs=plot_out)

    gr.Markdown("### 🧮 Predict Cluster for New Customer")
    with gr.Row():
        age = gr.Number(label="Age")
        income = gr.Number(label="Annual Income (k$)")
        score = gr.Number(label="Spending Score (1-100)")
    predict_btn = gr.Button("Predict Cluster")
    predict_out = gr.Textbox(label="Prediction")
    predict_btn.click(predict_cluster, inputs=[age, income, score], outputs=predict_out)

demo.launch()
