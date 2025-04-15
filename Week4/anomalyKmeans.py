import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
import gradio as gr

# 🎯 Generate Synthetic Bank Account Access Data
np.random.seed(42)
normal_users = np.random.normal(loc=[5, 1000, 10], scale=[2, 500, 3], size=(200, 3))  # Normal logins
anomaly_users = np.random.normal(loc=[20, 10000, 2], scale=[5, 2000, 1], size=(10, 3))  # Fraudulent logins
data = np.vstack((normal_users, anomaly_users))

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Login Frequency (per week)", "Transaction Amount ($)", "Session Duration (mins)"])

# 🔹 Scale Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 🔹 Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# 🔹 Identify Normal Cluster (Largest Cluster)
normal_cluster = np.bincount(labels).argmax()  # The cluster with the most points
df["Cluster"] = labels

# 🔹 Compute Mahalanobis Distance for Each Point
cov_matrix = np.cov(scaled_data.T)
inv_cov_matrix = np.linalg.inv(cov_matrix)

distances = np.array([mahalanobis(point, kmeans.cluster_centers_[normal_cluster], inv_cov_matrix) for point in scaled_data])
threshold = np.percentile(distances, 95)  # Adjust threshold dynamically

df["Anomaly"] = distances > threshold

# 🎯 Function to Detect Anomaly for New Bank Access
def detect_bank_anomaly(login_freq, trans_amt, session_time):
    point = scaler.transform([[login_freq, trans_amt, session_time]])  # Scale input
    dist = mahalanobis(point[0], kmeans.cluster_centers_[normal_cluster], inv_cov_matrix)  # Mahalanobis distance
    
    if dist > threshold:
        return "🚨 Suspicious Activity Detected!"
    else:
        return "✅ Normal Access"

# 🔹 Gradio Web Interface
iface = gr.Interface(
    fn=detect_bank_anomaly,
    inputs=[
        gr.Number(label="Login Frequency (per week)"),
        gr.Number(label="Transaction Amount ($)"),
        gr.Number(label="Session Duration (mins)")
    ],
    outputs="text",
    title="Bank Account Access Anomaly Detection",
    description="Enter user login details to check for suspicious activity."
)

iface.launch(share=True)
