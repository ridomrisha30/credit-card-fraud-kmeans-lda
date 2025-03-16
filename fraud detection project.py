import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

#Load the dataset
file_path = r"C:\Users\grigt\OneDrive\Documents\creditcard.csv"
df = pd.read_csv(file_path)

#Select relevant features
df_selected = df[['Amount', 'Time', 'Class']]  # Using Amount & Time

#Normalize the selected features
scaler = MinMaxScaler()
df_selected[['Amount', 'Time']] = scaler.fit_transform(df_selected[['Amount', 'Time']])

#Apply K-Means Clustering (Unsupervised Learning)
kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters (Fraud & Non-Fraud)
df_selected['cluster'] = kmeans.fit_predict(df_selected[['Amount', 'Time']])

#Visualizing the Clusters
plt.figure(figsize=(8, 6))

# Separate the clusters
cluster_0 = df_selected[df_selected['cluster'] == 0]
cluster_1 = df_selected[df_selected['cluster'] == 1]

# Scatter plot with labels
plt.scatter(cluster_0['Amount'], cluster_0['Time'], color='blue', alpha=0.5, label="Cluster 0 (Non-Fraud)")
plt.scatter(cluster_1['Amount'], cluster_1['Time'], color='red', alpha=0.5, label="Cluster 1 (Fraud)")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='*', label='Centroids')

# Labels and title
plt.xlabel("Transaction Amount (Normalized)")
plt.ylabel("Transaction Time (Normalized)")
plt.title("K-Means Clustering of Transactions")
plt.legend()
plt.show()

#Finding Optimal K Using the Elbow Method
sse = []
k_range = range(1, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_selected[['Amount', 'Time']])
    sse.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_range, sse, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method for Optimal K")
plt.show()

#Fisher’s LDA for Fraud Detection (Supervised Learning)
X = df_selected[['Amount', 'Time']]
y = df_selected['Class']  # Target (0 = Non-Fraud, 1 = Fraud)

# Splitting data into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train LDA Model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Make predictions
y_pred = lda.predict(X_test)

#Calculate & Print Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f" Fisher's LDA Accuracy on Test Set: {accuracy:.2%}")

#Visualizing LDA Decision Boundary
direction_v = lda.coef_[0]  # LDA direction vector

plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0]['Amount'], X_train[y_train == 0]['Time'], label="Non-Fraud", color="blue", alpha=0.5)
plt.scatter(X_train[y_train == 1]['Amount'], X_train[y_train == 1]['Time'], label="Fraud", color="red", alpha=0.5)
plt.quiver(0, 0, direction_v[0], direction_v[1], scale=10, color='black', width=0.005, label="LDA Direction")
plt.xlabel("Transaction Amount (Normalized)")
plt.ylabel("Transaction Time (Normalized)")
plt.legend()
plt.title("Fisher’s LDA for Fraud Detection")
plt.show()


#