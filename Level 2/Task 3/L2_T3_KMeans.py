import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Toggle: comment/uncomment the DATA/OUT lines for Colab vs local run
DATA = r'/content'
# DATA = r'C:\Users\ASUS\Downloads\CodeVeda_Internship_Projects\Level 1\Task 1'
OUT = r'/content'
# OUT = r'C:\Users\ASUS\Downloads\CodeVeda_Internship_Projects\Level 2\Task 3'

df = pd.read_csv(f'{DATA}/1) iris_cleaned.csv')
print('=== Iris Dataset for Clustering ===')
print(df.head())
print()

# Extract features (drop species for unsupervised learning)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# ---- Standardize ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('Feature means after standardization:')
print(f'  Mean ~ {np.mean(X_scaled, axis=0).round(6)}')
print(f'  Std  ~ {np.std(X_scaled, axis=0).round(6)}')
print()

# ---- Elbow Method ----
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUT}/kmeans_elbow.png', dpi=150)
plt.show()
print('Saved: kmeans_elbow.png')
print()

# ---- Apply K-Means with k=3 (known iris species) ----
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

print('=== Cluster Centers (standardized) ===')
centers = scaler.inverse_transform(kmeans.cluster_centers_)
print(pd.DataFrame(centers, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']))
print()

print('=== Cluster Distribution ===')
print(df['Cluster'].value_counts().sort_index())
print()

# ---- Cross-tab with actual species ----
print('=== Cluster vs Actual Species ===')
print(pd.crosstab(df['species'], df['Cluster'], margins=True))
print()

# ---- Visualize clusters using PCA (2D) ----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

# Left: actual species
plt.subplot(1, 2, 1)
species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
colors_actual = [species_map[s] for s in df['species']]
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_actual,
                      cmap='viridis', edgecolor='k', s=60)
plt.colorbar(scatter, ticks=[0, 1, 2], label='Species')
plt.title('Actual Species (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Right: K-Means clusters
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                      cmap='viridis', edgecolor='k', s=60)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
            pca.transform(kmeans.cluster_centers_)[:, 1],
            marker='X', s=200, c='red', edgecolor='k', label='Centroids')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Cluster')
plt.title('K-Means Clusters (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.tight_layout()
plt.savefig(f'{OUT}/kmeans_clusters.png', dpi=150)
plt.show()
print('Saved: kmeans_clusters.png')
print()

# ---- 2D scatter of top 2 features ----
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 2], X_scaled[:, 3], c=clusters,
            cmap='viridis', edgecolor='k', s=60, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
            marker='X', s=200, c='red', edgecolor='k', label='Centroids')
plt.xlabel('Petal Length (standardized)')
plt.ylabel('Petal Width (standardized)')
plt.title('K-Means Clusters (Petal Features)')
plt.legend()
plt.savefig(f'{OUT}/kmeans_petal_scatter.png', dpi=150)
plt.show()
print('Saved: kmeans_petal_scatter.png')