"""
Step 3: Clustering method comparison
Original capstone used K-means only, picked k via elbow + a single silhouette check.
Here we test whether K-means is actually the right model, comparing against
Gaussian Mixture Models and Agglomerative (hierarchical) clustering across k=2..7,
using silhouette score AND Davies-Bouldin index (K-means' choice metric alone
is biased toward K-means since it assumes spherical clusters).
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

df_pca = pd.read_csv('df_pca.csv')
X = df_pca.values

results = []
np.random.seed(42)

for k in range(2, 8):
    # K-Means
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km_labels = km.fit_predict(X)
    results.append({
        'method': 'KMeans', 'k': k,
        'silhouette': silhouette_score(X, km_labels),
        'davies_bouldin': davies_bouldin_score(X, km_labels)
    })

    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
    gmm_labels = gmm.fit_predict(X)
    results.append({
        'method': 'GMM', 'k': k,
        'silhouette': silhouette_score(X, gmm_labels),
        'davies_bouldin': davies_bouldin_score(X, gmm_labels)
    })

    # Agglomerative (hierarchical, Ward linkage)
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    agg_labels = agg.fit_predict(X)
    results.append({
        'method': 'Agglomerative', 'k': k,
        'silhouette': silhouette_score(X, agg_labels),
        'davies_bouldin': davies_bouldin_score(X, agg_labels)
    })

results_df = pd.DataFrame(results)
results_df.to_csv('method_comparison.csv', index=False)
print(results_df.to_string(index=False))

print("\nBest silhouette per method:")
print(results_df.loc[results_df.groupby('method')['silhouette'].idxmax()])
