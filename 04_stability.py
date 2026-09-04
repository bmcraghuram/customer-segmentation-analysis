"""
Step 4: Stability analysis
Question the original never asked: if we re-run clustering on resampled data,
do we get the same segments back? A "finding" that isn't stable under
resampling is a coin flip, not a business insight.
Method: bootstrap resample 80% of the data, re-cluster, compare label
assignments on the overlapping points via Adjusted Rand Index (ARI).
ARI = 1 means identical clustering; ARI ~ 0 means random agreement.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

df_pca = pd.read_csv('df_pca.csv')
X = df_pca.values
n = X.shape[0]

K = 5  # chosen from step 3
N_BOOTSTRAPS = 20
rng = np.random.RandomState(42)

# Reference clustering on full data
ref_model = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
ref_labels = ref_model.fit_predict(X)

ari_scores = []
for i in range(N_BOOTSTRAPS):
    idx = rng.choice(n, size=int(0.8 * n), replace=False)
    X_sample = X[idx]
    model = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=i)
    sample_labels = model.fit_predict(X_sample)
    ari = adjusted_rand_score(ref_labels[idx], sample_labels)
    ari_scores.append(ari)

ari_scores = np.array(ari_scores)
print(f"Bootstrap stability (K={K}, n={N_BOOTSTRAPS} resamples):")
print(f"  Mean ARI: {ari_scores.mean():.3f}")
print(f"  Std ARI:  {ari_scores.std():.3f}")
print(f"  Min/Max:  {ari_scores.min():.3f} / {ari_scores.max():.3f}")

pd.DataFrame({'ari': ari_scores}).to_csv('stability_scores.csv', index=False)

# Also compare against a lower K to show contrast (should be MORE stable — fewer, coarser groups)
K2 = 3
ref2 = KMeans(n_clusters=K2, init='k-means++', n_init=10, random_state=42).fit_predict(X)
ari_k3 = []
for i in range(N_BOOTSTRAPS):
    idx = rng.choice(n, size=int(0.8 * n), replace=False)
    model = KMeans(n_clusters=K2, init='k-means++', n_init=10, random_state=i)
    sample_labels = model.fit_predict(X[idx])
    ari_k3.append(adjusted_rand_score(ref2[idx], sample_labels))
print(f"\nFor comparison, K={K2} stability:")
print(f"  Mean ARI: {np.mean(ari_k3):.3f}  (Std: {np.std(ari_k3):.3f})")
