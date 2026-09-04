"""
Step 6: Sensitivity check
Feature importance in step 5 showed marital-status dummies dominate cluster
assignment. This tests whether that's an artifact of one-hot encoding
distorting Euclidean distance in PCA/K-means space, by re-clustering on
BEHAVIORAL features only (spend, purchases, engagement) with demographics
excluded, and checking whether a more informative segmentation emerges.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_features = pd.read_csv('df_features.csv')

behavioral_cols = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
                    'NumStorePurchases', 'NumWebVisitsMonth', 'TotalSpend',
                    'TotalPurchases', 'Age', 'Kidhome', 'Teenhome']

X_behav = df_features[behavioral_cols]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_behav)

pca = PCA(n_components=0.90)
X_pca_behav = pca.fit_transform(X_scaled)
print(f"Behavioral-only PCA: {X_pca_behav.shape[1]} components for 90% variance")

sil_scores = {}
for k in range(2, 8):
    labels = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit_predict(X_pca_behav)
    sil_scores[k] = silhouette_score(X_pca_behav, labels)

print("\nSilhouette scores, behavioral-only features:")
for k, s in sil_scores.items():
    print(f"  k={k}: {s:.3f}")

best_k = max(sil_scores, key=sil_scores.get)
labels = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42).fit_predict(X_pca_behav)
df_features['Behavioral_Cluster'] = labels

profile_cols = ['Income', 'Age', 'TotalSpend', 'TotalPurchases', 'NumWebVisitsMonth', 'Kidhome']
profile = df_features.groupby('Behavioral_Cluster')[profile_cols].mean().round(1)
profile['n'] = df_features.groupby('Behavioral_Cluster').size()
print(f"\nBest k={best_k}. Cluster profiles (behavioral-only):")
print(profile)
profile.to_csv('behavioral_cluster_profiles.csv')
