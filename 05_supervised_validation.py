"""
Step 5: Supervised validation
The original profiled clusters only by eyeballing crosstabs. Here we train a
classifier to predict cluster membership from the ORIGINAL (pre-PCA) features,
which does two things:
  1. If a classifier can predict clusters well, that's independent evidence
     the segments are real and separable (not a PCA artifact).
  2. Feature importances tell us which real-world variables actually drive
     the segmentation -- more defensible than eyeballing crosstabs.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

df_pca = pd.read_csv('df_pca.csv')
df_features = pd.read_csv('df_features.csv')

X_pca = df_pca.values
K = 5
labels = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42).fit_predict(X_pca)

X = df_features.values
feature_names = df_features.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Cluster-prediction classifier performance (holdout test set):")
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(clf, X, labels, cv=5)
print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 10 features driving cluster separation:")
print(importances.head(10))

importances.to_csv('feature_importances.csv')

# Cluster profiles: mean of key raw features per cluster
df_features['Cluster'] = labels
profile_cols = ['Income', 'Age', 'TotalSpend', 'TotalPurchases', 'Kidhome', 'Recency', 'NumWebVisitsMonth']
profile = df_features.groupby('Cluster')[profile_cols].mean().round(1)
profile['n'] = df_features.groupby('Cluster').size()
print("\nCluster profiles (mean values):")
print(profile)
profile.to_csv('cluster_profiles.csv')
