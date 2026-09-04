"""
Step 2: Encoding, correlation analysis, PCA
Mirrors original: Cramer's V for categorical correlation, one-hot + ordinal encoding, PCA
"""
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv('df_model.csv')

# --- Cramer's V correlation among categorical variables ---
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2_corr / denom)

cat_cols = ['Education', 'Marital_Status']
print("Cramer's V (categorical vars):")
print(f"  Education vs Marital_Status: {cramers_v(df['Education'], df['Marital_Status']):.3f}")

# --- Encoding ---
# Ordinal: Education has a natural order
education_order = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
df['Education_encoded'] = df['Education'].map(education_order)

# One-hot: Marital_Status is nominal
df = pd.get_dummies(df, columns=['Marital_Status'], prefix='Marital', drop_first=False)

df_features = df.drop(columns=['Education'])
df_features.to_csv('df_features.csv', index=False)
print(f"\nFeature matrix shape: {df_features.shape}")

# --- Scale + PCA ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_features)
scaled_df = pd.DataFrame(scaled, columns=df_features.columns)

pca_full = PCA()
pca_full.fit(scaled_df)
cum_var = pca_full.explained_variance_ratio_.cumsum()
n_components_90 = int((cum_var <= 0.90).sum()) + 1
print(f"\nComponents needed for >=90% variance: {n_components_90}")
print(f"Cumulative variance at that point: {cum_var[n_components_90-1]:.3f}")

pca = PCA(n_components=n_components_90)
pca_transformed = pca.fit_transform(scaled_df)
df_pca = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(n_components_90)])
df_pca.to_csv('df_pca.csv', index=False)

loadings = pd.DataFrame(pca.components_.T, columns=df_pca.columns, index=scaled_df.columns)
loadings.to_csv('pca_loadings.csv')

print("\nExplained variance ratio per component:")
print(np.round(pca.explained_variance_ratio_, 4))
print("\nTop 8 loadings on PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(8))
print("\nTop 8 loadings on PC2:")
print(loadings['PC2'].abs().sort_values(ascending=False).head(8))
