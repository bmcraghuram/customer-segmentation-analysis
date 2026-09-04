# Revisiting a Customer Segmentation Pipeline: Method Selection, Stability, and Validation

## Motivation

During my Master of Science in Business Analytics capstone (W.P. Carey School of Business, Spring 2024), I led the technical and modeling work on a customer segmentation project for an industry client, using K-means clustering on PCA-reduced demographic and behavioral data to identify customer segments for targeted marketing.

Reviewing that work afterward, I identified three gaps in the methodology: no comparison against alternative clustering algorithms, no test of whether the resulting segments were stable, and no independent validation of what was actually driving the segmentation beyond visual inspection of crosstabs. This report describes an independent extension addressing each gap, using a public dataset with a comparable structure (the original client data is confidential and could not be reused).

## Methodology

The pipeline mirrors the original's structure — data cleaning, categorical correlation analysis (Cramer's V), encoding, PCA, and clustering — with three additions:

**1. Clustering method comparison.** K-means assumes roughly spherical, evenly sized clusters, an assumption that should be tested rather than adopted by default. I compared K-means, Gaussian Mixture Models, and Agglomerative (Ward) clustering across k = 2–7, scoring each on silhouette score and the Davies-Bouldin index. All three methods agreed that k=5 outperformed k=4, the value selected in the original project from a narrower k=3-vs-4 comparison.

**2. Stability analysis.** I bootstrap-resampled 80% of the data over 20 iterations, re-clustering each time and comparing labels to the full-data clustering via Adjusted Rand Index (ARI). The k=5 segmentation proved highly stable (mean ARI = 0.96), giving confidence that the segments reflect real structure rather than an artifact of a single run.

**3. Supervised validation.** I trained a Random Forest classifier to predict cluster membership from the original (pre-PCA) features. The classifier achieved 98% cross-validated accuracy — strong evidence the clusters are genuinely separable — but feature importances showed that roughly 77% of predictive weight came from one-hot encoded marital-status dummy variables, not behavioral variables like spend or income.

## Key finding

The heavy weighting toward marital-status dummies indicated the clustering was substantially rediscovering demographic categories rather than uncovering behavioral patterns — a known risk when one-hot encoded categorical variables are fed directly into Euclidean-distance-based methods like K-means on PCA components, since dummy variables can dominate the distance metric. This is the same encoding approach the original project used, and the issue went untested there.

To confirm this, I re-ran the full pipeline using only behavioral and engagement features (spend, purchase channels, recency, web visits), excluding demographic dummies entirely. This produced a cleaner two-segment structure with a *higher* silhouette score (0.32 vs. 0.30 for the original 5-cluster demographic-driven solution):

- **Segment A** (high-value, low-engagement): higher income (~$70K), higher total spend (~$1,182), fewer web visits, fewer children
- **Segment B** (lower-value, high-engagement): lower income (~$38K), lower spend (~$154), more web visits, more children

This behavioral segmentation is more directly actionable for marketing purposes than a segmentation that largely reflects marital status, and it emerged specifically from diagnosing and correcting a methodological weakness rather than accepting the first plausible-looking result.

## Limitations

The public dataset used here (2,212 customers) is far smaller than the original client dataset (~988K records), so absolute metric values are not directly comparable between the two projects. The value of this extension is methodological: the diagnostic process — testing method choice, stability, and encoding sensitivity — generalizes directly back to the original, larger-scale problem.

## Summary

| | Original capstone | This extension |
|---|---|---|
| Clustering methods tested | K-means only | K-means, GMM, Agglomerative |
| k selection | Single comparison (k=3 vs. 4) | Full sweep k=2–7, two metrics, three methods |
| Stability tested | No | Yes (bootstrap ARI = 0.96) |
| Cluster validation | Manual inspection of crosstabs | Independent classifier + feature importance |
| Key insight | Segments described post-hoc | Diagnosed and corrected an encoding-driven bias in the clustering |

Full code and analysis: see accompanying Jupyter notebook.
