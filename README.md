# Extended Customer Segmentation Analysis

**Testing whether a K-means segmentation pipeline holds up under scrutiny — method comparison, stability testing, and independent validation.**

## Motivation

During my MSBA capstone (W.P. Carey School of Business, Spring 2024), I led the technical/modeling work on a customer segmentation project for an industry client, using K-means clustering on PCA-reduced demographic and behavioral data. Reviewing that work afterward, I identified three gaps: no comparison against alternative clustering algorithms, no test of whether the resulting segments were stable, and no independent validation of what was actually driving the segmentation beyond visual inspection of crosstabs.

This project re-approaches the same technical problem on a public, structurally comparable dataset (the original client data is confidential and not reusable), applying a more rigorous methodology — and in the process, surfaces a real methodological flaw in the original approach.

## What's different from the original

- **Method comparison**: K-means tested against Gaussian Mixture Models and Agglomerative (Ward) clustering across k=2–7, using silhouette score and Davies-Bouldin index, instead of K-means alone.
- **Stability testing**: Bootstrap resampling (20 iterations) with Adjusted Rand Index to check whether the segmentation is reproducible — not tested in the original.
- **Independent validation**: A Random Forest classifier trained to predict cluster membership from raw features, both as a separability check and to rank which features actually drive the segmentation.
- **A methodological finding**: feature importance revealed the clustering was largely driven by one-hot encoded marital-status dummies rather than behavior — a known risk when categorical dummies feed directly into Euclidean-distance-based clustering. A follow-up analysis on behavioral features only produced a cleaner, more interpretable 2-segment structure with a higher silhouette score.

## How to run

```bash
git clone <this-repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook Extended_Customer_Segmentation_Analysis.ipynb
```

Or run the individual pipeline scripts in order (`01_clean_features.py` through `06_sensitivity_check.py`) from the command line.

## Repo contents

| File | Description |
|---|---|
| `Extended_Customer_Segmentation_Analysis.ipynb` | Full analysis notebook, code + narrative + results |
| `01_clean_features.py` – `06_sensitivity_check.py` | Individual pipeline scripts |
| `technical_report.md` | Standalone written summary of methodology and findings |
| `figures/` | Summary figures (method comparison, stability, feature importance) |
| `marketing_campaign.csv` | Public dataset (see Data section) |
| `requirements.txt` | Python dependencies |

## Data

Public dataset used in place of the original (confidential) client data: 2,212 customers after cleaning, with demographic and purchase-behavior features — structurally analogous to the original project's mix of categorical and behavioral variables, though a much smaller sample.

**Source**: [Customer Personality Analysis dataset](https://github.com/andhikaw789/Customer-Personality-Analysis), originally compiled for a public marketing analytics case study.

## Limitations

Sample size here (~2,200) is far smaller than the original (~988K), so absolute metric values aren't directly comparable across the two projects. The contribution here is methodological — a more rigorous approach that generalizes back to the original problem.

## Future work

- Test whether cluster membership actually predicts campaign response (this dataset includes a `Response` field for a past marketing campaign), which would validate the segmentation against a real business outcome rather than internal clustering metrics alone.
- Extend the stability analysis to test sensitivity to feature scaling choices (MinMax vs. StandardScaler) and PCA variance thresholds.
- Apply the same diagnostic approach (testing whether categorical encoding dominates distance-based clustering) back to the original, larger-scale client dataset.

## License

MIT — see [LICENSE](LICENSE).
