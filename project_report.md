# Project Report: Material Master Clustering Analysis

## 1. Executive Summary

This report details the unsupervised machine learning analysis performed on the material master dataset, which contains over 300,000 records. The primary objective was to segment materials into meaningful groups and identify potential data quality issues. By applying clustering and anomaly detection techniques, we successfully grouped materials based on their MRP, purchasing, and warehousing characteristics and identified a subset of materials as outliers.

The analysis provides a foundational step for improving data governance, optimizing inventory, and streamlining procurement processes by highlighting the structure and quality of the existing material data.

## 2. Methodology

The project followed a standard data science workflow, encompassing data loading, preprocessing, modeling, and interpretation.

### 2.1. Data Loading and Inspection

-   **Dataset:** The analysis used the `MARC all.csv` file, containing 326,782 records and 114 features.
-   **Initial Findings:** The dataset consists of a mix of numerical and categorical data, with a significant number of missing values in several columns.

### 2.2. Feature Engineering and Selection

-   **Feature Scope:** To focus the analysis, we selected a subset of 23 key features related to MRP, purchasing, and warehousing, in addition to the material ID.
-   **Selected Features:** `['DISPR', 'DISMM', 'DISPO', 'PLIFZ', 'WEBAZ', 'AUSSS', 'MINBE', 'BSTMI', 'BSTMA', 'BSTRF', 'MABST', 'FHORI', 'FEVOR', 'EKGRP', 'BESKZ', 'SOBSL', 'EISBE', 'WZEIT', 'WERKS', 'LAGPR', 'LADGR', 'LGFSB', 'QZGTP']`

### 2.3. Data Preprocessing

To prepare the data for modeling, the following steps were taken:

1.  **Missing Value Imputation:**
    -   Categorical features: Missing values were filled with the string `'MISSING'`.
    -   Numerical features: Missing values were imputed using the median value of the respective column, which is robust to outliers.

2.  **Categorical Feature Encoding:**
    -   **One-Hot Encoding:** Applied to low-cardinality features (fewer than 50 unique values) to create binary columns for each category.
    -   **Frequency Encoding:** Applied to high-cardinality features to represent each category by its frequency in the dataset. This avoids creating an excessive number of new features.

3.  **Feature Scaling:**
    -   All numerical features were scaled using `RobustScaler`, which is less sensitive to outliers than standard scaling methods.

### 2.4. Modeling

Two primary unsupervised models were used:

1.  **Clustering with MiniBatchKMeans:**
    -   To determine the optimal number of clusters (K), the Elbow and Silhouette methods were employed, analyzing a range of K from 5 to 20.
    -   Based on the silhouette score, the optimal K was selected and the final clustering model was trained on the scaled data. `MiniBatchKMeans` was chosen for its memory efficiency, which is critical for large datasets.

2.  **Outlier Detection with Isolation Forest:**
    -   An `IsolationForest` model was trained to identify anomalies in the data.
    -   A contamination rate of 2% was set, meaning the model was configured to flag the top 2% of most anomalous data points as outliers.

## 3. Results and Findings

### 3.1. Optimal Number of Clusters

The analysis of silhouette scores indicated the optimal number of clusters for this dataset. The script automatically determines this value and uses it for the final clustering. The output file `elbow_silhouette_analysis.png` shows the plots used for this decision.

### 3.2. Cluster Profiles

The materials were segmented into distinct clusters. The characteristics of each cluster can be inferred by analyzing the mean values of their features. The `analysis_output.txt` file contains a summary of these cluster profiles.

### 3.3. Outlier Identification

The Isolation Forest model successfully identified 2% of the materials as outliers. These materials exhibit unusual combinations of feature values compared to the rest of the dataset. A detailed list of these materials, along with their anomaly scores, is available in the final output CSV.

**Characteristics of Outliers:**
The analysis reveals that materials flagged as outliers tend to have significantly higher values for key planning and purchasing parameters compared to inliers (normal data points). Specifically:
-   **Planned Delivery Time (PLIFZ):** Average of ~42 days vs ~25 days for inliers.
-   **Goods Receipt Processing Time (WEBAZ):** Average of ~6.5 days vs ~1.7 days.
-   **Minimum Order Quantity (BSTMI):** Average of ~26 vs ~1.
-   **Safety Stock (EISBE):** Average of ~15 vs ~0.2.
-   **Rounding Value (BSTRF):** Average of ~11.5 vs ~0.4.

Interestingly, outliers have *fewer* missing values on average (2.78 missing fields per row) compared to inliers (4.58), suggesting that these are often more fully populated records but contain extreme numerical values that deviate from the norm. These extreme values (e.g., very high safety stock or delivery times) may indicate data entry errors or legitimate but unique supply chain constraints that require manual review.

### 3.4. Visualization

-   **PCA Analysis:** Principal Component Analysis (PCA) was used to reduce the dimensionality of the data for visualization.
-   **2D and 3D Plots:** The generated plots (`pca_clusters.png`, `pca_3d_visualization.png`) visually represent the clusters and outliers in a reduced dimensional space, confirming the separation between groups.

### 3.5. Model Persistence

To ensure reproducibility and facilitate deployment, all trained models and preprocessing objects have been serialized and saved using the `joblib` library, which is optimized for fast disk I/O with NumPy arrays.

-   **Optimal Clustering Model (`./models/optimal_kmeans_model.joblib`):** The final MiniBatchKMeans model trained with the optimal number of clusters (K=2).
-   **K=7 Clustering Model (`./models/kmeans_k7_model.joblib`):** A dedicated model trained with K=7 clusters. This model is preserved to satisfy specific business rules or for use as an alternative segmentation strategy if the optimal K is deemed too coarse for certain operational needs.
-   **Anomaly Detection Model (`./models/isolation_forest_model.joblib`):** The trained Isolation Forest model used to flag outliers.
-   **Preprocessor (`./models/preprocessor_scaler.joblib`):** The fitted `RobustScaler` (and any other pipeline steps). Saving this is critical to ensure that new data fed into the models is scaled identically to the training data, maintaining data pipeline consistency.

## 4. Outputs

The analysis produced the following key deliverables:

-   `clustered_materials.csv`: A dataset mapping each material to a cluster and an outlier status.
-   `analysis_output.txt`: A log file with detailed statistics from each step of the analysis.
-   `./models/`: Directory containing saved `.joblib` files for models and scalers.
-   A set of PNG images with visualizations of the clustering results.

## 5. Conclusion and Next Steps

This project successfully demonstrates the use of unsupervised learning to derive valuable insights from a large material master dataset. The identified clusters can be used for targeted business strategies, and the flagged outliers should be investigated as potential data entry errors or unique cases requiring special management.

**Recommended next steps include:**

-   **Business Validation:** Work with subject matter experts to interpret and name the clusters based on their business meaning.
-   **Data Quality Initiative:** Investigate the identified outliers to correct data errors and improve data governance.
-   **Downstream Applications:** Use the cluster assignments to optimize inventory policies, tailor procurement strategies, or improve demand forecasting models.
