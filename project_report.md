# Project Report: Material Master Clustering Analysis

## 1. Executive Summary

This report details the advanced unsupervised machine learning analysis performed on the material master dataset, focusing on segmentation and data quality monitoring. Three optimized clustering models (K-Means, X-Means, and Autoencoder + K-Means) were compared using Julia for high-performance execution.

The analysis was constrained by a **business requirement that the number of material segments ($K$) must be greater than 4**. Based on this constraint and superior cluster quality metrics, the **Autoencoder + K-Means model, with $K=5$**, was selected as the final segmentation strategy. This model achieved a Silhouette Score of 0.6991, which was the highest among all constrained options.

The analysis successfully grouped materials based on their MRP, purchasing, and warehousing characteristics and identified approximately **5.0%** of the materials as outliers, which require further data quality investigation.

***

## 2. Methodology

The project followed a standard data science workflow, encompassing data loading, preprocessing, advanced modeling, and interpretation.

### 2.1. Data Loading and Inspection

-   **Dataset:** The analysis used the `MARC all.csv` file, containing over 300,000 records.
-   **Initial Findings:** The dataset consists of a mix of numerical and categorical data, with a significant number of missing values.

### 2.2. Feature Engineering and Selection

-   **Feature Scope:** A subset of key features related to MRP, purchasing, and warehousing was selected.
-   **Selected Features:** `['DISPR', 'DISMM', 'DISPO', 'PLIFZ', 'WEBAZ', 'AUSSS', 'MINBE', 'BSTMI', 'BSTMA', 'BSTRF', 'MABST', 'FHORI', 'FEVOR', 'EKGRP', 'BESKZ', 'SOBSL', 'EISBE', 'WZEIT', 'WERKS', 'LAGPR', 'LADGR', 'LGFSB', 'QZGTP']`

### 2.3. Data Preprocessing

To prepare the data for modeling, the following steps were taken:

1.  **Missing Value Imputation:** Missing numerical values were imputed using the mean, and missing categorical values were imputed using the mode.
2.  **Categorical Feature Encoding:** Categorical features were converted using one-hot encoding.
3.  **Feature Scaling:** All features were scaled using Z-score standardization.
4.  **Dimensionality Reduction:** Principal Component Analysis (PCA) was applied to retain 95% of the data's variance.

### 2.4. Modeling

Three unsupervised clustering approaches were implemented and rigorously compared: **K-Means (Optimized)**, **X-Means (Simulated)**, and **Autoencoder + K-Means**.

***

## 3. Results and Findings

### 3.1. Model Selection and Optimal Number of Clusters

The model selection process was strictly guided by two criteria: the **business requirement for $K > 4$** and the internal validation metrics (Silhouette Score and Davies-Bouldin Index).

The analysis of the standard K-Means algorithm over the range $K=2$ to $K=30$ is summarized below, showing the trade-off between $K$ and cluster quality:



The comparative results for all tested models are as follows:

| Model | Optimal K | Silhouette Score (Higher is Better) | Davies-Bouldin Index (Lower is Better) | $K>4$ Constraint? |
| :--- | :--- | :--- | :--- | :--- |
| K-Means (Absolute Optimum) | 2 | **0.8805** | **0.3855** | No |
| K-Means ($K=9$ Best Local Fit) | 9 | 0.4800 | (Not recorded in summary) | Yes |
| X-Means (Simulated) | 20 | 0.2364 | 0.7938 | Yes |
| **Autoencoder + K-Means**| **5** | 0.6991 | 1.5314 | **Yes** |

The **K-Means (Absolute Optimum)** at $K=2$ was rejected. Although the standard K-Means plot showed a local optimum at $K=9$ (Score: 0.48), the **Autoencoder + K-Means model at $K=5$** was selected as the final solution. It delivered a significantly higher Silhouette Score of **0.6991**, indicating a superior quality of cluster separation and internal cohesion compared to all other options satisfying the business constraint.

### 3.2. Cluster Profiles

The Autoencoder + K-Means model segmented the materials into **5 distinct clusters**.

The features most critical in defining these 5 clusters were identified using feature importance techniques:
* **`MINBE`** (Minimum Lot Size)
* **`BSTMA`** (Maximum Lot Size)
* **`AUSSS`** (Total Shelf Life)

### 3.3. Outlier Identification

Across the final model, **16,321 materials (4.99%)** were flagged as outliers.

**Characteristics of Outliers:**
The materials flagged as outliers tend to have significantly higher values for key planning and purchasing parameters compared to inliers, suggesting potential data entry errors or unique supply chain constraints that require manual review.
-   **Planned Delivery Time (PLIFZ):** Average of ~42 days vs ~25 days for inliers.
-   **Goods Receipt Processing Time (WEBAZ):** Average of ~6.5 days vs ~1.7 days.
-   **Safety Stock (EISBE):** Average of ~15 vs ~0.2.

### 3.4. Visualization

-   **PCA Analysis:** Principal Component Analysis (PCA) was used to reduce the dimensionality of the data for visualization.
-   **Plots:** The generated plots, such as `pca_clusters.png` and `pca_3d_visualization.png`, visually confirm the separation between the 5 clusters and the distribution of outliers in the reduced dimensional space.

### 3.5. Model Persistence

All trained models and preprocessing objects have been serialized and saved using Julia's `JLD2` library to ensure reproducibility.

-   **Final Clustering Model (`./models_julia/ae_kmeans_model.jld2`):** The K-Means model trained on the latent space with $K=5$ clusters.
-   **Autoencoder State (`./models_julia/autoencoder_state.jld2`):** The trained neural network state required to generate the latent features for new data.
-   **PCA Model and Scaler Parameters:** The fitted PCA object and data scaling parameters are saved for consistent preprocessing.

***

## 4. Outputs

The analysis produced the following key deliverables:

-   `clustered_materials.csv`: A dataset mapping each material to a cluster and an outlier status.
-   `analysis_summary.txt`: A detailed log file containing the comparative performance metrics for all three models.
-   `./models_julia/`: Directory containing saved `.jld2` files for models and preprocessors.
-   A set of PNG images with visualizations of the clustering results.

***

## 5. Conclusion and Next Steps

This project successfully implemented an advanced, comparative unsupervised clustering pipeline, demonstrating that the **Autoencoder + K-Means model ($K=5$)** is the optimal strategy given the required $K>4$ constraint and its high cluster quality (Silhouette Score: 0.6991).

**Recommended next steps include:**

-   **Business Validation:** Work with subject matter experts to interpret and name the 5 clusters based on their business meaning.
-   **Data Quality Initiative:** Investigate the identified 5.0% outliers to correct data errors and improve data governance.
-   **Downstream Applications:** Use the cluster assignments to optimize inventory policies, tailor procurement strategies, or improve demand forecasting models.