# Project Report: Material Master Clustering Analysis

## 1. Executive Summary

This report details the advanced unsupervised machine learning analysis performed on the material master dataset, focusing on segmentation and data quality monitoring. The project involved a three-model comparison pipeline in Julia, coupled with crucial optimizations to overcome severe memory limitations encountered on the server. The analysis was conducted on a dataset of **326,782 records**.

The analysis was constrained by a **business requirement that the number of material segments ($K$) must be greater than 4**. Based on this constraint and superior cluster quality metrics, the **Autoencoder + K-Means model, with $K=5$**, was selected as the final segmentation strategy, achieving a Silhouette Score of **0.6991**.

***

## 2. Methodology and Optimization

The project followed a robust data science workflow, heavily optimized for memory efficiency and job resumption.

### 2.3. Data Preprocessing & Memory Optimization

Preprocessing was refined to address the initial 160 GB memory overflow:
* **Feature Encoding:** For high-cardinality features (e.g., `LGPRO` with 53 unique values), standard One-Hot Encoding was replaced with **Frequency Encoding** to prevent feature space explosion and memory exhaustion.
* **Sparse Matrix Handling:** After encoding, the entire data matrix was converted and maintained as a **Sparse Matrix** to drastically reduce the memory footprint.
* **Dimensionality Reduction:** Principal Component Analysis (PCA) was applied to the sparse data to retain **95%** of the variance, resulting in a significantly reduced feature set (179 dimensions) for downstream modeling.

### 2.4. Modeling and Stability

Three unsupervised clustering approaches were implemented and compared: K-Means (Optimized), X-Means (Simulated), and Autoencoder + K-Means.

* **Training Loop Fix:** The Autoencoder training loop was updated from the deprecated `Flux.params` style to the modern **explicit gradient computation syntax** to resolve execution errors encountered during training.

***

## 3. Results and Findings

### 3.1. Model Selection Justification

The model selection adhered strictly to the business rule of $K>4$ and prioritized the maximum Silhouette Score.

The comparative analysis of all successful model runs is as follows:

| Model | Optimal K | Silhouette Score (Higher is Better) | Davies-Bouldin Index (Lower is Better) | $K>4$ Constraint? |
| :--- | :--- | :--- | :--- | :--- |
| K-Means (Absolute Optimum) | 2 | **0.8805** | **0.3855** | No (Rejected)|
| K-Means (Local Optimum) | 9 | 0.4856 | (Not recorded in summary) | Yes |
| X-Means (Simulated) | 20 | 0.2782 | 0.7938 | Yes|
| **Autoencoder + K-Means**| **5** | **0.6991** | 1.5314 | **Yes (Selected)**|

The **Autoencoder + K-Means model at $K=5$** was the decisive choice. Its Silhouette Score of $\mathbf{0.6991}$ significantly exceeded the next best constrained model (K-Means at $K=9$, score $0.4856$), confirming the highest cluster separation quality under the business requirement.

### 3.2. Cluster Profiles and Explainability

The final model segmented the materials into **5 distinct clusters**.

The features most critical in defining these 5 clusters were identified as **$\text{MINBE}$, $\text{BSTMA}$, and $\text{AUSSS}$**. **SHAP (SHapley Additive exPlanations)** analysis was performed using a surrogate model to provide robust, quantitative feature contribution scores for each cluster assignment.

### 3.3. Outlier Identification

Across the final model, approximately **$5.0\%$** of the dataset was flagged as anomalous. These materials often contain extreme numerical values that deviate from the norm.

### 3.4. Visualization



The Silhouette Score vs K chart provides empirical context for the model selection process, clearly showing the maximum at $K=2$ and the superior alternative $K=5$ selected after applying the constraint.

Further plots, such as `pca_clusters.png` and `pca_3d_visualization.png`, visually confirm the successful separation of the 5 clusters in the reduced dimensional space.

### 3.5. Model Persistence and Checkpointing

A persistent **checkpointing mechanism** was implemented using `JLD2.jl` after every resource-intensive step.

* **Intermediate Checkpoints:** PCA model and the results of the K-Means sweep (all K values) are saved to allow the `main_run.jl` script to resume immediately after a failure.
* **Final Models:** The final Autoencoder state, the PCA model, and the $K=5$ K-Means model are saved to the `models_julia/` directory.

***

## 5. Conclusion and Next Steps

The project successfully delivered a robust, memory-optimized clustering solution in Julia. The **Autoencoder + K-Means model ($K=5$)** is the officially recommended segmentation strategy.

**Recommended next steps include:**
* **SHAP Interpretation:** Detailed interpretation of the SHAP output to derive actionable business meanings for the top features and the final 5 clusters.
* **Data Quality Initiative:** Investigation of the identified $\approx 5.0\%$ outliers to ensure data governance improvements.
* **Downstream Applications:** Utilizing the cluster assignments for inventory and procurement optimization.