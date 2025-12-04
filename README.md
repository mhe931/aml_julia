# Material Master Unsupervised Clustering

This project performs an unsupervised clustering analysis on material master data to identify distinct material groups and flag potential data quality issues.

## Project Overview

The main goal of this project is to analyze over 300,000 records of material master data. By applying unsupervised machine learning techniques, we can:

-   **Group similar materials** into distinct clusters based on their MRP, Purchasing, and Warehousing features.
-   **Identify outliers** that may represent data quality problems or unique materials.
-   **Provide insights** into the material master data landscape.

The analysis uses `MiniBatchKMeans` for efficient clustering on a large dataset and `IsolationForest` for outlier detection.

## Setup and Installation

To run this analysis, you need to have Python 3 installed, as well as the libraries listed in the `requirements.txt` file.

1.  **Clone the repository** (or download the files).

2.  **Create a virtual environment** (recommended):
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Place the data file** `MARC all.csv` into the `data/` directory.

## How to Run the Analysis

Once the setup is complete, you can run the entire analysis by executing the main script:

```sh
python material_clustering.py
```

The script will perform all the steps from data loading and preprocessing to clustering and visualization.

## Outputs

The script will generate the following files:

-   `analysis_output.txt`: A text file containing all the logs and summary statistics from the analysis.
-   `clustered_materials.csv`: A CSV file with the original material IDs, their assigned cluster, anomaly score, and an outlier flag.
-   `elbow_silhouette_analysis.png`: A plot showing the Elbow and Silhouette scores used to determine the optimal number of clusters.
-   `pca_clusters.png`: A 2D visualization of the clusters and outliers using PCA.
-   `pca_3d_visualization.png`: A 3D visualization of the clusters and outliers.
-   `outlier_analysis.png`: A plot showing the distribution of anomaly scores and the count of outliers vs. inliers.

## Project Structure

```
.
├── data/
│   └── MARC all.csv
├── material_clustering.py
├── material_clustering.ipynb
├── project_report.md
├── README.md
└── requirements.txt
```
