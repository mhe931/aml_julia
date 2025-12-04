# Main Execution Script for Advanced Unsupervised Clustering (Julia)

import Pkg
# Ensure required packages are installed
# Pkg.add(["CSV", "DataFrames", "Clustering", "MultivariateStats", "Flux", "Statistics", "LinearAlgebra", "Random", "Distances", "JLD2", "FileIO", "DecisionTree"])

using CSV, DataFrames, Clustering, MultivariateStats, Flux, Statistics, LinearAlgebra, Random, Distances, JLD2, FileIO, DecisionTree

# Include modules
include("data_preprocessing.jl")
include("model_training.jl")
include("metrics_and_results.jl")

# Main execution
println("Starting Advanced Unsupervised Clustering Pipeline...")

if !isdir(MODEL_DIR)
    mkdir(MODEL_DIR)
end

# 1. Load and Preprocess Data
data_pca, df_encoded, pca_model, scaler_params = load_and_preprocess_data(DATA_FILE)

if data_pca !== nothing
    # 2. Train Models
    
    # Model 1: K-Means
    kmeans_model, best_k_kmeans, best_score_kmeans, _ = train_kmeans_julia(data_pca, 5:20)
    save(joinpath(MODEL_DIR, "kmeans_model.jld2"), "model", kmeans_model)

    # Model 2: X-Means (Simulated)
    xmeans_model, best_k_xmeans = train_xmeans_simulated(data_pca)
    save(joinpath(MODEL_DIR, "xmeans_model.jld2"), "model", xmeans_model)

    # Model 3: Autoencoder + K-Means
    ae_model, ae_kmeans, latent_data = train_autoencoder_clustering(data_pca)
    model_state = Flux.state(ae_model)
    save(joinpath(MODEL_DIR, "autoencoder_state.jld2"), "state", model_state)
    save(joinpath(MODEL_DIR, "ae_kmeans_model.jld2"), "model", ae_kmeans)

    # 3. Metrics and Results
    compile_and_save_results(data_pca, df_encoded, kmeans_model, best_k_kmeans, best_score_kmeans, xmeans_model, best_k_xmeans, ae_kmeans, latent_data, OUTPUT_FILE)
    
    println("Pipeline completed successfully.")
else
    println("Pipeline failed: Data loading error.")
end
