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
    # Model 1: K-Means
    kmeans_path = joinpath(MODEL_DIR, "kmeans_model.jld2")
    if isfile(kmeans_path)
        println("Loading saved K-Means model...")
        data = load(kmeans_path)
        if haskey(data, "best_k")
            kmeans_model = data["model"]
            best_k_kmeans = data["best_k"]
            best_score_kmeans = data["best_score"]
        else
            println("Metadata missing in checkpoint. Re-evaluating (fast via sweep checkpoint)...")
            kmeans_model, best_k_kmeans, best_score_kmeans, _ = train_kmeans_julia(data_pca, 5:20)
            save(kmeans_path, Dict("model" => kmeans_model, "best_k" => best_k_kmeans, "best_score" => best_score_kmeans))
        end
    else
        kmeans_model, best_k_kmeans, best_score_kmeans, _ = train_kmeans_julia(data_pca, 5:20)
        save(kmeans_path, Dict("model" => kmeans_model, "best_k" => best_k_kmeans, "best_score" => best_score_kmeans))
    end

    # Model 2: X-Means (Simulated)
    xmeans_path = joinpath(MODEL_DIR, "xmeans_model.jld2")
    if isfile(xmeans_path)
        println("Loading saved X-Means model...")
        xmeans_model = load(xmeans_path, "model")
        # We need best_k_xmeans. 
        best_k_xmeans = size(xmeans_model.centers, 2)
    else
        xmeans_model, best_k_xmeans = train_xmeans_simulated(data_pca)
        save(xmeans_path, "model", xmeans_model)
    end

    # Model 3: Autoencoder + K-Means
    ae_path = joinpath(MODEL_DIR, "ae_kmeans_model.jld2")
    if isfile(ae_path) && isfile(joinpath(MODEL_DIR, "autoencoder_state.jld2"))
        println("Loading saved Autoencoder model...")
        ae_kmeans = load(ae_path, "model")
        # We need latent_data for results.
        # We need to load the AE model to transform data.
        # Re-building model structure (should match training function)
        input_dim = size(data_pca, 1)
        encoding_dim = 10
        encoder = Chain(
            Dense(input_dim, 128, relu),
            Dense(128, 64, relu),
            Dense(64, encoding_dim, relu)
        )
        decoder = Chain(
            Dense(encoding_dim, 64, relu),
            Dense(64, 128, relu),
            Dense(128, input_dim)
        )
        ae_model = Chain(encoder, decoder)
        
        model_state = load(joinpath(MODEL_DIR, "autoencoder_state.jld2"), "state")
        Flux.loadmodel!(ae_model, model_state)
        
        latent_data = encoder(data_pca)
    else
        ae_model, ae_kmeans, latent_data = train_autoencoder_clustering(data_pca)
        model_state = Flux.state(ae_model)
        save(joinpath(MODEL_DIR, "autoencoder_state.jld2"), "state", model_state)
        save(joinpath(MODEL_DIR, "ae_kmeans_model.jld2"), "model", ae_kmeans)
    end

    # 3. Metrics and Results
    compile_and_save_results(data_pca, df_encoded, kmeans_model, best_k_kmeans, best_score_kmeans, xmeans_model, best_k_xmeans, ae_kmeans, latent_data, OUTPUT_FILE)
    
    println("Pipeline completed successfully.")
else
    println("Pipeline failed: Data loading error.")
end
