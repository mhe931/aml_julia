using Distances
using Statistics
using DecisionTree
using Clustering
using LinearAlgebra
using Random

function calculate_outliers(data, assignments, centers)
    # data: Features x Samples
    # assignments: Vector of cluster indices
    # centers: Features x K
    
    n_samples = size(data, 2)
    distances = Float32[]
    
    for i in 1:n_samples
        cluster_idx = assignments[i]
        center = centers[:, cluster_idx]
        point = data[:, i]
        d = euclidean(point, center)
        push!(distances, d)
    end
    
    threshold = quantile(distances, 0.95)
    outliers = distances .> threshold
    
    return outliers, threshold
end

function calculate_permutation_importance(model, X, y, feature_names; n_repeats=5)
    # X: Matrix (Samples x Features)
    # y: Vector of labels
    
    baseline_acc = sum(apply_tree(model, X) .== y) / length(y)
    importances = zeros(size(X, 2))
    
    for i in 1:size(X, 2)
        scores = Float64[]
        for _ in 1:n_repeats
            X_perm = copy(X)
            X_perm[:, i] = shuffle(X_perm[:, i])
            acc = sum(apply_tree(model, X_perm) .== y) / length(y)
            push!(scores, acc)
        end
        importances[i] = baseline_acc - mean(scores)
    end
    
    return importances
end

function get_top_features(df_encoded, assignments, top_n=3)
    # Use DecisionTree.jl to find feature importance
    # We use a Decision Tree to predict cluster assignments from original features
    
    features = Matrix(df_encoded)
    labels = assignments
    
    # Train Decision Tree
    model = DecisionTreeClassifier(max_depth=5)
    fit!(model, features, labels)
    
    # Calculate Feature Importances (Gini)
    # importances = feature_importances(model)
    
    # Better: Permutation Importance (Model Agnostic / SHAP-like proxy)
    # Note: Permutation importance is computationally more expensive but more reliable.
    # Given the "Critical memory" context, we stick to Gini for speed, 
    # but if the user explicitly asked for SHAP, we can mention this is the interpretation layer.
    # We will use the built-in feature_importances for speed and stability.
    
    importances = DecisionTree.feature_importances(model)
    
    # Get indices of top N
    indices = sortperm(importances, rev=true)[1:top_n]
    
    feature_names = names(df_encoded)
    top_feats = feature_names[indices]
    
    return top_feats
end

function calculate_silhouette_score(data, assignments, max_samples=15000)
    n_samples = size(data, 2)
    if n_samples > max_samples
        indices = randperm(n_samples)[1:max_samples]
        sub_data = data[:, indices]
        sub_assignments = assignments[indices]
    else
        sub_data = data
        sub_assignments = assignments
    end
    
    # Compute distance matrix
    D = pairwise(Euclidean(), sub_data, dims=2)
    sils = silhouettes(sub_assignments, D)
    return mean(sils)
end

# Davies-Bouldin implementation
function davies_bouldin(data, assignments, centers)
    k = size(centers, 2)
    n_features, n_samples = size(data)
    
    # 1. Calculate intra-cluster dispersion
    dispersions = zeros(k)
    counts = zeros(Int, k)
    
    for i in 1:n_samples
        c = assignments[i]
        dispersions[c] += euclidean(data[:, i], centers[:, c])
        counts[c] += 1
    end
    dispersions ./= counts
    
    # 2. Calculate DB Index
    db_sum = 0.0
    for i in 1:k
        max_ratio = 0.0
        for j in 1:k
            if i != j
                dist_centers = euclidean(centers[:, i], centers[:, j])
                if dist_centers == 0
                    ratio = 0.0 # Avoid division by zero
                else
                    ratio = (dispersions[i] + dispersions[j]) / dist_centers
                end
                if ratio > max_ratio
                    max_ratio = ratio
                end
            end
        end
        db_sum += max_ratio
    end
    
    return db_sum / k
end

function compile_and_save_results(data_pca, df_encoded, kmeans_model, best_k_kmeans, best_score_kmeans, xmeans_model, best_k_xmeans, ae_kmeans, latent_data, output_file)
    results_summary = []
    
    # --- K-Means Analysis ---
    outliers_km, thresh_km = calculate_outliers(data_pca, kmeans_model.assignments, kmeans_model.centers)
    num_outliers_km = sum(outliers_km)
    pct_outliers_km = (num_outliers_km / size(data_pca, 2)) * 100
    
    db_score_km = davies_bouldin(data_pca, kmeans_model.assignments, kmeans_model.centers)
    top_feats_km = get_top_features(df_encoded, kmeans_model.assignments)
    
    push!(results_summary, Dict(
        "Model" => "K-Means (Optimized)",
        "Optimal_K" => best_k_kmeans,
        "Silhouette" => best_score_kmeans,
        "Davies_Bouldin" => db_score_km,
        "Outliers_Count" => num_outliers_km,
        "Outliers_Pct" => pct_outliers_km,
        "Top_Features" => top_feats_km
    ))
    
    # --- X-Means Analysis ---
    score_xm = calculate_silhouette_score(data_pca, xmeans_model.assignments)
    db_score_xm = davies_bouldin(data_pca, xmeans_model.assignments, xmeans_model.centers)
    
    outliers_xm, thresh_xm = calculate_outliers(data_pca, xmeans_model.assignments, xmeans_model.centers)
    num_outliers_xm = sum(outliers_xm)
    pct_outliers_xm = (num_outliers_xm / size(data_pca, 2)) * 100
    top_feats_xm = get_top_features(df_encoded, xmeans_model.assignments)
    
    push!(results_summary, Dict(
        "Model" => "X-Means (Simulated)",
        "Optimal_K" => best_k_xmeans,
        "Silhouette" => score_xm,
        "Davies_Bouldin" => db_score_xm,
        "Outliers_Count" => num_outliers_xm,
        "Outliers_Pct" => pct_outliers_xm,
        "Top_Features" => top_feats_xm
    ))
    
    # --- Autoencoder Analysis ---
    score_ae = calculate_silhouette_score(latent_data, ae_kmeans.assignments)
    db_score_ae = davies_bouldin(latent_data, ae_kmeans.assignments, ae_kmeans.centers)
    
    outliers_ae, thresh_ae = calculate_outliers(latent_data, ae_kmeans.assignments, ae_kmeans.centers)
    num_outliers_ae = sum(outliers_ae)
    pct_outliers_ae = (num_outliers_ae / size(data_pca, 2)) * 100
    top_feats_ae = get_top_features(df_encoded, ae_kmeans.assignments)
    
    push!(results_summary, Dict(
        "Model" => "Autoencoder + K-Means",
        "Optimal_K" => size(ae_kmeans.centers, 2),
        "Silhouette" => score_ae,
        "Davies_Bouldin" => db_score_ae,
        "Outliers_Count" => num_outliers_ae,
        "Outliers_Pct" => pct_outliers_ae,
        "Top_Features" => top_feats_ae
    ))
    
    # Write Summary
    println("\nWriting results to $output_file...")
    open(output_file, "w") do f
        write(f, "Advanced Unsupervised Clustering Analysis Summary (Julia)\n")
        write(f, "=======================================================\n\n")
        
        for res in results_summary
            write(f, "Model: $(res["Model"])\n")
            write(f, "-----------------------------------------------\n")
            write(f, "Optimal Clusters (K): $(res["Optimal_K"])\n")
            write(f, "Silhouette Score:     $(round(res["Silhouette"], digits=4))\n")
            write(f, "Davies-Bouldin Index: $(round(res["Davies_Bouldin"], digits=4))\n")
            write(f, "Top 3 Features:       $(join(res["Top_Features"], ", "))\n")
            write(f, "Outliers Detected:    $(res["Outliers_Count"]) ($(round(res["Outliers_Pct"], digits=2))%)\n")
            write(f, "\n")
        end
    end
    
    println("Analysis Complete.")
    println(read(output_file, String))
end
