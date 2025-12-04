using Clustering
using Flux
using Statistics
using LinearAlgebra
using Random
using JLD2
using FileIO
using Distances

function train_kmeans_julia(data, k_range)
    println("Training K-Means...")
    best_score = -1.0
    best_k = -1
    best_model = nothing
    
    checkpoint_file = joinpath("models_julia", "kmeans_sweep.jld2")
    if isfile(checkpoint_file)
        println("Loading existing K-Means results...")
        results = load(checkpoint_file, "results")
    else
        results = Dict()
    end

    for k in k_range
        if haskey(results, k)
            println("Skipping K=$k (already computed). Silhouette=$(round(results[k], digits=4))")
            score = results[k]
            if score > best_score
                best_score = score
                best_k = k
            end
            continue
        end

        # Clustering.jl kmeans expects Features x Samples
        result = kmeans(data, k; maxiter=100, display=:none)
        
        # Calculate Silhouette on Subsample (max 15k)
        n_samples = size(data, 2)
        if n_samples > 15000
            indices = randperm(n_samples)[1:15000]
            sub_data = data[:, indices]
            sub_assignments = result.assignments[indices]
        else
            sub_data = data
            sub_assignments = result.assignments
        end
        
        # Compute distance matrix (required for silhouettes in some versions)
        D = pairwise(Euclidean(), sub_data, dims=2)
        sils = silhouettes(sub_assignments, D)
        score = mean(sils)
        
        results[k] = score
        save(checkpoint_file, "results", results)
        println("K=$k, Silhouette=$(round(score, digits=4))")
        
        if score > best_score
            best_score = score
            best_k = k
            best_model = result
        end
    end
    
    if best_k != -1 && best_model === nothing
        println("Retraining best K=$best_k to recover model...")
        best_model = kmeans(data, best_k; maxiter=100, display=:none)
    end
            
    return best_model, best_k, best_score, results
end

function train_xmeans_simulated(data, max_k=20)
    println("Training X-Means (Simulated via BIC)...")
    best_bic = Inf
    best_k = -1
    best_model = nothing
    
    d, n_samples = size(data)
    
    for k in 2:max_k
        result = kmeans(data, k; maxiter=100, display=:none)
        
        # Calculate BIC
        # WCSS (Within-Cluster Sum of Squares) is result.totalcost
        wcss = result.totalcost
        
        # Variance estimate
        bic = n_samples * log(wcss / n_samples) + k * log(n_samples)
        
        if bic < best_bic
            best_bic = bic
            best_k = k
            best_model = result
        end
    end
            
    return best_model, best_k
end

function train_autoencoder_clustering(data, encoding_dim=10, epochs=20)
    println("Training Autoencoder...")
    input_dim = size(data, 1)
    n_samples = size(data, 2)
    
    # Define Model
    encoder = Chain(
        Dense(input_dim, 128, relu),
        Dense(128, 64, relu),
        Dense(64, encoding_dim, relu)
    )
    
    decoder = Chain(
        Dense(encoding_dim, 64, relu),
        Dense(64, 128, relu),
        Dense(128, input_dim) # Linear output for scaled data
    )
    
    model = Chain(encoder, decoder)
    
    # Loss function (must accept model for explicit gradient)
    loss(m, x) = Flux.mse(m(x), x)
    
    # Optimizer
    # Modern Flux: Setup optimizer state
    opt_state = Flux.setup(Adam(0.001), model)
    
    # Data Loader
    # Flux expects batches. 
    batch_size = 256
    data_loader = Flux.DataLoader(data, batchsize=batch_size, shuffle=true)
    
    # Training Loop
    for epoch in 1:epochs
        for x_batch in data_loader
            # Explicit gradient calculation
            val, grads = Flux.withgradient(model) do m
                loss(m, x_batch)
            end
            Flux.update!(opt_state, model, grads[1])
        end
        
        # Calculate epoch loss
        current_loss = loss(model, data)
        println("Epoch $epoch/$epochs, Loss: $(round(current_loss, digits=4))")
    end
    
    # Extract Latent
    latent_data = encoder(data)
    
    println("Clustering on Latent Space...")
    kmeans_ae = kmeans(latent_data, 5; maxiter=100) # Fixed K=5 or search
    
    return model, kmeans_ae, latent_data
end
