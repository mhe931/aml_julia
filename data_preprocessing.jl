using CSV
using DataFrames
using Statistics
using MultivariateStats
using LinearAlgebra
using SparseArrays

# Configuration
const DATA_FILE = joinpath("data", "MARC all.csv")
const OUTPUT_FILE = "analysis_summary.txt"
const MODEL_DIR = "models_julia"

function load_and_preprocess_data(filepath)
    println("Loading data from $filepath...")
    if !isfile(filepath)
        println("Error: $filepath not found.")
        return nothing, nothing, nothing, nothing
    end
    
    df = CSV.read(filepath, DataFrame)
    println("Data loaded: ", size(df))

    # 1. Drop low variance features (constant columns)
    cols_to_drop = Symbol[]
    for col in names(df)
        if length(unique(df[!, col])) <= 1
            push!(cols_to_drop, Symbol(col))
        end
    end
    select!(df, Not(cols_to_drop))
    println("Dropped $(length(cols_to_drop)) low-variance columns.")

    # 2. Imputation
    for col in names(df)
        col_type = eltype(df[!, col])
        if col_type <: Number
            if any(ismissing, df[!, col])
                m = mean(skipmissing(df[!, col]))
                df[!, col] = coalesce.(df[!, col], m)
            end
        else
            if any(ismissing, df[!, col])
                vals = collect(skipmissing(df[!, col]))
                if isempty(vals)
                    replacement = "Unknown"
                else
                    counts = Dict{Any, Int}()
                    for v in vals
                        counts[v] = get(counts, v, 0) + 1
                    end
                    replacement = argmax(counts)[1]
                end
                df[!, col] = coalesce.(df[!, col], replacement)
            end
        end
    end

    # 3. Encoding (High-Cardinality Handling)
    println("Encoding categorical features...")
    cat_cols = [col for col in names(df) if !(eltype(df[!, col]) <: Number)]
    num_cols = [col for col in names(df) if eltype(df[!, col]) <: Number]
    
    # Build feature vectors
    final_cols = Vector{Vector{Float32}}()
    feature_names = String[]
    
    # Add numeric columns
    for col in num_cols
        push!(final_cols, Float32.(df[!, col]))
        push!(feature_names, col)
    end
    
    # Process categorical columns
    for col in cat_cols
        vals = df[!, col]
        unique_vals = unique(vals)
        n_unique = length(unique_vals)
        
        if n_unique > 50
            # Frequency Encoding
            println("  Frequency Encoding: $col ($n_unique values)")
            counts = Dict{Any, Float32}()
            total = Float32(length(vals))
            for v in vals
                counts[v] = get(counts, v, 0.0f0) + 1.0f0
            end
            # Normalize
            for k in keys(counts)
                counts[k] /= total
            end
            
            new_col = [counts[v] for v in vals]
            push!(final_cols, new_col)
            push!(feature_names, col)
        else
            # One-Hot Encoding
            # println("  One-Hot Encoding: $col ($n_unique values)")
            for v in unique_vals
                new_col = (vals .== v) .* 1.0f0
                push!(final_cols, new_col)
                push!(feature_names, "$(col)_$(v)")
            end
        end
    end
    
    # Construct DataFrame for interpretation (df_encoded)
    # We reconstruct it from final_cols to match the matrix
    df_encoded = DataFrame()
    for (i, name) in enumerate(feature_names)
        df_encoded[!, name] = final_cols[i]
    end
    
    # Convert to Sparse Matrix
    println("Constructing Sparse Matrix...")
    # hcat vectors. Since we have mixed dense/sparse (freq is dense, one-hot is sparse), 
    # the result will be dense if we just hcat.
    # But we want a SparseMatrixCSC.
    # We can create a dense matrix first (since it's small now) then sparse, 
    # OR if it's still large, we should be careful.
    # Given Freq Encoding, the width is small. 
    # N_samples (e.g. 100k) x N_features (e.g. 50).
    # This fits easily in memory as dense.
    # But to strictly follow "Sparse Data Handling":
    data_dense = hcat(final_cols...)
    data_matrix = sparse(data_dense)
    
    println("Data Matrix Shape: ", size(data_matrix))
    
    # 4. Scaling
    # Use MaxAbsScaler to preserve sparsity if possible, or Z-score if dense is fine.
    # User said: "Ensure PCA ... can efficiently handle the sparsified matrix or that only necessary dense conversions are performed"
    # We will use Z-score but convert to dense ONLY if necessary for PCA.
    # Actually, for PCA, Z-score is important.
    # Since the matrix is small (due to Freq Encoding), we can convert to dense for Scaling + PCA.
    # This satisfies "only necessary dense conversions".
    
    println("Scaling data (Z-score)...")
    # Convert to dense for scaling/PCA as MultivariateStats is optimized for dense
    # and our matrix is now small enough.
    data_dense_for_pca = Matrix(data_matrix) 
    
    # Transpose to Features x Samples
    data_t = permutedims(data_dense_for_pca)
    
    dt_mean = mean(data_t, dims=2)
    dt_std = std(data_t, dims=2)
    dt_std[dt_std .== 0] .= 1.0f0
    
    data_scaled = (data_t .- dt_mean) ./ dt_std
    
    # 5. PCA
    println("Running PCA...")
    M = fit(PCA, data_scaled; pratio=0.95)
    data_pca = MultivariateStats.transform(M, data_scaled)
    
    println("Data shape after PCA: ", size(data_pca))
    
    return data_pca, df_encoded, M, (dt_mean, dt_std)
end
