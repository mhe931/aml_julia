using CSV
using DataFrames
using Statistics
using MultivariateStats
using LinearAlgebra

# Configuration
const DATA_FILE = joinpath("data", "MARC all.csv")
const OUTPUT_FILE = "analysis_summary.txt"
const MODEL_DIR = "models_julia"

function load_and_preprocess_data(filepath)
    println("Loading data...")
    if !isfile(filepath)
        println("Error: $filepath not found.")
        return nothing, nothing, nothing, nothing
    end
    
    df = CSV.read(filepath, DataFrame)
    println("Data loaded: ", size(df))

    # 1. Drop low variance features (constant columns)
    # Identify columns with only 1 unique value
    cols_to_drop = Symbol[]
    for col in names(df)
        if length(unique(df[!, col])) <= 1
            push!(cols_to_drop, Symbol(col))
        end
    end
    select!(df, Not(cols_to_drop))
    println("Dropped $(length(cols_to_drop)) low-variance columns.")

    # 2. Imputation
    # Numeric: Mean, Categorical: Mode
    for col in names(df)
        col_type = eltype(df[!, col])
        if col_type <: Number
            # Handle missing for numeric
            if any(ismissing, df[!, col])
                m = mean(skipmissing(df[!, col]))
                df[!, col] = coalesce.(df[!, col], m)
            end
        else
            # Handle missing for categorical
            if any(ismissing, df[!, col])
                # Find mode
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

    # 3. Encoding
    println("Encoding categorical features...")
    # Identify categorical columns (String or non-Number)
    cat_cols = [col for col in names(df) if !(eltype(df[!, col]) <: Number)]
    
    # One-Hot Encoding manually or via package. 
    # For speed and control, we do a simple manual expansion or use Flux's onehot if needed, 
    # but DataFrames transformation is often easier.
    # We will use a simple approach: create dummy variables.
    
    df_encoded = copy(df)
    for col in cat_cols
        # Get unique values
        vals = unique(df[!, col])
        # Create dummy columns
        for v in vals
            # Avoid multicollinearity by dropping one? Usually yes, but for clustering distance it's debatable.
            # We'll keep all for now or drop first. Let's drop first implicitly by not creating it? 
            # Standard practice: drop first.
            if v == vals[1] continue end
            
            new_col_name = "$(col)_$(v)"
            df_encoded[!, new_col_name] = (df[!, col] .== v) .* 1.0
        end
    end
    select!(df_encoded, Not(cat_cols))
    
    # Convert to Matrix for Clustering/Flux
    # Ensure all are float
    data_matrix = Matrix{Float32}(df_encoded)
    
    # 4. Scaling (Z-score)
    println("Scaling data...")
    # Compute mean and std per column (feature)
    # data_matrix is N x Features. Clustering.jl expects Features x N usually.
    # Let's transpose now to Features x N (standard for Julia ML)
    data_t = permutedims(data_matrix)
    
    dt_mean = mean(data_t, dims=2)
    dt_std = std(data_t, dims=2)
    # Avoid division by zero
    dt_std[dt_std .== 0] .= 1.0
    
    data_scaled = (data_t .- dt_mean) ./ dt_std
    
    # 5. PCA
    println("Running PCA...")
    # Keep 95% variance
    M = fit(PCA, data_scaled; pratio=0.95)
    data_pca = MultivariateStats.transform(M, data_scaled)
    
    println("Data shape after PCA: ", size(data_pca))
    
    return data_pca, df_encoded, M, (dt_mean, dt_std)
end
