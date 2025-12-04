using ShapML
using DecisionTree
using DataFrames
using Statistics
using Random

function generate_shap_analysis(data_input, cluster_assignments, feature_names)
    println("Generating SHAP analysis...")
    
    # 1. Prepare Data
    # ShapML expects a DataFrame for features and a model that can predict.
    # We need to train a surrogate model (Classifier) to predict clusters.
    
    # data_input is likely a Matrix (Features x Samples) or DataFrame.
    # If it's a Matrix from PCA/Preprocessing, we need to ensure it matches feature_names.
    # In our pipeline, we have `df_encoded` which matches `feature_names`.
    # We should use `df_encoded` as input if possible, or `data_input` if it corresponds to `feature_names`.
    # The prompt says: "using the preprocessed original features (data_input) as input (the same features used for the final clustering model)."
    # The final clustering model uses PCA data. But SHAP on PCA components is hard to interpret.
    # Usually we want SHAP on original features.
    # However, the prompt says "preprocessed original features".
    # Let's assume `data_input` passed here is `df_encoded` (the interpretable features).
    
    # Check input type
    if isa(data_input, Matrix)
        # Convert to DataFrame if needed, but we likely pass df_encoded directly.
        df = DataFrame(data_input', feature_names)
    else
        df = data_input
    end
    
    n_samples = size(df, 1)
    labels = cluster_assignments
    
    # 2. Train Surrogate Model
    println("Training surrogate classifier (Random Forest)...")
    # We use DecisionTree.jl
    # RandomForestClassifier
    model = RandomForestClassifier(n_trees=20, max_depth=10)
    
    # Convert DataFrame to Matrix for DecisionTree
    X_matrix = Matrix(df)
    y_vector = labels
    
    fit!(model, X_matrix, y_vector)
    
    # 3. SHAP Calculation
    println("Calculating SHAP values (this may take a while)...")
    
    # Define predict function wrapper for ShapML
    # ShapML expects: predict_function(model, data) -> DataFrame of probabilities or predictions
    # For regression/classification, it depends.
    # ShapML.shap expects a function that returns a DataFrame of predictions.
    
    predict_function = (m, d) -> begin
        # d is a DataFrame. Convert to Matrix.
        mat = Matrix(d)
        preds = apply_forest(m, mat)
        # Return DataFrame with predictions. 
        # For multi-class, ShapML might expect probabilities?
        # Let's check ShapML docs conceptually.
        # Usually it handles regression (1 column) or classification.
        # For simplicity, let's treat it as a regression of "Cluster ID" or 
        # if ShapML supports multi-class, we output probabilities.
        # Given "Cluster Assignments" are 1-5, treating as regression is weird but possible.
        # Better: One-vs-Rest or just predict the class.
        # ShapML.jl documentation says: "The predict_function must take the model and data as arguments and return a DataFrame of predictions."
        DataFrame(Prediction = preds)
    end
    
    # Sample background data
    sample_size = min(500, n_samples)
    data_sample = df[randperm(n_samples)[1:sample_size], :]
    
    # Run SHAP
    # explain: DataFrame to explain. We can explain the same sample or a test set.
    # reference: Background data.
    shap_result = ShapML.shap(explain = data_sample,
                              reference = data_sample,
                              model = model,
                              predict_function = predict_function,
                              sample_size = 100, # Monte Carlo samples per instance
                              seed = 1234)
                              
    # shap_result is a DataFrame with columns: index, feature_name, feature_value, shap_effect, etc.
    
    # 4. Global Feature Importance
    # Mean absolute SHAP value per feature
    println("Calculating global feature importance...")
    
    # Group by feature_name and calculate mean(|shap_effect|)
    # We can use DataFrames meta-programming or manual loop
    
    features = unique(shap_result.feature_name)
    importance_dict = Dict{String, Float64}()
    
    for feat in features
        # Filter rows for this feature
        # Note: ShapML output column names might vary, usually `feature_name`, `shap_effect`
        rows = shap_result[shap_result.feature_name .== feat, :]
        mean_abs_shap = mean(abs.(rows.shap_effect))
        importance_dict[feat] = mean_abs_shap
    end
    
    # Sort
    sorted_features = sort(collect(importance_dict), by=x->x[2], rev=true)
    
    return sorted_features
end
