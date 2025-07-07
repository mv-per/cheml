from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import tempfile
import os


optimization_path = Path(__file__).parent

# Load and prepare the manufacturing quality data
df = pd.read_csv(optimization_path / "manufacturing_quality.csv")

# Prepare features and target
y = df["Quality Rating"].values
X = df.drop(columns=["Quality Rating"]).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create RandomForest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
)

# Set up MLflow tracking
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Manufacturing Quality RF Optimization")

mlflow.sklearn.autolog()

# Train the model with MLflow logging
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("max_depth", model.max_depth)
    mlflow.log_param("min_samples_split", model.min_samples_split)
    mlflow.log_param("min_samples_leaf", model.min_samples_leaf)
    mlflow.log_param("random_state", model.random_state)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("train_samples", len(X_tr))
    mlflow.log_param("val_samples", len(X_val))
    mlflow.log_param("test_samples", len(X_test))

    # Train the model
    model.fit(X_tr_scaled, y_tr)

    # Validate the model
    y_val_pred = model.predict(X_val_scaled)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    # Log validation metrics
    mlflow.log_metric("val_mse", val_mse)
    mlflow.log_metric("val_r2", val_r2)

    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Validation R²: {val_r2:.4f}")

    # Test the model
    y_test_pred = model.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Log test metrics
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_r2", test_r2)

    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # Get predictions for all test data
    sample_names = [f"sample_{i}" for i in range(len(y_test))]

    # Create DataFrame with results
    results_df = pd.DataFrame(
        {
            "sample_name": sample_names,
            "y_true": y_test,
            "y_predicted": y_test_pred,
            "absolute_error": np.abs(y_test - y_test_pred),
        }
    )

    # Log the predictions DataFrame as a table in MLflow
    mlflow.log_table(
        results_df,
        artifact_file="quality_predictions_randomforest.json",
    )

    # Log feature importance
    feature_names = df.columns[:-1].tolist()  # Exclude target column
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    mlflow.log_table(
        feature_importance, artifact_file="feature_importance.json"
    )

    # Save the trained model to MLflow
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="ManufacturingQualityRandomForest",
    )

    # Save the fitted scaler to MLflow
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        pickle.dump(scaler, f)
        scaler_path = f.name

    scaler_final_path = None
    try:
        # Rename the temp file to have a proper name before logging
        import shutil

        scaler_final_path = scaler_path + "_scaler.pkl"
        shutil.move(scaler_path, scaler_final_path)
        mlflow.log_artifact(scaler_final_path, artifact_path="preprocessing")
    finally:
        # Clean up - the file might have been moved, so check both paths
        if os.path.exists(scaler_path):
            os.unlink(scaler_path)
        if scaler_final_path and os.path.exists(scaler_final_path):
            os.unlink(scaler_final_path)

    # Log individual predictions as metrics with step index
    for idx, row in results_df.iterrows():
        mlflow.log_metric("y_true", row["y_true"], step=idx)
        mlflow.log_metric("y_predicted", row["y_predicted"], step=idx)
        mlflow.log_metric("absolute_error", row["absolute_error"], step=idx)
