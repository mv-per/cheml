from typing import Optional
import pickle
import mlflow
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

router = APIRouter(prefix="/manufacturing", tags=["manufacturing"])

# Global variables for model caching
_model: Optional[RandomForestRegressor] = None
_scaler: Optional[StandardScaler] = None
_model_error: Optional[float] = None


class ManufacturingInput(BaseModel):
    """Input model for manufacturing quality prediction"""

    temperature: float = Field(
        ..., description="Temperature in °C", ge=-50, le=500
    )
    pressure: float = Field(..., description="Pressure in kPa", ge=0, le=1000)
    temperature_x_pressure: Optional[float] = Field(
        None,
        description="Temperature x Pressure (will be calculated if not provided)",
    )
    material_fusion_metric: float = Field(
        ..., description="Material Fusion Metric", ge=0
    )
    material_transformation_metric: float = Field(
        ..., description="Material Transformation Metric", ge=0
    )

    @validator("temperature_x_pressure", always=True)
    def calculate_temp_pressure_interaction(cls, v, values):
        """Calculate temperature x pressure interaction if not provided"""
        if v is None:
            temp = values.get("temperature")
            pressure = values.get("pressure")
            if temp is not None and pressure is not None:
                return temp * pressure
        return v


class QualityPrediction(BaseModel):
    """Output model for quality prediction"""

    quality: float = Field(..., description="Predicted quality rating")
    error: float = Field(..., description="Estimated prediction error")


def load_model_and_scaler():
    """Load the trained model and scaler from MLflow"""
    global _model, _scaler, _model_error

    if _model is None or _scaler is None:
        try:
            # Set MLflow tracking URI

            # Load the latest version of the registered model
            model_name = "ManufacturingQualityRandomForest"
            model_version = "latest"
            model_uri = f"models:/{model_name}/{model_version}"

            # Load the model
            _model = mlflow.sklearn.load_model(model_uri)

            # Get the run ID from the model version
            client = mlflow.tracking.MlflowClient()
            model_version_details = client.get_latest_versions(
                model_name, stages=["None"]
            )
            if not model_version_details:
                raise ValueError(f"No model versions found for {model_name}")

            run_id = model_version_details[0].run_id

            # The scaler is saved in the preprocessing directory
            preprocessing_dir = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="preprocessing"
            )

            # Find the scaler file in the preprocessing directory
            scaler_files = [
                f
                for f in os.listdir(preprocessing_dir)
                if f.endswith("_scaler.pkl") or f.endswith(".pkl")
            ]
            if not scaler_files:
                raise ValueError(
                    "No scaler .pkl files found in preprocessing directory"
                )

            # Use the first scaler file found (should be only one)
            scaler_file_path = os.path.join(preprocessing_dir, scaler_files[0])
            with open(scaler_file_path, "rb") as f:
                _scaler = pickle.load(f)

            # Get the model error from MLflow metrics
            run = mlflow.get_run(run_id)
            _model_error = run.data.metrics.get(
                "test_mse", 1.0
            )  # Default to 1.0 if not found

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model or scaler: {str(e)}",
            )

    return _model, _scaler, _model_error


@router.post("/predict-quality", response_model=QualityPrediction)
async def predict_quality(input_data: ManufacturingInput):
    """
    Predict manufacturing quality based on process parameters

    Returns predicted quality rating and estimated error.
    """
    try:
        # Load model and scaler (cached after first call)
        model, scaler, model_error = load_model_and_scaler()

        # Prepare input features in the same order as training data
        # Expected order: Temperature, Pressure, Temperature x Pressure,
        # Material Fusion Metric, Material Transformation Metric
        features = np.array(
            [
                [
                    input_data.temperature,
                    input_data.pressure,
                    input_data.temperature_x_pressure,
                    input_data.material_fusion_metric,
                    input_data.material_transformation_metric,
                ]
            ]
        )

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Ensure prediction is within reasonable bounds (0-100 for quality rating)
        prediction = max(0, min(100, prediction))

        # Calculate prediction error estimate
        # Use a combination of model error and prediction uncertainty
        # For Random Forest, we can use the standard deviation of tree predictions
        tree_predictions = np.array(
            [tree.predict(features_scaled)[0] for tree in model.estimators_]
        )
        prediction_std = np.std(tree_predictions)

        # Combine model error (MSE) with prediction uncertainty
        estimated_error = np.sqrt(model_error + prediction_std**2)

        return QualityPrediction(
            quality=round(prediction, 4), error=round(estimated_error, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )


@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        model, scaler, model_error = load_model_and_scaler()

        return {
            "model_type": type(model).__name__,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "training_error_mse": model_error,
            "feature_names": [
                "Temperature (°C)",
                "Pressure (kPa)",
                "Temperature x Pressure",
                "Material Fusion Metric",
                "Material Transformation Metric",
            ],
            "scaler_type": type(scaler).__name__,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )
