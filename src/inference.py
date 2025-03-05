import hopsworks
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from hsfs.feature_store import FeatureStore
import joblib
from pathlib import Path
import src.config as config

def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the trained model to predict demand. Expects features to contain only the model input columns.
    """
    preds = model.predict(features)
    results = pd.DataFrame()
    results["predicted_demand"] = preds.round(0)
    return results

def load_model_from_registry():
    """
    Loads the latest model from the Hopsworks model registry.
    """
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()
    models = model_registry.get_models(name=config.MODEL_NAME)
    # Select the model with the highest version number
    model_obj = max(models, key=lambda m: m.version)
    model_dir = model_obj.download()
    model = joblib.load(Path(model_dir) / "lgb_model.pkl")
    return model
