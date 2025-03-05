from datetime import datetime, timedelta
import pandas as pd
import hopsworks
from hsfs.feature_store import FeatureStore
from pathlib import Path
import joblib
import src.config as config
from src.data_utils import transform_ts_data_info_features_aggregated

# --- Hopsworks Connection Functions ---
def get_hopsworks_project():
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store():
    project = get_hopsworks_project()
    return project.get_feature_store()

# --- Model Prediction Function ---
def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(features)
    results = pd.DataFrame()
    results["predicted_demand"] = preds.round(0)
    return results

# --- Batch Data Loading & Transformation ---
def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    fs = get_feature_store()
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    
    feature_view = fs.get_feature_view(name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION)
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
        read_options={"arrow_flight_config": {"use_spark": True, "timeout": 60000}}
    )
    # Filter to the desired window
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    ts_data.sort_values(by=["pickup_location_id", "pickup_hour"], inplace=True)
    ts_data = ts_data.reset_index(drop=True)
    ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)
    
    # Aggregate raw data into 5 features per location
    aggregated_features = transform_ts_data_info_features_aggregated(ts_data, feature_col="rides")
    return aggregated_features

# --- Model Registry Functions ---
def load_model_from_registry():
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()
    models = model_registry.get_models(name=config.MODEL_NAME)
    # Choose the model with the highest version
    model_obj = max(models, key=lambda m: m.version)
    model_dir = model_obj.download()
    model = joblib.load(Path(model_dir) / "lgb_model.pkl")
    return model

# --- Main Inference Pipeline ---
def main():
    # Get current UTC time
    current_date = pd.Timestamp.now(tz='Etc/UTC')
    
    # Load aggregated features from the feature store
    aggregated_features = load_batch_of_features_from_store(current_date)
    print("Aggregated features shape:", aggregated_features.shape)
    # Expect aggregated_features to have columns: 
    # ["pickup_location_id", "mean_rides", "min_rides", "max_rides", "std_rides", "sum_rides"]

    # Prepare model input by dropping the identifier column
    if "pickup_location_id" in aggregated_features.columns:
        model_input = aggregated_features.drop(columns=["pickup_location_id"])
        identifiers = aggregated_features[["pickup_location_id"]]
    else:
        model_input = aggregated_features
        identifiers = None
    
    print("Model input shape (should be 5 columns):", model_input.shape)
    
    # Load the trained model from the registry
    model = load_model_from_registry()
    
    # Get predictions using the model input
    predictions = get_model_predictions(model, model_input)
    
    # Add back the identifier and set pickup_hour to current time rounded up
    if identifiers is not None:
        predictions = pd.concat([identifiers.reset_index(drop=True), predictions], axis=1)
    predictions["pickup_hour"] = current_date.ceil("h")
    
    print("Predictions:")
    print(predictions.head())
    
    # Write predictions to a feature group in Hopsworks
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION,
        version=1,
        description="Predictions from LGBM Model",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour",
    )
    fg.insert(predictions, write_options={"wait_for_job": False})
    
if __name__ == "__main__":
    main()
