import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from pathlib import Path
import joblib
import hopsworks
import src.config as config
from src.data_utils import transform_ts_data_info_features  # This function returns aggregated features
from src.inference import get_model_predictions, get_feature_store, load_model_from_registry

# --- Batch Data Loading & Transformation ---
def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """
    Loads raw time-series data from the feature store for a given time window, then transforms it
    into aggregated features.
    """
    feature_store = get_feature_store()
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
    )
    
    # Retrieve batch data (using Spark if needed)
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
        read_options={"arrow_flight_config": {"use_spark": True, "timeout": 60000}}
    )
    # Filter data to the desired time window
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    ts_data.sort_values(by=["pickup_location_id", "pickup_hour"], inplace=True)
    ts_data = ts_data.reset_index(drop=True)
    # Remove any timezone information
    ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)
    
    # Transform raw data into aggregated features.
    # (Expected output: aggregated features DataFrame with 7 columns: 5 model input features + pickup_location_id + pickup_hour)
    features = transform_ts_data_info_features(ts_data, window_size=24*28, step_size=23)
    return features

# --- Example Inference Pipeline ---
if __name__ == "__main__":
    # Get current UTC time
    current_date = pd.Timestamp.now(tz='Etc/UTC')
    
    # Load raw batch features and transform them
    raw_features = load_batch_of_features_from_store(current_date)
    print("Transformed features shape:", raw_features.shape)  # Expect aggregated features shape (n_samples, 7)
    
    # Prepare model input by dropping the identifier columns so that we have exactly 5 features
    if "pickup_location_id" in raw_features.columns and "pickup_hour" in raw_features.columns:
        model_input = raw_features.drop(columns=["pickup_location_id", "pickup_hour"])
        identifier = raw_features["pickup_location_id"]
    else:
        model_input = raw_features
        identifier = None
    
    print("Model input shape (should be 5 columns):", model_input.shape)
    
    # Load your trained model from the registry
    model = load_model_from_registry()
    
    # Get predictions using the model input
    predictions = get_model_predictions(model, model_input)
    
    # Optionally add back the identifier to the predictions DataFrame
    if identifier is not None:
        predictions["pickup_location_id"] = identifier.values
    
    # Optionally add a pickup_hour column (e.g., set to the current hour rounded up)
    predictions["pickup_hour"] = current_date.ceil('h')
    
    print("Predictions:")
    print(predictions)
