#!/usr/bin/env python
import sys
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import joblib
import src.config as config

# Delay imports that may cause circular issues.
from src.inference import get_feature_store, get_model_predictions, load_model_from_registry
from src.data_utils import transform_ts_data_info_features

def main():
    # Get the current UTC time
    current_date = pd.Timestamp.now(tz="Etc/UTC")
    print(f"Current date and time (UTC): {current_date}")
    
    # Get the feature store
    feature_store = get_feature_store()
    
    # Define fetch window
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    
    # Get the feature view from Hopsworks
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    
    # Fetch raw batch data using Spark execution (with a generous timeout)
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
        read_options={"arrow_flight_config": {"use_spark": True, "timeout": 60000}}
    )
    # Filter data to the desired time window, sort, and reset the index
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    ts_data = ts_data.sort_values(by=["pickup_location_id", "pickup_hour"]).reset_index(drop=True)
    ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)
    print("Raw ts_data shape:", ts_data.shape)
    
    # Transform raw data into aggregated features (aggregated features should include extra columns)
    aggregated_features = transform_ts_data_info_features(ts_data, window_size=24 * 28, step_size=23)
    print("Aggregated features shape:", aggregated_features.shape)
    
    # Prepare model input: drop identifier columns so that only the 5 features remain.
    if "pickup_location_id" in aggregated_features.columns:
        model_input = aggregated_features.drop(columns=["pickup_location_id"])
    else:
        model_input = aggregated_features
    if "pickup_hour" in model_input.columns:
        model_input = model_input.drop(columns=["pickup_hour"])
    print("Model input shape (expected 5 columns):", model_input.shape)
    
    # Load the trained model from the registry
    model = load_model_from_registry()
    
    # Get predictions using the model input
    predictions = get_model_predictions(model, model_input)
    
    # Add back the identifier columns to the predictions DataFrame
    if "pickup_location_id" in aggregated_features.columns:
        predictions["pickup_location_id"] = aggregated_features["pickup_location_id"].values
    # Add a pickup_hour column (e.g., current time rounded up to the next hour)
    predictions["pickup_hour"] = current_date.ceil("h")
    
    print("Predictions:")
    print(predictions.head())
    
    # Save the predictions into the Hopsworks feature group for model predictions.
    fg = get_feature_store().get_or_create_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION,
        version=1,
        description="Predictions from LGBM Model",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour",
    )
    fg.insert(predictions, write_options={"wait_for_job": False})
    
if __name__ == "__main__":
    main()
