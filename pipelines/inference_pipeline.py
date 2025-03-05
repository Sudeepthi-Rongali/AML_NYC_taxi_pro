from datetime import datetime, timedelta
import pandas as pd
import hopsworks
import joblib
from pathlib import Path
import src.config as config
from src.inference import get_feature_store, get_model_predictions, load_model_from_registry

def transform_ts_data_info_features_aggregated(df, feature_col="rides"):
    """
    Aggregates raw time-series taxi ride data into 5 features per pickup_location_id:
      - avg_rides: mean rides count
      - min_rides: minimum rides count
      - max_rides: maximum rides count
      - std_rides: standard deviation of rides count
      - pickup_hour: the most recent pickup_hour (as a datetime)
    Assumes df contains columns: "pickup_hour", "pickup_location_id", and feature_col.
    """
    # Ensure pickup_hour is datetime and remove timezone info
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"]).dt.tz_localize(None)
    
    # Aggregate rides by location
    agg = df.groupby("pickup_location_id").agg(
        avg_rides = (feature_col, "mean"),
        min_rides = (feature_col, "min"),
        max_rides = (feature_col, "max"),
        std_rides = (feature_col, "std")
    ).reset_index()
    
    # Get the most recent pickup_hour per location
    last_time = df.groupby("pickup_location_id")["pickup_hour"].max().reset_index(name="pickup_hour")
    
    # Merge the aggregated features with the last pickup_hour
    merged = pd.merge(agg, last_time, on="pickup_location_id")
    return merged

def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """
    Loads raw time-series data from Hopsworks, then aggregates it into a feature table.
    """
    fs = get_feature_store()
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    
    feature_view = fs.get_feature_view(
        name=config.FEATURE_VIEW_NAME, 
        version=config.FEATURE_VIEW_VERSION
    )
    
    # Use Spark execution with a generous timeout
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
        read_options={"arrow_flight_config": {"use_spark": True, "timeout": 60000}}
    )
    
    # Filter raw data to the desired time window and sort
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    ts_data.sort_values(by=["pickup_location_id", "pickup_hour"], inplace=True)
    ts_data = ts_data.reset_index(drop=True)
    ts_data["pickup_hour"] = pd.to_datetime(ts_data["pickup_hour"]).dt.tz_localize(None)
    
    # Aggregate the raw data into a small set of features (5 columns)
    features = transform_ts_data_info_features_aggregated(ts_data, feature_col="rides")
    return features

def main():
    # Get current UTC time
    current_date = pd.Timestamp.now(tz="Etc/UTC")
    
    # Load and aggregate raw features from the feature store
    features = load_batch_of_features_from_store(current_date)
    print("Aggregated features shape:", features.shape)  # Expect (n_samples, 5)
    
    # Prepare model input by dropping the identifier column
    if "pickup_location_id" in features.columns:
        model_input = features.drop(columns=["pickup_location_id"])
        identifier = features["pickup_location_id"]
    else:
        model_input = features
        identifier = None
    print("Model input shape (should be 4 columns if TemporalFeatureEngineer adds 1):", model_input.shape)
    
    # Load the trained model from the registry
    model = load_model_from_registry()
    
    # Get predictions using the model input
    predictions = get_model_predictions(model, model_input)
    
    # Optionally add back the identifier and pickup_hour to predictions
    if identifier is not None:
        predictions["pickup_location_id"] = identifier.values
    # Set the pickup_hour for predictions as the current time rounded up to the next hour
    predictions["pickup_hour"] = current_date.ceil("h")
    
    print("Predictions:")
    print(predictions)
    
    # Insert predictions into the designated feature group in Hopsworks
    fs = get_feature_store()
    feature_group = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION,
        version=config.FEATURE_GROUP_MODEL_PREDICTION_VERSION,
        description="Predictions from LGBM Model",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour",
    )
    feature_group.insert(predictions, write_options={"wait_for_job": False})

if __name__ == "__main__":
    main()
