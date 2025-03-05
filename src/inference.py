from datetime import datetime, timedelta, timezone
import pandas as pd
import hopsworks
from hsfs.feature_store import FeatureStore
from pathlib import Path
import joblib
import src.config as config
from src.data_utils import transform_ts_data_info_features

# --- Hopsworks Connection ---
def get_hopsworks_project() -> hopsworks.project.Project:
return hopsworks.login(
project=config.HOPSWORKS_PROJECT_NAME,
api_key_value=config.HOPSWORKS_API_KEY
)

def get_feature_store() -> FeatureStore:
project = get_hopsworks_project()
return project.get_feature_store()

# --- Model Prediction Function ---
def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
"""
Uses the trained model to predict demand. Expects features to contain only the model input columns.
"""
preds = model.predict(features)
results = pd.DataFrame()
results["predicted_demand"] = preds.round(0)
return results


def fetch_predictions(hours: int) -> pd.DataFrame:
"""
Fetch predictions for the last 'hours' hours from the appropriate feature group.
Adjust the filtering logic as needed.
"""
current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")
fs = get_feature_store()
# Ensure the feature group name and version match what is used for predictions
fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=2)
return fg.filter(fg.pickup_hour >= current_hour).read()

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

# Force Spark execution if necessary:
ts_data = feature_view.get_batch_data(
start_time=(fetch_data_from - timedelta(days=1)),
end_time=(fetch_data_to + timedelta(days=1)),
read_options={"arrow_flight_config": {"use_spark": True, "timeout": 60000}}
)
# Filter data to desired window
ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
ts_data.sort_values(by=["pickup_location_id", "pickup_hour"], inplace=True)
ts_data = ts_data.reset_index(drop=True)
ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)

# Transform raw data into aggregated features (expected shape: (n_samples, 6))
features = transform_ts_data_info_features(ts_data, window_size=24*28, step_size=23)
return features

# --- Model Registry Functions ---
def load_model_from_registry():
"""
Loads the latest model from the Hopsworks model registry.
"""
project = get_hopsworks_project()
model_registry = project.get_model_registry()
models = model_registry.get_models(name=config.MODEL_NAME)
# Select the model with the highest version
model_obj = max(models, key=lambda m: m.version)
model_dir = model_obj.download()
model = joblib.load(Path(model_dir) / "lgb_model.pkl")
return model

# --- Example Inference Pipeline ---
if __name__ == "__main__":
# Get current UTC time
current_date = pd.Timestamp.now(tz='Etc/UTC')

# Load raw batch features and transform them
raw_features = load_batch_of_features_from_store(current_date)
print("Transformed features shape:", raw_features.shape) # Expect (n_samples, 6)

# Separate the model input (5 features) and identifier (pickup_location_id)
if "pickup_location_id" in raw_features.columns:
model_input = raw_features.drop(columns=["pickup_location_id"])
identifier = raw_features["pickup_location_id"]
else:
model_input = raw_features
identifier = None

print("Model input shape (should be 5 columns):", model_input.shape)

# Load your trained model
model = load_model_from_registry()

# Get predictions using the model input
predictions = get_model_predictions(model, model_input)

# Optionally add back the identifier to the predictions DataFrame
if identifier is not None:
predictions["pickup_location_id"] = identifier.values

# Optionally add a pickup_hour column (rounded up to the next hour)
predictions["pickup_hour"] = current_date.ceil('h')

print("Predictions:")
print(predictions)
