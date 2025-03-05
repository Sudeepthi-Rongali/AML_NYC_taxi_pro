import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file, if it exists
load_dotenv()

# Define directories
PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PARENT_DIR / "models"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Hopsworks Configuration
# If environment variables are not set, these defaults will be used.
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "NED5z1BDIN3LWVy1.zo5ROYxBl57FgFYTnVyXlFa26OUGFF4lpeaIgvbeiwiCG3Ow1oVUEmgATUXLOcGE")
HOPSWORKS_PROJECT_NAME = "NYC_taxi_data_pro"  # Based on screenshot

# Feature Group and View settings for Hopsworks
# config.py

FEATURE_GROUP_NAME = "aml_nyc_taxi"
FEATURE_GROUP_VERSION = 2


# If you use a Feature View, set its name and version
FEATURE_VIEW_NAME = "taxi_nyc_view"
FEATURE_VIEW_VERSION = 1

# Model registry settings (adjust as you prefer)
MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 1

# Additional feature group for model predictions (optional)
FEATURE_GROUP_MODEL_PREDICTION = "aml_nyc_taxi"
FEATURE_GROUP_MODEL_PREDICTION_VERSION = 2  # if that's the version
