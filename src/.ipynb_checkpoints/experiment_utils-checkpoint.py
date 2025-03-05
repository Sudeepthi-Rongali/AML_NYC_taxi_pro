import logging
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your MLflow tracking URI (DagsHub)
DAGSHUB_REPO = "Sudeepthi-Rongali/Taxi_NYC"  # Change if your repo name is different
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_REPO}.mlflow"
MLFLOW_TRACKING_USERNAME = "Sudeepthi-Rongali"  # Your DagsHub username
MLFLOW_TRACKING_PASSWORD = "a80f129d5bca56ee7e716ec464a5cb73a3ad602e"  # Your provided token

def set_mlflow_tracking():
    """
    Set up MLflow tracking server credentials and URI for DagsHub.
    """
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

    return mlflow


def log_model_to_mlflow(
    model,
    input_data,
    experiment_name,
    metric_name="metric",
    model_name=None,
    params=None,
    score=None,
):
    """
    Log a trained model, parameters, and metrics to MLflow.

    Parameters:
    - model: Trained model object (e.g., sklearn model).
    - input_data: Input data used for training (for signature inference).
    - experiment_name: Name of the MLflow experiment.
    - metric_name: Name of the metric to log (e.g., "RMSE", "accuracy").
    - model_name: Optional name for the registered model.
    - params: Optional dictionary of hyperparameters to log.
    - score: Optional evaluation metric to log.

    Returns:
    - model_info: Information about the logged model in MLflow.
    """
    try:
        # Set the experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment set to: {experiment_name}")

        # Start an MLflow run
        with mlflow.start_run():
            # Log hyperparameters if provided
            if params:
                mlflow.log_params(params)
                logger.info(f"Logged parameters: {params}")

            # Log metrics if provided
            if score is not None:
                mlflow.log_metric(metric_name, score)
                logger.info(f"Logged {metric_name}: {score}")

            # Infer the model signature
            signature = infer_signature(input_data, model.predict(input_data))
            logger.info("Model signature inferred.")

            # Determine the model name
            if not model_name:
                model_name = model.__class__.__name__

            # Log the model to MLflow
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifact",
                signature=signature,
                input_example=input_data,
                registered_model_name=model_name,
            )
            logger.info(f"Model logged with name: {model_name}")

            return model_info

    except Exception as e:
        logger.error(f"An error occurred while logging to MLflow: {e}")
        raise
