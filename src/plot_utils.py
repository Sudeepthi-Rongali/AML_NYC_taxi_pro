from datetime import timedelta
from typing import Optional
import pandas as pd
import numpy as np
import plotly.express as px


def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions: Optional[pd.DataFrame] = None,
):
    """
    Plots the time series data for a specific location from NYC taxi data.

    Args:
        features (pd.DataFrame): DataFrame containing feature data, including historical ride counts and metadata.
        targets (pd.Series): Series containing the target values (e.g., actual ride counts).
        row_id (int): Pickup location ID to plot.
        predictions (Optional[pd.DataFrame] or np.ndarray): DataFrame or NumPy array containing predicted values.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object showing the time series plot.
    """

    # Ensure required columns exist
    required_columns = ["pickup_location_id", "pickup_hour"]
    for col in required_columns:
        if col not in features.columns:
            raise KeyError(f"Column '{col}' not found in features DataFrame.")

    # Ensure the row_id exists in the dataset
    available_locations = features["pickup_location_id"].unique()
    if row_id not in available_locations:
        raise ValueError(
            f"Row ID {row_id} not found in 'pickup_location_id'.\n"
            f"Available values: {available_locations[:10]}... (Total: {len(available_locations)})"
        )

    # Extract the specific location's features and target
    location_features = features[features["pickup_location_id"] == row_id].iloc[0]
    actual_target = targets[features["pickup_location_id"] == row_id].values[0]

    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]
    time_series_values = location_features[time_series_columns].tolist() + [actual_target]

    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=location_features["pickup_hour"] - timedelta(hours=len(time_series_columns)),
        end=location_features["pickup_hour"],
        freq="h",
    )

    # Create the plot title with relevant metadata
    title = f"Pickup Hour: {location_features['pickup_hour']}, Location ID: {row_id}"

    # Create the base line plot
    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        template="plotly_white",
        markers=True,
        title=title,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    # Add the actual target value as a green marker
    fig.add_scatter(
        x=[time_series_dates[-1]],  # Last timestamp
        y=[actual_target],  # Actual target value
        line_color="green",
        mode="markers",
        marker_size=10,
        name="Actual Value",
    )

    # Optionally add the prediction as a red marker
    if predictions is not None:
        # Convert predictions to DataFrame if it is a NumPy array
        if isinstance(predictions, np.ndarray):
            predictions = predictions.reshape(-1, 2)  # Ensure it's 2D (N, 2)
            predictions = pd.DataFrame(predictions, columns=["pickup_location_id", "predicted_demand"])

        if "pickup_location_id" in predictions.columns:
            if row_id in predictions["pickup_location_id"].values:
                pred_values = predictions[predictions["pickup_location_id"] == row_id]["predicted_demand"].values
                if len(pred_values) > 0:
                    pred_value = pred_values[0]  # Ensure the value exists before indexing
                    fig.add_scatter(
                        x=[time_series_dates[-1]],  # Last timestamp
                        y=[pred_value],  # Predicted value
                        line_color="red",
                        mode="markers",
                        marker_symbol="x",
                        marker_size=15,
                        name="Prediction",
                    )
                else:
                    print(f"Warning: No prediction values found for location ID {row_id}.")
            else:
                print(f"Warning: Row ID {row_id} not found in predictions dataset.")
        else:
            print("Warning: 'pickup_location_id' column missing in predictions dataset.")

    return fig
