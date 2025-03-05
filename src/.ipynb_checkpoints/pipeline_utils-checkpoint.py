# pipeline_utils.py

import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    # We want columns for 1, 2, 3, 4 weeks ago
    hours_ago_list = [168, 336, 504, 672]  # 7*24, 14*24, 21*24, 28*24
    existing_cols = []

    for hours_ago in hours_ago_list:
        col_name = f"rides_t-{hours_ago}"
        if col_name in X.columns:
            existing_cols.append(col_name)

    if not existing_cols:
        # No columns for last 4 weeks found, skip
        print("Warning: None of the rides_t-xxx columns for last 4 weeks exist.")
        return X

    # Calculate the average across the existing columns
    X["average_rides_last_4_weeks"] = X[existing_cols].mean(axis=1)
    return X

# Wrap it in a FunctionTransformer
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek
        return X_.drop(columns=["pickup_hour", "pickup_location_id"], errors="ignore")

add_temporal_features = TemporalFeatureEngineer()

def get_pipeline(**hyper_params):
    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyper_params),
    )
    return pipeline
