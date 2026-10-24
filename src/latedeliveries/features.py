"""Feature selection and train/test splitting for late deliveries."""

import pandas as pd
from sklearn.model_selection import train_test_split

from latedeliveries.data import make_orders

RANDOM_STATE = 42
TEST_SIZE = 0.25

NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "courier_months",
]

CATEGORICAL_FEATURES = [
    "weather",
    "day_type",
    "vehicle",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "late"

EXCLUDED_FEATURES = [
    "order_id",
    "prep_minutes",
    "customer_rating",
]


def load_data() -> pd.DataFrame:
    """Generate the reproducible synthetic order dataset."""
    return make_orders(n=3000, seed=7)


def prepare_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select order-time features and the prediction target."""
    X = data[FEATURES].copy()
    y = data[TARGET].copy()
    return X, y


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified 75/25 train/test split."""
    X, y = prepare_features(data)

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
