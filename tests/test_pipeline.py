"""Tests for the preprocessing and prediction pipeline."""

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.features import load_data, split_data
from latedeliveries.pipeline import build_pipeline


@pytest.fixture
def trained_pipeline():
    """Create a fitted pipeline using training data only."""
    X_train, _, y_train, _ = split_data(load_data())

    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    return model, X_train


def test_valid_probabilities(trained_pipeline):
    """Predicted probabilities must be between zero and one."""
    model, X_train = trained_pipeline

    probabilities = model.predict_proba(X_train.head(10))

    assert probabilities.shape == (10, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_unseen_weather(trained_pipeline):
    """The pipeline must handle an unseen weather category."""
    model, X_train = trained_pipeline

    sample = X_train.head(1).copy()
    sample["weather"] = "unknown_weather"

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)


def test_missing_prep_minutes(trained_pipeline):
    """Prediction must not require the excluded prep_minutes column."""
    model, X_train = trained_pipeline

    sample = X_train.head(1).copy()

    assert "prep_minutes" not in sample.columns

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)


def test_joblib_roundtrip(trained_pipeline, tmp_path):
    """Saving and loading the pipeline must preserve predictions."""
    model, X_train = trained_pipeline

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    loaded_model = joblib.load(model_path)

    original = model.predict_proba(X_train.head(10))
    restored = loaded_model.predict_proba(X_train.head(10))

    np.testing.assert_allclose(original, restored)
