"""Tests for dataset generation and feature selection."""

from latedeliveries.features import (
    EXCLUDED_FEATURES,
    FEATURES,
    load_data,
    prepare_features,
    split_data,
)


def test_dataset_size():
    """The generator must produce 3000 orders."""
    data = load_data()

    assert data.shape[0] == 3000


def test_dataset_reproducibility():
    """Generating the dataset twice must give identical results."""
    first = load_data()
    second = load_data()

    assert first.equals(second)


def test_excluded_features():
    """Unavailable and leakage-prone columns must not be features."""
    data = load_data()
    X, _ = prepare_features(data)

    assert list(X.columns) == FEATURES

    for column in EXCLUDED_FEATURES:
        assert column not in X.columns


def test_stratified_split():
    """The split must preserve the target distribution."""
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    assert len(X_train) == 2250
    assert len(X_test) == 750

    overall_rate = data["late"].mean()
    train_rate = y_train.mean()
    test_rate = y_test.mean()

    assert abs(train_rate - overall_rate) < 0.01
    assert abs(test_rate - overall_rate) < 0.01


def test_split_reproducibility():
    """Repeated splits must produce identical train/test partitions."""
    data = load_data()

    first = split_data(data)
    second = split_data(data)

    for first_part, second_part in zip(first, second, strict=True):
        assert first_part.equals(second_part)
