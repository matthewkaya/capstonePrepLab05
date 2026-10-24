"""Tests for business costs and threshold optimization."""

import numpy as np
import pytest

from latedeliveries.cost import calculate_cost, find_best_threshold


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        ([0, 0, 1, 1], [0, 0, 0, 0], 30),
        ([0, 0, 1, 1], [1, 1, 1, 1], 16),
        ([0, 0, 1, 1], [0, 1, 1, 0], 23),
        ([0, 0, 0], [0, 0, 0], 0),
    ],
)
def test_calculate_cost(y_true, y_pred, expected):
    """Check business costs against manually calculated examples."""
    assert calculate_cost(y_true, y_pred) == expected


def test_flag_nobody_is_optimal():
    """Flagging nobody should be optimal when all orders are on time."""
    y_true = np.array([0, 0, 0, 0])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    best, _ = find_best_threshold(y_true, probabilities)

    assert best["cost"] == 0
    assert best["threshold"] == 0.45


def test_flag_everybody_is_optimal():
    """Flagging everybody should be optimal when all orders are late."""
    y_true = np.array([1, 1, 1, 1])
    probabilities = np.array([0.6, 0.7, 0.8, 0.9])

    best, _ = find_best_threshold(y_true, probabilities)

    assert best["cost"] == 16
    assert best["threshold"] == 0.05


def test_cost_rejects_mismatched_shapes():
    """Cost calculation must reject arrays with different shapes."""
    with pytest.raises(ValueError):
        calculate_cost([0, 1], [0])


def test_threshold_rejects_mismatched_lengths():
    """Threshold search must reject mismatched input lengths."""
    with pytest.raises(ValueError):
        find_best_threshold([0, 1], [0.2])
