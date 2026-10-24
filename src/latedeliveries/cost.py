"""Business cost calculations and threshold optimization."""

import numpy as np

VOUCHER_COST = 4
MISSED_LATE_COST = 15

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def calculate_cost(y_true, y_pred):
    """Calculate the total business cost of a prediction policy."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    flagged = np.sum(predicted == 1)
    missed_late = np.sum((actual == 1) & (predicted == 0))

    return int(VOUCHER_COST * flagged + MISSED_LATE_COST * missed_late)


def find_best_threshold(y_true, probabilities):
    """Find the threshold that minimizes business cost."""
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")

    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal lengths")

    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        cost = calculate_cost(y_true, predictions)

        results.append(
            {
                "threshold": float(threshold),
                "cost": cost,
            }
        )

    return min(results, key=lambda result: result["cost"]), results
