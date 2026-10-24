"""Train, evaluate, and save the final late-delivery model."""

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from latedeliveries.cost import calculate_cost, find_best_threshold
from latedeliveries.features import (
    FEATURES,
    RANDOM_STATE,
    load_data,
    split_data,
)
from latedeliveries.tuning import tune_model

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def main():
    """Run the complete training and evaluation workflow."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Split once. Keep test data untouched during model selection.
    X_train, X_test, y_train, y_test = split_data(load_data())

    # Tune hyperparameters using training data only.
    search = tune_model(X_train, y_train)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Out-of-fold probabilities for threshold selection.
    probabilities_oof = cross_val_predict(
        clone(search.best_estimator_),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    best_threshold, threshold_results = find_best_threshold(
        y_train,
        probabilities_oof,
    )

    threshold = best_threshold["threshold"]

    # Fit the final pipeline using all training rows.
    final_model = clone(search.best_estimator_)
    final_model.fit(X_train, y_train)

    # Evaluate the untouched test set once.
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions)),
        "f1": float(f1_score(y_test, test_predictions)),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test,
            test_predictions,
        ).tolist(),
        "costs": {
            "flag_none": calculate_cost(
                y_test,
                np.zeros(len(y_test), dtype=int),
            ),
            "flag_all": calculate_cost(
                y_test,
                np.ones(len(y_test), dtype=int),
            ),
            "model": calculate_cost(y_test, test_predictions),
        },
    }

    model_info = {
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "threshold": threshold,
        "features": FEATURES,
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "test_results": metrics,
    }

    joblib.dump(final_model, MODEL_DIR / "late_orders.joblib")

    (MODEL_DIR / "model_info.json").write_text(
        json.dumps(model_info, indent=2) + "\n",
        encoding="utf-8",
    )

    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    (REPORT_DIR / "threshold_results.json").write_text(
        json.dumps(threshold_results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Best parameters:", search.best_params_)
    print("Selected threshold:", threshold)
    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print("Model saved to:", MODEL_DIR / "late_orders.joblib")


if __name__ == "__main__":
    main()
