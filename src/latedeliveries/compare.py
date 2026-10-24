"""Compare classification models using stratified cross-validation."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from latedeliveries.features import RANDOM_STATE, load_data, split_data
from latedeliveries.pipeline import build_pipeline

REPORT_PATH = Path("reports/cv_results.csv")

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def get_candidates():
    """Return the baseline and candidate classification models."""
    return {
        "Dummy": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
        ),
        "Logistic Regression Balanced": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def compare_models() -> pd.DataFrame:
    """Compare models using five-fold stratified cross-validation."""
    X_train, _, y_train, _ = split_data(load_data())

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    for name, model in get_candidates().items():
        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
        )

        row = {"model": name}

        for metric in SCORING:
            values = scores[f"test_{metric}"]

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        rows.append(row)

    results = pd.DataFrame(rows)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_PATH,
        index=False,
    )

    return results


if __name__ == "__main__":
    results = compare_models()
    print(results.round(4).to_string(index=False))
