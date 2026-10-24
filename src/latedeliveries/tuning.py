"""Tune logistic regression using training data only."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from latedeliveries.features import RANDOM_STATE
from latedeliveries.pipeline import build_pipeline


def tune_model(X_train, y_train):
    """Find the best logistic regression hyperparameters."""
    pipeline = build_pipeline(
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    )

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    return search
