"""Classifier definitions and their (small) hyper-parameter grids.

Each experiment is a ``Pipeline(features -> clf)``, so feature extraction is
re-fitted inside every cross-validation fold: the TF-IDF vocabulary, IDF
weights and Min-Max ranges never see validation or test messages.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from . import config
from .features import make_features

MODEL_NAMES = ["naive_bayes", "logistic_regression", "random_forest"]
MODEL_LABELS = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
FEATURE_LABELS = {"tfidf": "TF-IDF", "handcrafted": "Hand-crafted", "combined": "Combined"}


def make_classifier(name: str):
    """Return an un-fitted classifier with fixed, documented defaults.

    No class re-weighting is used for any model: all three are compared at the
    same 0.5 probability threshold on the natural class balance (~12% spam),
    which favours precision, i.e. few legitimate messages lost to the filter.
    """
    if name == "naive_bayes":
        # Multinomial NB: the classic spam-filter baseline (Sahami et al., 1998).
        return MultinomialNB()
    if name == "logistic_regression":
        # L2-regularised linear model; liblinear is exact and deterministic.
        return LogisticRegression(solver="liblinear", max_iter=5000, random_state=config.SEED)
    if name == "random_forest":
        # Non-linear tree ensemble; 300 trees is past the point of diminishing
        # returns on this data size.
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=config.SEED)
    raise ValueError(f"unknown model: {name!r}")


# Grids are intentionally small: the goal is a fair comparison of
# representations, not a leaderboard. They are searched with stratified
# 5-fold CV on the training split only.
PARAM_GRIDS = {
    "naive_bayes": {"clf__alpha": [0.01, 0.03, 0.1, 0.3, 1.0]},
    "logistic_regression": {"clf__C": [1.0, 10.0, 100.0]},
    "random_forest": {"clf__max_features": ["sqrt", "log2"], "clf__min_samples_leaf": [1, 2]},
}


def make_pipeline(model: str, features: str) -> Pipeline:
    return Pipeline([("features", make_features(features)), ("clf", make_classifier(model))])
