"""Preprocessing and classifier live in one Pipeline.

Imputation and one-hot encoding are fit only when the pipeline is fit, so a
train/test split that happens first cannot leak those statistics.
"""

from streamloop import _threads  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from streamloop.util import plain, same_param

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class PrepareColumns(BaseEstimator, TransformerMixin):
    """Drop the identifier and coerce fields that are not model-ready.

    ``TotalCharges`` arrives as text and has blank strings. Coercion to
    numeric is stateless. The median that fills those blanks is a later
    ``SimpleImputer`` step, fit on whatever rows the pipeline sees.
    """

    def fit(self, X, y=None):
        self._check(X)
        return self

    def transform(self, X):
        self._check(X)
        frame = X.copy()
        if "customerID" in frame.columns:
            frame = frame.drop(columns=["customerID"])
        if "Churn" in frame.columns:
            frame = frame.drop(columns=["Churn"])
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        for column in CATEGORICAL_FEATURES:
            frame[column] = _as_category_text(frame[column])
        return frame

    def _check(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PrepareColumns expects a pandas DataFrame.")
        missing = [column for column in FEATURE_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")


def _as_category_text(series):
    missing = series.isna()
    text = series.astype(str).str.strip()
    text = text.mask(missing | text.eq("") | text.eq("<NA>") | text.eq("nan"), np.nan)
    return text


def build_pipeline(model_kwargs=None):
    """Default boosting model, except a fixed ``random_state`` for reproducibility."""
    params = {"random_state": 42}
    if model_kwargs:
        params.update(model_kwargs)
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("prepare", PrepareColumns()),
            ("preprocess", preprocess),
            ("model", HistGradientBoostingClassifier(**params)),
        ]
    )


def parameter_overrides(pipeline):
    """Hyperparameters that differ from a fresh default estimator."""
    model = pipeline.named_steps["model"]
    defaults = type(model)().get_params()
    current = model.get_params()
    overrides = {}
    for key in sorted(current):
        if not same_param(current[key], defaults[key]):
            overrides[key] = plain(current[key])
    return overrides


def fit_baseline(X_train, y_train, model_kwargs=None):
    pipeline = build_pipeline(model_kwargs)
    pipeline.fit(X_train, y_train)
    return pipeline
