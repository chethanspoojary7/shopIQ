"""Preprocessing: validation, feature engineering, ColumnTransformer.

Feature engineering (all from legitimate inputs, no target):
- TotalPages = Administrative + Informational + ProductRelated
- TotalDuration = Administrative_Duration + Informational_Duration
  + ProductRelated_Duration
- ProductShare = ProductRelated / max(1, TotalPages)
- ProductDurationShare = ProductRelated_Duration / max(1, TotalDuration)
Weekend (bool) is normalised to "TRUE"/"FALSE" strings so the
OneHotEncoder treats it as categorical consistently at train and
inference time. OperatingSystems/Browser/Region/TrafficType are
nominal codes, not ordered quantities, so they are one-hot encoded.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ORIGINAL_FEATURES, TARGET


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = (
        df["Administrative_Duration"] + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )
    df["ProductShare"] = df["ProductRelated"] / np.maximum(1, df["TotalPages"])
    df["ProductDurationShare"] = df["ProductRelated_Duration"] / np.maximum(1.0, df["TotalDuration"])
    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(str).str.upper()
        # pandas reads TRUE/FALSE as bool; str(True) is "True" -> "TRUE"
        df["Weekend"] = df["Weekend"].replace({"TRUE": "TRUE", "FALSE": "FALSE"})
    return df


def validate(df: pd.DataFrame) -> dict:
    missing_cols = [c for c in ORIGINAL_FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
        "purchase_rate": float(df[TARGET].astype(int).mean()),
    }


def build_preprocessor(include_pagevalues: bool = False) -> ColumnTransformer:
    numeric_cols = NUMERIC_FEATURES + (["PageValues"] if include_pagevalues else [])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def prepare_xy(df: pd.DataFrame, drop_pagevalues: bool):
    """Return X, y. drop_pagevalues=True gives the leakage-safe set."""
    df = add_engineered(df)
    y = df[TARGET].astype(int).to_numpy()
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[cols].copy()
    if not drop_pagevalues:
        X["PageValues"] = df["PageValues"]
    return X, y
