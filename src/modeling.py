"""Memory-aware helpers for direct Rossmann forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import mae, rmse, rmspe


def transform_target(sales):
    return np.log1p(np.asarray(sales, dtype="float64"))


def inverse_target(values):
    return np.clip(np.expm1(np.asarray(values, dtype="float64")), 0, None)


def encode_tree_frame(frame, feature_columns, medians=None, category_maps=None):
    """Encode categories as fold-fitted float32 codes without one-hot expansion."""
    result = pd.DataFrame(index=frame.index)
    medians = {} if medians is None else medians
    category_maps = {} if category_maps is None else category_maps
    for column in feature_columns:
        series = frame[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            mapping = category_maps.setdefault(
                column,
                {value: code for code, value in enumerate(series.dropna().unique())},
            )
            result[column] = series.map(mapping).fillna(-1).astype("float32")
        else:
            values = pd.to_numeric(series, errors="coerce")
            median = medians.setdefault(column, float(values.median()))
            result[column] = values.fillna(median).astype("float32")
    return result


def informative_columns(encoded_frame):
    """Return columns with at least two observed values for tree binning."""
    return [column for column in encoded_frame.columns if encoded_frame[column].nunique(dropna=True) > 1]


def score_predictions(actual, predicted):
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    return {"RMSPE": rmspe(actual, predicted), "MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted)}


def make_submission(test_frame, predictions, output_path):
    predictions = np.asarray(predictions, dtype="float64")
    if len(predictions) != len(test_frame):
        raise ValueError("prediction count must match test rows")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("predictions must be finite and non-negative")
    submission = pd.DataFrame({"Id": test_frame["Id"].to_numpy(), "Sales": predictions})
    if submission["Id"].duplicated().any() or submission.isna().any().any():
        raise ValueError("submission contains duplicate IDs or missing values")
    submission.to_csv(output_path, index=False)
    return submission