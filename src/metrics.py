"""Forecasting metrics used by the Rossmann modeling pipeline."""

import numpy as np


def _validate_pair(y_true, y_pred):
    actual = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must contain only finite values")
    return actual, predicted


def rmspe(y_true, y_pred):
    """Return official RMSPE, excluding only observations with zero Sales."""
    actual, predicted = _validate_pair(y_true, y_pred)
    nonzero = actual != 0
    if not nonzero.any():
        raise ValueError("RMSPE is undefined when every target is zero")
    percentage_error = (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
    return float(np.sqrt(np.mean(np.square(percentage_error))))


def mae(y_true, y_pred):
    actual, predicted = _validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true, y_pred):
    actual, predicted = _validate_pair(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))