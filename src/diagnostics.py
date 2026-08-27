"""Descriptive diagnostics for validation predictions."""

import numpy as np
import pandas as pd

from .metrics import mae, rmse, rmspe


def summarize_groups(frame, group_columns):
    """Calculate validation metrics for each requested subgroup."""
    required = set(group_columns) | {"Sales", "Prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing diagnostic columns: {sorted(missing)}")
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual = group["Sales"].to_numpy(dtype="float64")
        predicted = group["Prediction"].to_numpy(dtype="float64")
        group_rmspe = rmspe(actual, predicted) if (actual != 0).any() else np.nan
        rows.append({**dict(zip(group_columns, keys)), "Rows": len(group), "RMSPE": group_rmspe, "MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted)})
    return pd.DataFrame(rows)


def add_diagnostic_dimensions(frame):
    result = frame.copy()
    result["Month"] = pd.to_datetime(result["Date"]).dt.month.astype("int8")
    result["SalesMagnitude"] = pd.cut(result["Sales"], [-np.inf, 0, 1000, 5000, 10000, np.inf], labels=["zero", "low", "medium", "high", "very_high"])
    result["AbsoluteError"] = (result["Sales"] - result["Prediction"]).abs()
    result["SignedError"] = result["Prediction"] - result["Sales"]
    return result