"""Controlled Open-feature ablation helpers."""

import pandas as pd


def feature_columns_for_open_ablation(model_feature_columns, include_open):
    """Return the only feature-set difference allowed by the ablation."""
    columns = [column for column in model_feature_columns if column not in {"Open", "Open_missing"}]
    if include_open:
        columns.extend(["Open", "Open_missing"])
    return columns


def add_open_missing_indicator(frame):
    """Preserve Open and add an explicit missingness indicator."""
    result = frame.copy()
    result["Open_missing"] = result["Open"].isna().astype("int8")
    return result


def simulate_missing_open_block(frame, store, start, periods):
    """Remove one contiguous Open block without changing dates or targets."""
    result = frame.copy()
    dates = pd.date_range(start=pd.Timestamp(start), periods=periods, freq="D")
    mask = result["Store"].eq(store) & result["Date"].isin(dates)
    if int(mask.sum()) != periods:
        raise ValueError("requested Open block is not complete in the frame")
    result.loc[mask, "Open"] = pd.NA
    return result, dates