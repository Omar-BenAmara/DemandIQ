# DemandIQ

End-to-end retail demand forecasting project using the Rossmann Store Sales dataset.

## Objective

Build a reliable machine learning pipeline to forecast store-level sales.

## Project Status

Foundation verification complete. Reconnaissance, EDA, preparation, and temporal feature logic have been independently checked. Modeling now has a tested metric layer and a reproducible notebook pipeline; measured model results are produced by executing notebook 05.

## Dataset

The local raw files are the Rossmann Store Sales competition tables: `train.csv`, `test.csv`, `store.csv`, and `sample_submission.csv`. Their provenance and checksums are not currently recorded, so the acquisition source is not independently reproducible from this repository.

The raw data contains 11 missing `test.Open` values, all for Store 622 from 2015-09-05 through 2015-09-17. The repository preserves these values as missing. Their source meaning cannot be established from the supplied tables, so no imputation or inferred operating status is treated as verified foundation logic.

## Problem Definition

The authoritative Rossmann competition description defines `Sales` as the turnover for a store on a day and asks for Sales for every test `Store`-`Date` row. `Open` is an indicator where 0 means closed and 1 means open. Closed training rows have structural zero Sales, while the official RMSPE ignores any scored row whose actual Sales is zero. Source: the [Rossmann Store Sales competition data description](https://www.kaggle.com/c/rossmann-store-sales/data).

DemandIQ therefore defines the target as `Sales` for every Store-Date, including closed rows. `Open` is not required to define the target or official evaluation and must not be used as a deterministic zero-sales gate. It may be evaluated later as an optional prediction-time covariate, but the 11 missing Store 622 values remain unknown and must be represented explicitly if used.

The primary metric definition for later modeling is official RMSPE over rows with nonzero actual Sales, independent of `Open`. MAE and RMSE may be reported secondarily on all rows, with their population stated explicitly. The current historical `Open`-filtered RMSPE is numerically equivalent on this train data because every closed row has zero Sales, but the implementation must use the official Sales-based mask before modeling begins.

## Modeling

Notebook 05 consumes the verified direct multi-horizon features from notebook 04 and evaluates a naive rolling-mean baseline plus `HistGradientBoostingRegressor` on the same three chronological 48-day folds. Both direct `Sales` and `log1p(Sales)` targets are compared on the original Sales scale. The primary model excludes `Open`, so missing test `Open` values do not require imputation or a deterministic closed-store gate.

The reusable metric and submission checks are in [src/metrics.py](src/metrics.py) and [src/modeling.py](src/modeling.py), with independent tests in [tests/test_modeling_metrics.py](tests/test_modeling_metrics.py). Categorical features are fold-fitted integer codes and numeric features are `float32`; this avoids dense one-hot expansion of the large training matrices. Run `.venv\Scripts\python.exe -m unittest discover -s tests -v`, then execute notebook 05 from top to bottom to create `artifacts/fold_metrics.csv`, `artifacts/model_comparison.csv`, `artifacts/horizon_diagnostics.csv`, and `artifacts/submission.csv`.

## Verified Results

The bounded notebook run used three chronological 48-day folds and the two most recent eligible training origins per fold because the unrestricted six-origin run exhausted the available environment during tree fitting. Mean official RMSPE was `0.25304` for direct-Sales HistGradientBoosting, `0.33143` for the naive rolling-mean baseline, `0.45325` for HistGradientBoosting on `log1p(Sales)`, and `0.69493` for the lag-7 seasonal baseline. Direct-Sales HistGradientBoosting was selected because it had the lowest mean RMSPE, with mean MAE `1,336.75` and mean RMSE `2,130.71`; this is a bounded-window result, not a universal optimum claim.

The final local submission contains 41,088 rows in original `test.csv` order, with unique IDs, finite nonnegative predictions, and no missing values. `Open` was excluded from the primary model, including the unresolved Store 622 missing block. Known limitations include the global direct model's finite feature set, the computational cost of rebuilding frozen histories, the bounded recent-origin training window, and the unresolved semantic meaning of the 11 missing `Open` values.