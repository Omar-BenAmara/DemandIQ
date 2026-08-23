# DemandIQ

End-to-end retail demand forecasting project using the Rossmann Store Sales dataset.

## Objective

Build a reliable machine learning pipeline to forecast store-level sales.

## Project Status

Foundation verification complete. Reconnaissance, EDA, preparation, and temporal feature logic have been independently checked; modeling remains locked pending review.

## Dataset

The local raw files are the Rossmann Store Sales competition tables: `train.csv`, `test.csv`, `store.csv`, and `sample_submission.csv`. Their provenance and checksums are not currently recorded, so the acquisition source is not independently reproducible from this repository.

The raw data contains 11 missing `test.Open` values, all for Store 622 from 2015-09-05 through 2015-09-17. The repository preserves these values as missing. Their source meaning cannot be established from the supplied tables, so no imputation or inferred operating status is treated as verified foundation logic.

## Problem Definition

The authoritative Rossmann competition description defines `Sales` as the turnover for a store on a day and asks for Sales for every test `Store`-`Date` row. `Open` is an indicator where 0 means closed and 1 means open. Closed training rows have structural zero Sales, while the official RMSPE ignores any scored row whose actual Sales is zero. Source: the [Rossmann Store Sales competition data description](https://www.kaggle.com/c/rossmann-store-sales/data).

DemandIQ therefore defines the target as `Sales` for every Store-Date, including closed rows. `Open` is not required to define the target or official evaluation and must not be used as a deterministic zero-sales gate. It may be evaluated later as an optional prediction-time covariate, but the 11 missing Store 622 values remain unknown and must be represented explicitly if used.

The primary metric definition for later modeling is official RMSPE over rows with nonzero actual Sales, independent of `Open`. MAE and RMSE may be reported secondarily on all rows, with their population stated explicitly. The current historical `Open`-filtered RMSPE is numerically equivalent on this train data because every closed row has zero Sales, but the implementation must use the official Sales-based mask before modeling begins.