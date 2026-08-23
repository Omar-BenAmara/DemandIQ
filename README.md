# DemandIQ

End-to-end retail demand forecasting project using the Rossmann Store Sales dataset.

## Objective

Build a reliable machine learning pipeline to forecast store-level sales.

## Project Status

Foundation repair in progress. Reconnaissance, EDA, preparation, and temporal feature logic have been reviewed; modeling remains locked until the foundation is independently verified.

## Dataset

The local raw files are the Rossmann Store Sales competition tables: `train.csv`, `test.csv`, `store.csv`, and `sample_submission.csv`. Their provenance and checksums are not currently recorded, so the acquisition source is not independently reproducible from this repository.

The raw data contains 11 missing `test.Open` values, all for Store 622 from 2015-09-05 through 2015-09-17. The repository preserves these values as missing. Their source meaning cannot be established from the supplied tables, so no imputation or inferred operating status is treated as verified foundation logic.