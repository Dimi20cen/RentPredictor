# rent_predictor

Swiss rental price prediction project with:
- exploratory and feature engineering notebooks,
- a trained XGBoost rent model,
- and a Streamlit app for interactive inference.

## Project Status

This repository is a research/prototype workflow.  
Model artifacts are committed (`models/*.pkl`), and the app relies on local processed data and encoder/model files.

## What Is Included

- `app.py`: Streamlit UI and inference pipeline.
- `notebooks/`:
  - `01_eda.ipynb`
  - `02_features_and_baseline.ipynb`
  - `03_ml_and_interpretability.ipynb`
  - `archive/` (historical notebook snapshots, not canonical)
- `src/mappings.py`: city/commune mapping used by the tax feature merge workflow.
- `data/processed/`: engineered data used by app/notebooks.
- `data/external/tax_data_2025.csv`: tax enrichment source.
- `models/`: trained XGBoost model and ZIP target encoder.
- `environment.yml`: Conda environment specification.

## Model Notes

Based on notebook outputs:
- Baseline (Ridge): MAE ~579 CHF, RMSE ~1032 CHF, R2 ~0.592.
- Tuned XGBoost: MAE ~340 CHF, RMSE ~759 CHF, R2 ~0.779.

These are point-in-time notebook results and may change if data/features/training settings change.

## Canonical Notebook Flow

Use these as the primary workflow:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_features_and_baseline.ipynb`
3. `notebooks/03_ml_and_interpretability.ipynb`

Files under `notebooks/archive/` are retained for history/reference only.

## Setup

### 1) Create environment

```bash
conda env create -f environment.yml
conda activate swiss-rental
```

### 2) Verify required artifacts exist

The app expects these files:
- `models/xgb_rent_model.pkl`
- `models/zip_encoder.pkl`
- `data/processed/02_featured_data.pkl`

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (typically `http://localhost:8501`).

## Reproducible Training Workflow (CLI)

Notebook exploration remains useful, but you can now reproduce core training/evaluation from scripts.

### Train model + encoder

```bash
python scripts/train.py \
  --data data/processed/02_featured_data.pkl \
  --model-out models/xgb_rent_model.pkl \
  --encoder-out models/zip_encoder.pkl \
  --metrics-out models/training_metrics.json \
  --feature-columns-out models/feature_columns.json
```

### Evaluate saved artifacts

```bash
python scripts/evaluate.py \
  --data data/processed/02_featured_data.pkl \
  --model models/xgb_rent_model.pkl \
  --encoder models/zip_encoder.pkl \
  --metrics-out models/evaluation_metrics.json
```

### Batch prediction

```bash
python scripts/predict.py \
  --input-csv path/to/input_features.csv \
  --output-csv predictions.csv \
  --model models/xgb_rent_model.pkl \
  --encoder models/zip_encoder.pkl \
  --feature-columns models/feature_columns.json
```

`input_features.csv` should contain raw feature columns expected by preprocessing (including `Zip`, `Canton`, and `SubType`).

## How Prediction Works (App)

At runtime the app:
1. Loads model + ZIP encoder + reference dataframe.
2. Takes user inputs (area, rooms, floor, canton, zip, subtype, extras).
3. Adds geospatial features (distance to major Swiss rail hubs).
4. Uses tax rate by selected ZIP from reference data.
5. Target-encodes ZIP and aligns one-hot features with model feature names.
6. Runs XGBoost inference and returns estimated monthly rent (CHF).

## Repository Layout

```text
rent_predictor/
  app.py
  environment.yml
  README.md
  data/
    external/
    processed/
  models/
  notebooks/
  src/
```

## Known Limitations

- No automated test suite is currently present.
- Training is notebook-centric; there is no packaged training CLI/pipeline.
- Historical notebook snapshots are kept in `notebooks/archive/`.
- Inference uses a few proxy/default feature values for fields not explicitly collected in UI.

## Documentation

- `docs/project-review.md`: technical review and recommendations.
- `docs/changes.md`: repository change log.
