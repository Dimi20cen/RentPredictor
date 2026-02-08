# RentPredictor

Swiss rental price prediction project with:
- an XGBoost model,
- reproducible train/evaluate/predict scripts,
- and a Streamlit app for interactive inference.

## Navigation

- [Results and Demo](#results-and-demo)
- [Dataset](#dataset)
- [Replicate Locally](#replicate-locally)
- [CLI Workflow](#cli-workflow)
- [Canonical Notebooks](#canonical-notebooks)
- [Project Structure](#project-structure)
- [Notes](#notes)

## Results and Demo

### Key Results (current baseline run)

- MAE: `347.02 CHF`
- RMSE: `748.05 CHF`
- R2: `0.7856`

These metrics come from the reproducible CLI pipeline (`train.py` / `evaluate.py`) on `data/processed/02_featured_data.pkl`.

### Demo

- Local interactive demo:

```bash
streamlit run app.py
```

- App URL (local): `http://localhost:8501`

## Dataset

### Data lineage (from notebooks)

- Raw scrape (not committed): ~22,515 listings from ImmoScout24 (`notebooks/01_eda.ipynb`).
- Cleaned residential set: ~16,399 rows after filtering/cleaning (`data/processed/01_cleaned_data.pkl`).
- Featured modeling set: engineered dataset used for training/inference (`data/processed/02_featured_data.pkl`).
- External enrichment: Swiss municipal/cantonal tax data from `data/external/tax_data_2025.csv`.

### Files in this repository

- `data/processed/01_cleaned_data.pkl` / `.csv`: output of EDA + cleaning.
- `data/processed/02_featured_data.pkl`: canonical training/evaluation dataset.
- `data/processed/rentals_ready_for_modeling.csv`: modeling-ready export.
- `data/external/tax_data_2025.csv`: tax feature source.

### Important preprocessing notes

- Tax feature (`tax_rate`) is merged by city/commune mapping.
- If an exact city match fails, notebooks apply canton-level median fallback.
- Training and app inference both expect raw columns such as `Zip`, `Canton`, and `SubType` before encoding.

## Replicate Locally

### 1) Create environment

```bash
conda env create -f environment.dev.yml
conda activate swiss-rental
```

For Streamlit Cloud deployment, `requirements.txt` is provided with runtime dependencies.

### 2) Train artifacts

```bash
python scripts/train.py \
  --data data/processed/02_featured_data.pkl \
  --model-out models/xgb_rent_model.pkl \
  --encoder-out models/zip_encoder.pkl \
  --metrics-out models/training_metrics.json \
  --feature-columns-out models/feature_columns.json
```

### 3) Evaluate saved artifacts

```bash
python scripts/evaluate.py \
  --data data/processed/02_featured_data.pkl \
  --model models/xgb_rent_model.pkl \
  --encoder models/zip_encoder.pkl \
  --metrics-out models/evaluation_metrics.json
```

### 4) (Optional) Batch prediction

```bash
python scripts/predict.py \
  --input-csv path/to/input_features.csv \
  --output-csv predictions.csv \
  --model models/xgb_rent_model.pkl \
  --encoder models/zip_encoder.pkl \
  --feature-columns models/feature_columns.json
```

## CLI Workflow

- `scripts/train.py`: trains model + encoder and writes training metrics/feature columns.
- `scripts/evaluate.py`: evaluates saved model/encoder on deterministic split.
- `scripts/predict.py`: batch inference from input CSV.

## Canonical Notebook Flow

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_features_and_baseline.ipynb`
3. `notebooks/03_ml_and_interpretability.ipynb`

## Project Structure

```text
RentPredictor/
  app.py
  scripts/
    train.py
    evaluate.py
    predict.py
  src/
    ml_pipeline.py
    mappings.py
  data/
    external/
    processed/
  models/
  notebooks/
  docs/
```

## Notes

- Batch prediction input must include raw preprocessing columns, including `Zip`, `Canton`, and `SubType`.
- Current workflow is prototype-focused; CI/tests are not yet added.
