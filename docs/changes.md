# Changes

## 2026-02-08

Summary of change:
- Added complete project README with setup, run instructions, architecture summary, and known limitations.
- Added a project technical review document.
- Added this change log.

Affected files:
- `README.md`
- `docs/project-review.md`
- `docs/changes.md`

Migration notes:
- No code/runtime behavior changes.
- No data/model migration required.

Validation status:
- Documentation-only update.
- Confirmed file paths and commands against current repository structure.

## 2026-02-08 (Notebook cleanup)

Summary of change:
- Compared `-Copy1` notebooks against canonical versions and confirmed they differ in content (not metadata-only).
- Moved notebook duplicates from `notebooks/` to `notebooks/archive/`.
- Updated README to explicitly define canonical notebook flow.

Affected files:
- `notebooks/archive/02_features_and_baseline-Copy1.ipynb`
- `notebooks/archive/03_ml_and_interpretability-Copy1.ipynb`
- `README.md`
- `docs/changes.md`

Migration notes:
- No app/runtime path changed.
- Canonical workflow remains `01_eda -> 02_features_and_baseline -> 03_ml_and_interpretability`.

Validation status:
- Verified duplicate files are no longer in root `notebooks/`.
- Verified archived files exist under `notebooks/archive/`.

## 2026-02-08 (Archive pruning)

Summary of change:
- Deleted old notebook copy snapshots (`*Copy*.ipynb`) from `notebooks/archive/`.

Affected files:
- `notebooks/archive/01_eda-Copy1.ipynb` (deleted)
- `notebooks/archive/01_eda-Copy2.ipynb` (deleted)
- `notebooks/archive/02_features_and_baseline-Copy1.ipynb` (deleted)
- `notebooks/archive/03_ml_and_interpretability-Copy1.ipynb` (deleted)
- `docs/changes.md`

Migration notes:
- No runtime/app impact.
- Canonical notebooks remain unchanged.

Validation status:
- Confirmed no `*Copy*.ipynb` files remain in `notebooks/archive/`.

## 2026-02-08 (Reproducible CLI pipeline)

Summary of change:
- Added reusable ML pipeline utilities in `src/ml_pipeline.py`.
- Added CLI scripts for `train`, `evaluate`, and `predict` workflows.
- Updated README with reproducible command-line workflow examples.

Affected files:
- `src/ml_pipeline.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `README.md`
- `docs/changes.md`

Migration notes:
- No Streamlit app behavior changed.
- Notebook workflow remains available; CLI is an additional reproducible path.

Validation status:
- Syntax validation performed for new Python modules/scripts.
- Runtime execution requires project environment dependencies (`pandas`, `xgboost`, `category_encoders`, etc.).

## 2026-02-08 (CLI robustness fixes)

Summary of change:
- Fixed script import behavior by adding project-root path bootstrap in all CLI scripts.
- Fixed `evaluate.py` for models where XGBoost booster feature names are not serialized.
- Added persisted feature column metadata in training output and wired `predict.py` to use it.

Affected files:
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `README.md`
- `docs/changes.md`

Migration notes:
- Training now additionally writes `models/feature_columns.json` by default.
- Batch prediction should pass `--feature-columns models/feature_columns.json` (or keep default path).

Validation status:
- Executed successfully in `swiss-rental` env:
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `scripts/predict.py`
