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
