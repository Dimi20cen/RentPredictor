# Project Review

Date: 2026-02-08

## Scope Reviewed

- `app.py`
- `environment.yml`
- `src/mappings.py`
- `notebooks/` workflow and outputs
- repository structure/artifacts

## Architecture Summary

- Training and experimentation are notebook-driven.
- Inference is served through Streamlit (`app.py`) with pre-trained pickle artifacts.
- Feature generation includes:
  - geospatial hub distances,
  - tax enrichment (`tax_data_2025.csv` + city/commune mapping),
  - categorical encoding (ZIP target encoding + one-hot category alignment).

## Strengths

- End-to-end prototype is complete: data prep -> model -> interpretable notebook -> app.
- Model artifacts and required processed dataset are already versioned for local reproducibility.
- `app.py` aligns feature columns against model metadata, reducing inference mismatch risk.

## Risks and Gaps

- No automated tests, linting, or CI checks are present.
- Pipeline is not packaged into reproducible scripts/CLI; execution depends on notebook order and state.
- Some duplicate/archive notebooks increase maintenance overhead and possible drift.
- App infers several binary fields via defaults/proxies, which may impact prediction realism.
- Large committed binary/data artifacts may slow collaboration and inflate repository size.

## Recommended Next Steps

1. Add basic QA gates (lint + smoke test for model/app artifact loading).
2. Move notebook-critical preprocessing/training steps into versioned scripts.
3. Define a lightweight schema contract for model input features.
4. Add artifact/version metadata (model version, training date, feature set hash).
5. Clarify data refresh policy for tax/source data.
