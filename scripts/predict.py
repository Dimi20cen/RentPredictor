#!/usr/bin/env python3
"""Batch prediction using saved model + encoder."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import transform_features_for_model, predict_rent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference for rent prediction.")
    parser.add_argument("--input-csv", required=True, help="Input CSV with raw feature columns.")
    parser.add_argument("--output-csv", default="predictions.csv", help="Output CSV with predictions.")
    parser.add_argument("--model", default="models/xgb_rent_model.pkl", help="Path to trained model pickle.")
    parser.add_argument("--encoder", default="models/zip_encoder.pkl", help="Path to ZIP encoder pickle.")
    parser.add_argument(
        "--feature-columns",
        default="models/feature_columns.json",
        help="Path to JSON list of training feature column names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    with open(args.encoder, "rb") as f:
        encoder = pickle.load(f)

    feature_names: list[str]
    cols_path = Path(args.feature_columns)
    if cols_path.exists():
        feature_names = json.loads(cols_path.read_text(encoding="utf-8"))
    else:
        booster_names = model.get_booster().feature_names
        if booster_names is None:
            raise ValueError(
                "Feature names unavailable. Run scripts/train.py to generate "
                "models/feature_columns.json or pass --feature-columns."
            )
        feature_names = list(booster_names)
    raw = pd.read_csv(args.input_csv)
    x = transform_features_for_model(raw, encoder, feature_names)
    pred = predict_rent(model, x)

    out = raw.copy()
    out["predicted_rent_chf"] = pred
    out.to_csv(args.output_csv, index=False)

    print(f"Predictions complete: {len(out)} rows")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
