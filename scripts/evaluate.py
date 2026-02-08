#!/usr/bin/env python3
"""Evaluate saved model/encoder on deterministic holdout split."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import (
    ONE_HOT_COLUMNS,
    ZIP_COLUMN,
    load_featured_data,
    regression_metrics,
    save_json,
    split_features_target,
    predict_rent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rent prediction model.")
    parser.add_argument("--data", default="data/processed/02_featured_data.pkl", help="Path to featured dataset pickle.")
    parser.add_argument("--model", default="models/xgb_rent_model.pkl", help="Path to trained model pickle.")
    parser.add_argument("--encoder", default="models/zip_encoder.pkl", help="Path to ZIP encoder pickle.")
    parser.add_argument("--metrics-out", default="", help="Optional output JSON path for evaluation metrics.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    with open(args.encoder, "rb") as f:
        encoder = pickle.load(f)

    df = load_featured_data(args.data)
    x, y = split_features_target(df)
    x_train_raw, x_test_raw, _, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    x_train = x_train_raw.copy()
    x_test = x_test_raw.copy()

    if ZIP_COLUMN not in x_train.columns:
        raise ValueError(f"Missing required column for target encoding: {ZIP_COLUMN}")

    x_train["Zip_encoded"] = encoder.transform(x_train[ZIP_COLUMN])
    x_test["Zip_encoded"] = encoder.transform(x_test[ZIP_COLUMN])
    x_train = x_train.drop(columns=[ZIP_COLUMN])
    x_test = x_test.drop(columns=[ZIP_COLUMN])

    x_train = pd.get_dummies(x_train, columns=[c for c in ONE_HOT_COLUMNS if c in x_train.columns], drop_first=True)
    x_test = pd.get_dummies(x_test, columns=[c for c in ONE_HOT_COLUMNS if c in x_test.columns], drop_first=True)
    x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)

    preds = predict_rent(model, x_test)
    metrics = regression_metrics(y_test, preds)

    if args.metrics_out:
        save_json(
            args.metrics_out,
            {
                "data_path": args.data,
                "model_path": args.model,
                "encoder_path": args.encoder,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "test_rows": int(len(x_test)),
                "metrics": metrics,
            },
        )

    print("Evaluation complete")
    print(f"MAE:  {metrics['mae']:.2f} CHF")
    print(f"RMSE: {metrics['rmse']:.2f} CHF")
    print(f"R2:   {metrics['r2']:.4f}")
    if args.metrics_out:
        print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
