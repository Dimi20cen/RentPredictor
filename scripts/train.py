#!/usr/bin/env python3
"""Train XGBoost rent model from featured dataset."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import (
    encode_train_test,
    load_featured_data,
    regression_metrics,
    save_json,
    split_features_target,
    train_xgb_log_target,
    predict_rent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rent prediction model.")
    parser.add_argument("--data", default="data/processed/02_featured_data.pkl", help="Path to featured dataset pickle.")
    parser.add_argument("--model-out", default="models/xgb_rent_model.pkl", help="Output path for trained model pickle.")
    parser.add_argument("--encoder-out", default="models/zip_encoder.pkl", help="Output path for ZIP encoder pickle.")
    parser.add_argument("--metrics-out", default="models/training_metrics.json", help="Output path for training metrics JSON.")
    parser.add_argument(
        "--feature-columns-out",
        default="models/feature_columns.json",
        help="Output path for model feature column names JSON.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_featured_data(args.data)
    x, y = split_features_target(df)
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    x_train, x_test, encoder = encode_train_test(x_train_raw, x_test_raw, y_train)
    model = train_xgb_log_target(x_train, y_train, x_test, y_test)
    preds = predict_rent(model, x_test)
    metrics = regression_metrics(y_test, preds)

    model_out = Path(args.model_out)
    encoder_out = Path(args.encoder_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    encoder_out.parent.mkdir(parents=True, exist_ok=True)

    with model_out.open("wb") as f:
        pickle.dump(model, f)
    with encoder_out.open("wb") as f:
        pickle.dump(encoder, f)
    Path(args.feature_columns_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.feature_columns_out).write_text(json.dumps(list(x_train.columns), indent=2), encoding="utf-8")

    save_json(
        args.metrics_out,
        {
            "data_path": args.data,
            "random_state": args.random_state,
            "test_size": args.test_size,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "feature_count": int(len(x_train.columns)),
            "metrics": metrics,
        },
    )

    print("Training complete")
    print(f"MAE:  {metrics['mae']:.2f} CHF")
    print(f"RMSE: {metrics['rmse']:.2f} CHF")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Saved model to:   {args.model_out}")
    print(f"Saved encoder to: {args.encoder_out}")
    print(f"Saved columns to: {args.feature_columns_out}")
    print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
