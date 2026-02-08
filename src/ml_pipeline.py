"""Reusable ML pipeline utilities for training/evaluation/prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import category_encoders as ce
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DROP_COLUMNS = ["ID", "Description", "City", "Date_Created", "Link", "Title", "tax_source"]
ONE_HOT_COLUMNS = ["Canton", "SubType"]
TARGET_COLUMN = "Rent"
ZIP_COLUMN = "Zip"

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:absoluteerror",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
}


def load_featured_data(path: str | Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def prepare_model_df(df: pd.DataFrame) -> pd.DataFrame:
    drop = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=drop).copy()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_df = prepare_model_df(df)
    if TARGET_COLUMN not in model_df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    x = model_df.drop(columns=[TARGET_COLUMN]).copy()
    y = model_df[TARGET_COLUMN].copy()
    return x, y


def encode_train_test(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, ce.TargetEncoder]:
    if ZIP_COLUMN not in x_train.columns:
        raise ValueError(f"Missing required column for target encoding: {ZIP_COLUMN}")

    x_train = x_train.copy()
    x_test = x_test.copy()

    encoder = ce.TargetEncoder(cols=[ZIP_COLUMN])
    x_train["Zip_encoded"] = encoder.fit_transform(x_train[ZIP_COLUMN], y_train)
    x_test["Zip_encoded"] = encoder.transform(x_test[ZIP_COLUMN])

    x_train = x_train.drop(columns=[ZIP_COLUMN])
    x_test = x_test.drop(columns=[ZIP_COLUMN])

    x_train = pd.get_dummies(x_train, columns=[c for c in ONE_HOT_COLUMNS if c in x_train.columns], drop_first=True)
    x_test = pd.get_dummies(x_test, columns=[c for c in ONE_HOT_COLUMNS if c in x_test.columns], drop_first=True)
    x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)
    return x_train, x_test, encoder


def transform_features_for_model(
    x_raw: pd.DataFrame,
    encoder: ce.TargetEncoder,
    feature_names: list[str],
) -> pd.DataFrame:
    if ZIP_COLUMN not in x_raw.columns:
        raise ValueError(f"Missing required column for target encoding: {ZIP_COLUMN}")

    x_raw = x_raw.copy()
    x_raw["Zip_encoded"] = encoder.transform(x_raw[ZIP_COLUMN])
    x_raw = x_raw.drop(columns=[ZIP_COLUMN])
    x_raw = pd.get_dummies(x_raw, columns=[c for c in ONE_HOT_COLUMNS if c in x_raw.columns], drop_first=True)

    x_final = pd.DataFrame(0, index=x_raw.index, columns=feature_names, dtype=float)
    for col in x_raw.columns:
        if col in x_final.columns:
            x_final[col] = x_raw[col]
    return x_final


def train_xgb_log_target(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict[str, Any] | None = None,
) -> xgb.XGBRegressor:
    train_log = np.log1p(y_train)
    valid_log = np.log1p(y_valid)

    merged = dict(DEFAULT_XGB_PARAMS)
    if params:
        merged.update(params)

    model = xgb.XGBRegressor(**merged)
    model.fit(
        x_train,
        train_log,
        eval_set=[(x_valid, valid_log)],
        verbose=False,
    )
    return model


def predict_rent(model: xgb.XGBRegressor, x: pd.DataFrame) -> np.ndarray:
    pred_log = model.predict(x)
    return np.expm1(pred_log)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
