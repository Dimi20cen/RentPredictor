import unittest

import numpy as np
import pandas as pd

from src.ml_pipeline import (
    encode_train_test,
    predict_rent,
    regression_metrics,
    split_features_target,
    transform_features_for_model,
)


class DummyModel:
    def predict(self, x):
        return np.log1p(np.array([1000.0] * len(x)))


class MlPipelineTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                {
                    "Rent": 2000.0,
                    "Zip": 8001,
                    "Canton": "ZH",
                    "SubType": "FLAT",
                    "Rooms": 2.5,
                    "Area_m2": 60.0,
                },
                {
                    "Rent": 2200.0,
                    "Zip": 8001,
                    "Canton": "ZH",
                    "SubType": "FLAT",
                    "Rooms": 3.0,
                    "Area_m2": 70.0,
                },
                {
                    "Rent": 1800.0,
                    "Zip": 1200,
                    "Canton": "GE",
                    "SubType": "STUDIO",
                    "Rooms": 1.5,
                    "Area_m2": 35.0,
                },
            ]
        )

    def test_split_encode_and_transform(self):
        x, y = split_features_target(self.df)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)

        x_train_raw = x.iloc[:2].copy()
        x_test_raw = x.iloc[2:].copy()
        y_train = y.iloc[:2].copy()

        x_train, x_test, encoder = encode_train_test(x_train_raw, x_test_raw, y_train)
        self.assertIn("Zip_encoded", x_train.columns)
        self.assertNotIn("Zip", x_train.columns)

        feature_names = list(x_train.columns)
        x_transformed = transform_features_for_model(x_test_raw, encoder, feature_names)
        self.assertEqual(set(x_transformed.columns), set(feature_names))
        self.assertEqual(len(x_transformed), 1)

    def test_predict_and_metrics(self):
        model = DummyModel()
        x = pd.DataFrame({"a": [1.0, 2.0]})
        preds = predict_rent(model, x)
        self.assertTrue(np.allclose(preds, [1000.0, 1000.0]))

        metrics = regression_metrics(pd.Series([900.0, 1100.0]), preds)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)


if __name__ == "__main__":
    unittest.main()
