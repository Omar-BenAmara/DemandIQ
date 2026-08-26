import unittest

import numpy as np
import pandas as pd

from src.metrics import mae, rmse, rmspe
from src.modeling import inverse_target, make_submission, transform_target


class ModelingMetricTests(unittest.TestCase):
    def test_rmspe_uses_sales_mask_not_open(self):
        actual = pd.Series([0, 10, 20])
        predicted = pd.Series([999, 5, 30])
        self.assertAlmostEqual(rmspe(actual, predicted), np.sqrt((0.5**2 + 0.5**2) / 2))

    def test_metrics_and_shapes(self):
        self.assertEqual(mae([0, 2], [1, 4]), 1.5)
        self.assertEqual(rmse([0, 2], [1, 4]), np.sqrt(2.5))
        with self.assertRaises(ValueError):
            rmspe([[1]], [[1]])
        with self.assertRaises(ValueError):
            rmspe([1, 2], [1])
        with self.assertRaises(ValueError):
            rmspe([0, 0], [1, 1])

    def test_target_round_trip_and_submission(self):
        sales = np.array([0, 10, 1234], dtype="float64")
        np.testing.assert_allclose(inverse_target(transform_target(sales)), sales)
        frame = pd.DataFrame({"Id": [3, 1], "Date": pd.to_datetime(["2015-08-01", "2015-08-02"])})
        submission = make_submission(frame, [2.0, 4.0], "tests/_submission_test.csv")
        self.assertEqual(list(submission.columns), ["Id", "Sales"])


if __name__ == "__main__":
    unittest.main()