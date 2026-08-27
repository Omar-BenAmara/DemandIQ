import unittest

import pandas as pd

from src.open_ablation import (
    add_open_missing_indicator,
    feature_columns_for_open_ablation,
    simulate_missing_open_block,
)


class OpenAblationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Store": [622] * 5 + [1],
            "Date": pd.date_range("2015-08-01", periods=5, freq="D").tolist() + [pd.Timestamp("2015-08-01")],
            "Open": [1, 1, 1, 1, 1, 0],
            "Sales": [10, 11, 12, 13, 14, 0],
        })

    def test_feature_sets_differ_only_by_open_columns(self):
        columns = ["Store", "Promo", "Open"]
        self.assertEqual(feature_columns_for_open_ablation(columns, False), ["Store", "Promo"])
        self.assertEqual(feature_columns_for_open_ablation(columns, True), ["Store", "Promo", "Open", "Open_missing"])

    def test_missing_indicator_preserves_missing_open(self):
        frame = self.frame.copy()
        frame.loc[2, "Open"] = pd.NA
        result = add_open_missing_indicator(frame)
        self.assertTrue(pd.isna(result.loc[2, "Open"]))
        self.assertEqual(result.loc[2, "Open_missing"], 1)
        self.assertEqual(result.loc[0, "Open_missing"], 0)

    def test_missing_block_is_contiguous_and_targets_unchanged(self):
        result, dates = simulate_missing_open_block(self.frame, 622, "2015-08-02", 3)
        missing_dates = result.loc[result["Open"].isna(), "Date"].tolist()
        self.assertEqual(missing_dates, dates.tolist())
        self.assertEqual(result["Sales"].tolist(), self.frame["Sales"].tolist())
        self.assertEqual(result["Date"].tolist(), self.frame["Date"].tolist())


if __name__ == "__main__":
    unittest.main()