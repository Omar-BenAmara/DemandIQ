"""Independent, model-free checks for notebook 04 feature invariants."""

import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_feature_engineering.ipynb"


def load_feature_functions():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"np": np, "pd": pd}
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in code_cells:
        source = "".join(cell.get("source", []))
        tree = ast.parse(source, filename=str(NOTEBOOK_PATH))
        definitions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id in {"PROMO_MONTHS", "LAG_WINDOWS", "ROLLING_WINDOWS", "historical_features"}
                    for target in node.targets
                )
            )
        ]
        if definitions:
            tree.body = definitions
            exec(compile(tree, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


class FeatureEngineeringInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = load_feature_functions()
        dates = pd.date_range("2015-01-01", periods=150, freq="D")
        rows = []
        for store in [1, 2]:
            for date in dates:
                rows.append({
                    "Store": store,
                    "Date": date,
                    "Sales": store * date.day,
                    "Open": 1,
                    "Promo": 0,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                    "StoreType": "a",
                    "Assortment": "a",
                    "CompetitionDistance": 100.0,
                    "CompetitionOpenSinceMonth": 1.0,
                    "CompetitionOpenSinceYear": 2014.0,
                    "Promo2": 0,
                    "Promo2SinceWeek": np.nan,
                    "Promo2SinceYear": np.nan,
                    "PromoInterval": np.nan,
                })
        cls.frame = pd.DataFrame(rows)

    def test_lags_and_rolling_use_only_origin_history(self):
        origin = pd.Timestamp("2015-03-31")
        history = self.features["build_frozen_history"](self.frame, origin)
        row = history[history.Store.eq(1)].iloc[0]
        prior = self.frame[(self.frame.Store == 1) & (self.frame.Date <= origin)].set_index("Date")["Sales"]

        self.assertEqual(row["lag_1"], prior.loc[origin - pd.Timedelta(days=1)])
        self.assertEqual(row["lag_7"], prior.loc[origin - pd.Timedelta(days=7)])
        expected_window = prior.loc[origin - pd.Timedelta(days=6) : origin]
        self.assertEqual(row["rolling_mean_7"], expected_window.mean())
        self.assertEqual(row["rolling_median_7"], expected_window.median())
        self.assertEqual(row["rolling_std_7"], expected_window.std())

    def test_history_cutoff_excludes_future_sales(self):
        origin = pd.Timestamp("2015-03-31")
        altered = self.frame.copy()
        altered.loc[altered.Date > origin, "Sales"] = 999999
        original = self.features["build_frozen_history"](self.frame, origin)
        changed = self.features["build_frozen_history"](altered, origin)
        history_columns = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_median_7", "rolling_std_7"]
        pd.testing.assert_frame_equal(original[["Store", *history_columns]], changed[["Store", *history_columns]])

    def test_missing_calendar_observation_is_not_treated_as_zero(self):
        origin = pd.Timestamp("2015-03-31")
        incomplete = self.frame.loc[
            ~((self.frame.Store == 1) & (self.frame.Date == origin - pd.Timedelta(days=3)))
        ]
        history = self.features["build_frozen_history"](incomplete, origin)
        row = history[history.Store.eq(1)].iloc[0]

        self.assertTrue(pd.isna(row["rolling_mean_7"]))
        self.assertTrue(pd.isna(row["rolling_median_7"]))
        self.assertTrue(pd.isna(row["rolling_std_7"]))
        self.assertEqual(row["store_history_count"], len(incomplete[(incomplete.Store == 1) & (incomplete.Date <= origin)]))

    def test_validation_rows_are_after_origin_and_targets_are_separate(self):
        origin = pd.Timestamp("2015-03-31")
        base = self.features["add_store_time_features"](self.frame)
        validation = self.features["build_direct_rows"](base, self.frame[["Store", "Date", "Sales"]], origin, include_target=True)
        self.assertEqual(validation.Date.min(), origin + pd.Timedelta(days=1))
        self.assertEqual(validation.Date.max(), origin + pd.Timedelta(days=48))
        self.assertTrue((validation.HistoryOrigin == origin).all())
        self.assertTrue((validation.Date > validation.HistoryOrigin).all())

        history = self.frame[self.frame.Date <= origin]
        self.assertLessEqual(history.Date.max(), origin)
        self.assertNotIn("Sales", self.features["build_direct_rows"](base, history[["Store", "Date", "Sales"]], origin, include_target=False).columns)

    def test_test_rows_exclude_sales_and_preserve_missing_open(self):
        origin = pd.Timestamp("2015-03-31")
        base = self.features["add_store_time_features"](self.frame)
        test_rows = base[base.Date > origin].copy()
        test_rows.loc[test_rows.index[0], "Open"] = np.nan
        test_features = self.features["build_direct_rows"](
            test_rows,
            self.frame[["Store", "Date", "Sales"]],
            origin,
            include_target=False,
        )
        self.assertNotIn("Sales", test_features.columns)
        self.assertTrue(test_features.Open.isna().any())
        self.assertTrue(test_features.ForecastHorizon.between(1, 48).all())


if __name__ == "__main__":
    unittest.main()