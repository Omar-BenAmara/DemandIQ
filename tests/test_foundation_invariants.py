"""Independent checks for the verified Rossmann foundation assumptions."""

import unittest
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parents[1] / "data" / "raw" / "rossmann"


class FoundationInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(DATA_DIR / "train.csv", low_memory=False, parse_dates=["Date"])
        cls.test = pd.read_csv(DATA_DIR / "test.csv", low_memory=False, parse_dates=["Date"])
        cls.store = pd.read_csv(DATA_DIR / "store.csv", low_memory=False)

    def test_source_relationships_and_keys(self):
        self.assertEqual(self.train.shape, (1_017_209, 9))
        self.assertEqual(self.test.shape, (41_088, 8))
        self.assertEqual(self.store.shape, (1_115, 10))
        self.assertEqual(self.train.duplicated(["Store", "Date"]).sum(), 0)
        self.assertEqual(self.test.duplicated(["Store", "Date"]).sum(), 0)
        self.assertEqual(set(self.train.Store) - set(self.store.Store), set())
        self.assertEqual(set(self.test.Store) - set(self.store.Store), set())

    def test_temporal_split_is_strict(self):
        self.assertEqual(self.train.Date.max(), pd.Timestamp("2015-07-31"))
        self.assertEqual(self.test.Date.min(), pd.Timestamp("2015-08-01"))
        self.assertLess(self.train.Date.max(), self.test.Date.min())

    def test_missing_open_is_preserved_and_localized(self):
        missing = self.test[self.test.Open.isna()]
        self.assertEqual(len(missing), 11)
        self.assertEqual(missing.Store.nunique(), 1)
        self.assertEqual(missing.Store.iloc[0], 622)
        self.assertEqual(missing.Date.min(), pd.Timestamp("2015-09-05"))
        self.assertEqual(missing.Date.max(), pd.Timestamp("2015-09-17"))

    def test_independent_lag_and_rolling_cutoff(self):
        store_history = self.train[self.train.Store.eq(1)].set_index("Date")["Sales"].sort_index()
        origin = pd.Timestamp("2015-07-31")
        target_date = origin + pd.Timedelta(days=1)
        lag_date = target_date - pd.Timedelta(days=1)
        self.assertEqual(store_history.loc[lag_date], 5_263)
        prior_window = store_history.loc[origin - pd.Timedelta(days=6) : origin]
        self.assertEqual(len(prior_window), 7)
        self.assertNotIn(target_date, prior_window.index)
        self.assertEqual(prior_window.mean(), store_history.loc[prior_window.index].mean())


if __name__ == "__main__":
    unittest.main()