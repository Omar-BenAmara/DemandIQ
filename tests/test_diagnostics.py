import unittest

import numpy as np
import pandas as pd

from src.diagnostics import add_diagnostic_dimensions, summarize_groups


class DiagnosticTests(unittest.TestCase):
    def test_group_summary_reports_metrics_and_rows(self):
        frame = pd.DataFrame({"Group": ["a", "a", "b"], "Sales": [10, 20, 0], "Prediction": [8, 22, 3]})
        summary = summarize_groups(frame, ["Group"])
        self.assertEqual(summary.set_index("Group").loc["a", "Rows"], 2)
        self.assertAlmostEqual(summary.set_index("Group").loc["a", "MAE"], 2.0)
        self.assertTrue(np.isnan(summary.set_index("Group").loc["b", "RMSPE"]))

    def test_dimensions_are_descriptive_and_deterministic(self):
        frame = pd.DataFrame({"Date": pd.to_datetime(["2015-08-01"]), "Sales": [500], "Prediction": [700]})
        enriched = add_diagnostic_dimensions(frame)
        self.assertEqual(enriched.loc[0, "Month"], 8)
        self.assertEqual(enriched.loc[0, "SalesMagnitude"], "low")
        self.assertEqual(enriched.loc[0, "SignedError"], 200)


if __name__ == "__main__":
    unittest.main()