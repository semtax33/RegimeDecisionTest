from __future__ import annotations

import json
import unittest

import pandas as pd

from strategies.stage08_options.vix6_processed_input_experiment import (
    COMPARISON_PATH,
    MONTHLY_FEATURE_PATH,
    REPORT_PATH,
)


class Vix6ProcessedInputExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.monthly = pd.read_csv(MONTHLY_FEATURE_PATH, index_col=0)
        cls.comparison = pd.read_csv(COMPARISON_PATH)
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def test_all_requested_transform_families_exist(self) -> None:
        columns = set(self.monthly.columns)
        for component in self.report["decomposition_components"]:
            for transform in ("ma5", "ma21", "delta5", "delta21", "z63", "z126"):
                self.assertIn(f"{component}_{transform}_last", columns)

    def test_monthly_features_are_lagged(self) -> None:
        target = pd.PeriodIndex(self.monthly.index, freq="M")
        signal = pd.PeriodIndex(self.monthly["signal_month"], freq="M")
        self.assertTrue((signal < target).all())
        self.assertTrue(((target - signal).map(lambda value: value.n) == 1).all())

    def test_requested_full_period_is_measured(self) -> None:
        period = self.report["actual_common_test_period"]
        self.assertEqual(period["start"], "2007-04")
        self.assertGreaterEqual(period["end"], "2026-07")
        self.assertEqual(period["months"], 232)
        baseline = self.comparison.loc[
            self.comparison["candidate"].eq("Existing_Final_RobustVKOSPI")
        ].iloc[0]
        self.assertGreater(float(baseline["full_CAGR"]), 0)
        self.assertGreater(float(baseline["full_Sharpe"]), 0)
        self.assertLess(float(baseline["full_MDD"]), 0)

    def test_all_candidate_combinations_completed(self) -> None:
        expected = int(self.report["candidate_count_excluding_baseline"])
        actual = self.comparison["transform_family"].isin(
            ["raw_control", "processed"]
        ).sum()
        self.assertEqual(int(actual), expected)
        self.assertEqual(
            int(self.comparison["convergence_warning_count"].fillna(0).sum()),
            0,
        )

    def test_deployment_gate_is_consistent(self) -> None:
        processed = self.comparison.loc[
            self.comparison["transform_family"].eq("processed")
        ]
        any_winner = bool(processed["full_all_three_improve"].astype(bool).any())
        if any_winner:
            self.assertNotEqual(
                self.report["deployment_decision"],
                "Existing_Final_RobustVKOSPI",
            )
        else:
            self.assertEqual(
                self.report["deployment_decision"],
                "Existing_Final_RobustVKOSPI",
            )


if __name__ == "__main__":
    unittest.main()
