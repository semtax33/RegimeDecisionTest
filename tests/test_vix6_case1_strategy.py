from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import ASSETS
from strategies.stage08_options.vix6_case1_model_comparison import (
    FINAL_SELECTION_PATH,
    INPUT_DAILY_PATH,
    MONTHLY_INPUT_PATH,
    SELECTED_PATH,
    STANDALONE_DAILY_PATH,
)
from strategies.stage08_options.vix6_case1_strategy import (
    FEATURE_PATH,
    REPORT_PATH,
    _second_thursday,
)


ROOT = Path(__file__).resolve().parents[1]
class Vix6Case1StrategyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features = pd.read_csv(FEATURE_PATH, index_col=0, parse_dates=True)
        cls.selection = json.loads(FINAL_SELECTION_PATH.read_text(encoding="utf-8"))

    def test_second_thursday_expiry_approximation(self) -> None:
        month_starts = pd.Series(pd.to_datetime(["2026-01-01", "2026-08-01"]))
        actual = _second_thursday(month_starts)
        expected = pd.Series(pd.to_datetime(["2026-01-08", "2026-08-13"]))
        pd.testing.assert_series_equal(actual.reset_index(drop=True), expected)

    def test_six_factors_and_decomposition_identity_exist(self) -> None:
        expected = {
            "sticky_strike",
            "parallel_shift",
            "put_skew",
            "call_skew",
            "downside_convexity",
            "upside_convexity",
        }
        self.assertTrue(expected.issubset(self.features.columns))
        residual = self.features["decomposition_residual"].dropna().abs().max()
        self.assertLessEqual(float(residual), 1e-12)
        self.assertGreater(len(self.features), 4_000)

    def test_monthly_inputs_are_lagged_one_month(self) -> None:
        monthly = pd.read_csv(MONTHLY_INPUT_PATH, index_col=0)
        target = pd.PeriodIndex(monthly.index, freq="M")
        signal = pd.PeriodIndex(monthly["signal_month"], freq="M")
        self.assertTrue((signal < target).all())
        self.assertTrue(((target - signal).map(lambda value: value.n) == 1).all())

    def test_daily_signals_precede_action_open(self) -> None:
        for path in (INPUT_DAILY_PATH, STANDALONE_DAILY_PATH):
            daily = pd.read_csv(path, index_col=0, parse_dates=True)
            signal = pd.to_datetime(daily["signal_date"], errors="coerce")
            valid = signal.notna()
            self.assertTrue((daily.index[valid] > signal[valid]).all())

    def test_only_four_assets_are_allocated(self) -> None:
        expected_weights = {f"w_{asset}" for asset in ASSETS}
        for path in (INPUT_DAILY_PATH, STANDALONE_DAILY_PATH):
            columns = set(pd.read_csv(path, nrows=2).columns)
            actual_weights = {column for column in columns if column.startswith("w_")}
            self.assertEqual(actual_weights, expected_weights)
            self.assertFalse(any("option" in column.lower() for column in actual_weights))
        self.assertFalse(self.selection["option_is_allocated_asset"])

    def test_existing_strategy_is_kept_when_no_alternative_wins_all_three(self) -> None:
        self.assertEqual(
            self.selection["selected_strategy"], "Existing_Final_RobustVKOSPI"
        )
        self.assertFalse(
            any(
                candidate["all_three_improve"]
                for candidate in self.selection["alternatives"].values()
            )
        )
        selected = pd.read_csv(SELECTED_PATH, index_col=0)
        baseline = pd.read_csv(
            ROOT / "results" / "balanced_logistic_no_sjm_final_reconciled.csv",
            index_col=0,
        )
        common = selected.index.intersection(baseline.index)
        difference = (
            selected.loc[common, "return"] - baseline.loc[common, "return"]
        ).abs()
        self.assertLess(float(difference.max()), 1e-12)

    def test_reports_disclose_post_lock_exploration(self) -> None:
        hybrid_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(
            hybrid_report["lookahead_audit"]["locked_metrics_observed_during_development"]
        )
        self.assertIn("Post-lock", self.selection["development_status"])
        self.assertEqual(
            self.selection["input_candidate"]["added_vix6_features"],
            [
                "asymmetry_mean",
                "asymmetry_max",
                "breadth_z_mean",
                "breadth_z_max",
                "reaction_z_mean",
                "left_impulse_z_max",
            ],
        )


if __name__ == "__main__":
    unittest.main()
