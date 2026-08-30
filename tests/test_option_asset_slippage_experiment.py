from __future__ import annotations

import json
import unittest

import pandas as pd

from strategies.stage08_options.option_asset_slippage_experiment import (
    BEST_PATH,
    COMPARISON_PATH,
    OPTION_RETURNS_PATH,
    OPTION_TRADES_PATH,
    REPORT_PATH,
    SCENARIOS,
    effective_slippage,
)


class OptionAssetSlippageExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trades = pd.read_csv(
            OPTION_TRADES_PATH,
            parse_dates=["entry_date", "exit_date", "expiry"],
        )
        cls.returns = pd.read_csv(OPTION_RETURNS_PATH)
        cls.comparison = pd.read_csv(COMPARISON_PATH)
        cls.best = pd.read_csv(BEST_PATH, index_col=0)
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def test_one_tick_floor_is_enforced(self) -> None:
        conservative = next(item for item in SCENARIOS if item.name == "Conservative")
        effective, bucket, tick_floor, binds = effective_slippage(
            price=0.05,
            abs_delta=0.10,
            scenario=conservative,
            multiplier=1.0,
        )
        self.assertAlmostEqual(bucket, 0.10)
        self.assertAlmostEqual(tick_floor, 0.20)
        self.assertAlmostEqual(effective, 0.20)
        self.assertTrue(binds)

    def test_trades_are_liquid_and_in_requested_contract_range(self) -> None:
        self.assertTrue(self.trades["entry_volume"].gt(0).all())
        self.assertTrue(self.trades["exit_volume"].gt(0).all())
        self.assertTrue(self.trades["entry_abs_delta"].between(0.05, 0.15).all())
        self.assertTrue(self.trades["entry_dte"].between(30, 60).all())
        self.assertTrue((self.trades["entry_date"] < self.trades["exit_date"]).all())
        self.assertTrue(
            (
                pd.to_datetime(self.trades["entry_signal_date"])
                < self.trades["entry_date"]
            ).all()
        )
        self.assertTrue(
            (
                pd.to_datetime(self.trades["exit_signal_date"])
                < self.trades["exit_date"]
            ).all()
        )
        self.assertTrue(
            self.trades["exit_reason"].isin(["vix6_recovery", "month_end_roll"]).all()
        )

    def test_conservative_execution_is_never_better_than_optimistic(self) -> None:
        wide = self.returns.pivot(index="month", columns="scenario", values="option_return")
        self.assertTrue((wide["Conservative"] <= wide["Base"] + 1e-12).all())
        self.assertTrue((wide["Base"] <= wide["Optimistic"] + 1e-12).all())

    def test_candidate_grid_and_period_are_complete(self) -> None:
        self.assertEqual(len(self.comparison), 48)
        self.assertEqual(self.report["period"]["actual_start"], "2007-04")
        self.assertEqual(self.report["period"]["actual_end"], "2026-07")
        self.assertEqual(self.report["period"]["months"], 232)
        self.assertEqual(self.report["candidate_count"], 48)
        self.assertTrue(self.comparison["trigger"].str.startswith("vix6_").all())

    def test_option_is_a_funded_fifth_asset_sleeve(self) -> None:
        total = (
            self.best["w_existing_four_asset_sleeve"]
            + self.best["w_KOSPI200_put_option"]
        )
        self.assertLessEqual(float((total - 1.0).abs().max()), 1e-12)
        self.assertGreater(float(self.best["w_KOSPI200_put_option"].max()), 0.0)
        self.assertLessEqual(float(self.best["w_KOSPI200_put_option"].max()), 0.03)

    def test_selection_rule_matches_comparison(self) -> None:
        robust = []
        for trigger in self.comparison["trigger"].unique():
            for weight in self.comparison["max_option_weight"].unique():
                pair = self.comparison.loc[
                    self.comparison["trigger"].eq(trigger)
                    & self.comparison["max_option_weight"].eq(weight)
                    & self.comparison["scenario"].isin(["Base", "Conservative"])
                ]
                if len(pair) == 2 and pair["full_all_three_improve"].astype(bool).all():
                    robust.append((trigger, weight))
        self.assertEqual(len(robust), self.report["selection"]["robust_candidate_count"])
        if not robust:
            self.assertEqual(
                self.report["selection"]["selected_strategy"],
                "Existing_Final_RobustVKOSPI",
            )


if __name__ == "__main__":
    unittest.main()
