from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

import strategies.stage06_vkospi.vkospi_dynamic_risk_experiment as experiment
class VKOSPIDynamicRiskExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = experiment.load_reference_weights()
        cls.arrays = experiment.prepare_arrays(
            experiment.load_daily_open_levels(),
            cls.reference,
            experiment.build_daily_vkospi_signals(),
        )
        cls.report = json.loads(
            (experiment.RESULTS / "vkospi_dynamic_validation.json").read_text(
                encoding="utf-8"
            )
        )

    def test_every_daily_signal_precedes_earned_return(self) -> None:
        signal_dates = pd.DatetimeIndex(self.arrays["signal_dates"])
        return_dates = pd.DatetimeIndex(self.arrays["dates"])
        valid = signal_dates.notna()
        self.assertTrue((signal_dates[valid] < return_dates[valid]).all())

    def test_reconciliation_is_identity_without_an_overlay(self) -> None:
        _, baseline = experiment.simulate(self.arrays, None, keep_daily=False)
        reconciled = experiment.reconcile_to_monthly_reference(
            self.reference, baseline, baseline
        )
        np.testing.assert_allclose(
            reconciled["return"].to_numpy(),
            self.reference.loc[reconciled.index, "return"].to_numpy(),
            rtol=0,
            atol=1e-14,
        )

    def test_selected_weights_remain_long_only(self) -> None:
        daily = pd.read_csv(experiment.RESULTS / "vkospi_dynamic_daily.csv")
        weights = daily[[f"w_{asset}" for asset in experiment.ASSETS]]
        self.assertGreaterEqual(float(weights.min().min()), -1e-12)
        self.assertTrue(np.isfinite(weights.to_numpy()).all())

    def test_promoted_candidate_improves_all_locked_objectives(self) -> None:
        locked = self.report["locked"]
        self.assertTrue(locked["promoted"])
        for view in ("actual_daily", "monthly_reference_reconciled"):
            self.assertTrue(locked[view]["passes_all_three"])
            deltas = locked[view]["deltas"]
            self.assertGreater(deltas["CAGR"], 0)
            self.assertGreater(deltas["Sharpe"], 0)
            self.assertGreaterEqual(deltas["MDD"], 0)

    def test_two_stage_winner_is_reproducible(self) -> None:
        self.assertEqual(
            self.report["winner"],
            {
                "mode": "mean",
                "level_threshold": 0.8,
                "momentum_window": 5,
                "spike_threshold": 0.15,
                "max_risk_transfer": 0.25,
                "bond_share": 0.0,
                "rebalance_band": 0.15,
                "financing_rate": 0.04,
            },
        )
        bootstrap = self.report["locked"]["reconciled_multiobjective_bootstrap"]
        self.assertGreater(bootstrap["probability_sharpe_improves"], 0.95)
        self.assertGreater(bootstrap["probability_mdd_improves"], 0.90)
        self.assertGreater(bootstrap["probability_all_three_improve"], 0.55)

    def test_reconciled_full_period_matches_or_improves_all_objectives(self) -> None:
        comparison = pd.read_csv(experiment.RESULTS / "vkospi_dynamic_comparison.csv")
        full = comparison.loc[comparison["Period"] == "full_2007_2026"].set_index(
            "Strategy"
        )
        reference = full.loc["ReferenceMonthly"]
        dynamic = full.loc["VKOSPIDynamicReconciled"]
        self.assertGreater(dynamic["CAGR"], reference["CAGR"])
        self.assertGreater(dynamic["Sharpe"], reference["Sharpe"])
        self.assertGreaterEqual(dynamic["MDD"], reference["MDD"] - 1e-12)

    def test_standard_and_high_cost_results_are_robust(self) -> None:
        costs = pd.read_csv(experiment.RESULTS / "vkospi_dynamic_cost_sensitivity.csv")
        for period in ("cost_1.0x_locked", "cost_2.0x_locked"):
            view = costs.loc[costs["Period"] == period].set_index("Strategy")
            reference = view.loc["ReferenceDaily"]
            dynamic = view.loc["VKOSPIDynamic"]
            self.assertGreater(dynamic["Sharpe"], reference["Sharpe"])
            self.assertGreater(dynamic["MDD"], reference["MDD"])
            if period == "cost_1.0x_locked":
                self.assertGreater(dynamic["CAGR"], reference["CAGR"])
            else:
                self.assertGreater(dynamic["CAGR"], 0.995 * reference["CAGR"])


if __name__ == "__main__":
    unittest.main()
