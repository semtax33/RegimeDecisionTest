from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

import strategies.stage06_vkospi.vkospi_dynamic_risk_experiment as base
import strategies.stage06_vkospi.vkospi_robust_dynamic_attribution as attribution
import strategies.stage06_vkospi.vkospi_robust_dynamic_experiment as robust
class VKOSPIRobustDynamicExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = base.load_reference_weights()
        cls.arrays = base.prepare_arrays(
            base.load_daily_open_levels(),
            cls.reference,
            base.build_daily_vkospi_signals(),
        )
        cls.features = robust.align_features_to_arrays(
            robust.build_robust_daily_features(), cls.arrays
        )
        cls.report = json.loads(
            (robust.RESULTS / "vkospi_robust_dynamic_validation.json").read_text(
                encoding="utf-8"
            )
        )

    def test_signals_precede_returns_and_stress_is_bounded(self) -> None:
        signal_dates = pd.DatetimeIndex(self.arrays["signal_dates"])
        return_dates = pd.DatetimeIndex(self.arrays["dates"])
        valid = signal_dates.notna()
        self.assertTrue((signal_dates[valid] < return_dates[valid]).all())
        stress = robust.stress_from_features(
            self.features, "acceleration", level_threshold=0.9, shock_threshold=1.0
        )
        self.assertTrue(np.isfinite(stress).all())
        self.assertGreaterEqual(float(stress.min()), 0.0)
        self.assertLessEqual(float(stress.max()), 1.0)

    def test_stress_override_rejects_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "stress_override must have shape"):
            base.simulate(self.arrays, None, stress_override=np.zeros(3))

    def test_selected_configuration_is_reproducible_and_prelocked(self) -> None:
        self.assertEqual(self.report["calibration_end"], "2017-12")
        self.assertEqual(self.report["locked_start"], "2018-01")
        self.assertEqual(self.report["candidate_count"], 810)
        self.assertEqual(self.report["strict_eligible_count"], 2)
        self.assertEqual(
            self.report["winner"],
            {
                "mode": "acceleration",
                "level_threshold": 0.9,
                "shock_threshold": 1.0,
                "max_risk_transfer": 0.35,
                "bond_share": 0.0,
                "rebalance_band": 0.2,
                "financing_rate": 0.04,
            },
        )
        calibration = pd.read_csv(
            robust.RESULTS / "vkospi_robust_dynamic_calibration.csv"
        )
        selected = calibration.loc[calibration["Selected"]]
        self.assertEqual(len(selected), 1)
        row = selected.iloc[0]
        for prefix in ("Cal", "Validation"):
            self.assertGreater(row[f"{prefix}_CAGRDelta"], 0)
            self.assertGreater(row[f"{prefix}_SharpeDelta"], 0)
            self.assertGreaterEqual(row[f"{prefix}_MDDDelta"], 0)

    def test_reconciliation_applies_overlay_relative_factor(self) -> None:
        baseline = pd.read_csv(
            robust.RESULTS / "vkospi_robust_dynamic_monthly.csv", index_col=0
        )
        baseline.index = pd.PeriodIndex(baseline.index, freq="M")
        _, neutral = base.simulate(self.arrays, None, keep_daily=False)
        reconciled = base.reconcile_to_monthly_reference(
            self.reference, neutral, baseline
        )
        common = reconciled.index
        expected = (
            (1 + self.reference.loc[common, "return"])
            * (1 + baseline.loc[common, "return"])
            / (1 + neutral.loc[common, "return"])
            - 1
        )
        np.testing.assert_allclose(reconciled["return"], expected, atol=1e-14, rtol=0)

    def test_locked_reconciled_strategy_improves_all_objectives(self) -> None:
        locked = self.report["locked"]
        self.assertTrue(locked["passes_all_three"])
        for metric in ("CAGR", "Sharpe"):
            self.assertGreater(locked["deltas"][metric], 0)
        self.assertGreaterEqual(locked["deltas"]["MDD"], 0)

    def test_locked_actual_path_survives_double_costs(self) -> None:
        costs = pd.read_csv(
            robust.RESULTS / "vkospi_robust_dynamic_cost_sensitivity.csv"
        )
        for period in ("cost_1.0x_locked", "cost_2.0x_locked"):
            view = costs.loc[costs["Period"] == period].set_index("Strategy")
            existing = view.loc["ExistingDynamic"]
            candidate = view.loc["RobustDynamic"]
            self.assertGreater(candidate["CAGR"], existing["CAGR"])
            self.assertGreater(candidate["Sharpe"], existing["Sharpe"])
            self.assertGreater(candidate["MDD"], existing["MDD"])

    def test_attribution_uses_the_exact_promoted_stress_formula(self) -> None:
        components = attribution.diagnostic_components(
            self.features, level_threshold=0.9, shock_threshold=1.0
        )
        promoted = robust.stress_from_features(
            self.features, "acceleration", level_threshold=0.9, shock_threshold=1.0
        )
        np.testing.assert_allclose(
            components["SelectedAccelerationBlend"], promoted, atol=0, rtol=0
        )

    def test_attribution_final_row_matches_validation_report(self) -> None:
        stepwise = pd.read_csv(
            robust.RESULTS / "vkospi_robust_dynamic_stepwise_attribution.csv"
        )
        final = stepwise.loc[
            (stepwise["Experiment"] == "4_RobustWinner_Band20")
            & (stepwise["Period"] == "locked_2018_2026")
        ].iloc[0]
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar"):
            self.assertAlmostEqual(
                float(final[metric]), float(self.report["locked"]["robust"][metric])
            )

        stats = pd.read_csv(
            robust.RESULTS / "vkospi_robust_dynamic_signal_statistics.csv",
            index_col="Statistic",
        )
        self.assertLess(
            stats.loc["StressAbove025Days", "RobustWinner"],
            stats.loc["StressAbove025Days", "ExistingDynamic"],
        )


if __name__ == "__main__":
    unittest.main()
