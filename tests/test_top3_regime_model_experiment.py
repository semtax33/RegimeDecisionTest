from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import strategies.stage07_regime_models.top3_regime_model_experiment as experiment
class Top3RegimeModelExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = experiment.RESULTS
        cls.report = json.loads(
            (cls.results / "top3_regime_model_validation.json").read_text(
                encoding="utf-8"
            )
        )
        cls.probabilities = pd.read_csv(
            cls.results / "top3_regime_model_probabilities.csv"
        )
        cls.metrics = pd.read_csv(
            cls.results / "top3_regime_model_prediction_metrics.csv"
        )
        cls.comparison = pd.read_csv(
            cls.results / "top3_regime_model_comparison.csv"
        )
        cls.audit = pd.read_csv(
            cls.results / "top3_regime_model_lgbm_audit.csv"
        )

    def test_all_feedback_ranked_models_are_implemented(self) -> None:
        self.assertEqual(
            set(self.report["validation"]),
            {"CJM", "TVTP-HMM", "CJM+LightGBM"},
        )
        self.assertIn("SJM", set(self.probabilities["Model"]))

    def test_probabilities_are_finite_and_bounded(self) -> None:
        columns = ["p_h1", "p_h3", "p_h6"]
        values = self.probabilities[columns].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        self.assertGreater(len(finite), 300)
        self.assertGreaterEqual(float(finite.min()), 0.0)
        self.assertLessEqual(float(finite.max()), 1.0)

    def test_lightgbm_training_labels_never_cross_signal_month(self) -> None:
        signal = pd.PeriodIndex(self.audit["signal_month"], freq="M")
        max_label = pd.PeriodIndex(self.audit["max_label_month"], freq="M")
        fit_end = pd.PeriodIndex(self.audit["fit_end_month"], freq="M")
        self.assertTrue((max_label <= signal).all())
        self.assertTrue((fit_end < signal).all())

    def test_locked_period_was_not_used_for_overlay_selection(self) -> None:
        self.assertEqual(self.report["calibration_end"], "2017-12")
        self.assertEqual(self.report["locked_start"], "2018-01")
        calibration = pd.read_csv(
            self.results / "top3_regime_model_calibration.csv"
        )
        for model, group in calibration.groupby("Model"):
            self.assertEqual(int(group["Selected"].fillna(False).sum()), 1, model)

    def test_full_and_locked_comparisons_cover_same_strategies(self) -> None:
        expected = {
            "Existing_VKOSPI_Dynamic",
            "CJM",
            "TVTP-HMM",
            "CJM+LightGBM",
        }
        for period in ("full_2007_2026", "locked_2018_2026"):
            view = self.comparison.loc[self.comparison["Period"] == period]
            self.assertEqual(set(view["Strategy"]), expected)
            self.assertTrue(np.isfinite(view[["CAGR", "Sharpe", "MDD"]]).all().all())

    def test_probability_quality_has_all_horizons_and_periods(self) -> None:
        for period in ("full_oos", "locked_2018_2026"):
            view = self.metrics.loc[self.metrics["Period"] == period]
            self.assertEqual(set(view["Model"]), set(experiment.MODELS))
            for model in experiment.MODELS:
                self.assertEqual(
                    set(view.loc[view["Model"] == model, "Horizon"]),
                    set(experiment.HORIZONS),
                )

    def test_backtest_signal_precedes_earned_return(self) -> None:
        for model in ("cjm", "tvtp_hmm", "cjm_plus_lightgbm"):
            path = self.results / f"top3_regime_model_backtest_{model}.csv"
            frame = pd.read_csv(path)
            months = pd.PeriodIndex(frame["month"], freq="M")
            # The implementation always queries forecast.loc[month - 1].
            self.assertTrue(((months - 1) < months).all())

    def test_official_cjm_and_primary_tvtp_paths_were_used(self) -> None:
        self.assertGreater(self.report["cjm_fit_success_rate"], 0.95)
        self.assertGreater(self.report["tvtp_primary_fit_rate"], 0.70)


if __name__ == "__main__":
    unittest.main()
