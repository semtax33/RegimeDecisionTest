from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

import strategies.stage06_vkospi.vkospi_feature_experiment as experiment
class VKOSPIFeatureExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.daily = experiment.load_vkospi_daily()
        cls.kospi = experiment.load_kodex200_close()

    def test_source_schema_and_coverage(self) -> None:
        self.assertEqual(
            list(self.daily.columns), ["close", "change", "return_pct", "open", "high", "low"]
        )
        self.assertEqual(len(self.daily), 5_837)
        self.assertEqual(self.daily.index.min(), pd.Timestamp("2003-01-02"))
        self.assertGreaterEqual(self.daily["close"].notna().mean(), 0.999)
        self.assertTrue(self.daily.index.is_unique)

    def test_features_use_only_pre_month_information(self) -> None:
        months = pd.period_range("2017-12", "2018-02", freq="M")
        features = experiment.build_vkospi_features(months, self.daily, self.kospi)
        signal_month = features["vkospi_signal_date"].dt.to_period("M")
        self.assertTrue((signal_month < features.index).all())

    def test_future_data_cannot_change_an_earlier_feature_row(self) -> None:
        month = pd.PeriodIndex(["2018-01"], freq="M")
        original = experiment.build_vkospi_features(month, self.daily, self.kospi)
        changed = self.daily.copy()
        changed.loc[changed.index >= pd.Timestamp("2018-01-01"), "close"] *= 100
        mutated = experiment.build_vkospi_features(month, changed, self.kospi)
        numeric = experiment.RAW_FEATURES + experiment.DERIVED_FEATURES + experiment.OAP_FEATURES
        np.testing.assert_allclose(
            original[numeric].to_numpy(dtype=float),
            mutated[numeric].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_open_asset_pricing_lookbacks_match_documented_analogues(self) -> None:
        month = pd.PeriodIndex(["2018-01"], freq="M")
        features = experiment.build_vkospi_features(month, self.daily, self.kospi)
        cutoff = pd.Timestamp("2017-12-30")
        close = self.daily.loc[:cutoff, "close"]
        expected_mom12 = close.iloc[-22] / close.iloc[-253] - 1
        expected_short_reversal = close.iloc[-1] / close.iloc[-22] - 1
        self.assertAlmostEqual(features.iloc[0]["vkospi_oap_mom12m"], expected_mom12)
        self.assertAlmostEqual(
            features.iloc[0]["vkospi_oap_streversal"], expected_short_reversal
        )

    def test_failed_locked_gate_preserves_reference_strategy(self) -> None:
        report = json.loads(
            (experiment.RESULTS / "vkospi_validation.json").read_text(encoding="utf-8")
        )
        if report["selection"]["candidate_promoted"]:
            self.skipTest("The current candidate passed the locked promotion gate")
        selected = pd.read_csv(experiment.RESULTS / "vkospi_selected_backtest.csv")
        reference = pd.read_csv(
            experiment.RESULTS / "openassetpricing_medium_horizon_backtest.csv"
        )
        common = [column for column in selected if column in reference and column != "month"]
        np.testing.assert_allclose(
            selected[common].to_numpy(dtype=float),
            reference[common].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
