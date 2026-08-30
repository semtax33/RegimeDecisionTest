from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class VKOSPIExtendedDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.macro = pd.read_csv(RESULTS / "vkospi_macro_constant_sensitivity.csv")
        cls.features = pd.read_csv(RESULTS / "vkospi_tail_feature_diagnostics.csv")
        cls.prediction = pd.read_csv(RESULTS / "vkospi_tail_prediction_diagnostics.csv")
        cls.performance = pd.read_csv(RESULTS / "vkospi_extended_period_performance.csv")
        cls.overfit = json.loads(
            (RESULTS / "vkospi_overfitting_diagnostics.json").read_text(encoding="utf-8")
        )

    def test_tail_refit_reproduces_the_deployed_probabilities(self) -> None:
        reproduction = self.overfit["tail_probability_reproduction"]
        self.assertEqual(reproduction["observations"], 194)
        self.assertLess(reproduction["max_absolute_probability_difference"], 1e-12)

    def test_all_sixteen_non_vkospi_tail_inputs_have_two_period_diagnostics(self) -> None:
        self.assertEqual(self.features["feature"].nunique(), 16)
        self.assertEqual(set(self.features.groupby("feature").size()), {2})
        self.assertEqual(
            set(self.features["period"]),
            {"calibration_through_2017", "locked_2018_2026"},
        )

    def test_auc_and_brier_are_reported_without_hiding_bad_calibration(self) -> None:
        locked = self.prediction.set_index("period").loc["locked_2018_2026"]
        calibration = self.prediction.set_index("period").loc[
            "calibration_through_2017"
        ]
        self.assertGreater(locked["roc_auc"], 0.70)
        self.assertLess(calibration["roc_auc"], 0.50)
        for row in (calibration, locked):
            self.assertGreater(
                row["brier_score"], row["calibration_prevalence_brier"]
            )

    def test_requested_2005_strategy_period_is_marked_unavailable(self) -> None:
        requested = self.performance.loc[
            self.performance["period"] == "requested_2005_01_2026_07"
        ].set_index("strategy")
        for strategy in (
            "ReferenceMediumHorizonOAPVol15",
            "ExistingVKOSPIDynamic",
            "RobustVKOSPIDynamic",
        ):
            self.assertEqual(requested.loc[strategy, "status"], "unavailable_same_strategy")
        self.assertEqual(
            requested.loc["KODEX200ProxyBenchmark", "status"],
            "measured_benchmark_only",
        )

        available = self.performance.loc[
            (self.performance["period"] == "available_full_2007_04_2026_07")
            & (self.performance["strategy"] == "RobustVKOSPIDynamic")
        ].iloc[0]
        self.assertEqual(available["start"], "2007-04")
        self.assertEqual(int(available["Months"]), 232)

    def test_overfitting_audit_is_explicit_and_conservative(self) -> None:
        self.assertEqual(self.overfit["candidate_count"], 810)
        self.assertEqual(self.overfit["strict_pass_count"], 2)
        self.assertGreater(self.overfit["winner_runner_score_gap"], 0)
        self.assertLess(self.overfit["winner_runner_score_gap"], 0.01)
        self.assertEqual(self.overfit["locked_years_robust_outperformed"], 6)
        self.assertEqual(self.overfit["locked_years_total"], 9)
        self.assertIn("cannot be ruled out", self.overfit["conclusion"])

    def test_macro_sensitivity_records_that_deployed_constants_are_not_optimal(self) -> None:
        locked = self.macro.loc[self.macro["period"] == "locked_2018_2026"]
        deployed = locked.loc[locked["parameter"] == "deployed"].iloc[0]
        no_sjm = locked.loc[
            (locked["parameter"] == "sjm_weight") & (locked["value"] == 0.0)
        ].iloc[0]
        self.assertLess(no_sjm["mean_brier"], deployed["mean_brier"])
        self.assertGreater(no_sjm["quadrant_accuracy"], deployed["quadrant_accuracy"])

    def test_html_notebook_and_bundle_include_the_new_audit(self) -> None:
        html = (ROOT / "artifacts/reports/vkospi_robust_dynamic_technical_report.html").read_text(
            encoding="utf-8"
        )
        for required in (
            "0.55·0.20·0.10·0.85·0.15",
            "p_up_t = 0.50 − 0.15 × severity_t",
            "AUC와 Brier는 입력변수가 아니라 검증지표다",
            "여기서 꼬리손실은 로지스틱 회귀의 손실함수 이름이 아닙니다",
            "왜 새 분류기를 고르지 않고 기존 파이프라인을 재사용했나",
            "그대로 가져온 부분과 현재 목표에 맞게 바꾼 부분",
            "재사용은 최적성의 증거가 아니라 비교를 통제한 설계 선택입니다",
            "2005~2026을 요청했지만 동일 전략은 2007-04부터다",
            "오버피팅을 배제할 수 없다",
            "5 structures × 3 levels × 3 shocks × 3 transfer caps × 2 defensive splits × 3 bands = 810",
        ):
            self.assertIn(required, html)
        for placeholder in (
            "__EXTENDED_PERFORMANCE__",
            "__TAIL_PREDICTION__",
            "__MACRO_SENSITIVITY__",
            "__GRID_NEIGHBORHOOD__",
        ):
            self.assertNotIn(placeholder, html)

        notebook = json.loads(
            (ROOT / "artifacts/notebooks/vkospi_robust_dynamic_strategy_colab.ipynb").read_text(
                encoding="utf-8"
            )
        )
        notebook_source = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("vkospi_extended_diagnostics.py", notebook_source)
        self.assertIn("vkospi_tail_prediction_diagnostics.csv", notebook_source)

        with zipfile.ZipFile(ROOT / "artifacts/bundles/vkospi_robust_dynamic_colab_bundle.zip") as archive:
            names = set(archive.namelist())
        for member in (
            "RegimeDecisionTest/vkospi_extended_diagnostics.py",
            "RegimeDecisionTest/regime_research.py",
            "RegimeDecisionTest/results/vkospi_tail_prediction_diagnostics.csv",
            "RegimeDecisionTest/results/vkospi_overfitting_diagnostics.json",
        ):
            self.assertIn(member, names)


if __name__ == "__main__":
    unittest.main()
