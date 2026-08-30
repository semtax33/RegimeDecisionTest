from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage20_daily_technical_confidence import (
    daily_technical_confidence_slsqp as strategy,
)


def test_k_ratio_matches_manual_daily_log_price_regression() -> None:
    rng = np.random.default_rng(7)
    log_price = 4.0 + 0.001 * np.arange(strategy.K_RATIO_DAYS)
    log_price += rng.normal(0.0, 0.01, strategy.K_RATIO_DAYS)
    close = pd.Series(np.exp(log_price), index=pd.date_range("2024-01-01", periods=strategy.K_RATIO_DAYS))
    calculated = strategy.rolling_k_ratio(close).iloc[-1]

    x = np.arange(strategy.K_RATIO_DAYS, dtype=float)
    centered_x = x - x.mean()
    centered_y = log_price - log_price.mean()
    slope = float(centered_x @ centered_y / np.square(centered_x).sum())
    residual = centered_y - slope * centered_x
    variance = np.square(residual).sum() / (strategy.K_RATIO_DAYS - 2)
    standard_error = math.sqrt(variance / np.square(centered_x).sum())
    expected = slope / (standard_error * math.sqrt(strategy.K_RATIO_DAYS))
    assert calculated == pytest.approx(expected)
    assert strategy.rolling_k_ratio(close).iloc[:-1].isna().all()


def test_atr_uses_true_range_and_wilder_seed() -> None:
    index = pd.date_range("2024-01-01", periods=16)
    close = pd.Series(np.arange(100.0, 116.0), index=index)
    frame = pd.DataFrame(
        {"high": close + 2.0, "low": close - 1.0, "close": close},
        index=index,
    )
    atr = strategy.average_true_range(frame, period=14)
    assert atr.iloc[:14].isna().all()
    assert atr.iloc[14] == pytest.approx(3.0)
    assert atr.iloc[15] == pytest.approx(3.0)


def test_price_and_volume_relative_strength_have_expected_direction() -> None:
    index = pd.date_range("2024-01-01", periods=20)
    rising = pd.Series(np.arange(100.0, 120.0), index=index)
    falling = pd.Series(np.arange(120.0, 100.0, -1.0), index=index)
    volume = pd.Series(np.arange(1_000.0, 1_020.0), index=index)
    assert strategy.price_rsi(rising).iloc[-1] > 99.0
    assert strategy.price_rsi(falling).iloc[-1] < 1.0
    assert strategy.volume_rsi(rising, volume).iloc[-1] > 99.0
    assert strategy.volume_rsi(falling, volume).iloc[-1] < 1.0


def test_daily_data_cover_all_assets_before_backtest() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    audit = report["data_audit"]
    assert set(audit["assets"]) == set(ASSETS)
    assert all(
        pd.Timestamp(detail["feature_start"]) < pd.Timestamp("2007-04-01")
        for detail in audit["assets"].values()
    )
    assert audit["foreign_prices_converted_to_krw"]
    assert audit["bond_atr_uses_close_to_close_proxy"]
    assert audit["kodex_volume_segments_normalized_separately"]
    assert audit["k_ratio_days"] == 126
    assert audit["wilder_days"] == 14


@pytest.mark.parametrize("asset", ASSETS)
def test_saved_daily_files_contain_k_ratio_and_atr(asset: str) -> None:
    frame = pd.read_csv(
        strategy.OUTPUT_DIR / f"daily_technical_features_{asset}.csv"
    )
    assert {"k_ratio", "k_score", "atr", "natr", "atr_percentile"}.issubset(
        frame.columns
    )
    valid = frame.dropna(subset=["k_ratio", "atr_percentile"])
    assert len(valid) >= 5_000
    assert valid["k_score"].between(-1.0, 1.0).all()
    assert valid["atr_percentile"].between(0.0, 1.0).all()


def test_equity_daily_file_contains_price_and_volume_strength() -> None:
    frame = pd.read_csv(
        strategy.OUTPUT_DIR / "daily_technical_features_KODEX200.csv"
    )
    required = {
        "price_rsi",
        "volume_rsi",
        "price_strength",
        "volume_strength",
        "technical_direction",
    }
    assert required.issubset(frame.columns)
    valid = frame.dropna(subset=list(required))
    assert valid["price_rsi"].between(0.0, 100.0).all()
    assert valid["volume_rsi"].between(0.0, 100.0).all()


def test_every_technical_signal_precedes_investment_month() -> None:
    path = pd.read_csv(
        strategy.OUTPUT_DIR / "daily_technical_confidence_monthly.csv",
        index_col=0,
    )
    path.index = pd.PeriodIndex(path.index, freq="M")
    assert strategy.verify_technical_signal_dates(path)
    assert (
        pd.PeriodIndex(path["macro_signal_month"], freq="M") < path.index
    ).all()
    assert (
        pd.PeriodIndex(path["stress_signal_month"], freq="M") < path.index
    ).all()


def test_monthly_signals_equal_last_known_daily_observations() -> None:
    monthly = pd.read_csv(
        strategy.OUTPUT_DIR / "monthly_technical_signals.csv",
        index_col=0,
    )
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    daily = {}
    for asset in ASSETS:
        frame = pd.read_csv(
            strategy.OUTPUT_DIR / f"daily_technical_features_{asset}.csv",
            index_col=0,
            parse_dates=True,
        )
        daily[asset] = frame
    assert strategy.verify_monthly_signals_match_daily_features(monthly, daily)


def test_technical_inputs_change_mu_and_covariance_as_declared() -> None:
    signal = pd.Series(
        {
            **{f"technical_direction_{asset}": 1.0 for asset in ASSETS},
            **{f"atr_percentile_{asset}": 0.5 for asset in ASSETS},
        }
    )
    macro = np.array([0.02, 0.01, 0.03, 0.00])
    covariance = np.eye(4)
    detail = strategy.apply_technical_inputs(macro, covariance, signal)
    assert detail["macro_neutral_return"] == pytest.approx(0.015)
    assert np.allclose(detail["macro_confidence"], [1.0, 0.0, 1.0, 0.0])
    assert np.allclose(
        detail["filtered_macro_expected_return"], [0.02, 0.015, 0.03, 0.015]
    )
    assert np.allclose(detail["atr_variance_scale"], 1.5)
    assert np.allclose(detail["adjusted_covariance"], np.eye(4) * 1.5)


def test_saved_path_is_long_only_unlevered_and_solver_clean() -> None:
    path = pd.read_csv(
        strategy.OUTPUT_DIR / "daily_technical_confidence_monthly.csv"
    )
    weights = path[[f"w_{asset}" for asset in ASSETS]]
    assert len(path) == 232
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-9).all().all()
    assert (weights <= 1.0 + 1e-9).all().all()
    assert np.allclose(path["downside_risk_aversion_lambda"], 1.0)
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert (path["volatility_slack"] >= -1e-8).all()
    assert (path["cdar_slack"] >= -1e-8).all()


def test_full_and_locked_performance_are_reproduced() -> None:
    comparison = pd.read_csv(strategy.OUTPUT_DIR / "performance_comparison.csv")
    indexed = comparison.set_index(["Strategy", "Period"])
    full = indexed.loc[("Stage20_DailyTechnicalConfidence", "full_2007_2026")]
    locked = indexed.loc[
        ("Stage20_DailyTechnicalConfidence", "locked_2018_2026")
    ]
    assert full["CAGR"] == pytest.approx(0.0939740, abs=1e-6)
    assert full["Sharpe"] == pytest.approx(0.9865029, abs=1e-6)
    assert full["MDD"] == pytest.approx(-0.1401481, abs=1e-6)
    assert locked["CAGR"] == pytest.approx(0.1293925, abs=1e-6)
    assert locked["Sharpe"] == pytest.approx(1.2502705, abs=1e-6)
    assert locked["MDD"] == pytest.approx(-0.1141902, abs=1e-6)


def test_report_records_risk_return_tradeoff_and_no_search() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    changes = report["full_period_changes"]
    assert changes["cagr"] < 0.0
    assert changes["sharpe"] > 0.13
    assert changes["mdd"] > 0.09
    assert report["feature_policy"]["searched_parameters"] is None
    assert report["feature_policy"]["candidate_count"] == 1
    assert report["technical_concentration"]["months_above_50_percent"] == 34
    assert report["technical_concentration"]["months_above_90_percent"] == 0
    assert all(report["checks"].values())


def test_strategy_does_not_use_simple_monthly_momentum_or_post_overlay() -> None:
    source = inspect.getsource(strategy.solve_weights)
    assert "expected_return = filtered_macro + stress_adjustment" in source
    assert "covariance = np.asarray(technical[\"adjusted_covariance\"]" in source
    assert "momentum" not in source.lower()
    assert "post_optimizer" not in source.lower()
