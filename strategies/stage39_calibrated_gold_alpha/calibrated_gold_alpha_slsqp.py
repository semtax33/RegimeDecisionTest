from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.stage38_gold_state_alpha import (
    gold_state_alpha_slsqp as stage38,
)


stage36 = stage38.stage36
stage35 = stage38.stage35
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ASSETS = stage38.ASSETS
FULL_START = stage38.FULL_START
COMMON_START = stage38.COMMON_START
RESEARCH_END = stage38.RESEARCH_END
MIN_CAUSAL_MONTHS = stage38.MIN_CAUSAL_MONTHS
WEIGHT_COLUMNS = [f"w_{asset}" for asset in ASSETS]
SIGNIFICANCE_LEVEL = 0.10

SOURCE_FILES = stage38.SOURCE_FILES
FROZEN_FILES = (
    Path(stage36.__file__),
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
    stage36.OUTPUT_DIR / "validation_report.json",
    Path(stage38.__file__),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "last_write_time_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    }


def source_manifest() -> dict[str, dict[str, Any]]:
    return _manifest(SOURCE_FILES)


def frozen_manifest() -> dict[str, dict[str, Any]]:
    return _manifest(FROZEN_FILES)


def _causal_zscore(series: pd.Series) -> pd.Series:
    prior = series.shift(1)
    mean = prior.expanding(min_periods=MIN_CAUSAL_MONTHS).mean()
    std = prior.expanding(min_periods=MIN_CAUSAL_MONTHS).std(ddof=1)
    return ((series - mean) / std.where(std > 0.0)).replace(
        [np.inf, -np.inf], np.nan
    )


def build_calibrated_gold_signals(
    inputs: dict[str, pd.Series],
    target_months: pd.PeriodIndex,
    gold_returns: pd.Series,
) -> pd.DataFrame:
    """Build the fixed composite state and map it with a causal positive slope."""

    signals = stage38.build_monthly_gold_state_signals(
        inputs, target_months
    )
    raw_ranks: dict[str, pd.Series] = {}
    for column in (
        "real_yield_proxy_pct",
        "real_yield_change_3m_pctpt",
        "fx_momentum_63d",
        "gold_trend_252d",
    ):
        raw_ranks[column] = stage35.causal_expanding_midrank(
            signals[column]
        )
    signals["raw_real_yield_support"] = 0.5 * (
        1.0 - raw_ranks["real_yield_proxy_pct"]
        + 1.0 - raw_ranks["real_yield_change_3m_pctpt"]
    )
    signals["raw_fx_support"] = raw_ranks["fx_momentum_63d"]
    signals["raw_gold_trend_support"] = raw_ranks["gold_trend_252d"]
    signals["raw_gold_composite_state"] = signals[
        [
            "raw_real_yield_support",
            "raw_fx_support",
            "raw_gold_trend_support",
        ]
    ].mean(axis=1, skipna=False)
    signals["raw_gold_composite_z"] = _causal_zscore(
        signals["raw_gold_composite_state"]
    )

    rows: list[dict[str, Any]] = []
    for month in signals.index:
        history = signals.index[signals.index < month].intersection(
            gold_returns.index
        )
        feature = signals.loc[
            history, "raw_gold_composite_state"
        ]
        target = gold_returns.loc[history]
        complete = pd.concat([feature, target], axis=1).dropna()
        slope = stage35._nonnegative_univariate_slope(feature, target)
        current_z = float(signals.loc[month, "raw_gold_composite_z"])
        active = bool(
            signals.loc[month, "gold_composite_active"]
            and len(complete) >= MIN_CAUSAL_MONTHS
            and np.isfinite(current_z)
        )
        rows.append(
            {
                "target_month": month,
                "gold_calibration_observations": int(len(complete)),
                "gold_calibration_slope": float(slope),
                "raw_gold_composite_z_current": current_z,
                "calibrated_gold_alpha_active": active,
                "calibrated_gold_mu_adjustment": (
                    float(slope * current_z) if active else 0.0
                ),
            }
        )
    calibrated = pd.DataFrame(rows).set_index("target_month")
    calibrated.index = pd.PeriodIndex(calibrated.index, freq="M")
    return signals.join(calibrated).replace([np.inf, -np.inf], np.nan)


def _performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return stage38._performance_table(paths)


def _bootstrap_table(
    baseline: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return stage38._bootstrap_table(baseline, candidates)


def gate_decision(
    performance: pd.DataFrame,
    regressions: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    perf = performance.set_index(["Strategy", "Period"])
    base_name = "Stage36_Frozen"
    test_name = "Stage39_CalibratedGoldAlpha"
    full_base = perf.loc[(base_name, "full_2007_2026")]
    full_test = perf.loc[(test_name, "full_2007_2026")]
    common_base = perf.loc[(base_name, "common_2010_2026")]
    common_test = perf.loc[(test_name, "common_2010_2026")]
    locked_base = perf.loc[(base_name, "locked_2018_2026")]
    locked_test = perf.loc[(test_name, "locked_2018_2026")]
    performance_gates = {
        "full_cagr_higher": bool(full_test["CAGR"] > full_base["CAGR"]),
        "full_sharpe_higher": bool(
            full_test["Sharpe"] > full_base["Sharpe"]
        ),
        "full_mdd_not_worse": bool(full_test["MDD"] >= full_base["MDD"]),
        "common_cagr_higher": bool(
            common_test["CAGR"] > common_base["CAGR"]
        ),
        "common_sharpe_higher": bool(
            common_test["Sharpe"] > common_base["Sharpe"]
        ),
        "common_mdd_not_worse": bool(
            common_test["MDD"] >= common_base["MDD"]
        ),
        "locked_cagr_not_lower": bool(
            locked_test["CAGR"] >= locked_base["CAGR"]
        ),
    }
    mechanism = regressions.loc[
        regressions["Feature"].eq("GoldCompositeState")
        & regressions["Period"].eq("common_2010_2026")
        & regressions["Model"].eq("FullControls")
        & regressions["HorizonMonths"].isin([1, 3, 6])
    ]
    mechanism_gates = {
        "all_horizon_betas_positive": bool(
            mechanism["StandardizedBeta"].gt(0.0).all()
        ),
        "at_least_one_hac_p_below_10pct": bool(
            (
                mechanism["StandardizedBeta"].gt(0.0)
                & mechanism["HACPValue"].lt(SIGNIFICANCE_LEVEL)
            ).any()
        ),
    }
    boot = bootstrap.loc[
        bootstrap["Candidate"].eq(test_name)
        & bootstrap["Period"].eq("common_2010_2026")
    ].set_index("Metric")
    bootstrap_gates = {
        "cagr_improvement_probability_at_least_60pct": bool(
            boot.loc["delta_CAGR", "ProbabilityPositive"] >= 0.60
        ),
        "sharpe_improvement_probability_at_least_60pct": bool(
            boot.loc["delta_Sharpe", "ProbabilityPositive"] >= 0.60
        ),
        "mdd_improvement_probability_at_least_50pct": bool(
            boot.loc["delta_MDD", "ProbabilityPositive"] >= 0.50
        ),
    }
    promote = bool(
        all(performance_gates.values())
        and all(mechanism_gates.values())
        and all(bootstrap_gates.values())
    )
    return {
        "performance_gates": performance_gates,
        "mechanism_gates": mechanism_gates,
        "bootstrap_gates": bootstrap_gates,
        "promote": promote,
        "decision": (
            "promote_stage39_calibrated_gold_alpha"
            if promote
            else "retain_stage36_and_close_gold_alpha_branch"
        ),
        "promoted_strategy": test_name if promote else base_name,
    }


def _plot_performance(performance: pd.DataFrame, path: Path) -> None:
    strategies = ["Stage36_Frozen", "Stage39_CalibratedGoldAlpha"]
    labels = ["Stage36", "Stage39"]
    metrics = [
        ("CAGR", 100.0, "CAGR (%)"),
        ("Sharpe", 1.0, "Sharpe"),
        ("MDD", 100.0, "MDD (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    for row, period in enumerate(("full_2007_2026", "common_2010_2026")):
        view = performance.loc[
            performance["Period"].eq(period)
        ].set_index("Strategy")
        for column, (metric, scale, title) in enumerate(metrics):
            values = [
                float(view.loc[name, metric]) * scale
                for name in strategies
            ]
            axes[row, column].bar(
                labels, values, color=["#60788c", "#147869"]
            )
            axes[row, column].axhline(0.0, color="#333", linewidth=0.7)
            axes[row, column].set_title(
                f"{'2007-2026' if row == 0 else '2010-2026'} · {title}"
            )
            axes[row, column].grid(axis="y", alpha=0.22)
    fig.suptitle("Stage39 Causally Calibrated Gold Alpha")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_nav(paths: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    for name, color, width in (
        ("Stage36_Frozen", "#536d82", 2.2),
        ("Stage39_CalibratedGoldAlpha", "#12786a", 2.6),
    ):
        nav = (1.0 + paths[name]["return"].loc[
            FULL_START:RESEARCH_END
        ]).cumprod()
        ax.plot(
            nav.index.to_timestamp(),
            nav,
            label=name,
            color=color,
            linewidth=width,
        )
    ax.set_yscale("log")
    ax.set_title("Stage36 vs Stage39 Net NAV")
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _render_html(
    report: dict[str, Any],
    performance: pd.DataFrame,
    regressions: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    selected = performance.loc[
        performance["Strategy"].isin(
            ["Stage36_Frozen", "Stage39_CalibratedGoldAlpha"]
        ),
        [
            "Strategy",
            "Period",
            "CAGR",
            "Volatility",
            "Sharpe",
            "MDD",
            "AvgTurnover",
            "TotalCost",
        ],
    ].copy()
    for column in ("CAGR", "Volatility", "MDD", "AvgTurnover", "TotalCost"):
        selected[column] = selected[column].map(
            lambda value: f"{float(value) * 100:.3f}%"
        )
    selected["Sharpe"] = selected["Sharpe"].map(
        lambda value: f"{float(value):.4f}"
    )
    mechanism = regressions.loc[
        regressions["Feature"].eq("GoldCompositeState")
        & regressions["Model"].eq("FullControls"),
        [
            "Period",
            "HorizonMonths",
            "Observations",
            "StandardizedBeta",
            "HACPValue",
            "SpearmanIC",
        ],
    ]
    boot = bootstrap.loc[
        bootstrap["Candidate"].eq("Stage39_CalibratedGoldAlpha"),
        [
            "Period",
            "Metric",
            "Mean",
            "P05",
            "P50",
            "P95",
            "ProbabilityPositive",
        ],
    ]
    decision = report["decision"]
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage39 Calibrated Gold Alpha</title>
<style>body{{margin:0;background:#f3f6f8;color:#182534;font:16px/1.72 "Malgun Gothic",sans-serif}}header{{padding:48px 7%;background:linear-gradient(120deg,#17364f,#18786b);color:white}}main{{max-width:1120px;margin:24px auto;padding:0 20px}}section{{background:#fff;border:1px solid #dce3e8;border-radius:14px;padding:27px;margin:18px 0}}h1{{font-size:2.4rem}}h2{{color:#18445f;border-bottom:2px solid #e4ebef;padding-bottom:8px}}table{{border-collapse:collapse;width:100%;font-size:.86rem;display:block;overflow:auto}}th,td{{padding:9px;border-bottom:1px solid #e2e7eb;text-align:right;white-space:nowrap}}th{{background:#edf3f6}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;border:1px solid #dce3e8;border-radius:10px}}.note{{padding:14px;background:#edf7f4;border-left:5px solid #18786b}}.warn{{padding:14px;background:#fff5e5;border-left:5px solid #c98918}}code{{background:#edf1f4;padding:2px 5px}}</style></head><body>
<header><div>RegimeDecisionTest · Stage39</div><h1>인과적으로 보정한 금 복합상태 알파</h1><p>Stage36을 동결하고 실질금리·원달러·금 추세의 동일가중 상태를 expanding 양의 계수로 GLD μ에만 연결</p></header><main>
<section><h2>1. 최종 판정</h2><p><strong>{decision}</strong></p><div class="note">현재 월 이전의 상태와 GLD 수익률만으로 비음 기울기를 재추정한다. 부호·배율·기간 grid는 없다.</div></section>
<section><h2>2. 전략 구조</h2><p>실질금리가 낮고 하락하며, 원화가 약하고, USD 금 추세가 강할수록 상태점수가 높다. 현재 상태는 직전 60개월 이상이 있을 때만 거래되고, 과거 GLD 월수익률에 대한 expanding 단변량 기울기로 월간 μ 단위로 바뀐다.</p><pre>GoldState = mean(RealYieldSupport, FXSupport, GoldTrendSupport)
beta_t = max(expanding slope(GoldState_s, GLDReturn_s), 0), s &lt; t
DeltaMu_GLD,t = beta_t * causal_z(GoldState_t)</pre></section>
<section><h2>3. 성과</h2>{selected.to_html(index=False,escape=True,border=0)}<img src="outputs/performance_comparison.png" alt="performance"></section>
<section><h2>4. NAV</h2><img src="outputs/nav_comparison.png" alt="nav"></section>
<section><h2>5. 메커니즘 회귀</h2>{mechanism.to_html(index=False,escape=True,border=0)}</section>
<section><h2>6. 12개월 블록 부트스트랩</h2>{boot.to_html(index=False,escape=True,border=0)}</section>
<section><h2>7. 인과성과 통제</h2><ul><li>목표 월에는 직전 월말까지의 KTB10Y, 공개 CPI, USDKRW, GLD만 사용</li><li>현재 월 이후 수익률은 calibration에 포함하지 않음</li><li>60개월 전에는 μ 조정 0</li><li>원자료와 Stage36 hash 실행 전후 비교</li><li>long-only, no leverage, 13% vol·−16% CDaR guard 유지</li></ul></section>
<section><h2>8. 파일</h2><p><code>calibrated_gold_alpha_slsqp.py</code>, <code>outputs/stage39_calibratedgoldalpha_monthly.csv</code>, <code>outputs/validation_report.json</code></p></section>
<section><h2>9. 주의</h2><p class="warn">동일한 역사에서 Stage38의 메커니즘을 확인한 뒤 Stage39의 calibration을 설계했으므로 완전한 외부표본은 아니다. 2018 잠금구간과 부트스트랩을 별도로 공개하지만 미래 성과를 보장하지 않는다.</p></section>
</main></body></html>"""


def run_research(save: bool = True) -> dict[str, Any]:
    source_before = source_manifest()
    frozen_before = frozen_manifest()

    inputs, data_audit = stage38.load_gold_state_inputs()
    returns, _ = stage35.load_monthly_asset_returns(False)
    probabilities, _ = stage35.build_macro_probabilities(returns)
    stress = stage35.build_monthly_stress_signals(
        returns.index, stage35.build_daily_stress_features()
    )
    market, market_audit = stage35.stage20.load_daily_asset_ohlcv()
    gold_signals = build_calibrated_gold_signals(
        inputs, returns.index, returns["GLD"]
    )

    raw_fundamental, _ = stage35.load_fundamental_daily()
    fundamental = stage35.build_monthly_fundamental_signals(
        raw_fundamental
    )
    equity_close = market["KODEX200"]["close"].dropna()
    equity_monthly_close = equity_close.groupby(
        equity_close.index.to_period("M")
    ).last()
    calibrated_fundamental = stage35.add_causal_return_calibration(
        fundamental, equity_monthly_close.pct_change()
    )
    technical = stage35.stage34._load_period_csv(
        stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
        "target_month",
    )
    asset_vol_daily, _ = stage36.load_asset_implied_volatility_daily()
    asset_vol_signals = stage36.build_monthly_asset_volatility_signals(
        asset_vol_daily, returns.index
    )

    research = stage38.build_gold_research_frame(
        gold_signals,
        returns,
        probabilities,
        stress,
        market["GLD"]["close"].dropna(),
    )
    regressions = stage38.gold_predictive_regressions(research)
    candidate_paths = {
        "Stage39_NoChangeReproduction": stage38.run_backtest(
            returns,
            probabilities,
            stress,
            technical,
            calibrated_fundamental,
            asset_vol_signals,
            gold_signals,
            "baseline_reproduction",
        ),
        "Stage39_CalibratedGoldAlpha": stage38.run_backtest(
            returns,
            probabilities,
            stress,
            technical,
            calibrated_fundamental,
            asset_vol_signals,
            gold_signals,
            "calibrated_composite_mu",
        ),
    }
    stage36_path = stage35.stage34._load_period_csv(
        stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
        "month",
    )
    paths = {"Stage36_Frozen": stage36_path, **candidate_paths}
    performance = _performance_table(paths)
    bootstrap = _bootstrap_table(stage36_path, candidate_paths)
    gates = gate_decision(performance, regressions, bootstrap)

    reproduction = candidate_paths["Stage39_NoChangeReproduction"]
    common = stage36_path.index.intersection(reproduction.index)
    max_return_error = float(
        (
            stage36_path.loc[common, "return"]
            - reproduction.loc[common, "return"]
        )
        .abs()
        .max()
    )
    max_weight_error = float(
        (
            stage36_path.loc[common, WEIGHT_COLUMNS]
            - reproduction.loc[common, WEIGHT_COLUMNS]
        )
        .abs()
        .to_numpy()
        .max()
    )
    active = gold_signals.loc[
        gold_signals["calibrated_gold_alpha_active"]
    ]

    source_after = source_manifest()
    frozen_after = frozen_manifest()
    checks = {
        "source_files_unchanged": source_before == source_after,
        "frozen_files_unchanged": frozen_before == frozen_after,
        "signal_month_precedes_target": bool(
            (
                gold_signals["gold_state_signal_month"]
                < gold_signals.index
            ).all()
        ),
        "minimum_60_calibration_observations": bool(
            active["gold_calibration_observations"].min()
            >= MIN_CAUSAL_MONTHS
        ),
        "preactivation_mu_is_zero": bool(
            gold_signals.loc[
                ~gold_signals["calibrated_gold_alpha_active"],
                "calibrated_gold_mu_adjustment",
            ].eq(0.0).all()
        ),
        "nonnegative_expanding_slope": bool(
            gold_signals["gold_calibration_slope"].ge(0.0).all()
        ),
        "no_change_reproduces_stage36_returns": bool(
            max_return_error < 5e-7
        ),
        "no_change_reproduces_stage36_weights": bool(
            max_weight_error < 5e-6
        ),
        "no_leverage_long_only_sum_to_one": bool(
            all(
                np.allclose(path[WEIGHT_COLUMNS].sum(axis=1), 1.0)
                and (path[WEIGHT_COLUMNS] >= -1e-10).all().all()
                and (path[WEIGHT_COLUMNS] <= 1.0 + 1e-10).all().all()
                for path in candidate_paths.values()
            )
        ),
        "all_candidate_solvers_feasible": bool(
            all(
                path["solver_success"].all()
                and not path["used_fallback"].any()
                and path["volatility_slack"].min() >= -1e-7
                and path["cdar_slack"].min() >= -1e-7
                for path in candidate_paths.values()
            )
        ),
    }

    report = {
        "study": "Stage39_CausallyCalibratedGoldAlpha",
        "decision": gates["decision"],
        "promoted_strategy": gates["promoted_strategy"],
        "scope": (
            "Stage36 is frozen. The Stage38 economic composite is scaled "
            "only by a non-negative expanding GLD-return slope."
        ),
        "fixed_design": {
            "state_definition": (
                "equal real-yield, USDKRW, and USD gold-trend support"
            ),
            "calibration": (
                "non-negative univariate expanding slope to next-month "
                "KRW GLD return"
            ),
            "minimum_causal_months": MIN_CAUSAL_MONTHS,
            "mu_mapping": "slope_t times causal current state z-score",
            "searched_parameters": None,
            "selection_disclosure": (
                "Stage39 follows Stage38 mechanism evidence in the same "
                "historical dataset; not a pristine external holdout"
            ),
        },
        "data_audit": data_audit,
        "activation_audit": {
            "first_active_target": (
                str(active.index.min()) if not active.empty else None
            ),
            "neutral_backtest_months": int(
                (
                    ~gold_signals.loc[
                        FULL_START:RESEARCH_END,
                        "calibrated_gold_alpha_active",
                    ]
                ).sum()
            ),
        },
        "gate_results": gates,
        "performance_comparison": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "gold_predictive_regressions": json.loads(
            regressions.to_json(orient="records", force_ascii=False)
        ),
        "bootstrap_vs_stage36": json.loads(
            bootstrap.to_json(orient="records", force_ascii=False)
        ),
        "reproduction_audit": {
            "months": int(len(common)),
            "max_absolute_return_error": max_return_error,
            "max_absolute_weight_error": max_weight_error,
        },
        "solver_audit": {
            name: stage35.solver_summary(path)
            for name, path in candidate_paths.items()
        },
        "checks": checks,
        "source_manifest_before": source_before,
        "source_manifest_after": source_after,
        "frozen_manifest_before": frozen_before,
        "frozen_manifest_after": frozen_after,
        "market_audit": market_audit,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gold_signals.to_csv(
            OUTPUT_DIR / "monthly_calibrated_gold_signals.csv"
        )
        research.to_csv(
            OUTPUT_DIR / "monthly_calibrated_gold_research_frame.csv"
        )
        regressions.to_csv(
            OUTPUT_DIR / "gold_predictive_regressions.csv", index=False
        )
        performance.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage36.csv",
            index=False,
        )
        for name, candidate_path in candidate_paths.items():
            candidate_path.to_csv(
                OUTPUT_DIR / f"{name.lower()}_monthly.csv"
            )
        _plot_performance(
            performance, OUTPUT_DIR / "performance_comparison.png"
        )
        _plot_nav(paths, OUTPUT_DIR / "nav_comparison.png")
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (Path(__file__).resolve().parent / "stage39_gold_alpha_report.html").write_text(
            _render_html(report, performance, regressions, bootstrap),
            encoding="utf-8",
        )
    return report


def main() -> None:
    print(json.dumps(run_research(save=True), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
