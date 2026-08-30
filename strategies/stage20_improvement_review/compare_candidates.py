from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategies.core.regime_research import ASSETS, load_monthly_asset_returns


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_ROOT = ROOT / "strategies"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGET_CAGR = 0.10
TARGET_SHARPE = 1.0
BOOTSTRAP_BLOCK_MONTHS = 12
BOOTSTRAP_REPLICATIONS = 2_000
BOOTSTRAP_SEED = 20_260_829

CANDIDATES: dict[str, dict[str, Any]] = {
    "Stage20_DailyTechnicalConfidence": {
        "folder": "stage20_daily_technical_confidence",
        "path": "daily_technical_confidence_monthly.csv",
        "status": "frozen_baseline",
        "change": "none",
    },
    "Stage21_OneSidedConfidence": {
        "folder": "stage21_one_sided_confidence",
        "path": "one_sided_confidence_monthly.csv",
        "status": "predeclared_single_change",
        "change": "one-sided optimistic-view filter",
    },
    "Stage22_KRatioPrimary": {
        "folder": "stage22_k_ratio_primary",
        "path": "k_ratio_primary_monthly.csv",
        "status": "predeclared_single_change",
        "change": "K-Ratio direction with RSI magnitude confirmation",
    },
    "Stage23_RelativeATR": {
        "folder": "stage23_relative_atr",
        "path": "relative_atr_monthly.csv",
        "status": "predeclared_single_change",
        "change": "cross-sectionally normalized ATR covariance",
    },
    "Stage24_EquityKRatioOnly": {
        "folder": "stage24_equity_k_ratio_only",
        "path": "equity_k_ratio_only_monthly.csv",
        "status": "exploratory_single_change",
        "change": "K-Ratio is the sole KODEX200 direction input",
    },
    "Stage25_ConflictOnlyVeto": {
        "folder": "stage25_conflict_only_veto",
        "path": "conflict_only_veto_monthly.csv",
        "status": "exploratory_single_change",
        "change": "conflict-only confidence for all assets",
    },
    "Stage26_EquityConflictVeto": {
        "folder": "stage26_equity_conflict_veto",
        "path": "equity_conflict_veto_monthly.csv",
        "status": "exploratory_single_change",
        "change": "conflict-only confidence for KODEX200",
    },
    "Stage27_KRatioEquityVeto": {
        "folder": "stage27_k_ratio_equity_veto",
        "path": "k_ratio_equity_veto_monthly.csv",
        "status": "exploratory_combination",
        "change": "K-Ratio-only KODEX200 direction plus equity conflict veto",
    },
}


def _load_paths() -> dict[str, pd.DataFrame]:
    paths: dict[str, pd.DataFrame] = {}
    for name, detail in CANDIDATES.items():
        path = (
            STRATEGY_ROOT
            / str(detail["folder"])
            / "outputs"
            / str(detail["path"])
        )
        frame = pd.read_csv(path, parse_dates=["month"]).set_index("month")
        paths[name] = frame
    return paths


def _metrics(returns: np.ndarray) -> dict[str, float]:
    values = np.asarray(returns, dtype=float)
    nav = np.cumprod(1.0 + values)
    years = len(values) / 12.0
    cagr = float(nav[-1] ** (1.0 / years) - 1.0)
    volatility = float(np.std(values, ddof=1) * math.sqrt(12.0))
    sharpe = float(np.mean(values) / np.std(values, ddof=1) * math.sqrt(12.0))
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "MDD": float(drawdown.min()),
    }


def _comparison(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = {
        "full_2007_2026": pd.Timestamp("2007-04-01"),
        "locked_2018_2026": pd.Timestamp("2018-01-01"),
    }
    for period, start in periods.items():
        baseline_view = paths["Stage20_DailyTechnicalConfidence"].loc[start:]
        baseline = _metrics(baseline_view["return"].to_numpy())
        for name, path in paths.items():
            view = path.loc[start:]
            metric = _metrics(view["return"].to_numpy())
            cagr_pass = metric["CAGR"] >= TARGET_CAGR
            sharpe_pass = metric["Sharpe"] >= TARGET_SHARPE
            mdd_pass = metric["MDD"] >= baseline["MDD"] - 1e-12
            pareto = (
                metric["CAGR"] >= baseline["CAGR"] - 1e-12
                and metric["Sharpe"] >= baseline["Sharpe"] - 1e-12
                and metric["MDD"] >= baseline["MDD"] - 1e-12
                and (
                    metric["CAGR"] > baseline["CAGR"] + 1e-12
                    or metric["Sharpe"] > baseline["Sharpe"] + 1e-12
                    or metric["MDD"] > baseline["MDD"] + 1e-12
                )
            )
            rows.append(
                {
                    "Strategy": name,
                    "ResearchStatus": CANDIDATES[name]["status"],
                    "Change": CANDIDATES[name]["change"],
                    "Period": period,
                    "Start": view.index.min().strftime("%Y-%m"),
                    "End": view.index.max().strftime("%Y-%m"),
                    "Months": len(view),
                    **metric,
                    "DeltaCAGR": metric["CAGR"] - baseline["CAGR"],
                    "DeltaSharpe": metric["Sharpe"] - baseline["Sharpe"],
                    "DeltaMDD": metric["MDD"] - baseline["MDD"],
                    "PassCAGR10": cagr_pass,
                    "PassSharpe1": sharpe_pass,
                    "PassStage20MDD": mdd_pass,
                    "PassAllPointTargets": cagr_pass and sharpe_pass and mdd_pass,
                    "ParetoImprovesStage20": pareto,
                }
            )
    return pd.DataFrame(rows)


def _weight_and_attribution(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    asset_returns, _ = load_monthly_asset_returns()
    asset_returns.index = asset_returns.index.to_timestamp()
    rows: list[dict[str, Any]] = []
    for name, path in paths.items():
        common = path.index.intersection(asset_returns.index)
        weights = path.loc[common, [f"w_{asset}" for asset in ASSETS]].copy()
        weights.columns = ASSETS
        realized = asset_returns.loc[common, ASSETS]
        row: dict[str, Any] = {
            "Strategy": name,
            "AverageTurnover": float(path["turnover"].mean()),
            "TotalCost": float(path["trade_cost"].sum() + path["fx_cost"].sum()),
            "VolGuardBindingMonths": int((path["volatility_slack"].abs() < 1e-6).sum()),
            "CDaRGuardBindingMonths": int((path["cdar_slack"].abs() < 1e-6).sum()),
        }
        for asset in ASSETS:
            row[f"AverageWeight_{asset}"] = float(weights[asset].mean())
            row[f"AnnualArithmeticContribution_{asset}"] = float(
                (weights[asset] * realized[asset]).mean() * 12.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _drawdown_windows(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, path in paths.items():
        trough = path["drawdown"].idxmin()
        peak = path.loc[:trough, "nav"].idxmax()
        recovered = path.loc[trough:]
        recovered = recovered.loc[recovered["nav"] >= path.loc[peak, "nav"]]
        recovery = recovered.index.min() if len(recovered) else pd.NaT
        rows.append(
            {
                "Strategy": name,
                "Peak": peak.strftime("%Y-%m"),
                "Trough": trough.strftime("%Y-%m"),
                "Recovery": None if pd.isna(recovery) else recovery.strftime("%Y-%m"),
                "MDD": float(path["drawdown"].min()),
            }
        )
    return pd.DataFrame(rows)


def _paired_block_bootstrap(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline = paths["Stage20_DailyTechnicalConfidence"]["return"].to_numpy()
    n = len(baseline)
    block = BOOTSTRAP_BLOCK_MONTHS
    blocks_needed = math.ceil(n / block)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_indices: list[np.ndarray] = []
    for _ in range(BOOTSTRAP_REPLICATIONS):
        starts = rng.integers(0, n - block + 1, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + block) for start in starts]
        )[:n]
        sampled_indices.append(indices)

    rows: list[dict[str, Any]] = []
    baseline_point = _metrics(baseline)
    for name, path in paths.items():
        if name == "Stage20_DailyTechnicalConfidence":
            continue
        candidate = path["return"].to_numpy()
        point = _metrics(candidate)
        differences = {metric: [] for metric in ["CAGR", "Sharpe", "MDD"]}
        for indices in sampled_indices:
            base_metric = _metrics(baseline[indices])
            candidate_metric = _metrics(candidate[indices])
            for metric in differences:
                differences[metric].append(
                    candidate_metric[metric] - base_metric[metric]
                )
        for metric, values in differences.items():
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "Strategy": name,
                    "Metric": metric,
                    "PointDelta": point[metric] - baseline_point[metric],
                    "BootstrapMedianDelta": float(np.median(array)),
                    "BootstrapP05": float(np.quantile(array, 0.05)),
                    "BootstrapP95": float(np.quantile(array, 0.95)),
                    "ProbabilityDeltaPositive": float(np.mean(array > 0.0)),
                    "BlockMonths": block,
                    "Replications": BOOTSTRAP_REPLICATIONS,
                    "Interpretation": "descriptive; candidates include adaptive follow-ups",
                }
            )
    return pd.DataFrame(rows)


def run_review(save: bool = True) -> dict[str, Any]:
    paths = _load_paths()
    comparison = _comparison(paths)
    attribution = _weight_and_attribution(paths)
    drawdowns = _drawdown_windows(paths)
    bootstrap = _paired_block_bootstrap(paths)
    full = comparison.loc[comparison["Period"] == "full_2007_2026"]
    pass_all = full.loc[full["PassAllPointTargets"], "Strategy"].tolist()
    pareto = full.loc[full["ParetoImprovesStage20"], "Strategy"].tolist()
    report = {
        "baseline": "Stage20_DailyTechnicalConfidence",
        "point_targets": {
            "cagr_at_least": TARGET_CAGR,
            "sharpe_at_least": TARGET_SHARPE,
            "mdd_no_worse_than_stage20": True,
        },
        "candidates_passing_all_point_targets": pass_all,
        "candidates_pareto_improving_stage20": pareto,
        "recommended_deployment_candidate": (
            "Stage24_EquityKRatioOnly" if pareto == ["Stage24_EquityKRatioOnly"] else None
        ),
        "target_achieved": bool(pass_all),
        "selection_warning": (
            "Stage21-23 were predeclared. Stage24-27 are exploratory adaptive "
            "follow-ups and require fresh out-of-sample confirmation."
        ),
        "bootstrap": {
            "method": "paired moving-block bootstrap",
            "block_months": BOOTSTRAP_BLOCK_MONTHS,
            "replications": BOOTSTRAP_REPLICATIONS,
            "seed": BOOTSTRAP_SEED,
            "purpose": "descriptive uncertainty, not post-selection significance",
        },
    }
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(OUTPUT_DIR / "candidate_performance_and_gates.csv", index=False)
        attribution.to_csv(OUTPUT_DIR / "weight_and_return_attribution.csv", index=False)
        drawdowns.to_csv(OUTPUT_DIR / "worst_drawdown_windows.csv", index=False)
        bootstrap.to_csv(OUTPUT_DIR / "paired_block_bootstrap.csv", index=False)
        (OUTPUT_DIR / "review_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "comparison": comparison,
        "attribution": attribution,
        "drawdowns": drawdowns,
        "bootstrap": bootstrap,
        "report": report,
    }


def main() -> None:
    result = run_review(save=True)
    full = result["comparison"].loc[
        result["comparison"]["Period"] == "full_2007_2026"
    ]
    columns = [
        "Strategy",
        "CAGR",
        "Volatility",
        "Sharpe",
        "MDD",
        "DeltaCAGR",
        "DeltaSharpe",
        "DeltaMDD",
        "PassAllPointTargets",
        "ParetoImprovesStage20",
    ]
    print(full[columns].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
