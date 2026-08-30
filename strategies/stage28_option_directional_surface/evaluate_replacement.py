from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STAGE29_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "stage29_option_two_axis_confirmation"
    / "outputs"
)
BLOCK_MONTHS = 12
BOOTSTRAP_REPLICATIONS = 2_000
BOOTSTRAP_SEED = 28_2026


PATHS = {
    "Stage20_VIX6Decomposition": OUTPUT_DIR / "stage20_vix6_monthly.csv",
    "Stage28_ODS_Difference": OUTPUT_DIR / "option_directional_surface_monthly.csv",
    "Stage29_OptionTwoAxisConfirmation": (
        STAGE29_OUTPUT_DIR / "option_two_axis_monthly.csv"
    ),
}


def _metrics(returns: np.ndarray) -> dict[str, float]:
    values = np.asarray(returns, dtype=float)
    nav = np.cumprod(1.0 + values)
    years = len(values) / 12.0
    volatility = float(np.std(values, ddof=1) * math.sqrt(12.0))
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    return {
        "CAGR": float(nav[-1] ** (1.0 / years) - 1.0),
        "Volatility": volatility,
        "Sharpe": float(
            np.mean(values) / np.std(values, ddof=1) * math.sqrt(12.0)
        ),
        "MDD": float(drawdown.min()),
    }


def _load() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(path, parse_dates=["month"]).set_index("month")
        for name, path in PATHS.items()
    }


def _period_comparison(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    periods = {
        "full_2007_2026": (pd.Timestamp("2007-04-01"), pd.Timestamp("2026-07-01")),
        "early_2007_2017": (pd.Timestamp("2007-04-01"), pd.Timestamp("2017-12-01")),
        "locked_2018_2026": (pd.Timestamp("2018-01-01"), pd.Timestamp("2026-07-01")),
    }
    rows: list[dict[str, Any]] = []
    for period, (start, end) in periods.items():
        baseline = _metrics(
            paths["Stage20_VIX6Decomposition"].loc[start:end, "return"].to_numpy()
        )
        for name, path in paths.items():
            view = path.loc[start:end]
            metric = _metrics(view["return"].to_numpy())
            rows.append(
                {
                    "Strategy": name,
                    "Period": period,
                    "Start": view.index.min().strftime("%Y-%m"),
                    "End": view.index.max().strftime("%Y-%m"),
                    "Months": len(view),
                    **metric,
                    "DeltaCAGR": metric["CAGR"] - baseline["CAGR"],
                    "DeltaSharpe": metric["Sharpe"] - baseline["Sharpe"],
                    "DeltaMDD": metric["MDD"] - baseline["MDD"],
                    "ParetoVsStage20": bool(
                        metric["CAGR"] >= baseline["CAGR"]
                        and metric["Sharpe"] >= baseline["Sharpe"]
                        and metric["MDD"] >= baseline["MDD"]
                        and name != "Stage20_VIX6Decomposition"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_bootstrap(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline = paths["Stage20_VIX6Decomposition"]["return"].to_numpy()
    n = len(baseline)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = math.ceil(n / BLOCK_MONTHS)
    samples: list[np.ndarray] = []
    for _ in range(BOOTSTRAP_REPLICATIONS):
        starts = rng.integers(0, n - BLOCK_MONTHS + 1, size=blocks)
        samples.append(
            np.concatenate(
                [np.arange(start, start + BLOCK_MONTHS) for start in starts]
            )[:n]
        )
    baseline_point = _metrics(baseline)
    rows: list[dict[str, Any]] = []
    for name, path in paths.items():
        if name == "Stage20_VIX6Decomposition":
            continue
        candidate = path["return"].to_numpy()
        point = _metrics(candidate)
        differences = {key: [] for key in ["CAGR", "Sharpe", "MDD"]}
        for indices in samples:
            base_metric = _metrics(baseline[indices])
            candidate_metric = _metrics(candidate[indices])
            for key in differences:
                differences[key].append(candidate_metric[key] - base_metric[key])
        for key, values in differences.items():
            array = np.asarray(values)
            rows.append(
                {
                    "Strategy": name,
                    "Metric": key,
                    "PointDelta": point[key] - baseline_point[key],
                    "BootstrapMedianDelta": float(np.median(array)),
                    "BootstrapP05": float(np.quantile(array, 0.05)),
                    "BootstrapP95": float(np.quantile(array, 0.95)),
                    "ProbabilityDeltaPositive": float(np.mean(array > 0.0)),
                    "BlockMonths": BLOCK_MONTHS,
                    "Replications": BOOTSTRAP_REPLICATIONS,
                }
            )
    return pd.DataFrame(rows)


def run_evaluation(save: bool = True) -> dict[str, Any]:
    paths = _load()
    comparison = _period_comparison(paths)
    bootstrap = _paired_bootstrap(paths)
    full = comparison.loc[comparison["Period"] == "full_2007_2026"]
    report = {
        "decision": "retain_stage20",
        "reason": (
            "No option-direction replacement Pareto-improves full-period CAGR, "
            "Sharpe and MDD. Stage28 is strong only in the locked 2018+ period."
        ),
        "full_period_pareto_replacements": full.loc[
            full["ParetoVsStage20"], "Strategy"
        ].tolist(),
        "stage28_is_regime_unstable": True,
        "stage28_early_period_cagr": float(
            comparison.set_index(["Strategy", "Period"]).loc[
                ("Stage28_ODS_Difference", "early_2007_2017"), "CAGR"
            ]
        ),
        "stage28_locked_period_cagr": float(
            comparison.set_index(["Strategy", "Period"]).loc[
                ("Stage28_ODS_Difference", "locked_2018_2026"), "CAGR"
            ]
        ),
        "additional_feedback_assessment": {
            "signed_order_flow_available": False,
            "reason": (
                "The supplied KRX daily file has aggregate volume/value but no "
                "bid/ask quote or aggressor-side transaction field."
            ),
            "abnormal_surface_residual_implemented": False,
            "reason_not_added_post_hoc": (
                "It would introduce a newly trained residual model after seeing "
                "Stage28 performance and belongs in a separately preregistered study."
            ),
            "vix6_should_remain_risk_engine_in_next_study": True,
        },
        "bootstrap_is_descriptive_not_post_selection_significance": True,
    }
    if save:
        comparison.to_csv(
            OUTPUT_DIR / "replacement_performance_by_subperiod.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "replacement_paired_block_bootstrap.csv", index=False
        )
        (OUTPUT_DIR / "replacement_decision.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"comparison": comparison, "bootstrap": bootstrap, "report": report}


def main() -> None:
    result = run_evaluation(save=True)
    print(result["comparison"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
