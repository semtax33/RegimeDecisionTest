from __future__ import annotations

import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from strategies.core.regime_research import (
    ASSETS,
    SparseJump2,
    StrategyConfig,
    hard_regime_weights,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)
from strategies.stage06_vkospi.vkospi_extended_diagnostics import (
    DOMESTIC_FEATURES,
    OAP_COMPOSITES,
    TAIL_FEATURES,
    _ece,
    _macro_components,
    _macro_probabilities,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CAL_END = pd.Period("2017-12", freq="M")
VALIDATION_START = pd.Period("2013-01", freq="M")
TEST_START = pd.Period("2018-01", freq="M")
LOCKED_LATE_START = pd.Period("2022-01", freq="M")
RNG_SEED = 20260828
REPORT_SCHEMA_VERSION = 2

LOGISTIC_LONG_PATH = RESULTS / "vkospi_logistic_hyperparameter_robustness.csv"
LOGISTIC_SUMMARY_PATH = RESULTS / "vkospi_logistic_candidate_summary.csv"
LOGISTIC_REPORT_PATH = RESULTS / "vkospi_logistic_robustness.json"
SJM_LONG_PATH = RESULTS / "vkospi_sjm_robustness.csv"
SJM_SUMMARY_PATH = RESULTS / "vkospi_sjm_candidate_summary.csv"
SJM_REPORT_PATH = RESULTS / "vkospi_sjm_robustness.json"
SJM_CACHE_PATH = RESULTS / "vkospi_sjm_internal_paths.csv"
REPORT_PATH = RESULTS / "vkospi_model_robustness.json"

PERIODS = (
    ("calibration_2007_2017", None, CAL_END),
    ("validation_2013_2017", VALIDATION_START, CAL_END),
    ("locked_2018_2026", TEST_START, None),
    ("locked_early_2018_2021", TEST_START, pd.Period("2021-12", freq="M")),
    ("locked_late_2022_2026", LOCKED_LATE_START, None),
)


def causal_percentile(values: pd.Series, lookback: int = 60) -> pd.Series:
    """Map a value to its trailing empirical percentile without using the current row."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    history: list[float] = []
    for index, value in values.items():
        if not np.isfinite(value):
            continue
        reference = np.asarray(history[-lookback:], dtype=float)
        if len(reference) >= 12:
            result.loc[index] = (
                float(np.sum(reference <= value)) + 1.0
            ) / (len(reference) + 1.0)
        else:
            result.loc[index] = 0.5
        history.append(float(value))
    return result


def _transfer(
    weights: np.ndarray,
    donors: list[int],
    receivers: list[int],
    amount: float,
) -> np.ndarray:
    """Copy of the deployed factor-tilt transfer, kept local for Colab portability."""
    output = weights.copy()
    floors = np.array([0.02, 0.05, 0.02, 0.00])
    available = np.maximum(output[donors] - floors[donors], 0)
    transfer = min(float(amount), float(available.sum()))
    if transfer <= 0:
        return output
    output[donors] -= transfer * available / available.sum()
    receiver_base = np.maximum(output[receivers], 0.02)
    output[receivers] += transfer * receiver_base / receiver_base.sum()
    return output / output.sum()


def apply_factor_tilt(base: np.ndarray, score: float, max_shift: float) -> np.ndarray:
    if not np.isfinite(score) or max_shift <= 0:
        return base.copy()
    if score >= 0:
        return _transfer(base, donors=[1, 2], receivers=[0], amount=max_shift * score)
    return _transfer(base, donors=[0, 3], receivers=[1, 2], amount=max_shift * -score)


def run_factor_vol_target(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive_path: pd.DataFrame,
    factor: pd.DataFrame,
    max_shift: float,
    target_vol: float = 0.15,
    cost_multiplier: float = 1.0,
    financing_rate: float = 0.04,
    leverage_min: float = 0.50,
    leverage_max: float = 1.50,
    initial_leverage: float = 1.20,
) -> pd.DataFrame:
    """Reproduce the deployed medium-horizon allocation without heavy optional imports."""
    if not 0 < leverage_min <= leverage_max:
        raise ValueError("leverage bounds must satisfy 0 < minimum <= maximum")
    initial_leverage = float(np.clip(initial_leverage, leverage_min, leverage_max))
    months = signals.index.intersection(returns.index).intersection(defensive_path.index)
    rows: list[dict[str, float | pd.Period | str]] = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        hard = hard_regime_weights(signals.loc[month])
        defensive = defensive_path.loc[
            month, [f"w_{asset}" for asset in ASSETS]
        ].to_numpy(dtype=float)
        base = 0.40 * hard + 0.60 * defensive
        probability = float(factor.loc[month, "p_up"]) if month in factor.index else 0.5
        score = float(np.clip((probability - 0.5) / 0.15, -1, 1))
        unlevered = apply_factor_tilt(base, score, max_shift)

        history = returns.loc[returns.index < month, ASSETS].tail(24)
        if len(history) >= 12:
            conditional = history.to_numpy(dtype=float) @ unlevered
            observation_weights = np.exp(np.linspace(-2.0, 0.0, len(conditional)))
            observation_weights /= observation_weights.sum()
            mean = float(observation_weights @ conditional)
            variance = float(observation_weights @ (conditional - mean) ** 2)
            forecast_vol = math.sqrt(max(variance, 1e-8) * 12)
            leverage = float(
                np.clip(target_vol / forecast_vol, leverage_min, leverage_max)
            )
        else:
            forecast_vol = np.nan
            leverage = initial_leverage
        asset_weights = leverage * unlevered
        debt_weight = 1.0 - leverage
        delta = asset_weights - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = (
            abs((asset_weights[2] + asset_weights[3]) - (pretrade[2] + pretrade[3]))
            * 0.0005
            * cost_multiplier
        )
        financing = debt_weight * ((1 + financing_rate) ** (1 / 12) - 1)
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(asset_weights @ asset_return + financing)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = asset_weights * (1 + asset_return) / (1 + gross_return)
        first_trade = False
        rows.append(
            {
                "month": month,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "forecast_vol": forecast_vol,
                "leverage": leverage,
                "risk_score": score,
                **{
                    f"w_{asset}": asset_weights[index]
                    for index, asset in enumerate(ASSETS)
                },
            }
        )
    return pd.DataFrame(rows).set_index("month")


def _period_view(frame: pd.DataFrame | pd.Series, start: pd.Period | None, end: pd.Period | None):
    output = frame
    if start is not None:
        output = output.loc[start:]
    if end is not None:
        output = output.loc[:end]
    return output


def _metric_or_nan(function, y: pd.Series, values: pd.Series) -> float:
    if len(y) == 0 or y.nunique() < 2:
        return float("nan")
    return float(function(y, values))


def moving_block_draws(
    length: int,
    simulations: int = 2000,
    block_length: int = 6,
    seed: int = RNG_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((simulations, length), dtype=int)
    for row in range(simulations):
        indices: list[int] = []
        while len(indices) < length:
            start = int(rng.integers(0, max(length - block_length + 1, 1)))
            indices.extend(range(start, min(start + block_length, length)))
        output[row] = np.asarray(indices[:length], dtype=int)
    return output


def _sharpe_matrix(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1, ddof=1)
    return np.divide(mean, std, out=np.full_like(mean, np.nan), where=std > 0) * math.sqrt(12)


def add_locked_bootstrap(
    summary: pd.DataFrame,
    loss_matrix: pd.DataFrame,
    return_matrix: pd.DataFrame,
    deployed_id: str,
    simulations: int = 2000,
) -> pd.DataFrame:
    candidates = list(summary.index)
    loss = loss_matrix[candidates].dropna(how="any")
    returns = return_matrix[candidates].dropna(how="any")
    loss = loss.loc[loss.index >= TEST_START]
    returns = returns.loc[returns.index >= TEST_START]

    loss_draws = moving_block_draws(len(loss), simulations, seed=RNG_SEED + 1)
    return_draws = moving_block_draws(len(returns), simulations, seed=RNG_SEED + 2)
    loss_values = loss.to_numpy(dtype=float)[loss_draws].mean(axis=1)
    return_values = returns.to_numpy(dtype=float)[return_draws]
    sharpes = _sharpe_matrix(return_values)

    deployed_column = candidates.index(deployed_id)
    brier_delta = loss_values - loss_values[:, [deployed_column]]
    sharpe_delta = sharpes - sharpes[:, [deployed_column]]
    summary = summary.copy()
    summary["locked_boot_probability_brier_better_than_deployed"] = np.mean(
        brier_delta < 0, axis=0
    )
    summary["locked_boot_probability_sharpe_better_than_deployed"] = np.mean(
        sharpe_delta > 0, axis=0
    )
    summary["locked_boot_sharpe_delta_p05"] = np.nanquantile(sharpe_delta, 0.05, axis=0)
    summary["locked_boot_sharpe_delta_median"] = np.nanmedian(sharpe_delta, axis=0)
    summary["locked_boot_sharpe_delta_p95"] = np.nanquantile(sharpe_delta, 0.95, axis=0)
    return summary


def cscv_pbo(
    matrix: pd.DataFrame,
    kind: str,
    partitions: int = 8,
) -> dict[str, object]:
    clean = matrix.dropna(how="any")
    blocks = [np.asarray(block, dtype=int) for block in np.array_split(np.arange(len(clean)), partitions)]
    candidates = list(clean.columns)
    values = clean.to_numpy(dtype=float)
    selected: list[str] = []
    oos_percentiles: list[float] = []
    logits: list[float] = []

    def scores(rows: np.ndarray) -> np.ndarray:
        block = values[rows]
        if kind == "sharpe":
            std = block.std(axis=0, ddof=1)
            return np.divide(
                block.mean(axis=0),
                std,
                out=np.full(block.shape[1], -np.inf),
                where=std > 0,
            ) * math.sqrt(12)
        if kind == "mean":
            return block.mean(axis=0)
        raise ValueError(kind)

    half = partitions // 2
    for chosen in itertools.combinations(range(partitions), half):
        chosen_set = set(chosen)
        train_rows = np.concatenate([blocks[index] for index in chosen])
        test_rows = np.concatenate(
            [blocks[index] for index in range(partitions) if index not in chosen_set]
        )
        train_scores = scores(train_rows)
        winner = int(np.nanargmax(train_scores))
        test_scores = scores(test_rows)
        ranks = rankdata(test_scores, method="average")
        percentile = float((ranks[winner] - 0.5) / len(candidates))
        bounded = float(np.clip(percentile, 1e-6, 1 - 1e-6))
        selected.append(candidates[winner])
        oos_percentiles.append(percentile)
        logits.append(float(math.log(bounded / (1 - bounded))))

    counts = pd.Series(selected).value_counts()
    return {
        "kind": kind,
        "partitions": partitions,
        "splits": len(selected),
        "pbo": float(np.mean(np.asarray(logits) <= 0)),
        "median_oos_percentile": float(np.median(oos_percentiles)),
        "median_logit": float(np.median(logits)),
        "most_selected_candidate": str(counts.index[0]),
        "most_selected_share": float(counts.iloc[0] / len(selected)),
    }


def _rank_score(frame: pd.DataFrame, high: list[str], low: list[str]) -> pd.Series:
    ranks: list[pd.Series] = []
    for column in high:
        ranks.append(frame[column].rank(pct=True, method="average"))
    for column in low:
        ranks.append((-frame[column]).rank(pct=True, method="average"))
    return pd.concat(ranks, axis=1).mean(axis=1)


def _rank_correlation(first: pd.Series, second: pd.Series) -> float:
    joined = pd.concat([first, second], axis=1).dropna()
    if len(joined) < 3 or joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).statistic)


def logistic_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    c_values = (0.01, 0.03, 0.10, 0.30, 1.00, 3.00)
    for penalty, solver, class_weight in (
        ("l2", "liblinear", "balanced"),
        ("l1", "liblinear", "balanced"),
        ("l2", "lbfgs", "balanced"),
        ("l2", "liblinear", None),
    ):
        for c_value in c_values:
            weight_name = "balanced" if class_weight == "balanced" else "none"
            specs.append(
                {
                    "candidate": f"{penalty}_{solver}_c{c_value:g}_{weight_name}",
                    "penalty": penalty,
                    "solver": solver,
                    "C": c_value,
                    "class_weight": class_weight,
                    "l1_ratio": np.nan,
                }
            )
    for c_value in (0.03, 0.10, 0.30, 1.00):
        specs.append(
            {
                "candidate": f"elasticnet_saga_c{c_value:g}_r0.5_balanced",
                "penalty": "elasticnet",
                "solver": "saga",
                "C": c_value,
                "class_weight": "balanced",
                "l1_ratio": 0.5,
            }
        )
    return specs


def make_logistic_model(spec: dict[str, object]) -> Pipeline:
    arguments: dict[str, object] = {
        "C": float(spec["C"]),
        "penalty": str(spec["penalty"]),
        "solver": str(spec["solver"]),
        "class_weight": spec["class_weight"],
        "max_iter": 5000,
        "random_state": RNG_SEED,
    }
    if spec["penalty"] == "elasticnet":
        arguments["l1_ratio"] = float(spec["l1_ratio"])
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**arguments)),
        ]
    )


def tail_data() -> pd.DataFrame:
    domestic = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
    composites = pd.read_csv(RESULTS / "openassetpricing_composites.csv", index_col=0)
    factor = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_factor.csv", index_col=0
    )
    for frame in (domestic, composites, factor):
        frame.index = pd.PeriodIndex(frame.index, freq="M")
    return domestic[DOMESTIC_FEATURES].join(composites[OAP_COMPOSITES]).join(
        factor[["tail_event"]]
    )


def fit_logistic_candidate(
    data: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[pd.Series, dict[str, float]]:
    probability = pd.Series(np.nan, index=data.index, dtype=float)
    l1_norms: list[float] = []
    l2_norms: list[float] = []
    nonzero_counts: list[int] = []
    convergence_warnings = 0
    for number, month in enumerate(data.index):
        train_end = number - 2
        if train_end < 36:
            continue
        train = data.iloc[:train_end].dropna(subset=["tail_event"])
        y = train["tail_event"].astype(int)
        if y.sum() < 4 or (len(y) - y.sum()) < 12:
            continue
        model = make_logistic_model(spec)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train[TAIL_FEATURES], y)
        convergence_warnings += sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        probability.loc[month] = float(
            model.predict_proba(data.loc[[month], TAIL_FEATURES])[:, 1][0]
        )
        coefficient = model.named_steps["model"].coef_[0]
        l1_norms.append(float(np.abs(coefficient).sum()))
        l2_norms.append(float(np.sqrt(np.square(coefficient).sum())))
        nonzero_counts.append(int(np.sum(np.abs(coefficient) > 1e-8)))
    return probability, {
        "fit_count": float(len(l1_norms)),
        "convergence_warning_count": float(convergence_warnings),
        "median_coefficient_l1_norm": float(np.median(l1_norms)),
        "median_coefficient_l2_norm": float(np.median(l2_norms)),
        "median_nonzero_coefficients": float(np.median(nonzero_counts)),
    }


def make_tail_factor(probability: pd.Series, target: pd.Series) -> pd.DataFrame:
    output = pd.DataFrame({"p_tail_raw": probability, "tail_event": target})
    output["risk_percentile"] = causal_percentile(output["p_tail_raw"])
    output["risk_severity"] = (
        (output["risk_percentile"] - 0.80) / 0.20
    ).clip(0, 1)
    output["p_up"] = 0.50 - 0.15 * output["risk_severity"]
    output["score"] = -output["risk_severity"]
    return output


def logistic_period_record(
    candidate: str,
    spec: dict[str, object],
    fit_stats: dict[str, float],
    factor: pd.DataFrame,
    backtest: pd.DataFrame,
    period: str,
    start: pd.Period | None,
    end: pd.Period | None,
) -> dict[str, object]:
    view = _period_view(factor, start, end).dropna(
        subset=["p_tail_raw", "tail_event"]
    )
    y = view["tail_event"].astype(int)
    p = view["p_tail_raw"].clip(1e-6, 1 - 1e-6)
    triggered = view["risk_percentile"] >= 0.80
    path = _period_view(backtest, start, end)
    metrics = performance_summary(path["return"])
    return {
        "candidate": candidate,
        "period": period,
        **spec,
        **fit_stats,
        "prediction_observations": int(len(view)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "roc_auc": _metric_or_nan(roc_auc_score, y, p),
        "average_precision": _metric_or_nan(average_precision_score, y, p),
        "brier_score": float(brier_score_loss(y, p)),
        "prevalence_brier": float(y.mean() * (1 - y.mean())),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_5bin": _ece(y, p, bins=5),
        "policy_trigger_rate": float(triggered.mean()),
        "policy_recall": float(y[triggered].sum() / max(int(y.sum()), 1)),
        "policy_precision": float(y[triggered].mean()) if triggered.any() else np.nan,
        **metrics.to_dict(),
        "AvgTurnover": float(path["turnover"].mean()),
    }


def summarize_logistic(
    long: pd.DataFrame,
    loss_matrix: pd.DataFrame,
    return_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = long.drop_duplicates("candidate").set_index("candidate")[[
        "penalty",
        "solver",
        "C",
        "class_weight",
        "l1_ratio",
        "fit_count",
        "convergence_warning_count",
        "median_coefficient_l1_norm",
        "median_coefficient_l2_norm",
        "median_nonzero_coefficients",
    ]]
    summary = metadata.copy()
    metric_columns = (
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "ece_5bin",
        "policy_recall",
        "policy_precision",
        "CAGR",
        "Sharpe",
        "MDD",
        "Calmar",
        "AvgTurnover",
    )
    for period in ("calibration_2007_2017", "validation_2013_2017", "locked_2018_2026"):
        view = long.loc[long["period"] == period].set_index("candidate")
        prefix = {
            "calibration_2007_2017": "cal",
            "validation_2013_2017": "val",
            "locked_2018_2026": "locked",
        }[period]
        for column in metric_columns:
            summary[f"{prefix}_{column}"] = view[column]

    for prefix in ("cal", "locked"):
        summary[f"{prefix}_prediction_score"] = _rank_score(
            summary,
            [f"{prefix}_roc_auc", f"{prefix}_average_precision"],
            [f"{prefix}_brier_score", f"{prefix}_ece_5bin"],
        )
        summary[f"{prefix}_portfolio_score"] = _rank_score(
            summary,
            [f"{prefix}_CAGR", f"{prefix}_Sharpe", f"{prefix}_MDD"],
            [],
        )
        summary[f"{prefix}_prediction_rank"] = summary[
            f"{prefix}_prediction_score"
        ].rank(ascending=False, method="min")
        summary[f"{prefix}_portfolio_rank"] = summary[
            f"{prefix}_portfolio_score"
        ].rank(ascending=False, method="min")

    deployed_id = "l2_liblinear_c0.1_balanced"
    summary["is_deployed"] = summary.index == deployed_id
    summary = add_locked_bootstrap(summary, loss_matrix, return_matrix, deployed_id)

    cal_prediction_winner = str(summary["cal_prediction_score"].idxmax())
    cal_portfolio_winner = str(summary["cal_portfolio_score"].idxmax())
    deployed = summary.loc[deployed_id]
    cal = long.loc[long["period"] == "calibration_2007_2017"].set_index("candidate")
    val = long.loc[long["period"] == "validation_2013_2017"].set_index("candidate")
    deployed_cal = cal.loc[deployed_id]
    deployed_val = val.loc[deployed_id]
    strict = (
        (cal["roc_auc"] >= deployed_cal["roc_auc"])
        & (cal["brier_score"] <= deployed_cal["brier_score"])
        & (cal["Sharpe"] >= deployed_cal["Sharpe"])
        & (cal["MDD"] >= deployed_cal["MDD"])
        & (val["roc_auc"] >= deployed_val["roc_auc"])
        & (val["brier_score"] <= deployed_val["brier_score"])
        & (val["Sharpe"] >= deployed_val["Sharpe"])
        & (val["MDD"] >= deployed_val["MDD"])
    )
    strict_candidates = [str(candidate) for candidate in strict.index[strict]]
    warning_mask = summary["convergence_warning_count"] > 0
    warning_candidates = [
        {
            "candidate": str(candidate),
            "warning_count": int(summary.loc[candidate, "convergence_warning_count"]),
        }
        for candidate in summary.index[warning_mask]
    ]
    stable_candidates = [str(candidate) for candidate in summary.index[~warning_mask]]
    zero_coefficient_candidates = [
        str(candidate)
        for candidate in summary.index[summary["median_nonzero_coefficients"] == 0]
    ]

    report = {
        "candidate_count": int(len(summary)),
        "grid": {
            "C": [0.01, 0.03, 0.10, 0.30, 1.00, 3.00],
            "families": [
                "balanced L2 liblinear",
                "balanced L1 liblinear",
                "balanced L2 lbfgs",
                "unweighted L2 liblinear",
                "balanced ElasticNet saga",
            ],
        },
        "deployed_candidate": deployed_id,
        "deployed_calibration_prediction_rank": int(deployed["cal_prediction_rank"]),
        "deployed_locked_prediction_rank": int(deployed["locked_prediction_rank"]),
        "deployed_calibration_portfolio_rank": int(deployed["cal_portfolio_rank"]),
        "deployed_locked_portfolio_rank": int(deployed["locked_portfolio_rank"]),
        "calibration_prediction_winner": cal_prediction_winner,
        "calibration_prediction_winner_locked_rank": int(
            summary.loc[cal_prediction_winner, "locked_prediction_rank"]
        ),
        "calibration_portfolio_winner": cal_portfolio_winner,
        "calibration_portfolio_winner_locked_rank": int(
            summary.loc[cal_portfolio_winner, "locked_portfolio_rank"]
        ),
        "prediction_rank_correlation_calibration_vs_locked": _rank_correlation(
            summary["cal_prediction_score"], summary["locked_prediction_score"]
        ),
        "portfolio_rank_correlation_calibration_vs_locked": _rank_correlation(
            summary["cal_portfolio_score"], summary["locked_portfolio_score"]
        ),
        "strict_both_prelock_windows_count": int(strict.sum()),
        "strict_both_prelock_windows_candidates": strict_candidates,
        "candidates_with_convergence_warnings": int(warning_mask.sum()),
        "total_convergence_warnings": int(
            summary["convergence_warning_count"].sum()
        ),
        "convergence_warning_candidates": warning_candidates,
        "zero_coefficient_candidates": zero_coefficient_candidates,
        "locked_cscv_note": "CSCV/PBO is calculated only on pre-2018 data; locked data is reserved for rank validation.",
        "prelock_prediction_brier_pbo": cscv_pbo(
            -loss_matrix.loc[loss_matrix.index <= CAL_END], kind="mean"
        ),
        "prelock_portfolio_sharpe_pbo": cscv_pbo(
            return_matrix.loc[return_matrix.index <= CAL_END], kind="sharpe"
        ),
        "prelock_prediction_brier_pbo_excluding_warning_candidates": cscv_pbo(
            -loss_matrix.loc[loss_matrix.index <= CAL_END, stable_candidates],
            kind="mean",
        ),
        "prelock_portfolio_sharpe_pbo_excluding_warning_candidates": cscv_pbo(
            return_matrix.loc[return_matrix.index <= CAL_END, stable_candidates],
            kind="sharpe",
        ),
    }
    return summary.sort_values("cal_prediction_rank"), report


def run_logistic_robustness(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data = tail_data()
    stored_factor = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_factor.csv", index_col=0
    )
    stored_factor.index = pd.PeriodIndex(stored_factor.index, freq="M")
    stored_backtest = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_backtest.csv", index_col=0
    )
    stored_backtest.index = pd.PeriodIndex(stored_backtest.index, freq="M")

    long_rows: list[dict[str, object]] = []
    losses: dict[str, pd.Series] = {}
    portfolio_returns: dict[str, pd.Series] = {}
    reproduction: dict[str, float] = {}
    specs = logistic_specs()
    for number, spec in enumerate(specs, start=1):
        candidate = str(spec["candidate"])
        print(f"logistic {number:02d}/{len(specs)} {candidate}", flush=True)
        probability, fit_stats = fit_logistic_candidate(data, spec)
        factor = make_tail_factor(probability, data["tail_event"])
        backtest = run_factor_vol_target(
            returns,
            signals,
            defensive,
            factor,
            max_shift=0.20,
            target_vol=0.15,
        )
        valid = factor[["p_tail_raw", "tail_event"]].dropna()
        losses[candidate] = (
            valid["p_tail_raw"].clip(1e-6, 1 - 1e-6) - valid["tail_event"]
        ) ** 2
        portfolio_returns[candidate] = backtest["return"]
        for period, start, end in PERIODS:
            long_rows.append(
                logistic_period_record(
                    candidate,
                    spec,
                    fit_stats,
                    factor,
                    backtest,
                    period,
                    start,
                    end,
                )
            )
        if candidate == "l2_liblinear_c0.1_balanced":
            probability_comparison = factor[["p_tail_raw"]].join(
                stored_factor[["p_tail_raw"]],
                how="inner",
                lsuffix="_refit",
                rsuffix="_stored",
            ).dropna()
            common = backtest.index.intersection(stored_backtest.index)
            reproduction = {
                "probability_observations": int(len(probability_comparison)),
                "max_absolute_probability_difference": float(
                    (
                        probability_comparison["p_tail_raw_refit"]
                        - probability_comparison["p_tail_raw_stored"]
                    ).abs().max()
                ),
                "portfolio_observations": int(len(common)),
                "max_absolute_portfolio_return_difference": float(
                    (
                        backtest.loc[common, "return"]
                        - stored_backtest.loc[common, "return"]
                    ).abs().max()
                ),
            }

    long = pd.DataFrame(long_rows)
    loss_matrix = pd.concat(losses, axis=1).sort_index()
    return_matrix = pd.concat(portfolio_returns, axis=1).sort_index()
    summary, report = summarize_logistic(long, loss_matrix, return_matrix)
    report["deployed_reproduction"] = reproduction
    assert reproduction["probability_observations"] > 0
    assert reproduction["portfolio_observations"] > 0
    assert reproduction["max_absolute_probability_difference"] < 1e-12
    assert reproduction["max_absolute_portfolio_return_difference"] < 1e-12
    return long, summary, report


def sjm_internal_grid() -> tuple[tuple[float, ...], tuple[int, ...]]:
    return (0.0, 1.5, 3.0, 6.0), (2, 4, 6)


def build_or_load_sjm_paths(
    components: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    penalties, keeps = sjm_internal_grid()
    expected = {(float(penalty), int(keep)) for penalty in penalties for keep in keeps}
    if SJM_CACHE_PATH.exists():
        cached = pd.read_csv(SJM_CACHE_PATH)
        cached["target_month"] = pd.PeriodIndex(cached["target_month"], freq="M")
        cached["signal_month"] = pd.PeriodIndex(cached["signal_month"], freq="M")
        found = set(
            zip(cached["jump_penalty"].astype(float), cached["keep_features"].astype(int))
        )
        counts = cached.groupby(["jump_penalty", "keep_features"]).size()
        if found == expected and counts.eq(len(components)).all():
            print("SJM internal path cache reused", cached.shape, flush=True)
            return cached

    rows: list[dict[str, object]] = []
    combinations = list(itertools.product(penalties, keeps))
    for number, (penalty, keep) in enumerate(combinations, start=1):
        print(
            f"SJM internal {number:02d}/{len(combinations)} jump={penalty:g} keep={keep}",
            flush=True,
        )
        model = SparseJump2(jump_penalty=penalty, keep_features=keep)
        for target_month, component in components.iterrows():
            signal_month = component["signal_month"]
            history = macro.loc[: signal_month.to_timestamp("M")]
            pg, growth_detail = model.fit_predict_high(history["growth"])
            pi, inflation_detail = model.fit_predict_high(history["inflation"])
            rows.append(
                {
                    "target_month": target_month,
                    "signal_month": signal_month,
                    "jump_penalty": penalty,
                    "keep_features": keep,
                    "growth_sjm": pg,
                    "inflation_sjm": pi,
                    "growth_switches": growth_detail["switches"],
                    "inflation_switches": inflation_detail["switches"],
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(SJM_CACHE_PATH, index=False)
    return output


def build_macro_signals(
    probabilities: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    output = pd.DataFrame(index=probabilities.index)
    output["signal_month"] = components["signal_month"]
    output["p_growth_high"] = probabilities["growth"]
    output["p_inflation_high"] = probabilities["inflation"]
    output["p_Goldilocks"] = probabilities["growth"] * (1 - probabilities["inflation"])
    output["p_Overheating"] = probabilities["growth"] * probabilities["inflation"]
    output["p_Slowdown"] = (1 - probabilities["growth"]) * (1 - probabilities["inflation"])
    output["p_Stagflation"] = (1 - probabilities["growth"]) * probabilities["inflation"]
    regime_columns = [
        "p_Goldilocks",
        "p_Overheating",
        "p_Slowdown",
        "p_Stagflation",
    ]
    output["regime"] = output[regime_columns].idxmax(axis=1).str.removeprefix("p_")
    return output


def aligned_macro_targets(
    components: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    output = pd.DataFrame(index=components.index, dtype=float)
    for name in ("growth", "inflation"):
        output[name] = [
            targets.loc[signal_month.to_timestamp("M"), name]
            if signal_month.to_timestamp("M") in targets.index
            else np.nan
            for signal_month in components["signal_month"]
        ]
    return output


def macro_period_record(
    candidate: str,
    penalty: float,
    keep: int,
    weight: float,
    probabilities: pd.DataFrame,
    target: pd.DataFrame,
    soft: pd.DataFrame,
    proposed: pd.DataFrame | None,
    switches: tuple[float, float],
    period: str,
    start: pd.Period | None,
    end: pd.Period | None,
) -> dict[str, object]:
    pred = _period_view(probabilities.join(target, rsuffix="_target"), start, end).dropna()
    growth_y = pred["growth_target"].astype(int)
    inflation_y = pred["inflation_target"].astype(int)
    growth_p = pred["growth"].clip(1e-6, 1 - 1e-6)
    inflation_p = pred["inflation"].clip(1e-6, 1 - 1e-6)
    quadrant = (
        ((growth_p >= 0.5) == growth_y.astype(bool))
        & ((inflation_p >= 0.5) == inflation_y.astype(bool))
    )
    soft_view = _period_view(soft, start, end)
    soft_metrics = performance_summary(soft_view["return"])
    record: dict[str, object] = {
        "candidate": candidate,
        "period": period,
        "jump_penalty": penalty,
        "keep_features": keep,
        "sjm_weight": weight,
        "is_deployed": bool(
            math.isclose(penalty, 3.0)
            and keep == 4
            and math.isclose(weight, 0.10)
        ),
        "prediction_observations": int(len(pred)),
        "growth_auc": _metric_or_nan(roc_auc_score, growth_y, growth_p),
        "inflation_auc": _metric_or_nan(roc_auc_score, inflation_y, inflation_p),
        "mean_auc": float(
            np.nanmean(
                [
                    _metric_or_nan(roc_auc_score, growth_y, growth_p),
                    _metric_or_nan(roc_auc_score, inflation_y, inflation_p),
                ]
            )
        ),
        "growth_brier": float(brier_score_loss(growth_y, growth_p)),
        "inflation_brier": float(brier_score_loss(inflation_y, inflation_p)),
        "mean_brier": float(
            (
                brier_score_loss(growth_y, growth_p)
                + brier_score_loss(inflation_y, inflation_p)
            )
            / 2
        ),
        "quadrant_accuracy": float(quadrant.mean()),
        "growth_switches": switches[0],
        "inflation_switches": switches[1],
        **{f"Soft_{key}": value for key, value in soft_metrics.to_dict().items()},
        "Soft_AvgTurnover": float(soft_view["turnover"].mean()),
    }
    if proposed is not None:
        proposed_view = _period_view(proposed, start, end)
        proposed_metrics = performance_summary(proposed_view["return"])
        record.update(
            {f"Proposed_{key}": value for key, value in proposed_metrics.to_dict().items()}
        )
        record["Proposed_AvgTurnover"] = float(proposed_view["turnover"].mean())
    return record


def summarize_sjm(
    long: pd.DataFrame,
    loss_matrix: pd.DataFrame,
    return_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = long.drop_duplicates("candidate").set_index("candidate")[[
        "jump_penalty",
        "keep_features",
        "sjm_weight",
        "is_deployed",
        "growth_switches",
        "inflation_switches",
    ]]
    summary = metadata.copy()
    metric_columns = (
        "mean_auc",
        "mean_brier",
        "quadrant_accuracy",
        "Soft_CAGR",
        "Soft_Sharpe",
        "Soft_MDD",
        "Soft_Calmar",
        "Soft_AvgTurnover",
        "Proposed_CAGR",
        "Proposed_Sharpe",
        "Proposed_MDD",
    )
    for period in ("calibration_2007_2017", "validation_2013_2017", "locked_2018_2026"):
        view = long.loc[long["period"] == period].set_index("candidate")
        prefix = {
            "calibration_2007_2017": "cal",
            "validation_2013_2017": "val",
            "locked_2018_2026": "locked",
        }[period]
        for column in metric_columns:
            if column in view:
                summary[f"{prefix}_{column}"] = view[column]

    for prefix in ("cal", "locked"):
        summary[f"{prefix}_prediction_score"] = _rank_score(
            summary,
            [f"{prefix}_mean_auc", f"{prefix}_quadrant_accuracy"],
            [f"{prefix}_mean_brier"],
        )
        summary[f"{prefix}_soft_score"] = _rank_score(
            summary,
            [f"{prefix}_Soft_CAGR", f"{prefix}_Soft_Sharpe", f"{prefix}_Soft_MDD"],
            [],
        )
        summary[f"{prefix}_prediction_rank"] = summary[
            f"{prefix}_prediction_score"
        ].rank(ascending=False, method="min")
        summary[f"{prefix}_soft_rank"] = summary[f"{prefix}_soft_score"].rank(
            ascending=False, method="min"
        )

    deployed_id = "sjm_j3_k4_w0.1"
    summary = add_locked_bootstrap(summary, loss_matrix, return_matrix, deployed_id)
    cal_prediction_winner = str(summary["cal_prediction_score"].idxmax())
    cal_soft_winner = str(summary["cal_soft_score"].idxmax())
    deployed = summary.loc[deployed_id]

    cal = long.loc[long["period"] == "calibration_2007_2017"].set_index("candidate")
    val = long.loc[long["period"] == "validation_2013_2017"].set_index("candidate")
    deployed_cal = cal.loc[deployed_id]
    deployed_val = val.loc[deployed_id]
    strict = (
        (cal["mean_brier"] <= deployed_cal["mean_brier"])
        & (cal["quadrant_accuracy"] >= deployed_cal["quadrant_accuracy"])
        & (cal["Soft_Sharpe"] >= deployed_cal["Soft_Sharpe"])
        & (cal["Soft_MDD"] >= deployed_cal["Soft_MDD"])
        & (val["mean_brier"] <= deployed_val["mean_brier"])
        & (val["quadrant_accuracy"] >= deployed_val["quadrant_accuracy"])
        & (val["Soft_Sharpe"] >= deployed_val["Soft_Sharpe"])
        & (val["Soft_MDD"] >= deployed_val["Soft_MDD"])
    )
    strict_candidates = [str(candidate) for candidate in strict.index[strict]]

    report = {
        "candidate_count": int(len(summary)),
        "grid": {
            "jump_penalty": [0.0, 1.5, 3.0, 6.0],
            "keep_features": [2, 4, 6],
            "sjm_weight": [0.05, 0.10, 0.20, 0.30],
            "no_sjm_candidate": True,
        },
        "deployed_candidate": deployed_id,
        "deployed_calibration_prediction_rank": int(deployed["cal_prediction_rank"]),
        "deployed_locked_prediction_rank": int(deployed["locked_prediction_rank"]),
        "deployed_calibration_soft_rank": int(deployed["cal_soft_rank"]),
        "deployed_locked_soft_rank": int(deployed["locked_soft_rank"]),
        "calibration_prediction_winner": cal_prediction_winner,
        "calibration_prediction_winner_locked_rank": int(
            summary.loc[cal_prediction_winner, "locked_prediction_rank"]
        ),
        "calibration_soft_winner": cal_soft_winner,
        "calibration_soft_winner_locked_rank": int(
            summary.loc[cal_soft_winner, "locked_soft_rank"]
        ),
        "prediction_rank_correlation_calibration_vs_locked": _rank_correlation(
            summary["cal_prediction_score"], summary["locked_prediction_score"]
        ),
        "soft_rank_correlation_calibration_vs_locked": _rank_correlation(
            summary["cal_soft_score"], summary["locked_soft_score"]
        ),
        "strict_both_prelock_windows_count": int(strict.sum()),
        "strict_both_prelock_windows_candidates": strict_candidates,
        "prelock_macro_brier_pbo": cscv_pbo(
            -loss_matrix.loc[loss_matrix.index <= CAL_END], kind="mean"
        ),
        "prelock_soft_sharpe_pbo": cscv_pbo(
            return_matrix.loc[return_matrix.index <= CAL_END], kind="sharpe"
        ),
        "proposed_one_axis_candidate_count": int(
            summary["cal_Proposed_Sharpe"].notna().sum()
        ),
    }
    return summary.sort_values("cal_prediction_rank"), report


def run_sjm_robustness(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    components, macro_targets = _macro_components()
    macro, _ = load_macro_data()
    aligned_target = aligned_macro_targets(components, macro_targets)
    internal = build_or_load_sjm_paths(components, macro)

    candidates: list[tuple[str, float, int, float]] = [("no_sjm", 3.0, 4, 0.0)]
    penalties, keeps = sjm_internal_grid()
    for penalty, keep, weight in itertools.product(
        penalties, keeps, (0.05, 0.10, 0.20, 0.30)
    ):
        candidates.append(
            (f"sjm_j{penalty:g}_k{keep}_w{weight:g}", penalty, keep, weight)
        )

    long_rows: list[dict[str, object]] = []
    valid_long_rows: list[dict[str, object]] = []
    invalid_candidates: list[dict[str, object]] = []
    losses: dict[str, pd.Series] = {}
    soft_returns: dict[str, pd.Series] = {}
    deployed_reproduction: dict[str, float] = {}
    stored_signals = pd.read_csv(RESULTS / "regime_signals.csv", index_col=0)
    stored_signals.index = pd.PeriodIndex(stored_signals.index, freq="M")

    for number, (candidate, penalty, keep, weight) in enumerate(candidates, start=1):
        print(f"SJM candidate {number:02d}/{len(candidates)} {candidate}", flush=True)
        selected = internal.loc[
            np.isclose(internal["jump_penalty"], penalty)
            & (internal["keep_features"] == keep)
        ].set_index("target_month")
        selected.index = pd.PeriodIndex(selected.index, freq="M")
        variant_components = components.copy()
        variant_components["growth_sjm"] = selected["growth_sjm"]
        variant_components["inflation_sjm"] = selected["inflation_sjm"]
        probabilities = _macro_probabilities(
            variant_components,
            d3_weight=0.20,
            sigmoid_scale=0.55,
            sjm_weight=weight,
            current_weight=0.85,
        )
        nonfinite = ~np.isfinite(probabilities.to_numpy(dtype=float))
        if nonfinite.any():
            first_bad_position = int(np.argwhere(nonfinite)[0, 0])
            first_bad_month = str(probabilities.index[first_bad_position])
            reason = (
                "SparseJump2 produced a non-finite state probability; recursive "
                f"macro smoothing became non-finite from {first_bad_month}."
            )
            invalid_candidates.append(
                {
                    "candidate": candidate,
                    "jump_penalty": penalty,
                    "keep_features": keep,
                    "sjm_weight": weight,
                    "first_bad_month": first_bad_month,
                    "reason": reason,
                }
            )
            for period, _, _ in PERIODS:
                long_rows.append(
                    {
                        "candidate": candidate,
                        "period": period,
                        "jump_penalty": penalty,
                        "keep_features": keep,
                        "sjm_weight": weight,
                        "is_deployed": False,
                        "invalid_reason": reason,
                    }
                )
            print(f"  invalid: {reason}", flush=True)
            continue
        signals = build_macro_signals(probabilities, components)
        soft = run_backtest(returns, signals, StrategyConfig(), mode="soft")
        one_axis = (
            (math.isclose(penalty, 3.0) and keep == 4)
            or (
                math.isclose(weight, 0.10)
                and (math.isclose(penalty, 3.0) or keep == 4)
            )
        )
        proposed = (
            run_backtest(returns, signals, StrategyConfig(), mode="proposed")
            if one_axis
            else None
        )
        joined = probabilities.join(aligned_target, rsuffix="_target").dropna()
        losses[candidate] = (
            (joined["growth"] - joined["growth_target"]) ** 2
            + (joined["inflation"] - joined["inflation_target"]) ** 2
        ) / 2
        soft_returns[candidate] = soft["return"]
        switches = (
            float(selected["growth_switches"].iloc[-1]),
            float(selected["inflation_switches"].iloc[-1]),
        )
        for period, start, end in PERIODS:
            record = macro_period_record(
                    candidate,
                    penalty,
                    keep,
                    weight,
                    probabilities,
                    aligned_target,
                    soft,
                    proposed,
                    switches,
                    period,
                    start,
                    end,
                )
            record["invalid_reason"] = ""
            long_rows.append(record)
            valid_long_rows.append(record)
        if candidate == "sjm_j3_k4_w0.1":
            common = probabilities.index.intersection(stored_signals.index)
            deployed_reproduction = {
                "observations": int(len(common)),
                "max_absolute_growth_probability_difference": float(
                    (
                        probabilities.loc[common, "growth"]
                        - stored_signals.loc[common, "p_growth_high"]
                    ).abs().max()
                ),
                "max_absolute_inflation_probability_difference": float(
                    (
                        probabilities.loc[common, "inflation"]
                        - stored_signals.loc[common, "p_inflation_high"]
                    ).abs().max()
                ),
            }

    long = pd.DataFrame(long_rows)
    valid_long = pd.DataFrame(valid_long_rows)
    loss_matrix = pd.concat(losses, axis=1).sort_index()
    return_matrix = pd.concat(soft_returns, axis=1).sort_index()
    summary, report = summarize_sjm(valid_long, loss_matrix, return_matrix)
    if invalid_candidates:
        invalid_summary = pd.DataFrame(invalid_candidates).set_index("candidate")
        invalid_summary["is_deployed"] = False
        invalid_summary["invalid_reason"] = invalid_summary["reason"]
        summary["invalid_reason"] = ""
        summary = pd.concat([summary, invalid_summary], axis=0, sort=False)
    else:
        summary["invalid_reason"] = ""
    report["attempted_candidate_count"] = int(len(candidates))
    report["valid_candidate_count"] = int(len(candidates) - len(invalid_candidates))
    report["invalid_candidate_count"] = int(len(invalid_candidates))
    report["invalid_candidates"] = invalid_candidates
    report["deployed_reproduction"] = deployed_reproduction
    deployed = summary.loc["sjm_j3_k4_w0.1"]
    no_sjm = summary.loc["no_sjm"]
    report["no_sjm_vs_deployed"] = {
        "cal_mean_brier_delta": float(no_sjm["cal_mean_brier"] - deployed["cal_mean_brier"]),
        "locked_mean_brier_delta": float(
            no_sjm["locked_mean_brier"] - deployed["locked_mean_brier"]
        ),
        "locked_soft_sharpe_delta": float(
            no_sjm["locked_Soft_Sharpe"] - deployed["locked_Soft_Sharpe"]
        ),
        "locked_proposed_sharpe_delta": float(
            no_sjm["locked_Proposed_Sharpe"] - deployed["locked_Proposed_Sharpe"]
        ),
        "bootstrap_probability_no_sjm_brier_better": float(
            no_sjm["locked_boot_probability_brier_better_than_deployed"]
        ),
        "bootstrap_probability_no_sjm_soft_sharpe_better": float(
            no_sjm["locked_boot_probability_sharpe_better_than_deployed"]
        ),
    }
    assert deployed_reproduction["observations"] > 0
    assert deployed_reproduction["max_absolute_growth_probability_difference"] < 1e-12
    assert deployed_reproduction["max_absolute_inflation_probability_difference"] < 1e-12
    return long, summary, report


def main() -> None:
    returns, _ = load_monthly_asset_returns(False)
    signals = pd.read_csv(RESULTS / "regime_signals.csv", index_col=0)
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    defensive = pd.read_csv(RESULTS / "proposed_backtest.csv", index_col=0)
    defensive.index = pd.PeriodIndex(defensive.index, freq="M")

    cached_logistic_report = (
        json.loads(LOGISTIC_REPORT_PATH.read_text(encoding="utf-8"))
        if LOGISTIC_REPORT_PATH.exists()
        else {}
    )
    if (
        LOGISTIC_LONG_PATH.exists()
        and LOGISTIC_SUMMARY_PATH.exists()
        and cached_logistic_report.get("schema_version") == REPORT_SCHEMA_VERSION
    ):
        logistic_long = pd.read_csv(LOGISTIC_LONG_PATH)
        logistic_summary = pd.read_csv(LOGISTIC_SUMMARY_PATH).set_index("candidate")
        logistic_report = cached_logistic_report
        print("logistic robustness cache reused", logistic_long.shape, flush=True)
    else:
        logistic_long, logistic_summary, logistic_report = run_logistic_robustness(
            returns, signals, defensive
        )
        logistic_long.to_csv(LOGISTIC_LONG_PATH, index=False)
        logistic_summary.reset_index().to_csv(LOGISTIC_SUMMARY_PATH, index=False)
        logistic_report["schema_version"] = REPORT_SCHEMA_VERSION
        LOGISTIC_REPORT_PATH.write_text(
            json.dumps(logistic_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    sjm_long, sjm_summary, sjm_report = run_sjm_robustness(returns)
    sjm_long.to_csv(SJM_LONG_PATH, index=False)
    sjm_summary.reset_index().to_csv(SJM_SUMMARY_PATH, index=False)
    SJM_REPORT_PATH.write_text(
        json.dumps(sjm_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "method": {
            "selection_boundary": "all candidate ranking and CSCV/PBO use data through 2017-12 only",
            "locked_boundary": "2018-01 onward is used only for rank stability and failure audits",
            "bootstrap": "paired six-month moving-block bootstrap, 2,000 simulations",
            "cscv": "8 contiguous blocks and all 4-vs-4 combinations, 70 splits",
            "scope": "logistic medium-horizon allocation and SJM macro/regime layers; the final daily VKOSPI overlay is held outside this hyperparameter audit",
        },
        "logistic": logistic_report,
        "sjm": sjm_report,
        "conclusion": "Hyperparameter sensitivity, rank reversal, and CSCV/PBO quantify selection risk; they cannot prove the absence of overfitting. Locked data was not used to retune the deployed strategy.",
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("saved", LOGISTIC_LONG_PATH.name, logistic_long.shape, flush=True)
    print("saved", LOGISTIC_SUMMARY_PATH.name, logistic_summary.shape, flush=True)
    print("saved", SJM_LONG_PATH.name, sjm_long.shape, flush=True)
    print("saved", SJM_SUMMARY_PATH.name, sjm_summary.shape, flush=True)
    print("saved", REPORT_PATH.name, flush=True)


if __name__ == "__main__":
    main()
