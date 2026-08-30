from __future__ import annotations

import json
import math
import sqlite3
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)

ASSETS = ["KODEX200", "BOND", "GLD", "USO"]
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw_data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def get_path(directory: Path, filename: str) -> Path:
    target = unicodedata.normalize("NFC", filename)
    for path in directory.iterdir():
        if unicodedata.normalize("NFC", path.name) == target:
            return path
    raise FileNotFoundError(filename)


def rolling_zscore(series: pd.Series, window: int, clip: float = 3.0) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=1).replace(0, np.nan)
    return ((series - mean) / std).clip(-clip, clip)


def load_macro_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    gdp = pd.read_excel(get_path(RAW_DIR, "GDP 성장률.xlsx"), index_col=0, skiprows=6)
    gdp.columns = ["QoQ", "YoY"]
    gdp.index = pd.PeriodIndex(gdp.index, freq="Q").asfreq("M", how="end").to_timestamp("M") + pd.offsets.MonthEnd(1)
    gdp = gdp.resample("ME").ffill()

    trade = pd.read_excel(get_path(RAW_DIR, "수출입 총괄_20260816.xlsx"), index_col=0, skiprows=4)
    trade = trade[["수출 금액", "수입금액"]].iloc[1:].copy()
    for col in trade.columns:
        trade[col] = trade[col].astype(str).str.replace(",", "", regex=False).astype(float)
    trade.index = pd.to_datetime(trade.index, format="%Y.%m") + pd.offsets.MonthEnd(1)
    trade["Export_YoY"] = trade["수출 금액"].pct_change(12) * 100

    bsi = pd.read_csv(get_path(RAW_DIR, "기업경기조사(전망).csv"), encoding="cp949")
    bsi = bsi[(bsi["업종코드별"] == "제 조 업") & (bsi["BSI코드별"] == "업황전망BSI 1)")]
    bsi = bsi.iloc[:, 2:4].copy()
    bsi["시점"] = bsi["시점"].str.replace("월", "", regex=False).str.replace(" ", "", regex=False)
    bsi["시점"] = pd.to_datetime(bsi["시점"], format="%Y.%m") + pd.offsets.MonthEnd(1)
    bsi = bsi.set_index("시점")
    bsi.columns = ["BSI"]

    cpi = pd.read_excel(get_path(RAW_DIR, "소비자물가 상승률.xlsx"), index_col=0, skiprows=6)
    cpi.columns = ["CPI_QoQ", "CPI_YoY"]
    cpi.index = pd.to_datetime(cpi.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

    ppi = pd.read_excel(get_path(RAW_DIR, "생산자물가 상승률.xlsx"), index_col=0, skiprows=6)
    ppi.columns = ["PPI_QoQ", "PPI_YoY"]
    ppi.index = pd.to_datetime(ppi.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

    prices = pd.read_excel(get_path(RAW_DIR, "수출입물가 상승률.xlsx"), index_col=0, skiprows=6)
    prices.columns = ["ExportPrice_YoY", "ImportPrice_YoY"]
    prices.index = pd.to_datetime(prices.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

    core = pd.concat(
        {
            "GDP": rolling_zscore(gdp["YoY"], 72),
            "Export": rolling_zscore(trade["Export_YoY"], 36),
            "BSI": rolling_zscore(bsi["BSI"], 24),
            "CPI": rolling_zscore(cpi["CPI_YoY"], 36),
            "PPI": rolling_zscore(ppi["PPI_YoY"], 36),
            "ImportPrice": rolling_zscore(prices["ImportPrice_YoY"], 36),
        },
        axis=1,
    ).sort_index()

    # Levels describe the phase; 3-month changes help identify turning points.
    growth = core[["GDP", "Export", "BSI"]].copy()
    growth.columns = ["GDP_level", "Export_level", "BSI_level"]
    growth = pd.concat([growth, growth.diff(3).add_suffix("_d3")], axis=1)
    inflation = core[["CPI", "PPI", "ImportPrice"]].copy()
    inflation.columns = ["CPI_level", "PPI_level", "ImportPrice_level"]
    inflation = pd.concat([inflation, inflation.diff(3).add_suffix("_d3")], axis=1)
    features = pd.concat({"growth": growth, "inflation": inflation}, axis=1).dropna()
    return features, core


def download_market_cache(refresh: bool = False) -> pd.DataFrame:
    cache = CACHE_DIR / "market_daily.csv"
    if cache.exists() and not refresh:
        out = pd.read_csv(cache, parse_dates=["date"])
        return out

    import yfinance as yf

    rows: list[pd.DataFrame] = []
    for ticker, symbol in [("069500.KS", "KODEX200"), ("GLD", "GLD"), ("USO", "USO"), ("KRW=X", "USDKRW")]:
        data = yf.download(ticker, start="2000-01-01", auto_adjust=(symbol != "USDKRW"), progress=False, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index().rename(columns={"Date": "date", "Open": "open", "Close": "close"})
        data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_localize(None).dt.normalize()
        data["symbol"] = symbol
        rows.append(data[["date", "symbol", "open", "close"]])
    out = pd.concat(rows, ignore_index=True).dropna(subset=["date", "close"])
    out.to_csv(cache, index=False)
    return out


def load_monthly_asset_returns(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = download_market_cache(refresh)

    with sqlite3.connect(get_path(RAW_DIR, "compass.db")) as con:
        proxy = pd.read_sql(
            "select date, open, close from etf_prices where symbol = ? order by date",
            con,
            params=("1028",),
        )
    proxy["date"] = pd.to_datetime(proxy["date"])
    proxy[["open", "close"]] = proxy[["open", "close"]].apply(pd.to_numeric, errors="coerce")

    actual = market[market["symbol"] == "KODEX200"].copy().dropna(subset=["open"])
    # Yahoo contains a sparse early fragment followed by a long gap.  Use the
    # continuous KOSPI200 proxy through March 2009, then splice the ETF series.
    actual = actual[actual["date"] > pd.Timestamp("2009-03-31")]
    first_actual = actual["date"].min()
    actual_anchor = actual.loc[actual["date"] == first_actual, "open"].iloc[0]
    proxy_anchor = proxy.loc[proxy["date"] == first_actual, "open"]
    if proxy_anchor.empty:
        nearest = proxy.iloc[(proxy["date"] - first_actual).abs().argsort()[:1]]
        proxy_anchor_value = float(nearest["open"].iloc[0])
    else:
        proxy_anchor_value = float(proxy_anchor.iloc[0])
    proxy["open"] = proxy["open"] * float(actual_anchor) / proxy_anchor_value
    proxy["close"] = proxy["close"] * float(actual_anchor) / proxy_anchor_value
    proxy = proxy[proxy["date"] < first_actual]
    proxy["symbol"] = "KODEX200"
    kodex = pd.concat([proxy[["date", "symbol", "open", "close"]], actual], ignore_index=True)

    bond = pd.read_csv(get_path(RAW_DIR, "krx_bond_index.csv"), encoding="cp949")
    bond["date"] = pd.to_datetime(bond.iloc[:, 0])
    bond["open"] = bond.iloc[:, 1].astype(str).str.replace(",", "", regex=False).astype(float)
    bond["close"] = bond["open"]
    bond["symbol"] = "BOND"

    fx = market[market["symbol"] == "USDKRW"].set_index("date")["close"].sort_index()
    fx = fx.reindex(pd.date_range(fx.index.min(), fx.index.max(), freq="D")).ffill()

    first_open: dict[str, pd.Series] = {}
    trade_dates: dict[str, pd.Series] = {}
    for symbol, data in {
        "KODEX200": kodex,
        "BOND": bond,
        "GLD": market[market["symbol"] == "GLD"],
        "USO": market[market["symbol"] == "USO"],
    }.items():
        temp = data.dropna(subset=["open"]).sort_values("date").copy()
        temp["month"] = temp["date"].dt.to_period("M")
        first = temp.groupby("month", sort=True).first()
        value = first["open"].astype(float)
        if symbol in {"GLD", "USO"}:
            value = value * fx.reindex(pd.DatetimeIndex(first["date"]), method="ffill").to_numpy()
        first_open[symbol] = value
        trade_dates[symbol] = first["date"]

    levels = pd.concat(first_open, axis=1).sort_index()
    returns = levels.shift(-1).div(levels).sub(1.0).dropna(how="any")
    returns = returns[ASSETS]
    return returns, levels[ASSETS]


def _softmax(values: np.ndarray) -> np.ndarray:
    z = values - np.max(values)
    exp = np.exp(z)
    return exp / exp.sum()


class SparseJump2:
    """Small, transparent two-state sparse jump model.

    It alternates between sparse feature weighting/centroid estimation and a
    dynamic-programming state sequence with an explicit switching penalty.
    """

    def __init__(self, jump_penalty: float = 3.0, keep_features: int = 4, max_iter: int = 30):
        self.jump_penalty = float(jump_penalty)
        self.keep_features = int(keep_features)
        self.max_iter = int(max_iter)

    @staticmethod
    def _dp(dist: np.ndarray, jump: float) -> tuple[np.ndarray, np.ndarray]:
        n = len(dist)
        costs = np.zeros((n, 2))
        back = np.zeros((n, 2), dtype=int)
        costs[0] = dist[0]
        for t in range(1, n):
            for k in range(2):
                candidates = costs[t - 1] + jump * (np.arange(2) != k)
                back[t, k] = int(np.argmin(candidates))
                costs[t, k] = dist[t, k] + candidates[back[t, k]]
        states = np.zeros(n, dtype=int)
        states[-1] = int(np.argmin(costs[-1]))
        for t in range(n - 2, -1, -1):
            states[t] = back[t + 1, states[t + 1]]
        return states, costs

    def fit_predict_high(self, frame: pd.DataFrame) -> tuple[float, dict]:
        x_raw = frame.to_numpy(dtype=float)
        med = np.nanmedian(x_raw, axis=0)
        scale = np.nanpercentile(x_raw, 75, axis=0) - np.nanpercentile(x_raw, 25, axis=0)
        scale = np.where(scale < 0.15, np.nanstd(x_raw, axis=0), scale)
        scale = np.where(scale < 1e-6, 1.0, scale)
        x = np.clip((x_raw - med) / scale, -5, 5)

        score = np.nanmean(x[:, : min(3, x.shape[1])], axis=1)
        states = (score > np.nanmedian(score)).astype(int)
        weights = np.ones(x.shape[1]) / x.shape[1]
        for _ in range(self.max_iter):
            old = states.copy()
            centers = np.vstack([
                x[states == k].mean(axis=0) if np.any(states == k) else np.nanmean(x, axis=0)
                for k in range(2)
            ])
            within = np.vstack([
                np.nanvar(x[states == k], axis=0) if np.sum(states == k) > 1 else np.ones(x.shape[1])
                for k in range(2)
            ]).mean(axis=0)
            separation = (centers[1] - centers[0]) ** 2 / (within + 0.20)
            keep = np.argsort(separation)[-min(self.keep_features, len(separation)) :]
            weights = np.zeros_like(separation)
            weights[keep] = np.maximum(separation[keep], 1e-4)
            weights /= weights.sum()
            dist = np.stack([((x - centers[k]) ** 2 * weights).sum(axis=1) for k in range(2)], axis=1)
            states, costs = self._dp(dist, self.jump_penalty)
            if np.array_equal(states, old):
                break

        centers = np.vstack([x[states == k].mean(axis=0) for k in range(2)])
        high_state = int(np.argmax(centers[:, : min(3, x.shape[1])].mean(axis=1)))
        prev_state = int(states[-2]) if len(states) > 1 else int(states[-1])
        local_dist = np.array([((x[-1] - centers[k]) ** 2 * weights).sum() for k in range(2)])
        local_cost = local_dist + self.jump_penalty * 0.55 * (np.arange(2) != prev_state)
        probs = _softmax(-local_cost / 0.85)
        p_high = float(np.clip(probs[high_state], 0.03, 0.97))
        detail = {
            "p_high": p_high,
            "state": int(states[-1]),
            "high_state": high_state,
            "switches": int(np.sum(states[1:] != states[:-1])),
            "feature_weights": dict(zip(frame.columns, weights)),
        }
        return p_high, detail


def compute_regime_signals(features: pd.DataFrame, returns: pd.DataFrame, jump_penalty: float = 3.0, min_history: int = 24) -> pd.DataFrame:
    rows = []
    model = SparseJump2(jump_penalty=jump_penalty, keep_features=4)
    pg_prev = 0.5
    pi_prev = 0.5
    for target_month in returns.index:
        signal_month = target_month - 1
        hist = features.loc[: signal_month.to_timestamp("M")]
        if len(hist) < min_history:
            continue
        pg_sjm, gd = model.fit_predict_high(hist["growth"])
        pi_sjm, id_ = model.fit_predict_high(hist["inflation"])
        growth_now = float(hist["growth"].iloc[-1][["GDP_level", "Export_level", "BSI_level"]].mean())
        growth_mom = float(hist["growth"].iloc[-1][["GDP_level_d3", "Export_level_d3", "BSI_level_d3"]].mean())
        inflation_now = float(hist["inflation"].iloc[-1][["CPI_level", "PPI_level", "ImportPrice_level"]].mean())
        inflation_mom = float(hist["inflation"].iloc[-1][["CPI_level_d3", "PPI_level_d3", "ImportPrice_level_d3"]].mean())
        pg_composite = float(expit((growth_now + 0.20 * growth_mom) / 0.55))
        pi_composite = float(expit((inflation_now + 0.20 * inflation_mom) / 0.55))
        # The transparent composite is the primary forecast because the sample
        # is small; SJM contributes sparse selection and switch persistence.
        pg_raw = 0.10 * pg_sjm + 0.90 * pg_composite
        pi_raw = 0.10 * pi_sjm + 0.90 * pi_composite
        pg = 0.85 * pg_raw + 0.15 * pg_prev
        pi = 0.85 * pi_raw + 0.15 * pi_prev
        pg_prev, pi_prev = pg, pi
        probs = {
            "Goldilocks": pg * (1 - pi),
            "Overheating": pg * pi,
            "Slowdown": (1 - pg) * (1 - pi),
            "Stagflation": (1 - pg) * pi,
        }
        regime = max(probs, key=probs.get)
        rows.append(
            {
                "target_month": target_month,
                "signal_month": signal_month,
                "p_growth_high": pg,
                "p_inflation_high": pi,
                "p_growth_sjm": pg_sjm,
                "p_inflation_sjm": pi_sjm,
                "growth_composite": growth_now,
                "inflation_composite": inflation_now,
                "regime": regime,
                **{f"p_{k}": v for k, v in probs.items()},
                "growth_switches": gd["switches"],
                "inflation_switches": id_["switches"],
                "growth_features": json.dumps(gd["feature_weights"], ensure_ascii=False),
                "inflation_features": json.dumps(id_["feature_weights"], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).set_index("target_month")


REGIME_ANCHORS = pd.DataFrame(
    {
        "Goldilocks": [0.58, 0.22, 0.15, 0.05],
        "Overheating": [0.30, 0.12, 0.23, 0.35],
        "Slowdown": [0.12, 0.66, 0.20, 0.02],
        "Stagflation": [0.08, 0.24, 0.50, 0.18],
    },
    index=ASSETS,
).T
DEFENSIVE = np.array([0.05, 0.72, 0.23, 0.00])
STRATEGIC = np.array([0.20, 0.45, 0.30, 0.05])


def soft_anchor(signal: pd.Series) -> np.ndarray:
    p = np.array([signal[f"p_{r}"] for r in REGIME_ANCHORS.index])
    return p @ REGIME_ANCHORS.to_numpy()


def ewma_cov(history: pd.DataFrame, half_life: float = 12.0, leverage: float = 1.0) -> np.ndarray:
    x = history[ASSETS].to_numpy(dtype=float)
    if len(x) < 12:
        return np.cov(x, rowvar=False) + np.eye(len(ASSETS)) * 1e-6
    alpha = 1 - math.exp(math.log(0.5) / half_life)
    cov = np.cov(x[: min(24, len(x))], rowvar=False)
    mean = np.nanmean(x, axis=0)
    for row in x:
        shock = row - mean
        multiplier = 1.0 + leverage * min(max(-row[0], 0.0) / 0.08, 1.5)
        cov = (1 - alpha) * cov + alpha * multiplier * np.outer(shock, shock)
    return cov + np.eye(len(ASSETS)) * 1e-7


def cdar(returns: np.ndarray, alpha: float = 0.90) -> float:
    wealth = np.cumprod(1 + returns)
    dd = wealth / np.maximum.accumulate(np.r_[1.0, wealth])[-len(wealth):] - 1.0
    k = max(1, int(math.ceil((1 - alpha) * len(dd))))
    return float(np.mean(np.sort(dd)[:k]))


@dataclass
class StrategyConfig:
    name: str = "Proposed"
    target_vol: float = 0.08
    half_life: float = 12.0
    invvol_tilt: float = 0.35
    return_reward: float = 1.15
    vol_penalty: float = 0.18
    cdar_penalty: float = 0.25
    turnover_penalty: float = 0.05
    tracking_penalty: float = 0.32
    max_cdar: float = 0.16
    drawdown_guard: float = 0.75
    regime_strength: float = 0.75
    use_regime: bool = True
    use_risk_control: bool = True


def controlled_weights(
    signal: pd.Series,
    history: pd.DataFrame,
    pretrade: np.ndarray,
    current_dd: float,
    cfg: StrategyConfig,
) -> np.ndarray:
    anchor = soft_anchor(signal) if cfg.use_regime else STRATEGIC.copy()
    anchor = cfg.regime_strength * anchor + (1 - cfg.regime_strength) * STRATEGIC
    if not cfg.use_risk_control or len(history) < 24:
        return anchor

    cov = ewma_cov(history.tail(84), cfg.half_life, leverage=1.0)
    vols = np.sqrt(np.diag(cov)).clip(0.005, None)
    tilted = anchor * (np.median(vols) / vols) ** cfg.invvol_tilt
    tilted = tilted / tilted.sum()
    prior = 0.55 * anchor + 0.45 * tilted

    hist = history.tail(84)[ASSETS]
    long_mu = history[ASSETS].expanding(min_periods=24).mean().iloc[-1].to_numpy()
    recent_mu = hist.ewm(halflife=24, adjust=False).mean().iloc[-1].to_numpy()
    mu = 0.80 * long_mu + 0.20 * recent_mu
    mu = np.clip(mu, -0.006, 0.015)
    target = cfg.target_vol * (0.86 + 0.20 * float(signal["p_growth_high"]))

    def objective(w: np.ndarray) -> float:
        ann_return = 12 * float(w @ mu)
        ann_vol = math.sqrt(max(float(w @ cov @ w), 0.0) * 12)
        path_cdar = abs(cdar(hist.to_numpy() @ w, 0.90))
        turnover = 0.5 * np.sum(np.sqrt((w - pretrade) ** 2 + 1e-6))
        tracking = float(np.sum((w - prior) ** 2))
        return (
            -cfg.return_reward * ann_return
            + cfg.vol_penalty * ann_vol
            + cfg.cdar_penalty * path_cdar
            + cfg.turnover_penalty * turnover
            + cfg.tracking_penalty * tracking
        )

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: target - math.sqrt(max(float(w @ cov @ w), 0.0) * 12)},
        {"type": "ineq", "fun": lambda w: cfg.max_cdar + cdar(hist.to_numpy() @ w, 0.90)},
    ]
    bounds = [(0.02, 0.68), (0.05, 0.88), (0.02, 0.62), (0.0, 0.38)]
    result = minimize(objective, prior, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 80, "ftol": 1e-8})
    w = result.x if result.success and np.isfinite(result.x).all() else prior
    w = np.clip(w, 0, None)
    w /= w.sum()

    # MPC-style state-dependent risk aversion: react to realized drawdown, but
    # retain a floor in risky assets so recovery participation is not lost.
    if current_dd < -0.05:
        severity = min(max((-current_dd - 0.05) / 0.12, 0.0), 1.0)
        blend = cfg.drawdown_guard * (0.25 + 0.50 * severity)
        w = (1 - blend) * w + blend * DEFENSIVE
    return w / w.sum()


def hard_regime_weights(signal: pd.Series) -> np.ndarray:
    mapping = {
        "Goldilocks": np.array([1.0, 0.0, 0.0, 0.0]),
        "Overheating": np.array([0.0, 0.0, 0.0, 1.0]),
        "Slowdown": np.array([0.6, 0.4, 0.0, 0.0]),
        "Stagflation": np.array([0.0, 0.0, 1.0, 0.0]),
    }
    return mapping[signal["regime"]]


def run_backtest(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: StrategyConfig,
    mode: str = "proposed",
    start: str | None = None,
    end: str | None = None,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    months = signals.index.intersection(returns.index)
    if start:
        months = months[months >= pd.Period(start, "M")]
    if end:
        months = months[months <= pd.Period(end, "M")]
    rows = []
    pretrade = np.zeros(4)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        signal = signals.loc[month]
        history = returns.loc[returns.index < month]
        current_dd = nav / peak - 1.0
        if mode == "proposed":
            w = controlled_weights(signal, history, pretrade, current_dd, cfg)
        elif mode == "soft":
            w = soft_anchor(signal)
        elif mode == "hard":
            w = hard_regime_weights(signal)
        elif mode == "equal":
            w = np.full(4, 0.25)
        elif mode == "static_defensive":
            w = np.array([0.20, 0.45, 0.30, 0.05])
        elif mode == "kodex":
            w = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError(mode)

        delta = w - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = abs((w[2] + w[3]) - (pretrade[2] + pretrade[3])) * 0.0005 * cost_multiplier
        gross_return = float(w @ returns.loc[month, ASSETS].to_numpy())
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        end_w = w * (1 + returns.loc[month, ASSETS].to_numpy()) / (1 + gross_return)
        rows.append(
            {
                "month": month,
                "signal_month": signal["signal_month"],
                "regime": signal["regime"],
                "p_growth_high": signal["p_growth_high"],
                "p_inflation_high": signal["p_inflation_high"],
                "gross_return": gross_return,
                "return": net_return,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "nav": nav,
                "drawdown": nav / peak - 1,
                **{f"w_{a}": w[i] for i, a in enumerate(ASSETS)},
            }
        )
        pretrade = end_w
        first_trade = False
    return pd.DataFrame(rows).set_index("month")


def performance_summary(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).dropna()
    wealth = (1 + r).cumprod()
    years = len(r) / 12
    cagr = wealth.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=1) * math.sqrt(12)
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(12) if r.std(ddof=1) > 0 else np.nan
    dd = wealth / wealth.cummax() - 1
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * math.sqrt(12)
    sortino = r.mean() * 12 / downside if downside > 0 else np.nan
    return pd.Series(
        {
            "Months": len(r),
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MDD": mdd,
            "Calmar": calmar,
            "FinalMultiple": wealth.iloc[-1],
            "PositiveMonths": (r > 0).mean(),
        }
    )


def evaluate_regimes(signals: pd.DataFrame, core: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    composite = pd.DataFrame(
        {
            "growth_realized": core[["GDP", "Export", "BSI"]].mean(axis=1),
            "inflation_realized": core[["CPI", "PPI", "ImportPrice"]].mean(axis=1),
        }
    )
    # Ex-post target: average of the next three published monthly readings.
    future = pd.concat([composite.shift(-k) for k in (1, 2, 3)], axis=1)
    future.columns = pd.MultiIndex.from_product([[1, 2, 3], composite.columns])
    target = pd.DataFrame(index=composite.index)
    target["growth_high_realized"] = future.xs("growth_realized", axis=1, level=1).mean(axis=1) >= 0
    target["inflation_high_realized"] = future.xs("inflation_realized", axis=1, level=1).mean(axis=1) >= 0
    pred = signals.copy()
    pred.index = pred["signal_month"].apply(lambda x: x.to_timestamp("M"))
    joined = pred.join(target, how="inner").dropna(subset=["growth_high_realized", "inflation_high_realized"])
    joined["growth_pred"] = joined["p_growth_high"] >= 0.5
    joined["inflation_pred"] = joined["p_inflation_high"] >= 0.5
    joined["quadrant_hit"] = (joined["growth_pred"] == joined["growth_high_realized"]) & (joined["inflation_pred"] == joined["inflation_high_realized"])
    metrics = {
        "growth_balanced_accuracy": balanced_accuracy_score(joined["growth_high_realized"], joined["growth_pred"]),
        "inflation_balanced_accuracy": balanced_accuracy_score(joined["inflation_high_realized"], joined["inflation_pred"]),
        "quadrant_accuracy": float(joined["quadrant_hit"].mean()),
        "n_months": len(joined),
        "growth_confusion": confusion_matrix(joined["growth_high_realized"], joined["growth_pred"]).tolist(),
        "inflation_confusion": confusion_matrix(joined["inflation_high_realized"], joined["inflation_pred"]).tolist(),
    }
    return joined, metrics


def main(refresh: bool = False) -> None:
    features, core = load_macro_data()
    returns, levels = load_monthly_asset_returns(refresh=refresh)
    print("macro", features.index.min(), features.index.max(), features.shape)
    print("assets", returns.index.min(), returns.index.max(), returns.shape)
    signals = compute_regime_signals(features, returns)
    print("signals", signals.index.min(), signals.index.max(), signals.shape)

    cfg = StrategyConfig()
    modes = ["proposed", "soft", "hard", "equal", "static_defensive", "kodex"]
    tests = {mode: run_backtest(returns, signals, cfg, mode=mode) for mode in modes}
    summary = pd.DataFrame({mode: performance_summary(bt["return"]) for mode, bt in tests.items()}).T
    print(summary.round(4))
    _, regime_metrics = evaluate_regimes(signals, core)
    print(regime_metrics)

    summary.to_csv(RESULTS_DIR / "summary.csv")
    signals.to_csv(RESULTS_DIR / "regime_signals.csv")
    tests["proposed"].to_csv(RESULTS_DIR / "proposed_backtest.csv")
    with (RESULTS_DIR / "regime_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(regime_metrics, f, ensure_ascii=False, indent=2)
    with (RESULTS_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(refresh=False)
