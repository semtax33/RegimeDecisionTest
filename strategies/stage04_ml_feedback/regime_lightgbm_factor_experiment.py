from __future__ import annotations

import argparse
import json
import math
import sqlite3
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler

from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    compute_regime_signals,
    hard_regime_weights,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CAL_END = pd.Period("2017-12", "M")
TEST_START = pd.Period("2018-01", "M")
FACTOR_TEST_START = pd.Period("2005-01", "M")
HORIZON = 10

TICKERS = {
    "KODEX200": "069500.KS",
    "KOSPI200": "^KS200",
    "USDKRW": "KRW=X",
    "SPY": "SPY",
    "SOX": "^SOX",
    "BTC": "BTC-USD",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "HYG": "HYG",
    "LQD": "LQD",
    "TLT": "TLT",
    "SHY": "SHY",
    "GLD": "GLD",
    "UUP": "UUP",
    "USO": "USO",
}


@dataclass(frozen=True)
class FactorBlendConfig:
    max_shift: float
    probability_scale: float = 0.15
    hard_fraction: float = 0.40
    leverage: float = 1.20
    financing_rate: float = 0.04

    @property
    def name(self) -> str:
        return f"regime_lgbm_shift_{self.max_shift:.3f}"


def _normalise_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None).normalize()
    wanted = ["open", "high", "low", "close", "volume"]
    for col in wanted:
        if col not in frame:
            frame[col] = np.nan
    return frame[wanted].apply(pd.to_numeric, errors="coerce")


def download_ohlcv(refresh: bool = False) -> pd.DataFrame:
    cache_path = CACHE / "regime_lightgbm_ohlcv.csv"
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if set(TICKERS).issubset(set(cached["symbol"].unique())):
            return cached

    import yfinance as yf

    raw = yf.download(
        list(TICKERS.values()),
        start="2004-01-01",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=False,
    )
    rows: list[pd.DataFrame] = []
    if not isinstance(raw.columns, pd.MultiIndex):
        if len(TICKERS) != 1:
            raise RuntimeError("Unexpected yfinance column layout")
        symbol, ticker = next(iter(TICKERS.items()))
        frame = _normalise_yfinance_frame(raw)
        frame["symbol"] = symbol
        rows.append(frame.reset_index(names="date"))
    else:
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        for symbol, ticker in TICKERS.items():
            try:
                if ticker in level0:
                    frame = raw[ticker]
                elif ticker in level1:
                    frame = raw.xs(ticker, axis=1, level=1)
                else:
                    continue
                frame = _normalise_yfinance_frame(frame).dropna(how="all")
                frame["symbol"] = symbol
                rows.append(frame.reset_index(names="date"))
            except (KeyError, ValueError):
                continue
    if not rows:
        raise RuntimeError("No market data downloaded")
    out = pd.concat(rows, ignore_index=True)
    missing = sorted(set(TICKERS) - set(out["symbol"].unique()))
    if missing:
        raise RuntimeError(f"Required downloads missing: {missing}")
    out.to_csv(cache_path, index=False)
    return out


def market_frames(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol, data in raw.groupby("symbol"):
        frame = data.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frames[str(symbol)] = frame
    return frames


def splice_kospi200_proxy(actual: pd.DataFrame) -> pd.DataFrame:
    """Extend KODEX200 backward with the repository's KOSPI200 proxy.

    The same proxy is already used by regime_research.load_monthly_asset_returns.
    It provides a continuous OHLC history from 2000, allowing 2000-2004 to be
    an initial training window and 2005 onward to be genuinely out of sample.
    """
    with sqlite3.connect(ROOT / "raw_data" / "compass.db") as connection:
        proxy = pd.read_sql(
            "select date, open, high, low, close, volume from etf_prices where symbol = ? order by date",
            connection,
            params=("1028",),
        )
    proxy["date"] = pd.to_datetime(proxy["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        proxy[col] = pd.to_numeric(proxy[col], errors="coerce")
    proxy = proxy.set_index("date").sort_index()

    continuous_actual = actual.loc[actual.index > pd.Timestamp("2009-03-31")].dropna(subset=["close"]).copy()
    first_actual = continuous_actual.index.min()
    if pd.isna(first_actual):
        raise RuntimeError("Continuous KODEX200 history after 2009-03 is unavailable")
    nearest_proxy = proxy.iloc[proxy.index.get_indexer([first_actual], method="nearest")]
    scale = float(continuous_actual.loc[first_actual, "close"] / nearest_proxy["close"].iloc[0])
    for col in ["open", "high", "low", "close"]:
        proxy[col] *= scale
    # Index volume and ETF volume are not comparable. Leaving proxy volume
    # missing prevents a false structural break in the volume indicators.
    proxy["volume"] = np.nan
    proxy = proxy.loc[proxy.index < first_actual]
    return pd.concat([proxy, continuous_actual]).sort_index()


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def _rsi(close: pd.Series, window: int) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = _safe_div(gain, loss)
    return 100 - 100 / (1 + rs)


def _rolling_beta(target: pd.Series, market: pd.Series, window: int) -> pd.Series:
    covariance = target.rolling(window, min_periods=window).cov(market)
    variance = market.rolling(window, min_periods=window).var(ddof=1)
    return _safe_div(covariance, variance)


def _consecutive_count(mask: pd.Series) -> pd.Series:
    groups = (~mask.fillna(False)).cumsum()
    return mask.fillna(False).astype(int).groupby(groups).cumsum().astype(float)


def causal_hmm_features(frames: dict[str, pd.DataFrame], market: str) -> pd.DataFrame:
    if market == "kr":
        benchmark = frames["KOSPI200"]["close"].dropna()
        benchmark_return = benchmark.pct_change()
        usdkrw = frames["USDKRW"]["close"].reindex(benchmark.index).ffill(limit=5)
        obs = pd.DataFrame(
            {
                "market_ret20": benchmark.pct_change(20),
                "market_vol20": benchmark_return.rolling(20).std(ddof=1) * math.sqrt(252),
                "usdkrw_ret20": usdkrw.pct_change(20),
                "positive_fraction20": (benchmark_return > 0).rolling(20).mean(),
            }
        ).dropna()
        prefix = "kr_hmm"
    elif market == "us":
        benchmark = frames["SPY"]["close"].dropna()
        benchmark_return = benchmark.pct_change()
        vix = frames["VIX"]["close"].reindex(benchmark.index).ffill(limit=5)
        obs = pd.DataFrame(
            {
                "market_ret20": benchmark.pct_change(20),
                "market_vol20": benchmark_return.rolling(20).std(ddof=1) * math.sqrt(252),
                "vix_norm": vix / 100.0,
                "positive_fraction20": (benchmark_return > 0).rolling(20).mean(),
            }
        ).dropna()
        prefix = "us_hmm"
    else:
        raise ValueError(f"Unknown HMM market: {market}")

    output_columns = [
        f"{prefix}_p_bear",
        f"{prefix}_p_sideways",
        f"{prefix}_p_bull",
        f"{prefix}_score",
        f"{prefix}_regime",
    ]
    output = pd.DataFrame(index=obs.index, columns=output_columns, dtype=float)
    minimum = 252
    refit_step = 63
    context = 120
    max_history = 756
    for block_start in range(minimum - 1, len(obs), refit_step):
        history_start = max(0, block_start + 1 - max_history)
        history = obs.iloc[history_start : block_start + 1]
        scaler = RobustScaler().fit(history)
        x_history = scaler.transform(history)
        # The source paper uses a full covariance matrix. For the Korean
        # adaptation we use diagonal covariance: four economically distinct
        # inputs remain, while repeated causal expanding fits stay tractable.
        model = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=5,
            tol=1e-2,
            min_covar=1e-3,
            random_state=20260825,
        ).fit(x_history)
        fitted_states = model.predict(x_history)

        means = []
        for state in range(3):
            state_returns = history["market_ret20"].to_numpy()[fitted_states == state]
            means.append(float(np.nanmean(state_returns)) if len(state_returns) else float(model.means_[state, 0]))
        order = np.argsort(means)
        state_to_rank = {int(state): int(rank) for rank, state in enumerate(order)}

        block_end = min(block_start + refit_step, len(obs))
        for position in range(block_start, block_end):
            start = max(0, position - context + 1)
            x_context = scaler.transform(obs.iloc[start : position + 1])
            probabilities = model.predict_proba(x_context)[-1]
            ranked = np.zeros(3)
            for state, probability in enumerate(probabilities):
                ranked[state_to_rank[state]] = probability
            regime = int(np.argmax(ranked))
            output.iloc[position] = [ranked[0], ranked[1], ranked[2], ranked @ np.array([-1.0, 0.0, 1.0]), regime]
    output[f"{prefix}_regime"] = output[f"{prefix}_regime"].astype("Int64")
    return output


def build_features(
    frames: dict[str, pd.DataFrame],
    kr_hmm: pd.DataFrame,
    us_hmm: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    target = frames["KODEX200"].dropna(subset=["close"]).copy()
    target = target.loc[target.index >= pd.Timestamp("2000-01-01")]
    close = target["close"]
    high = target["high"].fillna(close)
    low = target["low"].fillna(close)
    volume = target["volume"].replace(0, np.nan)
    returns = close.pct_change()
    features: dict[str, pd.Series] = {}

    for window in [1, 5, 10, 20, 63]:
        features[f"ret_{window}"] = close.pct_change(window)
    features["lag_ret_1"] = returns.shift(1)
    features["lag_ret_2"] = returns.shift(2)
    features["lag_ret_5"] = returns.shift(5)

    for window in [5, 10, 20, 50, 200]:
        features[f"dist_sma_{window}"] = close / close.rolling(window).mean() - 1
    for window in [12, 26]:
        features[f"dist_ema_{window}"] = close / close.ewm(span=window, adjust=False).mean() - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features["macd"] = macd / close
    features["macd_signal"] = macd_signal / close
    features["macd_hist"] = (macd - macd_signal) / close

    for window in [3, 9, 14]:
        features[f"rsi_{window}"] = _rsi(close, window) / 100.0
    lowest14 = low.rolling(14).min()
    highest14 = high.rolling(14).max()
    stochastic_k = 100 * _safe_div(close - lowest14, highest14 - lowest14)
    features["stoch_k"] = stochastic_k / 100.0
    features["stoch_d"] = stochastic_k.rolling(3).mean() / 100.0
    features["williams_r"] = -_safe_div(highest14 - close, highest14 - lowest14)

    typical = (high + low + close) / 3
    typical_mean = typical.rolling(20).mean()
    mean_deviation = typical.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    features["cci_20"] = _safe_div(typical - typical_mean, 0.015 * mean_deviation)
    for window in [5, 10, 20]:
        features[f"roc_{window}"] = close.pct_change(window)

    momentum = close.diff()
    double_smoothed = momentum.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    double_abs = momentum.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    features["tsi"] = _safe_div(double_smoothed, double_abs)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=1)
    features["bb_width"] = _safe_div(4 * bb_std, bb_mid)
    features["bb_pct_b"] = _safe_div(close - (bb_mid - 2 * bb_std), 4 * bb_std)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    features["atr_14"] = atr14 / close
    ema20 = close.ewm(span=20, adjust=False).mean()
    features["keltner_width"] = 4 * atr14 / ema20

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100 * _safe_div(plus_dm.ewm(alpha=1 / 14, adjust=False).mean(), tr14)
    minus_di = 100 * _safe_div(minus_dm.ewm(alpha=1 / 14, adjust=False).mean(), tr14)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    features["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean() / 100.0
    features["plus_di"] = plus_di / 100.0
    features["minus_di"] = minus_di / 100.0

    aroon_up = high.rolling(25).apply(lambda x: (np.argmax(x) + 1) / len(x), raw=True)
    aroon_down = low.rolling(25).apply(lambda x: (np.argmin(x) + 1) / len(x), raw=True)
    features["aroon_osc"] = aroon_up - aroon_down
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    features["ichimoku_tenkan"] = tenkan / close - 1
    features["ichimoku_kijun"] = kijun / close - 1
    features["ichimoku_cloud_width"] = (tenkan - span_b).abs() / close

    signed_volume = np.sign(returns.fillna(0)) * volume.fillna(0)
    obv = signed_volume.cumsum()
    features["obv_slope_10"] = _safe_div(obv.diff(10), volume.rolling(10).mean() * 10)
    money_flow_multiplier = _safe_div((close - low) - (high - close), high - low)
    features["cmf_20"] = _safe_div((money_flow_multiplier * volume).rolling(20).sum(), volume.rolling(20).sum())
    features["volume_ratio_20"] = volume / volume.rolling(20).mean()
    vwap20 = _safe_div((typical * volume).rolling(20).sum(), volume.rolling(20).sum())
    features["vwap_deviation"] = close / vwap20 - 1

    features["consecutive_up"] = _consecutive_count(returns > 0).clip(upper=10) / 10
    features["consecutive_down"] = _consecutive_count(returns < 0).clip(upper=10) / 10
    features["drawdown_252"] = close / close.rolling(252, min_periods=60).max() - 1
    features["dist_high_20"] = close / high.rolling(20).max() - 1
    features["dist_low_20"] = close / low.rolling(20).min() - 1
    features["dist_high_252"] = close / high.rolling(252, min_periods=60).max() - 1
    for window in [20, 63, 126, 252]:
        features[f"realized_vol_{window}"] = returns.rolling(window, min_periods=max(10, window // 2)).std(ddof=1) * math.sqrt(252)

    downside = returns.clip(upper=0)
    features["downside_dev_10"] = np.sqrt(downside.pow(2).rolling(10).mean()) * math.sqrt(252)
    for window in [20, 60]:
        downside_risk = np.sqrt(downside.pow(2).rolling(window).mean()) * math.sqrt(252)
        features[f"sortino_{window}"] = _safe_div(returns.rolling(window).mean() * 252, downside_risk)

    aligned_close: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        aligned_close[symbol] = frame["close"].reindex(close.index).ffill(limit=5)
    # A Korean close occurs before the same calendar day's US close. Lag every
    # US-market input by one Korean session so the daily training rows only use
    # information that was actually known at the KODEX200 close.
    us_symbols = {"SPY", "SOX", "BTC", "VIX", "VIX3M", "HYG", "LQD", "TLT", "SHY", "GLD", "UUP", "USO"}
    known_close = {
        symbol: series.shift(1) if symbol in us_symbols else series
        for symbol, series in aligned_close.items()
    }

    kospi200 = known_close["KOSPI200"]
    usdkrw = known_close["USDKRW"].shift(1)
    for window in [1, 5, 20, 63]:
        features[f"krw_ret_{window}"] = usdkrw.pct_change(window)
    features["kospi200_ret_5"] = kospi200.pct_change(5)
    features["kospi200_ret_20"] = kospi200.pct_change(20)
    features["corr_krw_20"] = returns.rolling(20).corr(usdkrw.pct_change())

    spy = known_close["SPY"]
    spy_return = spy.pct_change()
    features["corr_spy_20"] = returns.rolling(20).corr(spy_return)
    features["beta_spy_63"] = _rolling_beta(returns, spy_return, 63)
    features["relative_spy_20"] = close.pct_change(20) - spy.pct_change(20)
    features["relative_spy_63"] = close.pct_change(63) - spy.pct_change(63)
    for window in [5, 10, 20]:
        features[f"spy_ret_{window}"] = spy.pct_change(window)

    sox = known_close["SOX"]
    features["sox_ret_1"] = sox.pct_change()
    features["sox_ret_5"] = sox.pct_change(5)
    features["sox_ret_20"] = sox.pct_change(20)
    features["sox_relative_spy_20"] = sox.pct_change(20) - spy.pct_change(20)

    vix = known_close["VIX"]
    vix3m = known_close["VIX3M"]
    features["vix_level"] = vix / 100
    features["vix_change_5"] = vix.pct_change(5)
    features["vix_change_10"] = vix.pct_change(10)
    features["vix3m_level"] = vix3m / 100
    features["vix_term_structure"] = _safe_div(vix3m - vix, vix)

    btc = known_close["BTC"]
    btc_return = btc.pct_change()
    for window in [5, 10, 20]:
        features[f"btc_ret_{window}"] = btc.pct_change(window)
    features["btc_rsi_14"] = _rsi(btc, 14) / 100
    features["btc_vol_20"] = btc_return.rolling(20).std(ddof=1) * math.sqrt(252)
    features["corr_btc_20"] = returns.rolling(20).corr(btc_return)

    features["credit_hyg_lqd"] = known_close["HYG"] / known_close["LQD"] - 1
    features["yield_tlt_shy"] = known_close["TLT"] / known_close["SHY"] - 1
    features["gold_spy"] = known_close["GLD"] / spy - 1
    features["dollar_ret_20"] = known_close["UUP"].pct_change(20)
    features["oil_ret_20"] = known_close["USO"].pct_change(20)

    aligned_kr_hmm = kr_hmm.reindex(close.index).ffill(limit=10)
    aligned_us_hmm = us_hmm.reindex(close.index).ffill(limit=10).shift(1)
    for col in ["kr_hmm_p_bear", "kr_hmm_p_sideways", "kr_hmm_p_bull", "kr_hmm_score"]:
        features[col] = aligned_kr_hmm[col].astype(float)
    for col in ["us_hmm_p_bear", "us_hmm_p_sideways", "us_hmm_p_bull", "us_hmm_score"]:
        features[col] = aligned_us_hmm[col].astype(float)

    frame = pd.DataFrame(features, index=close.index).replace([np.inf, -np.inf], np.nan)
    return frame, close


def feature_columns_for_variant(features: pd.DataFrame, variant: str) -> list[str]:
    domestic_prefixes = ("krw_", "kospi200_", "corr_krw_", "kr_hmm_")
    global_prefixes = (
        "corr_spy_",
        "beta_spy_",
        "relative_spy_",
        "spy_ret_",
        "sox_",
        "vix_",
        "vix3m_",
        "btc_",
        "corr_btc_",
        "credit_",
        "yield_",
        "gold_",
        "dollar_",
        "oil_",
        "us_hmm_",
    )
    domestic = [col for col in features if col.startswith(domestic_prefixes)]
    global_cross = [col for col in features if col.startswith(global_prefixes)]
    technical = [col for col in features if col not in set(domestic + global_cross)]
    if variant == "kr_only":
        chosen = technical + domestic
    elif variant == "kr_plus_global":
        chosen = technical + domestic + [col for col in global_cross if not col.startswith("us_hmm_")]
    elif variant == "us_paper_original":
        chosen = technical + global_cross
    else:
        raise ValueError(f"Unknown feature variant: {variant}")
    chosen_set = set(chosen)
    return [col for col in features if col in chosen_set]


def walk_forward_lightgbm(
    features: pd.DataFrame,
    close: pd.Series,
    target_months: pd.PeriodIndex,
    feature_columns: list[str],
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features[feature_columns].copy()
    label_date = pd.Series(features.index, index=features.index).shift(-HORIZON)
    forward_return = close.shift(-HORIZON) / close - 1
    target = (forward_return > 0).astype(float).where(forward_return.notna())
    importance = pd.Series(0.0, index=features.columns)
    importance_fits = 0
    rows = []

    for number, target_month in enumerate(target_months):
        signal_month = target_month - 1
        signal_end = signal_month.to_timestamp("M")
        candidates = features.index[features.index <= signal_end]
        if candidates.empty:
            continue
        signal_date = candidates[-1]
        known = (label_date <= signal_date) & target.notna()
        train_index = features.index[known.fillna(False)]
        if len(train_index) < 504:
            continue

        x_all = features.loc[train_index]
        y_all = target.loc[train_index].astype(int)
        usable = x_all.notna().mean() >= 0.70
        usable &= features.loc[signal_date].notna() | (x_all.notna().sum() >= 504)
        columns = list(x_all.columns[usable])
        if len(columns) < 20:
            continue

        validation_size = min(252, max(63, len(x_all) // 5))
        split = len(x_all) - validation_size
        x_train = x_all.iloc[:split][columns]
        y_train = y_all.iloc[:split]
        x_valid = x_all.iloc[split:][columns]
        y_valid = y_all.iloc[split:]
        medians = x_train.median()
        scaler = RobustScaler(quantile_range=(25, 75)).fit(x_train.fillna(medians))
        train_scaled = scaler.transform(x_train.fillna(medians))
        valid_scaled = scaler.transform(x_valid.fillna(medians))
        test_scaled = scaler.transform(features.loc[[signal_date], columns].fillna(medians))

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.80,
            colsample_bytree=0.70,
            reg_alpha=0.05,
            reg_lambda=0.05,
            min_child_samples=40,
            random_state=20260825,
            n_jobs=1,
            verbosity=-1,
        )
        callbacks = [lgb.early_stopping(50, verbose=False)] if y_valid.nunique() > 1 else []
        fit_kwargs = {"callbacks": callbacks} if callbacks else {}
        if callbacks:
            fit_kwargs["eval_set"] = [(valid_scaled, y_valid)]
            fit_kwargs["eval_metric"] = "binary_logloss"
        model.fit(train_scaled, y_train, **fit_kwargs)
        probability = float(model.predict_proba(test_scaled)[:, 1][0])
        gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=columns)
        if gain.sum() > 0:
            importance.loc[columns] += gain / gain.sum()
            importance_fits += 1
        rows.append(
            {
                "target_month": target_month,
                "signal_month": signal_month,
                "signal_date": signal_date,
                "p_up": probability,
                "feature_variant": variant,
                "score": float(np.clip((probability - 0.5) / 0.15, -1, 1)),
                "realized_10d_return": float(forward_return.loc[signal_date]) if pd.notna(forward_return.loc[signal_date]) else np.nan,
                "realized_up": float(target.loc[signal_date]) if pd.notna(target.loc[signal_date]) else np.nan,
                "hmm_score": float(
                    features.loc[signal_date, "us_hmm_score"]
                    if "us_hmm_score" in features
                    else features.loc[signal_date, "kr_hmm_score"]
                ),
                "train_rows": len(x_all),
                "feature_count": len(columns),
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
            }
        )
        if (number + 1) % 24 == 0:
            print(f"LightGBM predictions: {number + 1}/{len(target_months)}")

    predictions = pd.DataFrame(rows).set_index("target_month")
    predictions.index = pd.PeriodIndex(predictions.index, freq="M")
    importance_frame = (
        (importance / max(importance_fits, 1))
        .sort_values(ascending=False)
        .rename("mean_gain_share")
        .to_frame()
        .reset_index(names="feature")
    )
    return predictions, importance_frame


def _transfer(weights: np.ndarray, donors: list[int], receivers: list[int], amount: float) -> np.ndarray:
    out = weights.copy()
    floors = np.array([0.02, 0.05, 0.02, 0.00])
    available = np.maximum(out[donors] - floors[donors], 0)
    transfer = min(float(amount), float(available.sum()))
    if transfer <= 0:
        return out
    donor_share = available / available.sum()
    out[donors] -= transfer * donor_share
    receiver_base = np.maximum(out[receivers], 0.02)
    receiver_share = receiver_base / receiver_base.sum()
    out[receivers] += transfer * receiver_share
    return out / out.sum()


def apply_factor_tilt(base: np.ndarray, score: float, max_shift: float) -> np.ndarray:
    if not np.isfinite(score) or max_shift <= 0:
        return base.copy()
    if score >= 0:
        return _transfer(base, donors=[1, 2], receivers=[0], amount=max_shift * score)
    return _transfer(base, donors=[0, 3], receivers=[1, 2], amount=max_shift * -score)


def run_factor_blend(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive_path: pd.DataFrame,
    factor: pd.DataFrame,
    cfg: FactorBlendConfig,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    months = signals.index.intersection(returns.index).intersection(defensive_path.index)
    rows = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        hard = hard_regime_weights(signals.loc[month])
        defensive = defensive_path.loc[month, [f"w_{asset}" for asset in ASSETS]].to_numpy(dtype=float)
        base = cfg.hard_fraction * hard + (1 - cfg.hard_fraction) * defensive
        probability = float(factor.loc[month, "p_up"]) if month in factor.index else 0.5
        score = float(np.clip((probability - 0.5) / cfg.probability_scale, -1, 1))
        unlevered = apply_factor_tilt(base, score, cfg.max_shift)
        asset_weights = cfg.leverage * unlevered
        debt_weight = 1.0 - cfg.leverage

        delta = asset_weights - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = abs((asset_weights[2] + asset_weights[3]) - (pretrade[2] + pretrade[3])) * 0.0005 * cost_multiplier
        financing = debt_weight * ((1 + cfg.financing_rate) ** (1 / 12) - 1)
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
                "financing_return": financing,
                "regime": signals.loc[month, "regime"],
                "lgbm_probability": probability,
                "lgbm_score": score,
                "max_shift": cfg.max_shift,
                **{f"w_{asset}": asset_weights[i] for i, asset in enumerate(ASSETS)},
            }
        )
    return pd.DataFrame(rows).set_index("month")


def paired_block_bootstrap(
    baseline: pd.Series,
    candidate: pd.Series,
    block_length: int = 6,
    simulations: int = 5000,
) -> dict[str, float]:
    common = baseline.index.intersection(candidate.index)
    matrix = np.column_stack([baseline.loc[common], candidate.loc[common]]).astype(float)
    n = len(matrix)
    rng = np.random.default_rng(20260825 + block_length)
    deltas = []
    for _ in range(simulations):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n - block_length + 1))
            indices.extend(range(start, start + block_length))
        draw = matrix[np.asarray(indices[:n])]
        sharpes = []
        for column in range(2):
            values = draw[:, column]
            sharpes.append(float(values.mean() / values.std(ddof=1) * math.sqrt(12)))
        deltas.append(sharpes[1] - sharpes[0])
    delta = np.asarray(deltas)
    return {
        "block_length": block_length,
        "simulations": simulations,
        "probability_sharpe_improves": float(np.mean(delta > 0)),
        "sharpe_delta_p05": float(np.quantile(delta, 0.05)),
        "sharpe_delta_median": float(np.median(delta)),
        "sharpe_delta_p95": float(np.quantile(delta, 0.95)),
    }


def metric_record(period: str, strategy: str, backtest: pd.DataFrame) -> dict[str, float | str]:
    metrics = performance_summary(backtest["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        **metrics.to_dict(),
        "AvgTurnover": float(backtest["turnover"].mean()),
    }


def prediction_metrics(predictions: pd.DataFrame, start: pd.Period | None = None) -> dict[str, float]:
    view = predictions.loc[start:] if start is not None else predictions
    view = view.dropna(subset=["p_up", "realized_up"])
    if view.empty:
        return {}
    y = view["realized_up"].astype(int)
    probability = view["p_up"].clip(1e-6, 1 - 1e-6)
    return {
        "observations": int(len(view)),
        "auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
        "brier_score": float(brier_score_loss(y, probability)),
        "mean_probability": float(probability.mean()),
        "positive_rate": float(y.mean()),
    }


def main(refresh: bool = False, rebuild_factor: bool = False) -> None:
    raw = download_ohlcv(refresh=refresh)
    frames = market_frames(raw)
    frames["KODEX200"] = splice_kospi200_proxy(frames["KODEX200"])
    macro, _ = load_macro_data()
    asset_returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(macro, asset_returns)
    defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")

    variants = ["kr_only", "kr_plus_global", "us_paper_original"]
    prediction_paths = {variant: RESULTS / f"regime_lightgbm_factor_{variant}.csv" for variant in variants}
    importance_paths = {
        variant: RESULTS / f"regime_lightgbm_feature_importance_{variant}.csv" for variant in variants
    }
    predictions_by_variant: dict[str, pd.DataFrame] = {}
    importance_by_variant: dict[str, pd.DataFrame] = {}
    can_reuse = (
        not rebuild_factor
        and not refresh
        and all(path.exists() for path in prediction_paths.values())
        and all(path.exists() for path in importance_paths.values())
    )
    if can_reuse:
        for variant in variants:
            predictions = pd.read_csv(prediction_paths[variant], index_col=0)
            predictions.index = pd.PeriodIndex(predictions.index, freq="M")
            predictions["signal_date"] = pd.to_datetime(predictions["signal_date"])
            predictions_by_variant[variant] = predictions
            importance_by_variant[variant] = pd.read_csv(importance_paths[variant])
    else:
        kr_hmm_path = CACHE / "kr_hmm_diag756.csv"
        us_hmm_path = CACHE / "us_hmm_diag756.csv"
        if kr_hmm_path.exists() and us_hmm_path.exists() and not refresh:
            print("Loading cached causal Korean and US rolling HMM regimes...")
            kr_hmm = pd.read_csv(kr_hmm_path, index_col=0, parse_dates=True)
            us_hmm = pd.read_csv(us_hmm_path, index_col=0, parse_dates=True)
        else:
            print("Building causal Korean and US rolling HMM regimes...")
            kr_hmm = causal_hmm_features(frames, market="kr")
            us_hmm = causal_hmm_features(frames, market="us")
            kr_hmm.to_csv(kr_hmm_path)
            us_hmm.to_csv(us_hmm_path)
        print("Building paper-derived technical, Korean, and lagged global features...")
        features, close = build_features(frames, kr_hmm, us_hmm)
        print(f"Feature matrix: {features.shape[0]} rows x {features.shape[1]} total features")
        factor_months = pd.period_range(FACTOR_TEST_START, signals.index.max(), freq="M")
        for variant in variants:
            selected = feature_columns_for_variant(features, variant)
            print(f"Walk-forward LightGBM variant={variant}: {len(selected)} features")
            predictions, importance_frame = walk_forward_lightgbm(
                features,
                close,
                factor_months,
                selected,
                variant,
            )
            predictions.to_csv(prediction_paths[variant])
            importance_frame.to_csv(importance_paths[variant], index=False)
            predictions_by_variant[variant] = predictions
            importance_by_variant[variant] = importance_frame

    factor_start = max(predictions.index.min() for predictions in predictions_by_variant.values())
    # The factor is tested from 2005. Portfolio integration can only begin at
    # the first month available in the pre-existing four-asset FinalBlend.
    calibration_start = max(factor_start, signals.index.min())
    baseline_cfg = FactorBlendConfig(max_shift=0.0)
    baseline = run_factor_blend(
        asset_returns,
        signals,
        defensive,
        predictions_by_variant["kr_only"],
        baseline_cfg,
    )
    calibration_baseline = baseline.loc[calibration_start:CAL_END]
    baseline_cal_metrics = performance_summary(calibration_baseline["return"])

    calibration_rows = []
    backtests: dict[tuple[str, float], pd.DataFrame] = {}
    for variant in variants:
        predictions = predictions_by_variant[variant]
        for max_shift in [0.0, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.200]:
            cfg = FactorBlendConfig(max_shift=max_shift)
            backtest = run_factor_blend(asset_returns, signals, defensive, predictions, cfg)
            backtests[(variant, max_shift)] = backtest
            sample = backtest.loc[calibration_start:CAL_END]
            metrics = performance_summary(sample["return"])
            calibration_rows.append(
                {
                    "name": f"{variant}_{cfg.name}",
                    "feature_variant": variant,
                    **asdict(cfg),
                    **metrics.to_dict(),
                    "AvgTurnover": float(sample["turnover"].mean()),
                    "MDD15Pass": bool(metrics["MDD"] >= -0.15),
                    "CAGRRetention": float(metrics["CAGR"] / baseline_cal_metrics["CAGR"]),
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    eligible = calibration[(calibration["MDD15Pass"]) & (calibration["CAGRRetention"] >= 0.95)]
    if eligible.empty:
        eligible = calibration[calibration["max_shift"] == 0.0]
    winner_row = eligible.sort_values(["Sharpe", "Calmar", "CAGR"], ascending=False).iloc[0]
    winner_variant = str(winner_row["feature_variant"])
    winner_cfg = FactorBlendConfig(
        max_shift=float(winner_row["max_shift"]),
        probability_scale=float(winner_row["probability_scale"]),
        hard_fraction=float(winner_row["hard_fraction"]),
        leverage=float(winner_row["leverage"]),
        financing_rate=float(winner_row["financing_rate"]),
    )
    predictions = predictions_by_variant[winner_variant]
    importance_frame = importance_by_variant[winner_variant]
    winner = backtests[(winner_variant, winner_cfg.max_shift)]
    predictions.to_csv(RESULTS / "regime_lightgbm_factor.csv")
    importance_frame.to_csv(RESULTS / "regime_lightgbm_feature_importance.csv", index=False)

    comparison_rows = []
    periods = [
        ("calibration_common", calibration_start, CAL_END),
        ("locked_test", TEST_START, None),
        ("2007_2026", pd.Period("2007-01", "M"), None),
        ("full_common", factor_start, None),
        ("full_history", None, None),
    ]
    for period, start, end in periods:
        for strategy, backtest in [("FinalBlend", baseline), ("RegimeLightGBMFactor", winner)]:
            if start is None:
                sample = backtest.loc[:end] if end is not None else backtest
            elif end is None:
                sample = backtest.loc[start:]
            else:
                sample = backtest.loc[start:end]
            comparison_rows.append(metric_record(period, strategy, sample))
    comparison = pd.DataFrame(comparison_rows)

    locked_baseline = baseline.loc[TEST_START:]
    locked_winner = winner.loc[TEST_START:]
    baseline_locked_metrics = performance_summary(locked_baseline["return"])
    winner_locked_metrics = performance_summary(locked_winner["return"])
    sharpe_delta = float(winner_locked_metrics["Sharpe"] - baseline_locked_metrics["Sharpe"])
    calmar_delta = float(winner_locked_metrics["Calmar"] - baseline_locked_metrics["Calmar"])
    bootstrap = paired_block_bootstrap(locked_baseline["return"], locked_winner["return"])
    gates = {
        "nonzero_factor_selected_in_calibration": bool(winner_cfg.max_shift > 0),
        "locked_sharpe_delta_at_least_0_02": bool(sharpe_delta >= 0.02),
        "locked_calmar_delta_at_least_0_05": bool(calmar_delta >= 0.05),
        "locked_mdd_within_15": bool(winner_locked_metrics["MDD"] >= -0.15),
        "locked_cagr_retention_at_least_95pct": bool(
            winner_locked_metrics["CAGR"] >= 0.95 * baseline_locked_metrics["CAGR"]
        ),
    }
    gates["performance_improved"] = bool(
        gates["nonzero_factor_selected_in_calibration"]
        and (gates["locked_sharpe_delta_at_least_0_02"] or gates["locked_calmar_delta_at_least_0_05"])
        and gates["locked_mdd_within_15"]
        and gates["locked_cagr_retention_at_least_95pct"]
    )

    report = {
        "method": {
            "paper": "Pagliaro (2026), Regime-Aware LightGBM",
            "adaptation": "10-day KODEX200 direction probability used as a monthly allocation tilt in FinalBlend",
            "korean_hmm_inputs": [
                "KOSPI200 20-day return",
                "KOSPI200 20-day realized volatility",
                "USDKRW 20-day return",
                "KOSPI200 positive-day fraction over 20 days",
            ],
            "global_feature_timing": "All US-market features lagged by one Korean trading session",
            "hmm_covariance": "diagonal (Korean computational adaptation; source paper used full covariance)",
            "hmm_refit_interval_trading_days": 63,
            "hmm_rolling_history_trading_days": 756,
            "hmm_em_iteration_cap": 5,
            "horizon_trading_days": HORIZON,
            "factor_oos_start": str(FACTOR_TEST_START),
            "portfolio_common_start": str(calibration_start),
            "calibration_end": str(CAL_END),
            "locked_test_start": str(TEST_START),
            "calibration_common_start": str(calibration_start),
        },
        "winner_feature_variant": winner_variant,
        "winner": asdict(winner_cfg),
        "prediction_metrics_by_variant_full": {
            variant: prediction_metrics(variant_predictions)
            for variant, variant_predictions in predictions_by_variant.items()
        },
        "prediction_metrics_by_variant_locked": {
            variant: prediction_metrics(variant_predictions, TEST_START)
            for variant, variant_predictions in predictions_by_variant.items()
        },
        "prediction_metrics_full": prediction_metrics(predictions),
        "prediction_metrics_locked": prediction_metrics(predictions, TEST_START),
        "locked_deltas": {
            "CAGR": float(winner_locked_metrics["CAGR"] - baseline_locked_metrics["CAGR"]),
            "Sharpe": sharpe_delta,
            "MDD": float(winner_locked_metrics["MDD"] - baseline_locked_metrics["MDD"]),
            "Calmar": calmar_delta,
        },
        "bootstrap": bootstrap,
        "gates": gates,
    }

    calibration.to_csv(RESULTS / "regime_lightgbm_calibration.csv", index=False)
    comparison.to_csv(RESULTS / "regime_lightgbm_comparison.csv", index=False)
    winner.to_csv(RESULTS / "regime_lightgbm_backtest.csv")
    (RESULTS / "regime_lightgbm_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== CALIBRATION (2017-12 CUTOFF, COMMON SAMPLE) ===")
    print(
        calibration[["feature_variant", "max_shift", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "CAGRRetention"]]
        .round(4)
        .to_string(index=False)
    )
    print("\n=== PREDECLARED WINNER / LOCKED TEST ===")
    print(f"feature_variant={winner_variant}")
    print(winner_cfg)
    print(
        comparison[["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]]
        .round(4)
        .to_string(index=False)
    )
    print("\n=== PREDICTION QUALITY ===")
    print(json.dumps(report["prediction_metrics_locked"], ensure_ascii=False, indent=2))
    print("\n=== LOCKED DELTAS / GATES ===")
    print(json.dumps({"deltas": report["locked_deltas"], "gates": gates, "bootstrap": bootstrap}, indent=2))
    print("\n=== TOP FEATURE IMPORTANCE ===")
    print(importance_frame.head(15).round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--rebuild-factor", action="store_true")
    args = parser.parse_args()
    main(refresh=args.refresh, rebuild_factor=args.rebuild_factor)
