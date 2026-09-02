from __future__ import annotations

"""설명 가능한 거시 국면 기반 자산배분 전략.

이 파일은 Stage49에서 성과가 가장 단순했던 거시 국면 전략 하나만 독립적으로
실행한다. 다른 ``strategies.stageXX`` 모듈을 import하지 않는다. 처음 코드를 보는
사람은 아래의 한 방향 흐름만 따라가면 된다.

    원자료 -> 월별 자산수익률 / 거시 국면점수 / 일별 KRW 수익률
           -> Bayesian ridge 월 기대수익률
           -> Ledoit-Wolf 월 공분산
           -> 비용과 위험한도를 포함한 장기전용(long-only) 최적화
           -> 다음 한 달 보유수익률

리팩터링의 경계
----------------
이 구현은 "리팩터링"이므로 Stage49의 결과를 의도적으로 보존한다. 앞선 감사에서
발견된 통계적 한계를 조용히 고치면 더 이상 같은 전략이 아니기 때문이다. 특히
다음 세 항목은 코드 가까이에 명시하고 결과 보고서에도 남긴다.

1. Goldilocks를 기준범주로 제외한 ridge 회귀는 기준범주 불변이 아니다.
2. Bayesian 예측 표준편차는 기록하지만 배분 위험에 반영하지 않는다.
3. 거시 파일은 point-in-time vintage가 아니라 현재 보유한 최종 파일일 수 있다.

설계 원칙
---------
* SRP: 데이터, 신호, 평균, 공분산, 비용, 배분, 백테스트의 책임을 분리한다.
* OCP: BacktestEngine은 작은 Protocol 구현을 주입받아 새 모델에 열려 있다.
* LSP: 각 Protocol 구현은 같은 입출력 계약으로 교체할 수 있다.
* ISP: 평균·공분산·배분 인터페이스를 하나의 큰 인터페이스로 합치지 않는다.
* DIP: 고수준 백테스트는 sklearn/scipy 세부 구현이 아니라 Protocol에 의존한다.
* KISS: 실행 모드, overlay, ablation 분기가 없는 단일 전략 흐름만 둔다.
"""

import argparse
import json
import math
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import BayesianRidge


# ---------------------------------------------------------------------------
# 1. 도메인 계약과 선언된 가정
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepositoryPaths:
    """원자료 위치만 책임진다. 전략의 경제적 파라미터는 포함하지 않는다."""

    project_root: Path

    @classmethod
    def from_this_file(cls) -> "RepositoryPaths":
        return cls(Path(__file__).resolve().parents[2])

    @property
    def raw_data(self) -> Path:
        return self.project_root / "raw_data"

    @property
    def cache(self) -> Path:
        return self.project_root / "cache"

    @property
    def output(self) -> Path:
        return Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class StrategyConfig:
    """전략 결과에 영향을 주는 모든 숫자를 한곳에 노출한다.

    HEURISTIC 표시는 데이터에서 통계적으로 추정한 값이 아니라 연구자 또는
    투자정책이 선택한 값이라는 뜻이다. 숫자를 숨기지 않는 이유는 민감도 분석의
    대상을 분명히 하기 위해서다.
    """

    assets: tuple[str, ...] = ("KODEX200", "BOND", "GLD", "USO")
    regime_columns: tuple[str, ...] = (
        "p_Goldilocks",
        "p_Overheating",
        "p_Slowdown",
        "p_Stagflation",
    )

    # COMPATIBILITY HEURISTIC: 네 score의 합이 1이므로 intercept와 함께 네
    # 변수를 모두 쓰면 완전 다중공선성이 생긴다. Stage49는 Goldilocks를 기준으로
    # 제외했다. OLS 예측은 기준범주에 불변이지만 ridge prior는 불변이 아니다.
    # 결과 재현을 위해 유지하며, 새 경제 모형에서는 대칭 contrast가 더 적절하다.
    forecast_features: tuple[str, ...] = (
        "p_Overheating",
        "p_Slowdown",
        "p_Stagflation",
    )

    full_start: str = "2007-04"
    locked_start: str = "2018-01"
    research_end: str = "2026-07"

    # HEURISTIC: 252일과 21일은 각각 통상적인 1년/1개월 거래일 수다. 데이터가
    # 선택한 최적 창이 아니며, 253번째 관측치를 갑자기 버리는 rolling-window
    # 가정을 포함한다.
    covariance_lookback_days: int = 252
    trading_days_per_month: float = 21.0

    # HEURISTIC: p개 설명변수와 intercept 하나당 5개 관측치를 요구한다. 이는
    # Bayesian ridge의 정리가 아니라 너무 짧은 표본에서 모형을 켜지 않기 위한
    # 보수적인 sufficiency rule이다. 현재 p=3이므로 최소 20개월이다.
    observations_per_parameter: int = 5

    # GOVERNANCE HEURISTIC: 아래 값은 미래분포에서 추정된 최적값이 아니라 투자
    # 위원회가 정해야 할 위험 예산이다. Stage49 결과 재현을 위해 Stage36 값을
    # 유지한다. 실제 운용 전에는 반드시 별도 민감도 분석을 해야 한다.
    annual_volatility_cap: float = 0.13
    maximum_cdar: float = 0.16
    cdar_confidence: float = 0.90

    # COST HEURISTIC: 고정 거래비용은 시기별 spread, 세금, 시장충격을 설명하지
    # 않는다. 국내/해외 전 자산의 절대 비중변화에 15bp를 적용하고 GLD+USO의
    # 순 USD 흐름에만 5bp 환전비용을 추가한다.
    proportional_trade_cost: float = 0.0015
    foreign_exchange_cost: float = 0.0005

    solver_max_iterations: int = 300
    solver_tolerance: float = 1e-9
    numerical_epsilon: float = 1e-12

    @property
    def start_month(self) -> pd.Period:
        return pd.Period(self.full_start, freq="M")

    @property
    def locked_month(self) -> pd.Period:
        return pd.Period(self.locked_start, freq="M")

    @property
    def end_month(self) -> pd.Period:
        return pd.Period(self.research_end, freq="M")


@dataclass(frozen=True)
class ForecastEstimate:
    mean: float
    predictive_std: float
    observations: int
    last_training_month: pd.Period | None
    used_unconditional_fallback: bool


@dataclass(frozen=True)
class CovarianceEstimate:
    matrix: np.ndarray
    shrinkage: float
    cutoff: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    observations: int


@dataclass(frozen=True)
class AllocationResult:
    weights: np.ndarray
    diagnostics: dict[str, Any]


class ExpectedReturnForecaster(Protocol):
    """ISP: 기대수익률 모형이 제공해야 하는 최소 계약."""

    def forecast(
        self, feature_history: pd.DataFrame, current_features: pd.Series
    ) -> dict[str, ForecastEstimate]: ...


class CovarianceForecaster(Protocol):
    """ISP: 위험 모형이 제공해야 하는 최소 계약."""

    def forecast(
        self, daily_returns: pd.DataFrame, decision_month: pd.Period
    ) -> CovarianceEstimate: ...


class PortfolioAllocator(Protocol):
    """ISP: 배분 정책이 제공해야 하는 최소 계약."""

    def allocate(
        self,
        historical_monthly_returns: pd.DataFrame,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        pretrade_weights: np.ndarray,
    ) -> AllocationResult: ...


def _find_unicode_safe_file(directory: Path, filename: str) -> Path:
    """Windows에서 한글 파일명의 NFC/NFD 차이를 흡수한다."""

    target = unicodedata.normalize("NFC", filename)
    for path in directory.iterdir():
        if unicodedata.normalize("NFC", path.name) == target:
            return path
    raise FileNotFoundError(f"Required source not found: {directory / filename}")


# ---------------------------------------------------------------------------
# 2. 데이터 저장소: 모델 로직과 파일 형식을 격리한다(SRP)
# ---------------------------------------------------------------------------


class MonthlyAssetReturnRepository:
    """월초 시가 기준 네 자산의 원화 수익률을 만든다.

    경제적 의미
    -----------
    의사결정월 t의 수익률은 t월 첫 거래일 시가에서 t+1월 첫 거래일 시가까지다.
    GLD와 USO는 각 첫 거래일 USDKRW로 원화 환산한다. KODEX200 상장 전 구간은
    KOSPI200 proxy를 실제 ETF 시가에 연결한다.

    DATA HEURISTIC: 서로 다른 공급원의 지수/ETF를 한 점에서 비례 연결하면 수준은
    연속이지만 상품구조·배당·추적오차가 같다는 뜻은 아니다. Stage49와 동일한
    연구 표본을 재현하기 위한 데이터 계약이다.
    """

    def __init__(self, paths: RepositoryPaths, config: StrategyConfig) -> None:
        self.paths = paths
        self.config = config

    def _market_cache(self, refresh: bool) -> pd.DataFrame:
        cache = self.paths.cache / "market_daily.csv"
        if cache.exists() and not refresh:
            return pd.read_csv(cache, parse_dates=["date"])

        # yfinance는 cache 재생성이 필요할 때만 선택적으로 import한다.
        import yfinance as yf

        rows: list[pd.DataFrame] = []
        instruments = (
            ("069500.KS", "KODEX200"),
            ("GLD", "GLD"),
            ("USO", "USO"),
            ("KRW=X", "USDKRW"),
        )
        for ticker, symbol in instruments:
            data = yf.download(
                ticker,
                start="2000-01-01",
                auto_adjust=symbol != "USDKRW",
                progress=False,
                threads=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.reset_index().rename(
                columns={"Date": "date", "Open": "open", "Close": "close"}
            )
            data["date"] = (
                pd.to_datetime(data["date"], utc=True)
                .dt.tz_localize(None)
                .dt.normalize()
            )
            data["symbol"] = symbol
            rows.append(data[["date", "symbol", "open", "close"]])
        result = pd.concat(rows, ignore_index=True).dropna(
            subset=["date", "close"]
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache, index=False)
        return result

    def load(self, refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        market = self._market_cache(refresh)
        compass = _find_unicode_safe_file(self.paths.raw_data, "compass.db")
        with sqlite3.connect(compass) as connection:
            proxy = pd.read_sql(
                "select date, open, close from etf_prices "
                "where symbol = ? order by date",
                connection,
                params=("1028",),
            )
        proxy["date"] = pd.to_datetime(proxy["date"])
        proxy[["open", "close"]] = proxy[["open", "close"]].apply(
            pd.to_numeric, errors="coerce"
        )

        actual = market.loc[market["symbol"].eq("KODEX200")].copy()
        actual = actual.dropna(subset=["open"])
        actual = actual.loc[actual["date"] > pd.Timestamp("2009-03-31")]
        if actual.empty:
            raise ValueError("Continuous KODEX200 ETF history is unavailable.")
        first_actual = actual["date"].min()
        actual_anchor = float(
            actual.loc[actual["date"].eq(first_actual), "open"].iloc[0]
        )
        exact_proxy = proxy.loc[proxy["date"].eq(first_actual), "open"]
        if exact_proxy.empty:
            nearest = proxy.iloc[(proxy["date"] - first_actual).abs().argsort()[:1]]
            proxy_anchor = float(nearest["open"].iloc[0])
        else:
            proxy_anchor = float(exact_proxy.iloc[0])
        proxy[["open", "close"]] *= actual_anchor / proxy_anchor
        proxy = proxy.loc[proxy["date"] < first_actual].copy()
        proxy["symbol"] = "KODEX200"
        kodex = pd.concat(
            [proxy[["date", "symbol", "open", "close"]], actual],
            ignore_index=True,
        )

        bond_source = _find_unicode_safe_file(
            self.paths.raw_data, "krx_bond_index.csv"
        )
        bond = pd.read_csv(bond_source, encoding="cp949")
        bond["date"] = pd.to_datetime(bond.iloc[:, 0])
        bond["open"] = (
            bond.iloc[:, 1]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
        bond["close"] = bond["open"]
        bond["symbol"] = "BOND"

        fx = (
            market.loc[market["symbol"].eq("USDKRW")]
            .set_index("date")["close"]
            .sort_index()
        )
        fx = fx.reindex(pd.date_range(fx.index.min(), fx.index.max(), freq="D"))
        fx = fx.ffill()

        sources = {
            "KODEX200": kodex,
            "BOND": bond,
            "GLD": market.loc[market["symbol"].eq("GLD")],
            "USO": market.loc[market["symbol"].eq("USO")],
        }
        first_open: dict[str, pd.Series] = {}
        for asset, source in sources.items():
            clean = source.dropna(subset=["open"]).sort_values("date").copy()
            clean["month"] = clean["date"].dt.to_period("M")
            first = clean.groupby("month", sort=True).first()
            value = first["open"].astype(float)
            if asset in {"GLD", "USO"}:
                aligned_fx = fx.reindex(
                    pd.DatetimeIndex(first["date"]), method="ffill"
                )
                value = value * aligned_fx.to_numpy(dtype=float)
            first_open[asset] = value

        levels = pd.concat(first_open, axis=1).sort_index()
        returns = levels.shift(-1).div(levels).sub(1.0).dropna(how="any")
        returns = returns.loc[:, self.config.assets]
        audit = {
            "market_cache": str(self.paths.cache / "market_daily.csv"),
            "monthly_return_definition": "first-open(t+1)/first-open(t)-1",
            "foreign_assets_in_krw": True,
            "rows": int(len(returns)),
            "start": str(returns.index.min()),
            "end": str(returns.index.max()),
        }
        return returns, audit


class DailyRiskReturnRepository:
    """공분산 추정에 사용할 일별 원화 종가수익률만 만든다.

    DATA HEURISTIC: 한국·미국 시장을 하나의 영업일 달력에 놓고 최대 5일
    forward-fill한다. 휴장일에는 0 수익률이 생기고 다음 거래일에 움직임이 몰려
    비동시 거래 공분산 편향이 생길 수 있다. Stage49 결과 보존을 위해 유지한다.
    """

    def __init__(self, paths: RepositoryPaths, config: StrategyConfig) -> None:
        self.paths = paths
        self.config = config

    @staticmethod
    def _numeric_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output.index = pd.to_datetime(output.index).normalize()
        columns = ["open", "high", "low", "close", "volume"]
        for column in columns:
            if column not in output:
                output[column] = np.nan
            output[column] = pd.to_numeric(output[column], errors="coerce")
        output = output[columns].sort_index()
        return output.loc[~output.index.duplicated(keep="last")]

    def load(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        cache = self.paths.cache / "regime_lightgbm_ohlcv.csv"
        raw = pd.read_csv(cache, parse_dates=["date"])
        market = {
            str(symbol): self._numeric_ohlcv(group.set_index("date"))
            for symbol, group in raw.groupby("symbol")
        }
        required = {"KODEX200", "GLD", "USO", "USDKRW"}
        missing = sorted(required.difference(market))
        if missing:
            raise ValueError(f"OHLCV cache is missing symbols: {missing}")

        actual = market["KODEX200"].loc[
            market["KODEX200"].index > pd.Timestamp("2009-03-31")
        ].dropna(subset=["close"])
        compass = _find_unicode_safe_file(self.paths.raw_data, "compass.db")
        with sqlite3.connect(compass) as connection:
            proxy = pd.read_sql(
                "select date, open, high, low, close, volume "
                "from etf_prices where symbol = ? order by date",
                connection,
                params=("1028",),
            )
        proxy["date"] = pd.to_datetime(proxy["date"])
        proxy = self._numeric_ohlcv(proxy.set_index("date"))
        first_actual = actual.index.min()
        nearest = proxy.index.get_indexer([first_actual], method="nearest")[0]
        scale = float(actual.loc[first_actual, "close"] / proxy.iloc[nearest]["close"])
        proxy[["open", "high", "low", "close"]] *= scale
        proxy = proxy.loc[proxy.index < first_actual]
        kodex = pd.concat([proxy, actual]).sort_index()

        bond_source = _find_unicode_safe_file(
            self.paths.raw_data, "krx_bond_index.csv"
        )
        bond_raw = pd.read_csv(bond_source, encoding="cp949")
        bond_close = pd.to_numeric(
            bond_raw.iloc[:, 1].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        bond = pd.DataFrame(
            {"close": bond_close.to_numpy()},
            index=pd.to_datetime(bond_raw.iloc[:, 0]).dt.normalize(),
        ).sort_index()

        fx = market["USDKRW"]["close"].dropna().sort_index()
        foreign: dict[str, pd.DataFrame] = {}
        for asset in ("GLD", "USO"):
            frame = market[asset].dropna(subset=["close"]).copy()
            aligned_fx = fx.reindex(frame.index, method="ffill")
            if aligned_fx.isna().any():
                raise ValueError(f"USDKRW does not cover full {asset} history")
            frame["close"] = frame["close"] * aligned_fx
            foreign[asset] = frame

        frames = {
            "KODEX200": kodex,
            "BOND": bond,
            "GLD": foreign["GLD"],
            "USO": foreign["USO"],
        }
        close = pd.concat(
            {asset: frames[asset]["close"].dropna().astype(float) for asset in self.config.assets},
            axis=1,
        ).sort_index()
        calendar = pd.date_range(close.index.min(), close.index.max(), freq="B")
        levels = close.reindex(calendar).ffill(limit=5)
        daily_returns = levels.pct_change(fill_method=None)
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan)
        daily_returns = daily_returns.loc[:, self.config.assets]
        audit = {
            "ohlcv_cache": str(cache),
            "foreign_assets_in_krw": True,
            "business_day_forward_fill_limit": 5,
            "rows": int(len(daily_returns)),
            "start": str(daily_returns.index.min().date()),
            "end": str(daily_returns.index.max().date()),
        }
        return daily_returns, audit


class MacroRegimeRepository:
    """여섯 거시지표를 네 개의 합계 1 국면 score로 변환한다.

    통계적 의미
    -----------
    각 지표의 값이 그 시점까지의 과거 분포에서 차지하는 expanding percentile을
    계산한다. 성장 3개와 물가 3개를 각각 동일가중 평균한 뒤 2x2 국면으로 곱한다.

    HEURISTIC: 이 값은 calibrated posterior probability가 아니다. 동일가중과
    성장/물가 score의 곱은 단순성과 해석가능성을 위한 구성 규칙이다. 따라서
    코드에서는 '확률 추정'이 아니라 '정규화된 국면 score'로 해석한다.

    VINTAGE WARNING: 파일의 과거 값이 수정된 최종치라면 날짜를 한 달 늦춰도
    revision look-ahead를 제거하지 못한다. 실운용 검증에는 vintage 데이터가 필요하다.
    """

    def __init__(self, paths: RepositoryPaths, config: StrategyConfig) -> None:
        self.paths = paths
        self.config = config

    @staticmethod
    def _causal_expanding_percentile(series: pd.Series) -> pd.Series:
        output = pd.Series(np.nan, index=series.index, dtype=float)
        history: list[float] = []
        for index, value in series.items():
            if not np.isfinite(value):
                continue
            history.append(float(value))
            reference = np.asarray(history, dtype=float)
            less = float(np.sum(reference < value))
            equal = float(np.sum(reference == value))
            output.loc[index] = (less + 0.5 * equal) / len(reference)
        return output

    def _load_levels(self) -> pd.DataFrame:
        raw = self.paths.raw_data
        gdp = pd.read_excel(
            _find_unicode_safe_file(raw, "GDP 성장률.xlsx"),
            index_col=0,
            skiprows=6,
        )
        gdp.columns = ["GDP_QoQ", "GDP_YoY"]
        gdp.index = (
            pd.PeriodIndex(gdp.index, freq="Q")
            .asfreq("M", how="end")
            .to_timestamp("M")
            + pd.offsets.MonthEnd(1)
        )
        gdp = gdp.resample("ME").ffill()

        trade = pd.read_excel(
            _find_unicode_safe_file(raw, "수출입 총괄_20260816.xlsx"),
            index_col=0,
            skiprows=4,
        )
        trade = trade[["수출 금액", "수입금액"]].iloc[1:].copy()
        for column in trade:
            trade[column] = (
                trade[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
            )
        trade.index = pd.to_datetime(trade.index, format="%Y.%m") + pd.offsets.MonthEnd(1)
        trade["Export_YoY"] = trade["수출 금액"].pct_change(12) * 100.0

        bsi = pd.read_csv(
            _find_unicode_safe_file(raw, "기업경기조사(전망).csv"), encoding="cp949"
        )
        bsi = bsi.loc[
            bsi["업종코드별"].eq("제 조 업")
            & bsi["BSI코드별"].eq("업황전망BSI 1)")
        ].iloc[:, 2:4]
        bsi["시점"] = (
            bsi["시점"]
            .str.replace("월", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        bsi["시점"] = pd.to_datetime(bsi["시점"], format="%Y.%m") + pd.offsets.MonthEnd(1)
        bsi = bsi.set_index("시점")
        bsi.columns = ["BSI"]

        cpi = pd.read_excel(
            _find_unicode_safe_file(raw, "소비자물가 상승률.xlsx"),
            index_col=0,
            skiprows=6,
        )
        cpi.columns = ["CPI_QoQ", "CPI_YoY"]
        cpi.index = pd.to_datetime(cpi.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

        ppi = pd.read_excel(
            _find_unicode_safe_file(raw, "생산자물가 상승률.xlsx"),
            index_col=0,
            skiprows=6,
        )
        ppi.columns = ["PPI_QoQ", "PPI_YoY"]
        ppi.index = pd.to_datetime(ppi.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

        prices = pd.read_excel(
            _find_unicode_safe_file(raw, "수출입물가 상승률.xlsx"),
            index_col=0,
            skiprows=6,
        )
        prices.columns = ["ExportPrice_YoY", "ImportPrice_YoY"]
        prices.index = pd.to_datetime(prices.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

        return pd.concat(
            [
                gdp["GDP_YoY"],
                trade["Export_YoY"],
                bsi["BSI"],
                cpi["CPI_YoY"],
                ppi["PPI_YoY"],
                prices["ImportPrice_YoY"],
            ],
            axis=1,
        ).sort_index()

    def build_scores(
        self, decision_months: pd.PeriodIndex
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        levels = self._load_levels()
        ranks = levels.apply(self._causal_expanding_percentile)
        rows: list[dict[str, Any]] = []
        for decision_month in decision_months:
            signal_month = decision_month - 1
            known = ranks.loc[: signal_month.to_timestamp("M")]
            if known.empty or known.iloc[-1].isna().any():
                continue
            current = known.iloc[-1]
            growth_high = float(current[["GDP_YoY", "Export_YoY", "BSI"]].mean())
            inflation_high = float(
                current[["CPI_YoY", "PPI_YoY", "ImportPrice_YoY"]].mean()
            )
            rows.append(
                {
                    "target_month": decision_month,
                    "signal_month": signal_month,
                    "p_growth_high": growth_high,
                    "p_inflation_high": inflation_high,
                    "p_Goldilocks": growth_high * (1.0 - inflation_high),
                    "p_Overheating": growth_high * inflation_high,
                    "p_Slowdown": (1.0 - growth_high) * (1.0 - inflation_high),
                    "p_Stagflation": (1.0 - growth_high) * inflation_high,
                }
            )
        scores = pd.DataFrame(rows).set_index("target_month")
        scores.index = pd.PeriodIndex(scores.index, freq="M")
        audit = {
            "source_columns": list(levels.columns),
            "equal_weight_within_growth_and_inflation": True,
            "calibrated_probabilities": False,
            "point_in_time_vintages_verified": False,
            "rows": int(len(scores)),
        }
        return scores, audit


# ---------------------------------------------------------------------------
# 3. 통계 모형: 평균과 공분산은 서로 다른 책임이다(SRP/ISP)
# ---------------------------------------------------------------------------


class BayesianRegimeReturnForecaster:
    """거시 국면 score로 각 자산의 다음 월 수익률 평균을 예측한다.

    모형은 ``r(i,t) = intercept(i) + beta(i)' x(t) + error(i,t)``다.
    BayesianRidge는 coefficient/noise precision을 marginal likelihood로 추정한다.
    이는 작은 표본에서 OLS 계수를 0 방향으로 수축해 극단적인 평균 예측을 줄인다.

    통계적 한계
    -----------
    * Gaussian, 선형, 고정계수, 조건부 등분산을 가정한다.
    * sklearn 기본 Gamma hyperprior(1e-6)를 명시적으로 적었지만 경제법칙은 아니다.
    * predictive_std에는 잔차와 coefficient 불확실성이 함께 들어 있다. Stage49
      결과 재현을 위해 이를 기록만 하고 배분 공분산에는 넣지 않는다.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def _fit_one(
        self,
        history: pd.DataFrame,
        target: str,
        current: pd.Series,
    ) -> ForecastEstimate:
        features = self.config.forecast_features
        complete = history[[target, *features]].dropna()
        minimum = self.config.observations_per_parameter * (len(features) + 1)
        if len(complete) < minimum or current[list(features)].isna().any():
            available = history[target].dropna()
            return ForecastEstimate(
                mean=float(available.mean()),
                predictive_std=float(available.std(ddof=1)),
                observations=int(len(available)),
                last_training_month=available.index.max() if len(available) else None,
                used_unconditional_fallback=True,
            )

        predictors = complete.loc[:, features].astype(float)
        means = predictors.mean()
        standard_deviations = predictors.std(ddof=1)
        usable = standard_deviations.gt(self.config.numerical_epsilon)
        if not usable.any():
            available = history[target].dropna()
            return ForecastEstimate(
                mean=float(available.mean()),
                predictive_std=float(available.std(ddof=1)),
                observations=int(len(available)),
                last_training_month=available.index.max() if len(available) else None,
                used_unconditional_fallback=True,
            )

        columns = tuple(standard_deviations.index[usable])
        x = (predictors.loc[:, columns] - means.loc[list(columns)])
        x = x / standard_deviations.loc[list(columns)]
        current_x = (current.loc[list(columns)] - means.loc[list(columns)])
        current_x = current_x / standard_deviations.loc[list(columns)]

        model = BayesianRidge(
            fit_intercept=True,
            compute_score=True,
            tol=1e-7,
            max_iter=1_000,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
        )
        model.fit(
            x.to_numpy(dtype=float),
            complete.loc[x.index, target].to_numpy(dtype=float),
        )
        prediction, prediction_std = model.predict(
            current_x.to_numpy(dtype=float).reshape(1, -1), return_std=True
        )
        return ForecastEstimate(
            mean=float(prediction[0]),
            predictive_std=float(prediction_std[0]),
            observations=int(len(complete)),
            last_training_month=complete.index.max(),
            used_unconditional_fallback=False,
        )

    def forecast(
        self, feature_history: pd.DataFrame, current_features: pd.Series
    ) -> dict[str, ForecastEstimate]:
        return {
            asset: self._fit_one(
                feature_history, f"return_{asset}", current_features
            )
            for asset in self.config.assets
        }


class LedoitWolfMonthlyCovarianceForecaster:
    """직전 252개 일별 수익률의 constant-correlation shrinkage 공분산.

    표본 공분산의 각 분산은 유지하고 모든 상관계수를 공통 평균상관으로 수축한다.
    수축강도는 Ledoit-Wolf 식으로 데이터에서 추정하므로 임의의 혼합비가 아니다.
    이후 21을 곱해 월 공분산으로 환산하고 작은 음의 고유값을 수치적으로 보정한다.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def _constant_correlation(self, values: np.ndarray) -> tuple[np.ndarray, float]:
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            raise ValueError("values must be a finite T by N matrix")
        if not np.isfinite(x).all():
            raise ValueError("values must be finite")

        observations, assets = x.shape
        centered = x - x.mean(axis=0, keepdims=True)
        sample = (centered.T @ centered) / observations
        variances = np.maximum(np.diag(sample).copy(), 1e-16)
        standard_deviations = np.sqrt(variances)
        outer_std = np.outer(standard_deviations, standard_deviations)
        off_diagonal = ~np.eye(assets, dtype=bool)
        average_correlation = (
            float((sample / outer_std)[off_diagonal].mean()) if assets > 1 else 0.0
        )
        target = average_correlation * outer_std
        np.fill_diagonal(target, variances)

        squared = centered * centered
        pi_matrix = (squared.T @ squared) / observations - sample * sample
        pi_hat = float(pi_matrix.sum())
        cross = ((squared * centered).T @ centered) / observations
        theta_ii = cross - variances[:, None] * sample
        theta_jj = cross.T - variances[None, :] * sample
        ratio = np.outer(1.0 / standard_deviations, standard_deviations)
        rho_off = (average_correlation / 2.0) * (
            ratio * theta_ii + ratio.T * theta_jj
        )
        rho_hat = float(
            np.diag(pi_matrix).sum() + rho_off[off_diagonal].sum()
        )
        distance = target - sample
        gamma_hat = float((distance * distance).sum())
        shrinkage = (
            0.0
            if gamma_hat <= 1e-30
            else float(
                np.clip(
                    (pi_hat - rho_hat) / gamma_hat / observations,
                    0.0,
                    1.0,
                )
            )
        )
        covariance = shrinkage * target + (1.0 - shrinkage) * sample
        covariance.flat[:: assets + 1] += self.config.numerical_epsilon
        return covariance, shrinkage

    def _nearest_positive_definite(self, covariance: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        scale = max(
            float(np.trace(symmetric)) / len(symmetric),
            self.config.numerical_epsilon,
        )
        floor = scale * 1e-10
        return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T

    def forecast(
        self, daily_returns: pd.DataFrame, decision_month: pd.Period
    ) -> CovarianceEstimate:
        cutoff = (decision_month - 1).to_timestamp("M")
        window = daily_returns.loc[:cutoff, self.config.assets]
        window = window.dropna(how="any").tail(
            self.config.covariance_lookback_days
        )
        if len(window) < self.config.covariance_lookback_days:
            raise ValueError(
                f"{decision_month}: {len(window)} complete daily rows, "
                f"{self.config.covariance_lookback_days} required"
            )
        daily_covariance, shrinkage = self._constant_correlation(
            window.to_numpy(dtype=float)
        )
        monthly_covariance = self._nearest_positive_definite(
            daily_covariance * self.config.trading_days_per_month
        )
        return CovarianceEstimate(
            matrix=monthly_covariance,
            shrinkage=shrinkage,
            cutoff=cutoff,
            window_start=window.index.min(),
            window_end=window.index.max(),
            observations=int(len(window)),
        )


# ---------------------------------------------------------------------------
# 4. 비용·위험·배분: 경제적 목적함수를 한곳에 둔다
# ---------------------------------------------------------------------------


class TransactionCostModel:
    """최적화용 매끄러운 비용과 사후 실제 선형 비용을 함께 정의한다."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.foreign_indices = np.array(
            [config.assets.index("GLD"), config.assets.index("USO")]
        )

    def expected_cost(
        self, weights: np.ndarray, pretrade_weights: np.ndarray
    ) -> float:
        change = weights - pretrade_weights
        # sqrt(x^2 + epsilon)은 |x|의 미분 가능한 근사다. 경제 파라미터가 아니라
        # SLSQP가 0에서 안정적으로 gradient를 계산하기 위한 수치 장치다.
        smooth_change = np.sqrt(change**2 + self.config.numerical_epsilon)
        trading = float(smooth_change.sum()) * self.config.proportional_trade_cost
        net_foreign_change = float(change[self.foreign_indices].sum())
        foreign_exchange = math.sqrt(
            net_foreign_change**2 + self.config.numerical_epsilon
        ) * self.config.foreign_exchange_cost
        return trading + foreign_exchange

    def realized_costs(
        self, weights: np.ndarray, pretrade_weights: np.ndarray
    ) -> tuple[float, float]:
        change = weights - pretrade_weights
        trading = float(np.abs(change).sum()) * self.config.proportional_trade_cost
        foreign_exchange = (
            abs(float(change[self.foreign_indices].sum()))
            * self.config.foreign_exchange_cost
        )
        return trading, foreign_exchange


def _project_to_long_only_simplex(weights: np.ndarray) -> np.ndarray:
    """유클리드 거리 기준으로 비중을 {w>=0, sum(w)=1}에 사영한다."""

    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Weights must be finite before simplex projection")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    candidates = ordered - cumulative / np.arange(1, len(values) + 1) > 0
    rho = int(np.flatnonzero(candidates)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - threshold, 0.0)
    return projected / projected.sum()


def _conditional_drawdown_at_risk(
    returns: np.ndarray, confidence: float
) -> float:
    """과거 경로의 최악 (1-confidence) drawdown 평균을 음수로 반환한다.

    이는 현재 비중을 과거 전체에 고정 보유했다고 가정한 역사적 stress measure다.
    동적 전략의 미래 CDaR 예측 또는 통계적 신뢰구간은 아니다.
    """

    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    running_peak = np.maximum.accumulate(np.r_[1.0, wealth])[-len(wealth) :]
    drawdowns = wealth / running_peak - 1.0
    tail_count = max(1, int(math.ceil((1.0 - confidence) * len(drawdowns))))
    return float(np.mean(np.sort(drawdowns)[:tail_count]))


class RiskConstrainedAllocator:
    """평균-분산 certainty equivalent가 가장 큰 실현 가능한 비중을 고른다.

    목적함수
    --------
        CE(w) = w' mu - 0.5 w' Sigma w - expected_transaction_cost(w)

    경제적 해석은 위험회피계수 1의 mean-variance 효용이다. 단순수익률의 정확한
    로그효용은 아니며 2차 근사에서도 -0.5*mu^2 항을 생략한다. 월 mu가 작다는
    근사에 기대는 Stage49 호환 선택이다.

    제약은 완전투자, long-only, 연 변동성 13% 이하, 역사적 90%-CDaR 16% 이하이다.
    전자는 투자 가능집합, 후자의 숫자는 통계 추정이 아닌 위험정책이다.
    """

    def __init__(
        self, config: StrategyConfig, transaction_cost: TransactionCostModel
    ) -> None:
        self.config = config
        self.transaction_cost = transaction_cost

    def allocate(
        self,
        historical_monthly_returns: pd.DataFrame,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        pretrade_weights: np.ndarray,
    ) -> AllocationResult:
        historical = historical_monthly_returns.loc[:, self.config.assets]
        historical_values = historical.to_numpy(dtype=float)
        initial = (
            _project_to_long_only_simplex(pretrade_weights)
            if np.isfinite(pretrade_weights).all() and pretrade_weights.sum() > 0.99
            else np.repeat(1.0 / len(self.config.assets), len(self.config.assets))
        )

        def portfolio_values(weights: np.ndarray) -> dict[str, float]:
            mean = float(weights @ expected_returns)
            variance = max(float(weights @ covariance @ weights), 0.0)
            cost = self.transaction_cost.expected_cost(weights, pretrade_weights)
            certainty_equivalent = mean - 0.5 * variance - cost
            return {
                "expected_monthly_return": mean,
                "expected_monthly_variance": variance,
                "expected_annual_log_growth": 12.0 * (mean - 0.5 * variance),
                "estimated_transaction_cost": cost,
                "monthly_certainty_equivalent": certainty_equivalent,
            }

        def annual_volatility(weights: np.ndarray) -> float:
            variance = max(float(weights @ covariance @ weights), 0.0)
            return math.sqrt(variance * 12.0)

        def cdar_slack(weights: np.ndarray) -> float:
            historical_portfolio = historical_values @ weights
            cdar = _conditional_drawdown_at_risk(
                historical_portfolio, self.config.cdar_confidence
            )
            return self.config.maximum_cdar + cdar

        constraints = [
            {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
            {
                "type": "ineq",
                "fun": lambda weights: self.config.annual_volatility_cap
                - annual_volatility(weights),
            },
            {"type": "ineq", "fun": cdar_slack},
        ]
        bounds = [(0.0, 1.0)] * len(self.config.assets)
        result = minimize(
            lambda weights: -portfolio_values(weights)[
                "monthly_certainty_equivalent"
            ],
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self.config.solver_max_iterations,
                "ftol": self.config.solver_tolerance,
            },
        )

        used_fallback = False
        if result.success and np.isfinite(result.x).all():
            weights = _project_to_long_only_simplex(result.x)
        else:
            # 수치 실패 시 기대수익률을 임의로 고치지 않고 같은 위험제약 안의
            # 최소분산 포트폴리오로 후퇴한다. 안전하고 결정적인 fallback이다.
            fallback = minimize(
                lambda weights: float(weights @ covariance @ weights),
                initial,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": self.config.solver_max_iterations,
                    "ftol": self.config.solver_tolerance,
                },
            )
            if not fallback.success or not np.isfinite(fallback.x).all():
                raise RuntimeError(
                    f"Economic and fallback solves failed: {result.message}; "
                    f"{fallback.message}"
                )
            result = fallback
            weights = _project_to_long_only_simplex(fallback.x)
            used_fallback = True

        values = portfolio_values(weights)
        annual_vol = annual_volatility(weights)
        historical_cdar = _conditional_drawdown_at_risk(
            historical_values @ weights, self.config.cdar_confidence
        )
        diagnostics = {
            **values,
            "solver_success": bool(result.success),
            "used_fallback": used_fallback,
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "solver_iterations": int(result.nit),
            "objective_value": float(result.fun),
            "expected_annual_volatility": annual_vol,
            "historical_cdar": historical_cdar,
            "sum_error": abs(float(weights.sum()) - 1.0),
            "volatility_slack": self.config.annual_volatility_cap - annual_vol,
            "cdar_slack": self.config.maximum_cdar + historical_cdar,
        }
        return AllocationResult(weights=weights, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# 5. 단일 백테스트 흐름: 고수준 코드는 Protocol에만 의존한다(DIP)
# ---------------------------------------------------------------------------


class BacktestEngine:
    """월별 의사결정 순서와 포트폴리오 상태 전이만 책임진다."""

    def __init__(
        self,
        config: StrategyConfig,
        return_forecaster: ExpectedReturnForecaster,
        covariance_forecaster: CovarianceForecaster,
        allocator: PortfolioAllocator,
        transaction_cost: TransactionCostModel,
    ) -> None:
        self.config = config
        self.return_forecaster = return_forecaster
        self.covariance_forecaster = covariance_forecaster
        self.allocator = allocator
        self.transaction_cost = transaction_cost

    def _feature_frame(
        self, monthly_returns: pd.DataFrame, regime_scores: pd.DataFrame
    ) -> pd.DataFrame:
        frame = regime_scores.loc[:, self.config.regime_columns].copy()
        for asset in self.config.assets:
            frame[f"return_{asset}"] = monthly_returns[asset]
        frame.index = pd.PeriodIndex(frame.index, freq="M")
        return frame.sort_index().replace([np.inf, -np.inf], np.nan)

    def run(
        self,
        monthly_returns: pd.DataFrame,
        regime_scores: pd.DataFrame,
        daily_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        feature_frame = self._feature_frame(monthly_returns, regime_scores)
        months = monthly_returns.index.intersection(regime_scores.index)
        months = months.intersection(feature_frame.index)
        months = months[
            (months >= self.config.start_month) & (months <= self.config.end_month)
        ]

        pretrade = np.zeros(len(self.config.assets), dtype=float)
        first_trade = True
        nav = 1.0
        peak = 1.0
        rows: list[dict[str, Any]] = []

        for month in months:
            historical_returns = monthly_returns.loc[
                monthly_returns.index < month, self.config.assets
            ]
            if len(historical_returns) < 12:
                continue
            try:
                covariance = self.covariance_forecaster.forecast(
                    daily_returns, month
                )
            except ValueError:
                continue

            feature_history = feature_frame.loc[feature_frame.index < month]
            estimates = self.return_forecaster.forecast(
                feature_history, feature_frame.loc[month]
            )
            expected_returns = np.array(
                [estimates[asset].mean for asset in self.config.assets], dtype=float
            )
            allocation = self.allocator.allocate(
                historical_returns,
                expected_returns,
                covariance.matrix,
                pretrade,
            )
            weights = allocation.weights
            change = weights - pretrade
            turnover = (
                float(np.abs(change).sum())
                if first_trade
                else 0.5 * float(np.abs(change).sum())
            )
            trade_cost, fx_cost = self.transaction_cost.realized_costs(
                weights, pretrade
            )
            asset_return = monthly_returns.loc[
                month, list(self.config.assets)
            ].to_numpy(dtype=float)
            gross_return = float(weights @ asset_return)
            net_return = gross_return - trade_cost - fx_cost
            nav *= 1.0 + net_return
            peak = max(peak, nav)

            row: dict[str, Any] = {
                "month": month,
                "macro_signal_month": regime_scores.loc[month, "signal_month"],
                "covariance_cutoff": covariance.cutoff,
                "covariance_start": covariance.window_start,
                "covariance_end": covariance.window_end,
                "covariance_observations": covariance.observations,
                "ledoit_wolf_shrinkage": covariance.shrinkage,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "covariance_min_eigenvalue": float(
                    np.linalg.eigvalsh(covariance.matrix).min()
                ),
                **{
                    column: float(regime_scores.loc[month, column])
                    for column in self.config.regime_columns
                },
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(self.config.assets)
                },
                **allocation.diagnostics,
            }
            for asset in self.config.assets:
                estimate = estimates[asset]
                row[f"expected_return_{asset}"] = estimate.mean
                row[f"predictive_std_{asset}"] = estimate.predictive_std
                row[f"training_observations_{asset}"] = estimate.observations
                row[f"last_training_month_{asset}"] = estimate.last_training_month
                row[f"used_mean_fallback_{asset}"] = (
                    estimate.used_unconditional_fallback
                )
            rows.append(row)

            # 월말 drift: 다음 리밸런싱 직전 비중은 보유자산별 실현수익률에 따라
            # 변한다. 비용은 현금에서 빠지지만 상대 자산비중 drift에는 gross return을
            # 사용한 Stage49 회계 규칙을 그대로 유지한다.
            pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
            first_trade = False

        result = pd.DataFrame(rows).set_index("month")
        result.index = pd.PeriodIndex(result.index, freq="M")
        return result


# ---------------------------------------------------------------------------
# 6. 검증·성과·실행 조립점
# ---------------------------------------------------------------------------


def _performance_summary(returns: pd.Series) -> dict[str, float]:
    clean = pd.Series(returns).dropna()
    wealth = (1.0 + clean).cumprod()
    years = len(clean) / 12.0
    cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0
    volatility = float(clean.std(ddof=1) * math.sqrt(12.0))
    sharpe = (
        float(clean.mean() / clean.std(ddof=1) * math.sqrt(12.0))
        if clean.std(ddof=1) > 0.0
        else math.nan
    )
    drawdown = wealth / wealth.cummax() - 1.0
    maximum_drawdown = float(drawdown.min())
    downside = float(
        np.sqrt(np.mean(np.minimum(clean, 0.0) ** 2)) * math.sqrt(12.0)
    )
    return {
        "Months": float(len(clean)),
        "CAGR": float(cagr),
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": float(clean.mean() * 12.0 / downside) if downside > 0 else math.nan,
        "MDD": maximum_drawdown,
        "Calmar": float(cagr / abs(maximum_drawdown))
        if maximum_drawdown < 0
        else math.nan,
        "FinalMultiple": float(wealth.iloc[-1]),
        "PositiveMonths": float((clean > 0.0).mean()),
    }


def performance_table(
    monthly_result: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    periods = {
        "full_common": (monthly_result.index.min(), monthly_result.index.max()),
        "locked_2018_2026": (config.locked_month, monthly_result.index.max()),
    }
    rows: list[dict[str, Any]] = []
    for name, (start, end) in periods.items():
        view = monthly_result.loc[start:end]
        rows.append(
            {
                "Strategy": "Stage50_ExplainableRegimeAllocator",
                "Period": name,
                "Start": str(view.index.min()),
                "End": str(view.index.max()),
                **_performance_summary(view["return"]),
                "AvgTurnover": float(view["turnover"].mean()),
                "TotalCost": float(view[["trade_cost", "fx_cost"]].sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def validation_report(
    result: pd.DataFrame,
    config: StrategyConfig,
    data_audit: dict[str, Any],
) -> dict[str, Any]:
    weights = result[[f"w_{asset}" for asset in config.assets]]
    training_lag_checks = []
    for asset in config.assets:
        available = result[f"last_training_month_{asset}"].notna()
        last_month = pd.PeriodIndex(
            result.loc[available, f"last_training_month_{asset}"], freq="M"
        )
        training_lag_checks.extend(last_month < result.index[available])

    checks = {
        "finite_returns": bool(np.isfinite(result["return"]).all()),
        "finite_weights": bool(np.isfinite(weights.to_numpy()).all()),
        "long_only": bool(weights.min().min() >= -1e-8),
        "fully_invested": bool(
            np.allclose(weights.sum(axis=1), 1.0, atol=1e-8)
        ),
        "macro_signal_strictly_lagged": bool(
            (pd.PeriodIndex(result["macro_signal_month"], freq="M") < result.index).all()
        ),
        "training_strictly_precedes_decision": bool(all(training_lag_checks)),
        "covariance_strictly_lagged": bool(
            (
                pd.to_datetime(result["covariance_cutoff"])
                <= pd.DatetimeIndex(
                    [(month - 1).to_timestamp("M") for month in result.index]
                )
            ).all()
        ),
        "positive_definite_covariance": bool(
            result["covariance_min_eigenvalue"].min() > 0.0
        ),
        "volatility_policy_satisfied": bool(
            result["volatility_slack"].min() >= -1e-7
        ),
        "cdar_policy_satisfied": bool(result["cdar_slack"].min() >= -1e-7),
    }
    return {
        "strategy": "Stage50_ExplainableRegimeAllocator",
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "months": int(len(result)),
        "start": str(result.index.min()),
        "end": str(result.index.max()),
        "fallback_solver_months": int(result["used_fallback"].sum()),
        "binding_constraints": {
            "annual_volatility": int(
                result["volatility_slack"].abs().lt(1e-6).sum()
            ),
            "historical_cdar": int(result["cdar_slack"].abs().lt(1e-6).sum()),
        },
        "known_statistical_limitations": {
            "regime_scores_are_calibrated_probabilities": False,
            "ridge_prediction_is_reference_category_invariant": False,
            "posterior_mean_uncertainty_enters_allocator": False,
            "macro_point_in_time_vintages_verified": False,
        },
        "declared_heuristics": {
            "covariance_lookback_days": config.covariance_lookback_days,
            "trading_days_per_month": config.trading_days_per_month,
            "observations_per_parameter": config.observations_per_parameter,
            "annual_volatility_cap": config.annual_volatility_cap,
            "maximum_cdar": config.maximum_cdar,
            "cdar_confidence": config.cdar_confidence,
            "proportional_trade_cost": config.proportional_trade_cost,
            "foreign_exchange_cost": config.foreign_exchange_cost,
        },
        "data_audit": data_audit,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Period, pd.Timestamp, Path)):
        return str(value)
    return value


def run_research(
    *, save: bool = True, refresh_monthly_market_cache: bool = False
) -> dict[str, Any]:
    """Composition root: 구체 구현을 한 번 조립하고 단일 전략을 실행한다."""

    paths = RepositoryPaths.from_this_file()
    config = StrategyConfig()
    monthly_returns, monthly_audit = MonthlyAssetReturnRepository(
        paths, config
    ).load(refresh_monthly_market_cache)
    daily_returns, daily_audit = DailyRiskReturnRepository(paths, config).load()
    regime_scores, macro_audit = MacroRegimeRepository(paths, config).build_scores(
        monthly_returns.index
    )

    transaction_cost = TransactionCostModel(config)
    engine = BacktestEngine(
        config=config,
        return_forecaster=BayesianRegimeReturnForecaster(config),
        covariance_forecaster=LedoitWolfMonthlyCovarianceForecaster(config),
        allocator=RiskConstrainedAllocator(config, transaction_cost),
        transaction_cost=transaction_cost,
    )
    monthly_result = engine.run(monthly_returns, regime_scores, daily_returns)
    performance = performance_table(monthly_result, config)
    report = validation_report(
        monthly_result,
        config,
        {
            "monthly_returns": monthly_audit,
            "daily_risk_returns": daily_audit,
            "macro_regime_scores": macro_audit,
        },
    )

    if save:
        paths.output.mkdir(parents=True, exist_ok=True)
        monthly_result.to_csv(paths.output / "monthly_results.csv")
        performance.to_csv(paths.output / "performance.csv", index=False)
        with (paths.output / "validation_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                _json_ready(report), handle, ensure_ascii=False, indent=2
            )
    return {
        "monthly_result": monthly_result,
        "performance": performance,
        "validation_report": report,
        "config": asdict(config),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the standalone explainable regime allocation strategy."
    )
    parser.add_argument(
        "--refresh-monthly-market-cache",
        action="store_true",
        help="Download and replace cache/market_daily.csv before the run.",
    )
    args = parser.parse_args(argv)
    research = run_research(
        save=True,
        refresh_monthly_market_cache=args.refresh_monthly_market_cache,
    )
    print(research["performance"].to_string(index=False))
    print(
        "\nValidation:",
        "PASS" if research["validation_report"]["all_checks_pass"] else "FAIL",
    )


if __name__ == "__main__":
    main()
