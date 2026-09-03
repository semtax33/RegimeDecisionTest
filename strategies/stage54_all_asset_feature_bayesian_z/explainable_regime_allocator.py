from __future__ import annotations

"""자산별 feature를 추가한 Bayesian z-score Stage54 자산배분 전략.

이 파일은 Stage51의 GSG universe와 Bayesian ridge 기대수익 추정은 유지하되,
각 자산 regression에 해당 자산군 전용 feature를 추가한다.
SLSQP 최적화 대신 forecast z-score로 직접 long-only 비중을 정하는 연구 변형이다.
다른 ``strategies.stageXX`` 모듈을 import하지 않는다. 처음 코드를 보는 사람은
아래의 한 방향 흐름만 따라가면 된다.

    원자료 -> 월별 자산수익률 / 거시 국면점수 / 일별 KRW 수익률
           -> 자산별 fundamental/technical/option/curve/volatility feature
           -> Bayesian ridge 월 기대수익률
           -> z = expected_return / predictive_std
           -> betting_size = clip(2 * NormalCDF(z) - 1, 0, 1)
           -> betting_size 합계로 정규화한 long-only 비중
           -> 다음 한 달 보유수익률

리팩터링의 경계
----------------
이 구현은 Stage54의 simple return 기본값, GSG universe, Bayesian z allocator를
보존하되 각 자산 기대수익 regression의 feature set을 확장한다. 따라서 Stage50/51의
SLSQP 위험예산 충족 성과를 보존하지 않는다. 앞선
감사에서 발견된 통계적 한계는 조용히 고치지 않고 코드 가까이에 남긴다.

1. Goldilocks를 기준범주로 제외한 ridge 회귀는 기준범주 불변이 아니다.
2. Bayesian 예측 표준편차는 투자 강도에 쓰지만 공분산 불확실성으로 반영하지 않는다.
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
from scipy.stats import norm
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

    assets: tuple[str, ...] = ("KODEX200", "BOND", "GLD", "GSG")
    regime_columns: tuple[str, ...] = (
        "p_Goldilocks",
        "p_Overheating",
        "p_Slowdown",
        "p_Stagflation",
    )

    # COMPATIBILITY HEURISTIC: 네 score의 합이 1이므로 intercept와 함께 네
    # 변수를 모두 쓰면 완전 다중공선성이 생긴다. Stage50은 Goldilocks를 기준으로
    # 제외했다. OLS 예측은 기준범주에 불변이지만 ridge prior는 불변이 아니다.
    # 결과 재현을 위해 유지하며, 새 경제 모형에서는 대칭 contrast가 더 적절하다.
    forecast_features: tuple[str, ...] = (
        "p_Overheating",
        "p_Slowdown",
        "p_Stagflation",
    )
    kospi_feature_columns: tuple[str, ...] = (
        "kospi_eps_revision_z",
        "kospi_valuation_gap_z",
        "kospi_credit_widening_z",
        "kospi_k_score",
        "kospi_option_direction_score",
        "kospi_abnormal_bear_pressure_fast",
        "kospi_abnormal_bear_pressure_slow",
    )
    bond_feature_columns: tuple[str, ...] = (
        "bond_carry_5y_monthly_z",
        "bond_roll_down_5y_to_3y_monthly_z",
        "bond_rate_momentum_5y_monthly_z",
        "bond_curve_slope_10y_minus_3y_pctpt_z",
        "bond_curve_total_return_proxy_z",
        "bond_k_score",
        "bond_atr_percentile",
    )
    gld_feature_columns: tuple[str, ...] = (
        "gld_real_yield_support",
        "gld_fx_support",
        "gld_gold_trend_support",
        "gld_gold_composite_state",
        "gld_gvz_causal_rank",
        "gld_k_score",
        "gld_atr_percentile",
    )
    gsg_feature_columns: tuple[str, ...] = (
        "gsg_ovx_causal_rank",
        "gsg_return_momentum_3m",
        "gsg_return_momentum_12m",
        "gsg_return_volatility_3m_rank",
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
    # 위원회가 정해야 할 위험 예산이다. Stage50과 같은 Stage36 값을
    # 유지한다. 실제 운용 전에는 반드시 별도 민감도 분석을 해야 한다.
    annual_volatility_cap: float = 0.13
    maximum_cdar: float = 0.16
    cdar_confidence: float = 0.90

    # COST HEURISTIC: 고정 거래비용은 시기별 spread, 세금, 시장충격을 설명하지
    # 않는다. 국내/해외 전 자산의 절대 비중변화에 15bp를 적용하고 GLD+GSG의
    # 순 USD 흐름에만 5bp 환전비용을 추가한다.
    proportional_trade_cost: float = 0.0015
    foreign_exchange_cost: float = 0.0005

    # RETURN DEFINITION: 기본은 단순 자산수익률이다. True로 명시한 실험에서만
    # Bayesian regression과 실현 PnL에 들어가는 자산수익률을 CD(91일) 기준
    # 초과수익률로 바꾼다.
    use_risk_free_rate: bool = False
    risk_free_rate_item: str = "CD(91일)"
    risk_free_day_count: float = 365.0

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

    def forecast_features_for_asset(self, asset: str) -> tuple[str, ...]:
        return (*self.forecast_features, *self.asset_feature_columns(asset))

    def asset_feature_columns(self, asset: str) -> tuple[str, ...]:
        if asset == "KODEX200":
            return self.kospi_feature_columns
        if asset == "BOND":
            return self.bond_feature_columns
        if asset == "GLD":
            return self.gld_feature_columns
        if asset == "GSG":
            return self.gsg_feature_columns
        return ()

    @property
    def all_asset_feature_columns(self) -> tuple[str, ...]:
        return (
            *self.kospi_feature_columns,
            *self.bond_feature_columns,
            *self.gld_feature_columns,
            *self.gsg_feature_columns,
        )


@dataclass(frozen=True)
class ForecastEstimate:
    mean: float
    predictive_std: float
    observations: int
    last_training_month: pd.Period | None
    used_unconditional_fallback: bool
    intercept: float
    standardized_intercept: float
    coefficients: dict[str, float]
    standardized_coefficients: dict[str, float]
    feature_means: dict[str, float]
    feature_standard_deviations: dict[str, float]


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
        predictive_standard_deviations: np.ndarray,
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


class RiskFreeRateRepository:
    """일별 시장금리 파일에서 CD(91일) risk-free return을 만든다."""

    def __init__(self, paths: RepositoryPaths, config: StrategyConfig) -> None:
        self.paths = paths
        self.config = config

    def _source_path(self) -> Path:
        return _find_unicode_safe_file(
            self.paths.raw_data, "시장금리(일별)_28181724.csv"
        )

    def annual_rate(self) -> pd.Series:
        source = self._source_path()
        raw = pd.read_csv(source, encoding="utf-8-sig")
        matched = raw.loc[
            raw["계정항목"].astype(str).eq(self.config.risk_free_rate_item)
        ]
        if matched.empty:
            raise ValueError(
                f"{source} has no row for {self.config.risk_free_rate_item}"
            )
        metadata = {"통계표", "계정항목", "단위", "변환"}
        values = matched.drop(columns=list(metadata.intersection(raw.columns))).iloc[0]
        dates = pd.to_datetime(values.index, errors="coerce")
        rates = pd.to_numeric(values.to_numpy(), errors="coerce")
        series = pd.Series(rates, index=dates, dtype=float).dropna()
        series = series.loc[~series.index.duplicated(keep="last")].sort_index()
        if series.empty:
            raise ValueError(f"{source} has no usable {self.config.risk_free_rate_item}")
        return series / 100.0

    def daily_returns(self, index: pd.DatetimeIndex) -> pd.Series:
        annual = self.annual_rate()
        dates = pd.DatetimeIndex(index).normalize()
        aligned = annual.reindex(dates, method="ffill")
        if aligned.isna().any():
            first_missing = dates[aligned.isna()][0]
            raise ValueError(
                f"{self.config.risk_free_rate_item} does not cover {first_missing.date()}"
            )
        return pd.Series(
            aligned.to_numpy(dtype=float) / self.config.risk_free_day_count,
            index=index,
            name="risk_free_return",
        )

    def monthly_returns(self, months: pd.PeriodIndex) -> pd.Series:
        annual = self.annual_rate()
        calendar = pd.date_range(annual.index.min(), annual.index.max(), freq="D")
        daily_rate = annual.reindex(calendar, method="ffill")
        daily_return = daily_rate / self.config.risk_free_day_count
        monthly = (1.0 + daily_return).groupby(calendar.to_period("M")).prod() - 1.0
        monthly = monthly.reindex(pd.PeriodIndex(months, freq="M"))
        if monthly.isna().any():
            first_missing = monthly.index[monthly.isna()][0]
            raise ValueError(
                f"{self.config.risk_free_rate_item} monthly return is missing "
                f"for {first_missing}"
            )
        monthly.name = "risk_free_return"
        return monthly

    def audit(self) -> dict[str, Any]:
        annual = self.annual_rate()
        return {
            "source": str(self._source_path()),
            "rate_item": self.config.risk_free_rate_item,
            "unit": "annual percent in source, converted to decimal return",
            "day_count": self.config.risk_free_day_count,
            "start": str(annual.index.min().date()),
            "end": str(annual.index.max().date()),
            "rows": int(len(annual)),
        }


class AssetFeatureRepository:
    """이전 stage가 만든 자산별 월별 feature를 읽고 결합한다."""

    def __init__(self, paths: RepositoryPaths, config: StrategyConfig) -> None:
        self.paths = paths
        self.config = config

    @staticmethod
    def _read_monthly_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Required asset feature file not found: {path}")
        frame = pd.read_csv(path)
        index_column = "target_month" if "target_month" in frame.columns else "month"
        if index_column not in frame.columns:
            raise ValueError(f"{path} has no target_month or month column")
        frame[index_column] = pd.PeriodIndex(frame[index_column], freq="M")
        return frame.set_index(index_column).sort_index()

    @staticmethod
    def _causal_expanding_midrank(series: pd.Series) -> pd.Series:
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

    def build(
        self,
        decision_months: pd.PeriodIndex,
        monthly_returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        output = pd.DataFrame(index=pd.PeriodIndex(decision_months, freq="M"))
        root = self.paths.project_root / "strategies"

        fundamental_path = (
            root
            / "stage35_earnings_credit_fundamentals"
            / "outputs"
            / "monthly_earnings_credit_signals.csv"
        )
        fundamental = self._read_monthly_csv(fundamental_path)
        output["kospi_eps_revision_z"] = fundamental["eps_revision_z"]
        output["kospi_valuation_gap_z"] = fundamental["valuation_gap_z"]
        output["kospi_credit_widening_z"] = fundamental["credit_widening_z"]

        technical_path = (
            root
            / "stage24_equity_k_ratio_only"
            / "outputs"
            / "monthly_technical_signals.csv"
        )
        technical = self._read_monthly_csv(technical_path)
        output["kospi_k_score"] = technical["k_score_KODEX200"]
        output["bond_k_score"] = technical["k_score_BOND"]
        output["bond_atr_percentile"] = technical["atr_percentile_BOND"]
        output["gld_k_score"] = technical["k_score_GLD"]
        output["gld_atr_percentile"] = technical["atr_percentile_GLD"]

        ods_path = (
            root
            / "stage28_option_directional_surface"
            / "outputs"
            / "monthly_option_direction_signals.csv"
        )
        ods = self._read_monthly_csv(ods_path)
        output["kospi_option_direction_score"] = ods["option_direction_score"]

        abnormal_path = (
            root
            / "stage30_abnormal_surface_erp"
            / "outputs"
            / "monthly_option_alpha_signals.csv"
        )
        abnormal = self._read_monthly_csv(abnormal_path)
        output["kospi_abnormal_bear_pressure_fast"] = abnormal[
            "abnormal_bear_pressure_fast"
        ]
        output["kospi_abnormal_bear_pressure_slow"] = abnormal[
            "abnormal_bear_pressure_slow"
        ]

        bond_path = (
            root
            / "stage37_bond_curve_alpha"
            / "outputs"
            / "monthly_bond_curve_signals.csv"
        )
        bond = self._read_monthly_csv(bond_path)
        output["bond_carry_5y_monthly_z"] = bond["carry_5y_monthly_z"]
        output["bond_roll_down_5y_to_3y_monthly_z"] = bond[
            "roll_down_5y_to_3y_monthly_z"
        ]
        output["bond_rate_momentum_5y_monthly_z"] = bond[
            "rate_momentum_5y_monthly_z"
        ]
        output["bond_curve_slope_10y_minus_3y_pctpt_z"] = bond[
            "curve_slope_10y_minus_3y_pctpt_z"
        ]
        output["bond_curve_total_return_proxy_z"] = bond[
            "curve_total_return_proxy_z"
        ]

        gold_path = (
            root
            / "stage38_gold_state_alpha"
            / "outputs"
            / "monthly_gold_state_signals.csv"
        )
        gold = self._read_monthly_csv(gold_path)
        output["gld_real_yield_support"] = gold["real_yield_support"]
        output["gld_fx_support"] = gold["fx_support"]
        output["gld_gold_trend_support"] = gold["gold_trend_support"]
        output["gld_gold_composite_state"] = gold["gold_composite_state"]

        asset_vol_path = (
            root
            / "stage36_asset_implied_volatility_risk"
            / "outputs"
            / "monthly_asset_volatility_signals.csv"
        )
        asset_vol = self._read_monthly_csv(asset_vol_path)
        output["gld_gvz_causal_rank"] = asset_vol["gvz_causal_rank"]
        output["gsg_ovx_causal_rank"] = asset_vol["ovx_causal_rank"]

        gsg_returns = monthly_returns.loc[:, "GSG"].astype(float)
        known_gsg_returns = gsg_returns.shift(1)
        log_gsg_returns = np.log1p(known_gsg_returns)
        output["gsg_return_momentum_3m"] = (
            np.exp(log_gsg_returns.rolling(3, min_periods=3).sum()) - 1.0
        )
        output["gsg_return_momentum_12m"] = (
            np.exp(log_gsg_returns.rolling(12, min_periods=12).sum()) - 1.0
        )
        gsg_volatility_3m = known_gsg_returns.rolling(3, min_periods=3).std(ddof=1)
        output["gsg_return_volatility_3m_rank"] = self._causal_expanding_midrank(
            gsg_volatility_3m
        )

        output = output.replace([np.inf, -np.inf], np.nan)
        audit = {
            "feature_scope": "asset-specific regression features",
            "features_by_asset": {
                asset: list(self.config.asset_feature_columns(asset))
                for asset in self.config.assets
            },
            "source_files": {
                "fundamental": str(fundamental_path),
                "technical": str(technical_path),
                "option_directional_surface": str(ods_path),
                "abnormal_option_surface": str(abnormal_path),
                "bond_curve": str(bond_path),
                "gold_state": str(gold_path),
                "asset_implied_volatility": str(asset_vol_path),
                "gsg_momentum": "derived from lagged historical_monthly_returns.csv",
            },
            "rows": int(len(output)),
            "start": str(output.index.min()),
            "end": str(output.index.max()),
            "complete_rows": int(output.dropna().shape[0]),
        }
        return output, audit


class MonthlyAssetReturnRepository:
    """월초 시가 기준 네 자산의 원화 수익률을 만든다.

    경제적 의미
    -----------
    의사결정월 t의 수익률은 t월 첫 거래일 시가에서 t+1월 첫 거래일 시가까지다.
    GLD와 GSG는 각 첫 거래일 USDKRW로 원화 환산한다. KODEX200 상장 전 구간은
    KOSPI200 proxy를 실제 ETF 시가에 연결한다.

    DATA HEURISTIC: 서로 다른 공급원의 지수/ETF를 한 점에서 비례 연결하면 수준은
    연속이지만 상품구조·배당·추적오차가 같다는 뜻은 아니다. Stage50과 비교 가능한
    연구 표본을 만들기 위한 데이터 계약이다.
    """

    def __init__(self, paths: RepositoryPaths, config: StrategyConfig) -> None:
        self.paths = paths
        self.config = config

    def _market_cache(self, refresh: bool) -> pd.DataFrame:
        cache = self.paths.cache / "market_daily.csv"
        required_symbols = {"KODEX200", "GLD", "GSG", "USDKRW"}
        if cache.exists() and not refresh:
            cached = pd.read_csv(cache, parse_dates=["date"])
            missing = required_symbols.difference(
                set(cached["symbol"].dropna().astype(str))
            )
            if missing:
                raise ValueError(
                    f"{cache} is missing {sorted(missing)}. "
                    "Run Stage52 with --refresh-monthly-market-cache to download GSG."
                )
            return cached

        # yfinance는 cache 재생성이 필요할 때만 선택적으로 import한다.
        try:
            import yfinance as yf
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Refreshing Stage52 market cache requires yfinance. "
                "Install requirements.txt or the research extra before running "
                "--refresh-monthly-market-cache."
            ) from error

        rows: list[pd.DataFrame] = []
        instruments = (
            ("069500.KS", "KODEX200"),
            ("GLD", "GLD"),
            ("GSG", "GSG"),
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
        if cache.exists():
            cached = pd.read_csv(cache, parse_dates=["date"])
            result = pd.concat(
                [
                    cached.loc[~cached["symbol"].astype(str).isin(required_symbols)],
                    result,
                ],
                ignore_index=True,
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
            "GSG": market.loc[market["symbol"].eq("GSG")],
        }
        first_open: dict[str, pd.Series] = {}
        for asset, source in sources.items():
            clean = source.dropna(subset=["open"]).sort_values("date").copy()
            clean["month"] = clean["date"].dt.to_period("M")
            first = clean.groupby("month", sort=True).first()
            value = first["open"].astype(float)
            if asset in {"GLD", "GSG"}:
                aligned_fx = fx.reindex(
                    pd.DatetimeIndex(first["date"]), method="ffill"
                )
                value = value * aligned_fx.to_numpy(dtype=float)
            first_open[asset] = value

        levels = pd.concat(first_open, axis=1).sort_index()
        raw_returns = levels.shift(-1).div(levels).sub(1.0).dropna(how="any")
        raw_returns = raw_returns.loc[:, self.config.assets]
        returns = raw_returns.copy()
        risk_free_audit: dict[str, Any] | None = None
        if self.config.use_risk_free_rate:
            risk_free = RiskFreeRateRepository(
                self.paths, self.config
            ).monthly_returns(raw_returns.index)
            returns = raw_returns.sub(risk_free, axis=0)
            risk_free_audit = {
                **RiskFreeRateRepository(self.paths, self.config).audit(),
                "monthly_return_definition": "calendar-month compounded daily CD return",
                "monthly_start": str(risk_free.index.min()),
                "monthly_end": str(risk_free.index.max()),
            }
        audit = {
            "market_cache": str(self.paths.cache / "market_daily.csv"),
            "monthly_return_definition": "first-open(t+1)/first-open(t)-1",
            "return_type": "excess_return" if self.config.use_risk_free_rate else "simple_return",
            "risk_free_enabled": self.config.use_risk_free_rate,
            "risk_free": risk_free_audit,
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
    비동시 거래 공분산 편향이 생길 수 있다. Stage50 구조와 비교 가능하도록 유지한다.
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

    def _market_daily_supplement(self, symbol: str) -> pd.DataFrame:
        cache = self.paths.cache / "market_daily.csv"
        if not cache.exists():
            return pd.DataFrame()
        raw = pd.read_csv(cache, parse_dates=["date"])
        source = raw.loc[raw["symbol"].astype(str).eq(symbol)].copy()
        if source.empty:
            return pd.DataFrame()
        source = source.set_index("date").sort_index()
        supplement = pd.DataFrame(index=source.index)
        for column in ("open", "high", "low", "close", "volume"):
            if column in source:
                supplement[column] = pd.to_numeric(source[column], errors="coerce")
            else:
                supplement[column] = np.nan
        return self._numeric_ohlcv(supplement)

    def load(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        cache = self.paths.cache / "regime_lightgbm_ohlcv.csv"
        raw = pd.read_csv(cache, parse_dates=["date"])
        market = {
            str(symbol): self._numeric_ohlcv(group.set_index("date"))
            for symbol, group in raw.groupby("symbol")
        }
        required = {"KODEX200", "GLD", "GSG", "USDKRW"}
        for symbol in sorted(required.difference(market)):
            supplement = self._market_daily_supplement(symbol)
            if not supplement.empty:
                market[symbol] = supplement
        missing = sorted(required.difference(market))
        if missing:
            raise ValueError(
                f"OHLCV cache is missing symbols: {missing}. "
                "For GSG, run Stage52 with --refresh-monthly-market-cache first."
            )

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
        for asset in ("GLD", "GSG"):
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
            "GSG": foreign["GSG"],
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
        risk_free_audit: dict[str, Any] | None = None
        if self.config.use_risk_free_rate:
            risk_free = RiskFreeRateRepository(
                self.paths, self.config
            ).daily_returns(pd.DatetimeIndex(daily_returns.index))
            daily_returns = daily_returns.sub(risk_free, axis=0)
            risk_free_audit = {
                **RiskFreeRateRepository(self.paths, self.config).audit(),
                "daily_return_definition": "annual CD rate / day_count",
            }
        audit = {
            "ohlcv_cache": str(cache),
            "foreign_assets_in_krw": True,
            "return_type": "excess_return" if self.config.use_risk_free_rate else "simple_return",
            "risk_free_enabled": self.config.use_risk_free_rate,
            "risk_free": risk_free_audit,
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
    * predictive_std는 ``x @ model.sigma_ @ x.T``로 계산한 posterior mean
      uncertainty다. sklearn의 ``return_std=True``처럼 잔차/noise까지 더한 전체
      예측분산이 아니다.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def _fit_one(
        self,
        history: pd.DataFrame,
        asset: str,
        target: str,
        current: pd.Series,
    ) -> ForecastEstimate:
        features = self.config.forecast_features_for_asset(asset)
        complete = history[[target, *features]].dropna()
        minimum = self.config.observations_per_parameter * (len(features) + 1)
        zero_coefficients = {feature: 0.0 for feature in features}
        empty_feature_statistics = {feature: float("nan") for feature in features}
        if len(complete) < minimum or current[list(features)].isna().any():
            available = history[target].dropna()
            fallback_mean = float(available.mean())
            return ForecastEstimate(
                mean=fallback_mean,
                predictive_std=float(available.std(ddof=1)),
                observations=int(len(available)),
                last_training_month=available.index.max() if len(available) else None,
                used_unconditional_fallback=True,
                intercept=fallback_mean,
                standardized_intercept=fallback_mean,
                coefficients=zero_coefficients.copy(),
                standardized_coefficients=zero_coefficients.copy(),
                feature_means=empty_feature_statistics.copy(),
                feature_standard_deviations=empty_feature_statistics.copy(),
            )

        predictors = complete.loc[:, features].astype(float)
        means = predictors.mean()
        standard_deviations = predictors.std(ddof=1)
        usable = standard_deviations.gt(self.config.numerical_epsilon)
        if not usable.any():
            available = history[target].dropna()
            fallback_mean = float(available.mean())
            return ForecastEstimate(
                mean=fallback_mean,
                predictive_std=float(available.std(ddof=1)),
                observations=int(len(available)),
                last_training_month=available.index.max() if len(available) else None,
                used_unconditional_fallback=True,
                intercept=fallback_mean,
                standardized_intercept=fallback_mean,
                coefficients=zero_coefficients.copy(),
                standardized_coefficients=zero_coefficients.copy(),
                feature_means=empty_feature_statistics.copy(),
                feature_standard_deviations=empty_feature_statistics.copy(),
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
        current_x_array = current_x.to_numpy(dtype=float).reshape(1, -1)
        prediction = model.predict(current_x_array)
        mean_variance = float(
            (current_x_array @ model.sigma_ @ current_x_array.T).item()
        )
        mean_std = float(np.sqrt(max(mean_variance, 0.0)))
        standardized_coefficients = {feature: 0.0 for feature in features}
        coefficients = {feature: 0.0 for feature in features}
        feature_means = {feature: float(means.loc[feature]) for feature in features}
        feature_standard_deviations = {
            feature: float(standard_deviations.loc[feature]) for feature in features
        }
        raw_intercept = float(model.intercept_)
        for feature, standardized_beta in zip(columns, model.coef_):
            feature_mean = float(means.loc[feature])
            feature_std = float(standard_deviations.loc[feature])
            raw_beta = float(standardized_beta / feature_std)
            standardized_coefficients[feature] = float(standardized_beta)
            coefficients[feature] = raw_beta
            raw_intercept -= raw_beta * feature_mean
        return ForecastEstimate(
            mean=float(prediction[0]),
            predictive_std=mean_std,
            observations=int(len(complete)),
            last_training_month=complete.index.max(),
            used_unconditional_fallback=False,
            intercept=raw_intercept,
            standardized_intercept=float(model.intercept_),
            coefficients=coefficients,
            standardized_coefficients=standardized_coefficients,
            feature_means=feature_means,
            feature_standard_deviations=feature_standard_deviations,
        )

    def forecast(
        self, feature_history: pd.DataFrame, current_features: pd.Series
    ) -> dict[str, ForecastEstimate]:
        return {
            asset: self._fit_one(
                feature_history, asset, f"return_{asset}", current_features
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
            [config.assets.index("GLD"), config.assets.index("GSG")]
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


class BayesianZBettingSizeAllocator:
    """Bayesian forecast z-score만으로 long-only 비중을 만든다.

    사용자가 제안한 규칙을 그대로 따른다.

        z_i = expected_return_i / predictive_std_i
        betting_size_i = clip(2 * Phi(z_i) - 1, 0, 1)
        w_i = betting_size_i / sum_j betting_size_j

    모든 betting size가 0이면 정규화할 수 없으므로 같은 비중으로 후퇴하고 이를
    diagnostics에 기록한다. 이 allocator는 SLSQP를 쓰지 않으므로 Stage50/51의
    변동성·CDaR 한도는 제약이 아니라 사후 진단값이다.
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
        predictive_standard_deviations: np.ndarray,
        covariance: np.ndarray,
        pretrade_weights: np.ndarray,
    ) -> AllocationResult:
        historical = historical_monthly_returns.loc[:, self.config.assets]
        historical_values = historical.to_numpy(dtype=float)

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

        z_score = np.divide(
            expected_returns,
            predictive_standard_deviations,
            out=np.zeros_like(expected_returns, dtype=float),
            where=predictive_standard_deviations > self.config.numerical_epsilon,
        )
        z_score = np.where(np.isfinite(z_score), z_score, 0.0)
        betting_size = np.clip(2.0 * norm.cdf(z_score) - 1.0, 0.0, 1.0)
        betting_sum = float(betting_size.sum())
        used_equal_weight_fallback = betting_sum <= self.config.numerical_epsilon
        if used_equal_weight_fallback:
            weights = np.repeat(1.0 / len(self.config.assets), len(self.config.assets))
        else:
            weights = betting_size / betting_sum

        values = portfolio_values(weights)
        annual_vol = annual_volatility(weights)
        historical_cdar = _conditional_drawdown_at_risk(
            historical_values @ weights, self.config.cdar_confidence
        )
        diagnostics = {
            **values,
            "solver_success": True,
            "used_fallback": used_equal_weight_fallback,
            "used_equal_weight_fallback": used_equal_weight_fallback,
            "solver_status": 0,
            "solver_message": "closed-form Bayesian z-score betting size",
            "solver_iterations": 0,
            "objective_value": -values["monthly_certainty_equivalent"],
            "expected_annual_volatility": annual_vol,
            "historical_cdar": historical_cdar,
            "sum_error": abs(float(weights.sum()) - 1.0),
            "volatility_slack": self.config.annual_volatility_cap - annual_vol,
            "cdar_slack": self.config.maximum_cdar + historical_cdar,
            **{
                f"z_score_{asset}": float(z_score[index])
                for index, asset in enumerate(self.config.assets)
            },
            **{
                f"betting_size_{asset}": float(betting_size[index])
                for index, asset in enumerate(self.config.assets)
            },
            "betting_size_sum": betting_sum,
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
        feature_columns = [
            *self.config.regime_columns,
            *self.config.all_asset_feature_columns,
        ]
        available_features = [
            column for column in feature_columns if column in regime_scores.columns
        ]
        frame = regime_scores.loc[:, available_features].copy()
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
            predictive_standard_deviations = np.array(
                [estimates[asset].predictive_std for asset in self.config.assets],
                dtype=float,
            )
            allocation = self.allocator.allocate(
                historical_returns,
                expected_returns,
                predictive_standard_deviations,
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
                    column: float(regime_scores.loc[month, column])
                    if column in regime_scores and pd.notna(regime_scores.loc[month, column])
                    else math.nan
                    for column in self.config.all_asset_feature_columns
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
                row[f"regime_intercept_{asset}"] = estimate.intercept
                row[f"standardized_regime_intercept_{asset}"] = (
                    estimate.standardized_intercept
                )
                for feature in self.config.forecast_features_for_asset(asset):
                    feature_name = feature.removeprefix("p_")
                    row[f"regime_beta_{asset}_{feature_name}"] = (
                        estimate.coefficients[feature]
                    )
                    row[f"standardized_regime_beta_{asset}_{feature_name}"] = (
                        estimate.standardized_coefficients[feature]
                    )
                    row[f"regime_feature_mean_{asset}_{feature_name}"] = (
                        estimate.feature_means[feature]
                    )
                    row[f"regime_feature_std_{asset}_{feature_name}"] = (
                        estimate.feature_standard_deviations[feature]
                    )
            rows.append(row)

            # 월말 drift: 다음 리밸런싱 직전 비중은 보유자산별 실현수익률에 따라
            # 변한다. 비용은 현금에서 빠지지만 상대 자산비중 drift에는 gross return을
            # 사용한 Stage50 회계 규칙을 그대로 유지한다.
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
    periods = analysis_periods(monthly_result, config)
    rows: list[dict[str, Any]] = []
    for name, view in periods.items():
        rows.append(
            {
                "Strategy": "Stage54_AllAssetFeatureBayesianZ",
                "Period": name,
                "Start": str(view.index.min()),
                "End": str(view.index.max()),
                **_performance_summary(view["return"]),
                "AvgTurnover": float(view["turnover"].mean()),
                "TotalCost": float(view[["trade_cost", "fx_cost"]].sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def analysis_periods(
    monthly_result: pd.DataFrame, config: StrategyConfig
) -> dict[str, pd.DataFrame]:
    """Return the common reporting windows used by performance and charts."""

    periods = {
        "full_common": (monthly_result.index.min(), monthly_result.index.max()),
        "locked_2018_2026": (config.locked_month, monthly_result.index.max()),
    }
    return {
        name: monthly_result.loc[start:end]
        for name, (start, end) in periods.items()
    }


def weight_summary_table(
    monthly_result: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    """Summarize each asset's realized portfolio weights by reporting period."""

    rows: list[dict[str, Any]] = []
    for period, view in analysis_periods(monthly_result, config).items():
        for asset in config.assets:
            weights = view[f"w_{asset}"].astype(float)
            rows.append(
                {
                    "Period": period,
                    "Asset": asset,
                    "Start": str(view.index.min()),
                    "End": str(view.index.max()),
                    "Months": int(len(view)),
                    "AverageWeight": float(weights.mean()),
                    "MedianWeight": float(weights.median()),
                    "MinWeight": float(weights.min()),
                    "MaxWeight": float(weights.max()),
                    "LastWeight": float(weights.iloc[-1]),
                    "ZeroWeightMonths": int(weights.le(1e-12).sum()),
                }
            )
    return pd.DataFrame(rows)


def save_weight_artifacts(
    monthly_result: pd.DataFrame, config: StrategyConfig, output: Path
) -> pd.DataFrame:
    """Save period-level weight paths as CSV files and stacked area charts."""

    import os

    plot_config = output / ".matplotlib"
    plot_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for period, view in analysis_periods(monthly_result, config).items():
        weights = view[[f"w_{asset}" for asset in config.assets]].copy()
        weights.columns = list(config.assets)
        weights.index = weights.index.astype(str)
        weights.rename_axis("month").to_csv(output / f"weights_{period}.csv")

        dates = pd.PeriodIndex(view.index, freq="M").to_timestamp()
        series = [
            view[f"w_{asset}"].to_numpy(dtype=float)
            for asset in config.assets
        ]
        fig, ax = plt.subplots(figsize=(12.0, 5.0))
        ax.stackplot(dates, series, labels=config.assets, alpha=0.92)
        ax.set_title(f"Stage54 Portfolio Weights - {period}")
        ax.set_ylabel("Portfolio weight")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", ncol=len(config.assets), frameon=False)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output / f"weights_{period}.png", dpi=180)
        plt.close(fig)

    weight_summary = weight_summary_table(monthly_result, config)
    weight_summary.to_csv(output / "weight_summary_by_period.csv", index=False)
    return weight_summary


def individual_asset_performance_table(
    monthly_result: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Compare the strategy with single-asset buy-and-hold benchmarks."""

    rows: list[dict[str, Any]] = []
    for period, view in analysis_periods(monthly_result, config).items():
        rows.append(
            {
                "Period": period,
                "Series": "Stage54_AllAssetFeatureBayesianZ",
                "Type": "strategy_net",
                "Start": str(view.index.min()),
                "End": str(view.index.max()),
                **_performance_summary(view["return"]),
            }
        )
        benchmark_returns = monthly_returns.loc[view.index, list(config.assets)]
        for asset in config.assets:
            asset_returns = benchmark_returns[asset].astype(float).dropna()
            rows.append(
                {
                    "Period": period,
                    "Series": asset,
                    "Type": "single_asset_buy_hold",
                    "Start": str(asset_returns.index.min()),
                    "End": str(asset_returns.index.max()),
                    **_performance_summary(asset_returns),
                }
            )
    return pd.DataFrame(rows)


def save_pnl_comparison_artifacts(
    monthly_result: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    config: StrategyConfig,
    output: Path,
) -> pd.DataFrame:
    """Save strategy-vs-asset cumulative PnL CSVs and line charts."""

    import os

    plot_config = output / ".matplotlib"
    plot_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for period, view in analysis_periods(monthly_result, config).items():
        returns = pd.DataFrame(index=view.index)
        returns["Stage54"] = view["return"].astype(float)
        for asset in config.assets:
            returns[asset] = monthly_returns.loc[view.index, asset].astype(float)

        nav = (1.0 + returns).cumprod()
        pnl = nav - 1.0
        comparison = pd.concat(
            [
                nav.add_suffix("_NAV"),
                pnl.add_suffix("_PnL"),
            ],
            axis=1,
        )
        comparison.index = comparison.index.astype(str)
        comparison.rename_axis("month").to_csv(
            output / f"pnl_comparison_{period}.csv"
        )

        dates = pd.PeriodIndex(view.index, freq="M").to_timestamp()
        fig, ax = plt.subplots(figsize=(12.0, 5.0))
        for column in pnl.columns:
            ax.plot(dates, pnl[column], label=column, linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_title(f"Stage54 vs Single-Asset Buy-and-Hold PnL - {period}")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", ncol=3, frameon=False)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output / f"pnl_comparison_{period}.png", dpi=180)
        plt.close(fig)

    asset_performance = individual_asset_performance_table(
        monthly_result, monthly_returns, config
    )
    asset_performance.to_csv(
        output / "individual_asset_performance.csv", index=False
    )
    return asset_performance


def regime_beta_table(
    monthly_result: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    """Bayesian ridge regime coefficients in long form.

    ``regime_beta`` is on the original 0~1 regime-score scale. ``standardized``
    beta is the coefficient learned on the internally standardized feature.
    """

    rows: list[dict[str, Any]] = []
    for month, row in monthly_result.iterrows():
        for asset in config.assets:
            for feature in config.forecast_features_for_asset(asset):
                feature_name = feature.removeprefix("p_")
                rows.append(
                    {
                        "month": month,
                        "asset": asset,
                        "feature": feature,
                        "feature_name": feature_name,
                        "regime_beta": row[f"regime_beta_{asset}_{feature_name}"],
                        "standardized_regime_beta": row[
                            f"standardized_regime_beta_{asset}_{feature_name}"
                        ],
                        "regime_intercept": row[f"regime_intercept_{asset}"],
                        "standardized_regime_intercept": row[
                            f"standardized_regime_intercept_{asset}"
                        ],
                        "feature_mean": row[
                            f"regime_feature_mean_{asset}_{feature_name}"
                        ],
                        "feature_std": row[
                            f"regime_feature_std_{asset}_{feature_name}"
                        ],
                        "expected_return": row[f"expected_return_{asset}"],
                        "predictive_std": row[f"predictive_std_{asset}"],
                        "training_observations": row[
                            f"training_observations_{asset}"
                        ],
                        "last_training_month": row[
                            f"last_training_month_{asset}"
                        ],
                        "used_mean_fallback": row[
                            f"used_mean_fallback_{asset}"
                        ],
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
    }
    advisory_checks = {
        "volatility_policy_satisfied": bool(result["volatility_slack"].min() >= -1e-7),
        "cdar_policy_satisfied": bool(result["cdar_slack"].min() >= -1e-7),
    }
    return {
        "strategy": "Stage54_AllAssetFeatureBayesianZ",
        "checks": checks,
        "advisory_checks": advisory_checks,
        "all_checks_pass": bool(all(checks.values())),
        "months": int(len(result)),
        "start": str(result.index.min()),
        "end": str(result.index.max()),
        "fallback_solver_months": int(result["used_fallback"].sum()),
        "equal_weight_fallback_months": int(
            result["used_equal_weight_fallback"].sum()
        ),
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
            "use_risk_free_rate": config.use_risk_free_rate,
            "risk_free_rate_item": config.risk_free_rate_item,
            "risk_free_day_count": config.risk_free_day_count,
            "asset_feature_columns": {
                asset: list(config.asset_feature_columns(asset))
                for asset in config.assets
            },
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
    *,
    save: bool = True,
    refresh_monthly_market_cache: bool = False,
    use_risk_free_rate: bool = False,
) -> dict[str, Any]:
    """Composition root: 구체 구현을 한 번 조립하고 단일 전략을 실행한다."""

    paths = RepositoryPaths.from_this_file()
    config = StrategyConfig(use_risk_free_rate=use_risk_free_rate)
    monthly_returns, monthly_audit = MonthlyAssetReturnRepository(
        paths, config
    ).load(refresh_monthly_market_cache)
    if config.use_risk_free_rate:
        monthly_risk_free_returns = RiskFreeRateRepository(
            paths, config
        ).monthly_returns(monthly_returns.index)
    else:
        monthly_risk_free_returns = pd.Series(
            0.0, index=monthly_returns.index, name="risk_free_return"
        )
    daily_returns, daily_audit = DailyRiskReturnRepository(paths, config).load()
    regime_scores, macro_audit = MacroRegimeRepository(paths, config).build_scores(
        monthly_returns.index
    )
    asset_features, asset_feature_audit = AssetFeatureRepository(
        paths, config
    ).build(monthly_returns.index, monthly_returns)
    regime_scores = regime_scores.join(asset_features, how="left")

    transaction_cost = TransactionCostModel(config)
    engine = BacktestEngine(
        config=config,
        return_forecaster=BayesianRegimeReturnForecaster(config),
        covariance_forecaster=LedoitWolfMonthlyCovarianceForecaster(config),
        allocator=BayesianZBettingSizeAllocator(config, transaction_cost),
        transaction_cost=transaction_cost,
    )
    monthly_result = engine.run(monthly_returns, regime_scores, daily_returns)
    performance = performance_table(monthly_result, config)
    regime_betas = regime_beta_table(monthly_result, config)
    weight_summary = weight_summary_table(monthly_result, config)
    individual_asset_performance = individual_asset_performance_table(
        monthly_result, monthly_returns, config
    )
    report = validation_report(
        monthly_result,
        config,
        {
            "monthly_returns": monthly_audit,
            "daily_risk_returns": daily_audit,
            "macro_regime_scores": macro_audit,
            "asset_features": asset_feature_audit,
        },
    )

    if save:
        paths.output.mkdir(parents=True, exist_ok=True)
        monthly_returns.rename_axis("month").to_csv(
            paths.output / "historical_monthly_returns.csv"
        )
        daily_returns.rename_axis("date").to_csv(
            paths.output / "historical_daily_returns.csv"
        )
        monthly_risk_free_returns.rename_axis("month").to_csv(
            paths.output / "monthly_cd_risk_free_returns.csv"
        )
        asset_features.to_csv(paths.output / "asset_feature_inputs.csv")
        monthly_result.to_csv(paths.output / "monthly_results.csv")
        regime_betas.to_csv(paths.output / "regime_betas.csv", index=False)
        weight_summary = save_weight_artifacts(
            monthly_result, config, paths.output
        )
        individual_asset_performance = save_pnl_comparison_artifacts(
            monthly_result, monthly_returns, config, paths.output
        )
        performance.to_csv(paths.output / "performance.csv", index=False)
        with (paths.output / "validation_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                _json_ready(report), handle, ensure_ascii=False, indent=2
            )
    return {
        "historical_monthly_returns": monthly_returns,
        "historical_daily_returns": daily_returns,
        "monthly_risk_free_returns": monthly_risk_free_returns,
        "asset_features": asset_features,
        "monthly_result": monthly_result,
        "regime_betas": regime_betas,
        "weight_summary": weight_summary,
        "individual_asset_performance": individual_asset_performance,
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
    parser.add_argument(
        "--enable-risk-free",
        action="store_true",
        help="Use CD(91d) excess returns instead of simple asset returns.",
    )
    parser.add_argument(
        "--disable-risk-free",
        action="store_true",
        help="Keep simple asset returns. This is the default.",
    )
    args = parser.parse_args(argv)
    research = run_research(
        save=True,
        refresh_monthly_market_cache=args.refresh_monthly_market_cache,
        use_risk_free_rate=args.enable_risk_free and not args.disable_risk_free,
    )
    print(research["performance"].to_string(index=False))
    print(
        "\nValidation:",
        "PASS" if research["validation_report"]["all_checks_pass"] else "FAIL",
    )


if __name__ == "__main__":
    main()
