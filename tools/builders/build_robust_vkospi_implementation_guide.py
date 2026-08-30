from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "artifacts/reports/robust_vkospi_implementation_guide.html"


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}%"


def pp(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):+.{digits}f}%p"


def num(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def href(path: str, label: str | None = None) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith(("http://", "https://", "#")):
        target = normalized
    elif "/" not in normalized and normalized.endswith(".ipynb"):
        target = f"../notebooks/{normalized}"
    elif "/" not in normalized and normalized.endswith(".zip"):
        target = f"../bundles/{normalized}"
    elif normalized.startswith(("strategies/", "tests/", "results/", "raw_data/", "cache/")):
        target = f"../../{normalized}"
    else:
        target = normalized
    safe_path = html.escape(target, quote=True)
    safe_label = html.escape(label or path)
    return f'<a href="{safe_path}"><code>{safe_label}</code></a>'


def line_of(path: str, token: str) -> int:
    for number, line in enumerate((ROOT / path).read_text(encoding="utf-8").splitlines(), 1):
        if token in line:
            return number
    raise ValueError(f"{token!r} not found in {path}")


def source_ref(path: str, token: str, label: str) -> str:
    line = line_of(path, token)
    return f'{href(path, label)} <span class="line">L{line}</span>'


def find_row(
    frame: pd.DataFrame,
    *,
    period: str,
    strategy: str,
) -> pd.Series:
    view = frame.loc[frame["Period"].eq(period) & frame["Strategy"].eq(strategy)]
    if len(view) != 1:
        raise ValueError(f"Expected one row for {period=} {strategy=}, found {len(view)}")
    return view.iloc[0]


def metric_cells(row: pd.Series) -> str:
    return (
        f"<td>{int(float(row['Months']))}</td>"
        f"<td>{pct(row['CAGR'])}</td>"
        f"<td>{num(row['Sharpe'])}</td>"
        f"<td>{pct(row['MDD'])}</td>"
        f"<td>{num(row['Calmar'])}</td>"
    )


def file_rows() -> str:
    files = (
        (
            "핵심 실행",
            "strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py",
            "현재 기준 경로",
            "무SJM 거시 신호 → 16변수 균형 로지스틱 → 월간 기준 비중 → 고정 Robust VKOSPI 오버레이를 한 번에 재구성한다.",
            "def main()",
        ),
        (
            "핵심 신호",
            "strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py",
            "오버레이 정의·선정",
            "VKOSPI robust 특징 15개를 만들고 5개 스트레스 조합, 810개 정책 후보를 2017년 말 이전 자료로 비교한다.",
            "def build_robust_daily_features",
        ),
        (
            "핵심 엔진",
            "strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py",
            "일별 체결·비용·조정",
            "전일 신호 정렬, 주식·원유 위험 이전, 리밸런싱 밴드, 거래·환전·조달비용, 월간 상대수익 조정을 담당한다.",
            "def simulate",
        ),
        (
            "월간 모델",
            "strategies/stage06_vkospi/vkospi_model_robustness.py",
            "로지스틱·비중 엔진",
            "중앙값 대치, 표준화, expanding walk-forward 학습, 꼬리위험 비중 이동, 변동성 목표와 로지스틱/SJM 강건성 감사를 제공한다.",
            "def fit_logistic_candidate",
        ),
        (
            "설명·진단",
            "strategies/stage06_vkospi/vkospi_robust_dynamic_attribution.py",
            "효과 분해",
            "새 신호, 최대 이전 35%, 리밸런싱 밴드 20%를 단계별로 분리하고 성분 절삭실험과 월별 기여도를 만든다.",
            "def main()",
        ),
        (
            "옵션 확장",
            "strategies/stage08_options/vix6_case1_strategy.py",
            "VIX6 decomposition",
            "KOSPI200 옵션 표면을 30일 만기로 보간하고 sticky-strike, parallel shift, skew, convexity를 분해한다.",
            "def decompose_surface",
        ),
        (
            "옵션 매매",
            "strategies/stage08_options/option_asset_slippage_experiment.py",
            "풋 매수·청산·비중",
            "VIX6 기반 진입·회복 점수, 10델타 풋 선택, 실제 보유기간, 슬리피지와 5번째 자산 비중을 평가한다.",
            "def build_vix6_trade_signals",
        ),
        (
            "확장 비교",
            "strategies/stage08_options/vix6_case1_model_comparison.py",
            "VIX6 대안 3종",
            "기존 Robust VKOSPI와 VIX6 입력, 단독 오버레이, 혼합 오버레이를 같은 기간·비용 기준으로 비교한다.",
            "def main()",
        ),
        (
            "테스트",
            "tests/test_vkospi_robust_dynamic_experiment.py",
            "오버레이 회귀검증",
            "신호 선행성, 810개 후보·승자 재현, 상대수익 조정, 잠금성과, 비용 2배와 attribution 일치를 검사한다.",
            "class VKOSPIRobustDynamicExperimentTest",
        ),
        (
            "테스트",
            "tests/test_balanced_logistic_no_sjm_strategy.py",
            "현재 기준 경로 검증",
            "무SJM·균형 L2 설정, 2개월 embargo, 일별 선행성, 232개월 정렬, 잠금 결과와 사전구간 게이트를 검사한다.",
            "def test_requested_model_configuration_is_explicit",
        ),
        (
            "테스트",
            "tests/test_option_asset_slippage_experiment.py",
            "옵션 매매 검증",
            "틱 하한, 거래량, 델타·DTE, 진입/청산 순서, 48개 후보, 자금조달 비중합과 채택 규칙을 검사한다.",
            "class OptionAssetSlippageExperimentTest",
        ),
    )
    rows = []
    for kind, path, status, role, token in files:
        line = line_of(path, token)
        rows.append(
            "<tr>"
            f'<td><span class="pill">{html.escape(kind)}</span></td>'
            f'<td>{href(path)} <span class="line">L{line}</span></td>'
            f"<td><b>{html.escape(status)}</b></td>"
            f"<td>{html.escape(role)}</td>"
            "</tr>"
        )
    return "".join(rows)


def result_file_rows() -> str:
    files = (
        ("raw_data/VKOSPIData.csv", "2003–2026 VKOSPI 일별 OHLC 원자료"),
        ("raw_data/KOSPI200OptionPrice.csv", "VIX6 표면과 풋 실물수익을 만드는 통합 KOSPI200 옵션 원자료"),
        ("results/openassetpricing_composites.csv", "Open Asset Pricing 신호를 네 개 스트레스 합성변수로 묶은 월간 입력"),
        ("results/vkospi_selected_backtest.csv", "일간 오버레이가 받아오는 검증된 월간 기준 비중·수익"),
        ("results/balanced_logistic_no_sjm_final_reconciled.csv", "현재 기준 월별 수익·NAV·낙폭"),
        ("results/balanced_logistic_no_sjm_final_daily.csv", "현재 기준 일별 비중·스트레스·비용·신호일"),
        ("results/balanced_logistic_no_sjm_validation.json", "무SJM·로지스틱·잠금성과·부트스트랩 설정"),
        ("results/vkospi_robust_dynamic_calibration.csv", "Robust VKOSPI 810개 후보의 사전구간 비교"),
        ("results/vkospi_robust_dynamic_validation.json", "오버레이 승자와 2018년 이후 잠금 결과"),
        ("results/vkospi_robust_dynamic_stepwise_attribution.csv", "효과가 생긴 단계를 분해한 결과"),
        ("results/vkospi_robust_dynamic_component_ablation.csv", "수준·충격·가속 성분 절삭실험"),
        ("results/vkospi_robust_dynamic_cost_sensitivity.csv", "기본비용·2배 비용 비교"),
        ("results/vkospi_model_robustness.json", "로지스틱 28개·SJM 37개 유효 후보·CSCV/PBO 감사"),
        ("results/option_asset_monthly_trades.csv", "VIX6 신호로 완결된 옵션 진입·청산 39건"),
        ("results/option_asset_slippage_comparison_2007_2026.csv", "슬리피지 3종×신호 4종×최대비중 4종=48개 후보"),
        ("results/option_asset_slippage_validation.json", "옵션 후보 최종 채택/기각 판정"),
        ("results/vix6_case1_final_selection.json", "VIX6 입력·단독·혼합 대안 최종 판정"),
        ("vkospi_robust_dynamic_strategy_colab.ipynb", "Google Colab에서 전략을 재현하는 기존 노트북"),
        ("vkospi_robust_dynamic_technical_report.html", "모형·통계 진단을 더 넓게 다룬 기존 기술보고서"),
    )
    return "".join(
        f"<tr><td>{href(path)}</td><td>{html.escape(role)}</td></tr>"
        for path, role in files
    )


def feature_rows(features: list[str], descriptions: dict[str, str]) -> str:
    return "".join(
        f"<tr><td><code>{html.escape(name)}</code></td><td>{html.escape(descriptions.get(name, '코드 산출 입력'))}</td></tr>"
        for name in features
    )


def main() -> None:
    robust = read_json("vkospi_robust_dynamic_validation.json")
    balanced = read_json("balanced_logistic_no_sjm_validation.json")
    model_audit = read_json("vkospi_model_robustness.json")
    option = read_json("option_asset_slippage_validation.json")
    vix6_selection = read_json("vix6_case1_final_selection.json")

    balanced_comparison = pd.read_csv(RESULTS / "balanced_logistic_no_sjm_comparison.csv")
    robust_comparison = pd.read_csv(RESULTS / "vkospi_robust_dynamic_comparison.csv")
    stepwise = pd.read_csv(RESULTS / "vkospi_robust_dynamic_stepwise_attribution.csv")
    components = pd.read_csv(RESULTS / "vkospi_robust_dynamic_component_ablation.csv")
    cost = pd.read_csv(RESULTS / "vkospi_robust_dynamic_cost_sensitivity.csv")
    option_comparison = pd.read_csv(RESULTS / "option_asset_slippage_comparison_2007_2026.csv")

    current_name = "NoSJM_BalancedLogistic_RobustVKOSPI"
    legacy_name = "Deployed_SJM10_BalancedLogistic_RobustVKOSPI"
    current_full = find_row(
        balanced_comparison, period="full_2007_2026", strategy=current_name
    )
    current_locked = find_row(
        balanced_comparison, period="locked_2018_2026", strategy=current_name
    )
    legacy_full = find_row(
        balanced_comparison, period="full_2007_2026", strategy=legacy_name
    )
    legacy_locked = find_row(
        balanced_comparison, period="locked_2018_2026", strategy=legacy_name
    )
    no_sjm_cal = find_row(
        balanced_comparison, period="calibration_2007_2017", strategy=current_name
    )
    sjm_cal = find_row(
        balanced_comparison, period="calibration_2007_2017", strategy=legacy_name
    )

    current_rows = "".join(
        (
            f'<tr class="winner"><td><b>현재 기준 · 전체 2007-04–2026-07</b></td>{metric_cells(current_full)}</tr>',
            f'<tr class="winner"><td><b>현재 기준 · 잠금표기 2018-01–2026-07</b></td>{metric_cells(current_locked)}</tr>',
            f'<tr><td>구 배포 SJM10 · 전체</td>{metric_cells(legacy_full)}</tr>',
            f'<tr><td>구 배포 SJM10 · 2018 이후</td>{metric_cells(legacy_locked)}</tr>',
        )
    )

    robust_locked_rows = robust_comparison.loc[
        robust_comparison["Period"].eq("locked_2018_2026")
        & robust_comparison["Strategy"].isin(
            ["ExistingDynamicReconciled", "RobustDynamicReconciled"]
        )
    ].set_index("Strategy")
    before = robust_locked_rows.loc["ExistingDynamicReconciled"]
    after = robust_locked_rows.loc["RobustDynamicReconciled"]
    overlay_rows = (
        f"<tr><td>기존 동적 VKOSPI</td><td>{pct(before['CAGR'])}</td><td>{num(before['Sharpe'])}</td><td>{pct(before['MDD'])}</td><td>{num(before['Calmar'])}</td></tr>"
        f'<tr class="winner"><td><b>Robust VKOSPI</b></td><td>{pct(after["CAGR"])}</td><td>{num(after["Sharpe"])}</td><td>{pct(after["MDD"])}</td><td>{num(after["Calmar"])}</td></tr>'
        f'<tr class="delta"><td>차이</td><td>{pp(after["CAGR"] - before["CAGR"])}</td><td>{num(after["Sharpe"] - before["Sharpe"], 4)}</td><td>{pp(after["MDD"] - before["MDD"])}</td><td>{num(after["Calmar"] - before["Calmar"], 4)}</td></tr>'
    )

    stepwise_locked = stepwise.loc[stepwise["Period"].eq("locked_2018_2026")].set_index("Experiment")
    step_labels = {
        "1_ExistingDynamic": "① 기존 동적 VKOSPI",
        "2_RobustSignal_OldPolicy": "② Robust 신호만 교체",
        "3_RobustSignal_Transfer35": "③ 최대 이전 35% 적용",
        "4_RobustWinner_Band20": "④ 리밸런싱 밴드 20%까지 적용",
    }
    step_rows = ""
    for name, label in step_labels.items():
        row = stepwise_locked.loc[name]
        klass = ' class="winner"' if name.startswith("4_") else ""
        step_rows += (
            f"<tr{klass}><td><b>{label}</b></td><td>{pct(row['CAGR'])}</td>"
            f"<td>{num(row['Sharpe'])}</td><td>{pct(row['MDD'])}</td>"
            f"<td>{num(row['AvgTurnover'])}</td></tr>"
        )

    comp_labels = {
        "LevelOnly": "수준만",
        "ShockOnly": "충격만",
        "AccelerationOnly": "가속만",
        "LevelShockRenormalized": "수준+충격",
        "SelectedAccelerationBlend": "선정식 · 수준+충격+가속",
    }
    comp_rows = ""
    for name, label in comp_labels.items():
        cal = components.loc[
            components["Experiment"].eq(name)
            & components["Period"].eq("calibration_2007_2017")
        ].iloc[0]
        lock = components.loc[
            components["Experiment"].eq(name)
            & components["Period"].eq("locked_2018_2026")
        ].iloc[0]
        klass = ' class="winner"' if name == "SelectedAccelerationBlend" else ""
        comp_rows += (
            f"<tr{klass}><td>{label}</td>"
            f"<td>{pct(cal['CAGR'])} / {num(cal['Sharpe'])} / {pct(cal['MDD'])}</td>"
            f"<td>{pct(lock['CAGR'])} / {num(lock['Sharpe'])} / {pct(lock['MDD'])}</td></tr>"
        )

    cost_rows = ""
    for period, label in (("cost_1.0x_locked", "기본비용"), ("cost_2.0x_locked", "비용 2배")):
        view = cost.loc[cost["Period"].eq(period)].set_index("Strategy")
        for strategy, strategy_label in (("ExistingDynamic", "기존"), ("RobustDynamic", "Robust")):
            row = view.loc[strategy]
            klass = ' class="winner"' if strategy == "RobustDynamic" else ""
            cost_rows += (
                f"<tr{klass}><td>{label}</td><td>{strategy_label}</td><td>{pct(row['CAGR'])}</td>"
                f"<td>{num(row['Sharpe'])}</td><td>{pct(row['MDD'])}</td><td>{num(row['AvgTurnover'])}</td></tr>"
            )

    option_base = option["best_base_candidate"]
    option_rows = (
        f'<tr class="winner"><td><b>옵션 없음 · 현재 Robust 기준</b></td><td>{pct(option["baseline"]["CAGR"])}</td><td>{num(option["baseline"]["Sharpe"])}</td><td>{pct(option["baseline"]["MDD"])}</td><td>채택</td></tr>'
        f'<tr><td>Base · <code>{html.escape(option_base["candidate"])}</code></td><td>{pct(option_base["CAGR"])}</td><td>{num(option_base["Sharpe"])}</td><td>{pct(option_base["MDD"])}</td><td>기각</td></tr>'
        f'<tr class="delta"><td>옵션 후보 − 기준</td><td>{pp(option_base["CAGR_delta"])}</td><td>{num(option_base["Sharpe_delta"], 4)}</td><td>{pp(option_base["MDD_delta"])}</td><td>3개 동시개선 실패</td></tr>'
    )

    vix6_alt_rows = ""
    for name, info in vix6_selection["alternatives"].items():
        m = info["locked_metrics"]
        d = info["delta_minus_existing"]
        vix6_alt_rows += (
            f"<tr><td><code>{html.escape(name)}</code></td><td>{pct(m['CAGR'])}</td>"
            f"<td>{num(m['Sharpe'])}</td><td>{pct(m['MDD'])}</td>"
            f"<td>{pp(d['CAGR'])} / {num(d['Sharpe'], 4)} / {pp(d['MDD'])}</td>"
            f"<td>{'통과' if info['all_three_improve'] else '실패'}</td></tr>"
        )

    macro_features = balanced["implementation"]["macro"]["features"]
    logistic_features = balanced["implementation"]["logistic"]["features"]
    macro_descriptions = {
        "GDP_level": "GDP 성장 수준의 표준화 상태",
        "GDP_level_d3": "GDP 성장 수준의 3개월 변화",
        "Export_level": "수출 성장 수준",
        "Export_level_d3": "수출 성장의 3개월 변화",
        "BSI_level": "기업경기실사지수 수준",
        "BSI_level_d3": "기업심리의 3개월 변화",
        "CPI_level": "소비자물가 수준",
        "CPI_level_d3": "소비자물가의 3개월 변화",
        "PPI_level": "생산자물가 수준",
        "PPI_level_d3": "생산자물가의 3개월 변화",
        "ImportPrice_level": "수입물가 수준",
        "ImportPrice_level_d3": "수입물가의 3개월 변화",
    }
    logistic_descriptions = {
        "base_USO": "중립 기준 포트폴리오의 원유 비중",
        "base_GLD": "중립 기준 포트폴리오의 금 비중",
        "base_KODEX200": "중립 기준 포트폴리오의 주식 비중",
        "p_inflation_high": "다음 달 고물가 국면 확률",
        "proxy_mom1": "국내 위험자산 1개월 모멘텀 대용치",
        "proxy_mom6": "국내 위험자산 6개월 모멘텀 대용치",
        "proxy_vol6": "6개월 변동성 대용치",
        "daily_mom21": "일별 자료로 계산한 21거래일 모멘텀",
        "daily_mom252": "일별 자료로 계산한 252거래일 모멘텀",
        "daily_vol21": "21거래일 실현변동성",
        "daily_downvol21": "21거래일 하방변동성",
        "daily_mean_corr63": "자산 간 63거래일 평균 상관",
        "oap_momentum_trend_stress": "Open Asset Pricing 모멘텀·추세 합성 스트레스",
        "oap_reversal_crowding_stress": "단기반전·혼잡 합성 스트레스",
        "oap_low_risk_tail_stress": "저위험·꼬리위험 합성 스트레스",
        "oap_liquidity_activity_stress": "유동성·거래활동 합성 스트레스",
    }
    robust_features = [
        "close",
        "percentile_126",
        "percentile_252",
        "robust_z_63",
        "robust_z_252",
        "shock_5",
        "shock_10",
        "shock_21",
        "acceleration_5",
        "acceleration_z5",
        "distance_high21",
        "close_location21",
        "positive_fraction5",
        "positive_fraction21",
        "fast_slow",
    ]
    robust_descriptions = {
        "close": "VKOSPI 종가",
        "percentile_126": "현재 값의 인과적 126거래일 분위수",
        "percentile_252": "현재 값의 인과적 252거래일 분위수",
        "robust_z_63": "로그 VKOSPI의 63일 median/MAD z점수",
        "robust_z_252": "로그 VKOSPI의 252일 median/MAD z점수",
        "shock_5": "5일 로그변화를 직전 63일 변동성으로 나눈 충격",
        "shock_10": "10일 정규화 충격",
        "shock_21": "21일 정규화 충격",
        "acceleration_5": "최근 5일 변화 − 그 전 5일 변화",
        "acceleration_z5": "5일 가속도를 63일 변동성으로 정규화",
        "distance_high21": "현재 값 / 21일 고가 − 1",
        "close_location21": "21일 고저 범위 안에서 현재 종가 위치",
        "positive_fraction5": "최근 5일 VKOSPI 상승일 비율",
        "positive_fraction21": "최근 21일 VKOSPI 상승일 비율",
        "fast_slow": "5일 변화 − 5/21×21일 변화",
    }

    winner = robust["winner"]
    bootstrap = robust["locked"]["bootstrap"]
    logistic_audit = model_audit["logistic"]
    option_audit = option["trade_audit"]

    replacements = {
        "__GENERATED__": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "__CURRENT_CAGR__": pct(current_full["CAGR"]),
        "__CURRENT_SHARPE__": num(current_full["Sharpe"]),
        "__CURRENT_MDD__": pct(current_full["MDD"]),
        "__LOCKED_CAGR__": pct(current_locked["CAGR"]),
        "__LOCKED_SHARPE__": num(current_locked["Sharpe"]),
        "__LOCKED_MDD__": pct(current_locked["MDD"]),
        "__CURRENT_ROWS__": current_rows,
        "__OVERLAY_ROWS__": overlay_rows,
        "__STEP_ROWS__": step_rows,
        "__COMP_ROWS__": comp_rows,
        "__COST_ROWS__": cost_rows,
        "__OPTION_ROWS__": option_rows,
        "__VIX6_ALT_ROWS__": vix6_alt_rows,
        "__FILE_ROWS__": file_rows(),
        "__RESULT_FILE_ROWS__": result_file_rows(),
        "__MACRO_ROWS__": feature_rows(macro_features, macro_descriptions),
        "__LOGISTIC_ROWS__": feature_rows(logistic_features, logistic_descriptions),
        "__ROBUST_ROWS__": feature_rows(robust_features, robust_descriptions),
        "__ROBUST_COUNT__": str(len(robust_features)),
        "__WINNER_MODE__": html.escape(str(winner["mode"])),
        "__WINNER_LEVEL__": num(winner["level_threshold"], 2),
        "__WINNER_SHOCK__": num(winner["shock_threshold"], 2),
        "__WINNER_TRANSFER__": pct(winner["max_risk_transfer"], 0),
        "__WINNER_BOND_SHARE__": pct(winner["bond_share"], 0),
        "__WINNER_BAND__": pct(winner["rebalance_band"], 0),
        "__BOOT_ALL__": pct(bootstrap["probability_all_three_improve"], 1),
        "__BOOT_CAGR__": pct(bootstrap["probability_cagr_improves"], 1),
        "__BOOT_SHARPE__": pct(bootstrap["probability_sharpe_improves"], 1),
        "__BOOT_MDD__": pct(bootstrap["probability_mdd_improves"], 1),
        "__LOGISTIC_PBO__": pct(logistic_audit["prelock_portfolio_sharpe_pbo"]["pbo"], 1),
        "__LOGISTIC_RANK__": str(logistic_audit["deployed_locked_portfolio_rank"]),
        "__LOGISTIC_CANDIDATES__": str(logistic_audit["candidate_count"]),
        "__NO_SJM_CAL_CAGR__": pct(no_sjm_cal["CAGR"]),
        "__SJM_CAL_CAGR__": pct(sjm_cal["CAGR"]),
        "__NO_SJM_CAL_SHARPE__": num(no_sjm_cal["Sharpe"]),
        "__SJM_CAL_SHARPE__": num(sjm_cal["Sharpe"]),
        "__OPTION_TRADES__": str(option_audit["completed_monthly_trades"]),
        "__OPTION_RECOVERY_EXITS__": str(option_audit["vix6_recovery_exits"]),
        "__OPTION_ROLL_EXITS__": str(option_audit["month_end_roll_exits"]),
        "__OPTION_DELTA_RANGE__": f"{option_audit['entry_delta_min']:.3f}–{option_audit['entry_delta_max']:.3f}",
        "__OPTION_DTE_RANGE__": f"{option_audit['entry_dte_min']:.0f}–{option_audit['entry_dte_max']:.0f}일",
        "__REF_FIXED_OVERLAY__": source_ref("strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py", "def fixed_robust_overlay", "fixed_robust_overlay()"),
        "__REF_FEATURES__": source_ref("strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py", "def build_robust_daily_features", "build_robust_daily_features()"),
        "__REF_STRESS__": source_ref("strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py", "def stress_from_features", "stress_from_features()"),
        "__REF_SIM__": source_ref("strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py", "def simulate", "simulate()"),
        "__REF_RECON__": source_ref("strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py", "def reconcile_to_monthly_reference", "reconcile_to_monthly_reference()"),
        "__REF_FIT__": source_ref("strategies/stage06_vkospi/vkospi_model_robustness.py", "def fit_logistic_candidate", "fit_logistic_candidate()"),
        "__REF_OPTION_SIGNAL__": source_ref("strategies/stage08_options/option_asset_slippage_experiment.py", "def build_vix6_trade_signals", "build_vix6_trade_signals()"),
        "__REF_OPTION_SELECT__": source_ref("strategies/stage08_options/option_asset_slippage_experiment.py", "def select_monthly_put_trades", "select_monthly_put_trades()"),
        "__REF_OPTION_ALLOC__": source_ref("strategies/stage08_options/option_asset_slippage_experiment.py", "def run_option_allocation", "run_option_allocation()"),
    }

    template = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robust VKOSPI 구현·알고리즘·옵션 매매 가이드</title>
<style>
:root{--bg:#07111f;--panel:#0d1b2b;--panel2:#11243a;--ink:#eaf2fb;--muted:#a9bdd1;--line:#28425d;--cyan:#5de4d3;--blue:#73a7ff;--amber:#ffcc70;--red:#ff8193;--green:#87e59a;--shadow:0 18px 44px rgba(0,0,0,.22)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 80% 0,#14335c 0,transparent 32rem),var(--bg);color:var(--ink);font-family:Pretendard,"Noto Sans KR","Malgun Gothic",sans-serif;line-height:1.72}a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}code{font-family:"Cascadia Code",Consolas,monospace;font-size:.92em;color:#b9f6ee}button{font:inherit}.shell{max-width:1480px;margin:auto;padding:0 24px}.hero{padding:76px 0 44px;border-bottom:1px solid var(--line)}.eyebrow{letter-spacing:.16em;text-transform:uppercase;color:var(--cyan);font-size:.78rem;font-weight:800}.hero h1{font-size:clamp(2.35rem,5vw,5.2rem);line-height:1.03;max-width:1150px;margin:.45rem 0 1.25rem;letter-spacing:-.045em}.hero p{max-width:970px;color:var(--muted);font-size:1.08rem}.meta{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}.tag,.pill{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:999px;padding:5px 10px;color:var(--muted);font-size:.78rem}.tag.good{border-color:#2f7e66;color:var(--green)}.tag.warn{border-color:#80612d;color:var(--amber)}.tag.stop{border-color:#853b49;color:#ffb1bd}.layout{display:grid;grid-template-columns:250px minmax(0,1fr);gap:42px;align-items:start}.toc{position:sticky;top:0;padding:28px 0;max-height:100vh;overflow:auto}.toc b{display:block;margin-bottom:10px}.toc a{display:block;color:var(--muted);padding:5px 0;font-size:.88rem}.toc a.active{color:var(--cyan)}main{min-width:0;padding:42px 0 100px}.section{scroll-margin-top:24px;margin:0 0 58px}.section>h2{font-size:clamp(1.7rem,3vw,2.6rem);letter-spacing:-.035em;margin:0 0 10px}.lead{color:var(--muted);max-width:980px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px;margin-top:22px}.card{grid-column:span 4;background:linear-gradient(150deg,rgba(17,36,58,.94),rgba(9,23,38,.98));border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:var(--shadow)}.card.wide{grid-column:span 8}.card.full{grid-column:1/-1}.card.half{grid-column:span 6}.card h3{margin:0 0 9px;font-size:1.08rem}.card p{margin:0;color:var(--muted)}.metric strong{display:block;font-size:2rem;color:var(--cyan);line-height:1.15;margin:.3rem 0}.metric span{color:var(--muted);font-size:.86rem}.callout{border-left:4px solid var(--amber);background:#211d15;padding:16px 18px;border-radius:0 12px 12px 0;margin:20px 0;color:#f4dfb4}.callout.good{border-color:var(--green);background:#10241b}.callout.red{border-color:var(--red);background:#29151b}.flow{display:flex;gap:10px;align-items:stretch;overflow:auto;padding:8px 2px 16px}.flow .node{min-width:190px;flex:1;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:15px}.flow .arrow{align-self:center;color:var(--cyan);font-size:1.4rem}.node small{display:block;color:var(--muted)}.node b{display:block;margin:3px 0}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:14px;margin:18px 0;background:rgba(10,25,41,.8)}table{width:100%;border-collapse:collapse;min-width:760px}th,td{padding:11px 13px;text-align:left;border-bottom:1px solid rgba(40,66,93,.7);vertical-align:top}th{position:sticky;top:0;background:#10243a;color:#c7d7e7;font-size:.78rem;letter-spacing:.035em}tr:last-child td{border-bottom:0}tr.winner{background:rgba(93,228,211,.075)}tr.delta{color:var(--amber)}.line{color:#7490aa;font-family:Consolas,monospace;font-size:.78rem}.formula{background:#06101c;border:1px solid var(--line);border-radius:14px;padding:16px 18px;overflow:auto;color:#c8f8f1;font-family:"Cascadia Code",Consolas,monospace;white-space:pre;line-height:1.55}.decision{display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center}.decision .branch{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:18px}.decision .yes{border-color:#337d66}.decision .no{border-color:#814252}.decision .gate{font-size:2rem;color:var(--amber)}details{background:rgba(10,25,41,.72);border:1px solid var(--line);border-radius:14px;margin:12px 0}summary{cursor:pointer;padding:15px 17px;font-weight:750;color:#dce9f5}details>div{padding:0 17px 17px;color:var(--muted)}ul,ol{padding-left:1.3rem}li{margin:.35rem 0}.kicker{color:var(--cyan);font-weight:800}.subtle{color:var(--muted);font-size:.9rem}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.statusbar{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:15px 0}.print{background:transparent;border:1px solid var(--line);border-radius:10px;padding:7px 11px;color:var(--ink);cursor:pointer}.print:hover{border-color:var(--cyan)}footer{border-top:1px solid var(--line);padding:35px 0 60px;color:var(--muted)}
@media(max-width:980px){.layout{grid-template-columns:1fr}.toc{position:static;max-height:none;columns:2;border-bottom:1px solid var(--line)}.card,.card.half,.card.wide{grid-column:1/-1}.two{grid-template-columns:1fr}.decision{grid-template-columns:1fr}.decision .gate{transform:rotate(90deg);text-align:center}}
@media(max-width:640px){.shell{padding:0 15px}.hero{padding-top:50px}.toc{columns:1}.flow{flex-direction:column}.flow .arrow{transform:rotate(90deg)}.hero h1{font-size:2.35rem}}
@media print{body{background:white;color:#111}.toc,.print{display:none}.layout{display:block}.shell{max-width:none}.card,.table-wrap,details{box-shadow:none;background:white;color:#111;border-color:#bbb}.card p,.lead,.subtle,details>div{color:#333}a,code{color:#064f47}.section{break-inside:avoid}.hero{padding:25px 0}.hero h1{font-size:34px}.formula{white-space:pre-wrap;color:#111;background:#f5f5f5}}
</style>
</head>
<body>
<header class="hero"><div class="shell">
  <div class="statusbar"><span class="eyebrow">Implementation & decision audit · 2026</span><button class="print" onclick="print()">인쇄 / PDF</button></div>
  <h1>Robust VKOSPI<br>구현·알고리즘·옵션 매매 가이드</h1>
  <p>이 보고서는 “Robust VKOSPI”라는 이름이 실제로 가리키는 파일과 실행 경로를 먼저 분리한다. 이어서 월간 무SJM·균형 로지스틱, 일간 VKOSPI 위험 이전, 성과가 생긴 지점, VIX6 decomposition을 이용한 옵션 매수·청산·비중 결정까지 코드와 결과 파일을 기준으로 연결한다.</p>
  <div class="meta"><span class="tag good">현재 선택: Existing Final Robust VKOSPI</span><span class="tag stop">옵션 자산: 최종 비중 0%</span><span class="tag warn">2007-04–2026-07 · 232개월</span><span class="tag">생성 __GENERATED__ KST</span></div>
</div></header>

<div class="shell layout">
<nav class="toc" aria-label="목차"><b>목차</b>
  <a href="#verdict">1. 결론</a><a href="#naming">2. 이름과 버전</a><a href="#architecture">3. 전체 구조</a><a href="#files">4. 구현 파일</a><a href="#monthly">5. 월간 알고리즘</a><a href="#inputs">6. 입력변수 전부</a><a href="#overlay">7. Robust VKOSPI</a><a href="#search">8. 반복 후보 탐색</a><a href="#execution">9. 체결·비용·조정</a><a href="#effects">10. 효과의 원인</a><a href="#option">11. 옵션 매수·매도</a><a href="#results">12. 성과</a><a href="#validation">13. 검증과 한계</a><a href="#run">14. 실행 방법</a>
</nav>
<main>

<section class="section" id="verdict"><h2>1. 결론부터</h2><p class="lead">현재 작업공간에서 최종 비교 기준으로 쓰는 경로는 <b>무SJM 거시 국면 + 균형 L2 로지스틱 + Robust VKOSPI 일간 오버레이</b>다. 옵션 풋 매수·청산 로직은 구현과 검증까지 끝났지만, 비용을 넣은 48개 후보 중 CAGR·Sharpe·MDD를 동시에 개선한 후보가 없어 최종 전략에서는 옵션 비중을 0%로 유지한다.</p>
<div class="grid">
 <article class="card metric"><span>전체 CAGR</span><strong>__CURRENT_CAGR__</strong><span>2007-04–2026-07</span></article>
 <article class="card metric"><span>전체 Sharpe</span><strong>__CURRENT_SHARPE__</strong><span>월수익, 무위험 0 가정 요약치</span></article>
 <article class="card metric"><span>전체 MDD</span><strong>__CURRENT_MDD__</strong><span>덜 음수일수록 양호</span></article>
 <article class="card metric"><span>2018 이후 CAGR</span><strong>__LOCKED_CAGR__</strong><span>표기상 locked 구간</span></article>
 <article class="card metric"><span>2018 이후 Sharpe</span><strong>__LOCKED_SHARPE__</strong><span>103개월</span></article>
 <article class="card metric"><span>2018 이후 MDD</span><strong>__LOCKED_MDD__</strong><span>2018-01–2026-07</span></article>
</div>
<div class="callout red"><b>중요한 연구상 주의.</b> Robust VKOSPI 오버레이 승자는 2017년 말 이전 자료로 고른 뒤 2018년 이후를 잠갔다. 하지만 현재 기준인 <b>무SJM 변형</b>과 뒤이은 VIX6·옵션 실험은 사용자의 추가 요청에 따른 사후 비교다. 따라서 현재 기준의 2018년 이후 숫자는 편의상 “locked”라고 표기하지만, 전체 연구가 끝난 뒤에도 완전히 손대지 않은 최종 홀드아웃이라고 해석하면 안 된다.</div>
</section>

<section class="section" id="naming"><h2>2. “Robust VKOSPI”는 세 층으로 나눠 봐야 한다</h2>
<div class="grid">
 <article class="card"><h3>① 오버레이 모듈</h3><p>VKOSPI 수준·충격·가속으로 0–1 스트레스를 만들고 주식·원유 비중 일부를 안전자산으로 옮기는 일간 규칙 자체다. 핵심은 __REF_FEATURES__와 __REF_STRESS__다.</p></article>
 <article class="card"><h3>② 구 배포 결합경로</h3><p>SJM 가중치 10%의 거시 모델, 균형 로지스틱, Robust VKOSPI를 결합한 과거 기준이다. 결과는 <code>vkospi_robust_dynamic_reconciled_monthly.csv</code>에 남아 있다.</p></article>
 <article class="card"><h3>③ 현재 비교 기준</h3><p>SJM을 0%로 제거하고 같은 균형 로지스틱과 같은 오버레이를 다시 계산한 경로다. __REF_FIXED_OVERLAY__가 조립하고 <code>balanced_logistic_no_sjm_final_reconciled.csv</code>가 현재 기준 결과다.</p></article>
</div>
<p class="callout">이 구분이 필요한 이유는 단순하다. “오버레이가 얼마나 개선했는가”와 “무SJM까지 포함한 최신 전체 전략 성과”는 서로 다른 비교다. 앞의 개선폭은 오버레이만 바꾼 공정한 절삭실험이고, 뒤의 성과는 월간 모델 변경까지 합친 최종 경로다.</p>
</section>

<section class="section" id="architecture"><h2>3. 전체 알고리즘 구조</h2><p class="lead">월간 엔진이 기본 비중을 정하고, 일간 엔진이 급격한 변동성 스트레스 때만 그 비중을 방어적으로 덮어쓴다. 옵션 확장은 이 네 자산 경로의 일부를 풋 프리미엄으로 재배분하는 별도 5번째 슬리브다.</p>
<div class="flow" aria-label="전략 데이터 흐름">
 <div class="node"><small>월간 입력</small><b>거시 12개 + 시장/OAP 16개</b><small>전월까지 관측</small></div><div class="arrow">→</div>
 <div class="node"><small>월간 모델</small><b>무SJM 국면 + 균형 로지스틱</b><small>2개월 꼬리손실 예측</small></div><div class="arrow">→</div>
 <div class="node"><small>기준 비중</small><b>KODEX200·채권·금·원유</b><small>15% 변동성 목표</small></div><div class="arrow">→</div>
 <div class="node"><small>일간 입력</small><b>Robust VKOSPI</b><small>수준·충격·가속</small></div><div class="arrow">→</div>
 <div class="node"><small>오버레이</small><b>위험 이전 + 20% 밴드</b><small>주식·원유 → 금</small></div>
</div>
<div class="flow" aria-label="옵션 확장 흐름">
 <div class="node"><small>옵션 원자료</small><b>KOSPI200 콜·풋 표면</b><small>30일 만기 보간</small></div><div class="arrow">→</div>
 <div class="node"><small>VIX6</small><b>shift·skew·convexity</b><small>6 decomposition</small></div><div class="arrow">→</div>
 <div class="node"><small>매매 규칙</small><b>10델타 풋 진입·회복 청산</b><small>신호 1관측치 지연</small></div><div class="arrow">→</div>
 <div class="node"><small>승격 게이트</small><b>CAGR·Sharpe·MDD 동시 개선?</b><small>Base와 Conservative 비용</small></div><div class="arrow">→</div>
 <div class="node"><small>현재 판정</small><b>아니오 → 옵션 0%</b><small>기존 Robust 유지</small></div>
</div>
</section>

<section class="section" id="files"><h2>4. 어떤 파일이 실제 구현인가</h2><p class="lead">“보고서 생성기”와 “전략 엔진”을 섞어 보면 실행 경로를 오해하기 쉽다. 아래 표의 앞 세 파일이 Robust VKOSPI 코어이고, <code>balanced_logistic_no_sjm_strategy.py</code>가 현재 전체 경로의 진입점이다.</p>
<div class="table-wrap"><table><thead><tr><th>분류</th><th>파일·시작점</th><th>상태</th><th>역할</th></tr></thead><tbody>__FILE_ROWS__</tbody></table></div>
<details><summary>결과 파일 지도</summary><div><div class="table-wrap"><table><thead><tr><th>파일</th><th>내용</th></tr></thead><tbody>__RESULT_FILE_ROWS__</tbody></table></div></div></details>
<p class="subtle"><code>build_vkospi_robust_dynamic_technical_report.py</code>, <code>build_vkospi_robust_dynamic_deliverables.py</code>, 이 문서를 만드는 <code>build_robust_vkospi_implementation_guide.py</code>는 설명 자료 생성기다. 백테스트 비중이나 수익을 결정하는 실행 경로에는 들어가지 않는다.</p>
<div class="callout good"><b>가장 짧은 추적 순서:</b> <code>balanced_logistic_no_sjm_strategy.py:main()</code> → <code>fixed_robust_overlay()</code> → <code>build_robust_daily_features()</code> → <code>stress_from_features()</code> → <code>simulate()</code> → <code>reconcile_to_monthly_reference()</code>.</div>
</section>

<section class="section" id="monthly"><h2>5. 월간 기본전략: 무SJM + 균형 로지스틱</h2>
<div class="grid">
 <article class="card half"><h3>5-1. 국면 확률</h3><p>성장축은 GDP·수출·BSI, 물가축은 CPI·PPI·수입물가의 수준과 3개월 변화를 사용한다. 현재 상태 가중 85%, 3개월 변화 가중 20%, sigmoid scale 0.55는 그대로 두되 SJM 기여를 0으로 만든다. 신호월은 항상 투자월보다 앞선다.</p></article>
 <article class="card half"><h3>5-2. 4자산 기준 비중</h3><p>하드 국면 비중 40%와 방어 최적화 경로 60%를 섞는다. 자산 순서는 KODEX200, BOND, GLD, USO다. 이 단계가 “어떤 자산을 기본적으로 얼마나 들고 갈지”를 정한다.</p></article>
 <article class="card half"><h3>5-3. 꼬리위험 라벨</h3><p>중립 기준 수익의 앞으로 2개월 누적 경로 중 최저값이 −5% 아래면 <code>tail_event=1</code>이다. 즉 다음 달 하나의 방향보다 두 달 안에 나타날 수 있는 경로 손실을 분류한다.</p></article>
 <article class="card half"><h3>5-4. expanding 학습</h3><p>__REF_FIT__은 중앙값 대치 → 표준화 → class-balanced L2 로지스틱(<code>C=0.1</code>, <code>liblinear</code>)을 매월 다시 적합한다. 최소 36개월, 양성 4건, 음성 12건을 요구하고 현재 월 앞의 2개월을 embargo한다.</p></article>
</div>
<details open><summary>4개 국면 분류와 하드 비중</summary><div>
<p><code>p_growth_high</code>와 <code>p_inflation_high</code>의 곱으로 네 사분면 확률을 만든 뒤 가장 큰 확률을 국면으로 택한다. 자산 순서는 KODEX200 / BOND / GLD / USO다.</p>
<div class="table-wrap"><table><thead><tr><th>국면</th><th>확률식</th><th>하드 비중</th><th>경제적 해석</th></tr></thead><tbody>
<tr><td>Goldilocks</td><td><code>g × (1-i)</code></td><td>100% / 0% / 0% / 0%</td><td>고성장·저물가 → 주식</td></tr>
<tr><td>Overheating</td><td><code>g × i</code></td><td>0% / 0% / 0% / 100%</td><td>고성장·고물가 → 원유</td></tr>
<tr><td>Slowdown</td><td><code>(1-g) × (1-i)</code></td><td>60% / 40% / 0% / 0%</td><td>저성장·저물가 → 주식+채권</td></tr>
<tr><td>Stagflation</td><td><code>(1-g) × i</code></td><td>0% / 0% / 100% / 0%</td><td>저성장·고물가 → 금</td></tr>
</tbody></table></div>
<p>방어 경로는 이 하드 비중과 별개로 soft regime anchor, 역변동성 기울기, 기대수익, 연율 변동성, CDaR, turnover, anchor 추적오차를 함께 넣은 SLSQP 최적화다. 자산별 범위와 변동성·CDaR 제약을 적용하고, 실현 낙폭이 −5%를 넘으면 방어 앵커 쪽으로 추가 혼합한다. 최종 월간 기준은 이 방어 경로 60%와 하드 비중 40%의 혼합이다.</p>
</div></details>
<div class="formula">train_end(t) = position(t) - 2
train = observations[0 : train_end(t)]
model_t = MedianImputer → StandardScaler → LogisticRegression(L2, C=0.1, class_weight="balanced")
p_tail(t) = model_t.predict_proba(X_t)

risk_percentile(t) = percentile of p_tail(t) versus prior 60 valid probabilities
severity(t) = clip((risk_percentile(t) - 0.80) / 0.20, 0, 1)
p_up(t) = 0.50 - 0.15 × severity(t)
risk_score(t) = -severity(t)</div>
<p>로지스틱이 학습을 “안 하는” 것이 아니다. 매월 과거 자료만으로 다시 학습한다. 반면 <b>일간 Robust VKOSPI와 옵션 VIX6 진입·청산은 학습모델이 아니라 사전 정의된 연속 규칙</b>이다. 두 종류를 분리해 봐야 한다.</p>
<div class="formula">base = 0.40 × hard_regime_weights + 0.60 × defensive_weights

if risk_score &lt; 0:
    transfer up to 20% from KODEX200 and USO → BOND and GLD

forecast_vol = exponentially weighted volatility of prior 24 months (minimum 12)
leverage = clip(0.15 / forecast_vol, 0.50, 1.50)
final_monthly_weights = leverage × tilted_base</div>
</section>

<section class="section" id="inputs"><h2>6. 사용한 입력변수 전체 목록</h2><p class="lead">아래 목록은 “현재 전체 경로에 실제로 쓰는 월간 입력”, “Robust VKOSPI가 생성하는 일간 입력”, “옵션 매매에 추가로 쓰는 VIX6 입력”을 구분한다. CAGR·Sharpe·MDD는 입력변수가 아니라 후보를 평가하는 목적함수다.</p>
<div class="two">
<details open><summary>거시 국면 입력 12개</summary><div><div class="table-wrap"><table><thead><tr><th>변수</th><th>의미</th></tr></thead><tbody>__MACRO_ROWS__</tbody></table></div></div></details>
<details open><summary>균형 로지스틱 입력 16개</summary><div><div class="table-wrap"><table><thead><tr><th>변수</th><th>의미</th></tr></thead><tbody>__LOGISTIC_ROWS__</tbody></table></div></div></details>
</div>
<details open><summary>Robust VKOSPI 생성변수 __ROBUST_COUNT__개</summary><div><p>최종 승자식이 직접 쓰는 것은 <code>percentile_252</code>(초기에는 126일로 보충), <code>shock_5</code>, <code>acceleration_z5</code> 세 축이다. 나머지는 후보 모드와 진단·대체식에 쓰인다.</p><div class="table-wrap"><table><thead><tr><th>변수</th><th>의미</th></tr></thead><tbody>__ROBUST_ROWS__</tbody></table></div></div></details>
<details open><summary>VIX6 decomposition 및 옵션 매매 입력</summary><div>
<ul><li><b>기초 6성분:</b> <code>sticky_strike</code>, <code>parallel_shift</code>, <code>put_skew</code>, <code>call_skew</code>, <code>downside_convexity</code>, <code>upside_convexity</code>.</li><li><b>수준 합성:</b> <code>left_tail</code> = put-skew와 downside-convexity의 robust z 평균, <code>right_tail</code> = call 쪽 평균, <code>asymmetry</code> = left − right.</li><li><b>경로 합성:</b> <code>breadth_z</code>, <code>reaction_z</code>, <code>left_impulse_z</code>, <code>left_change_5</code>, 옵션 내재 forward의 5·21일 수익.</li><li><b>실제 옵션 진입:</b> <code>left_tail</code>, <code>asymmetry</code>, <code>left_impulse_z</code>, <code>breadth_z</code>.</li><li><b>실제 옵션 청산:</b> <code>left_change_5</code>, <code>left_impulse_z</code>, <code>right_tail - left_tail</code>.</li><li><b>체결비용만 보정:</b> 전일 <code>VKOSPI percentile</code>과 <code>shock_5</code>가 위기 슬리피지 배수를 1.0–2.0배로 높인다.</li></ul>
</div></details>
</section>

<section class="section" id="overlay"><h2>7. Robust VKOSPI 오버레이는 구체적으로 무엇인가</h2><p class="lead">오버레이는 VKOSPI 자체를 매수하는 전략이 아니다. 월간 모델이 만든 4자산 비중을 유지하되, 한국판 VIX인 VKOSPI가 “높고, 빠르게 오르고, 상승 속도까지 빨라질 때” 주식과 원유 일부를 금으로 옮기는 일간 위험예산 규칙이다.</p>
<div class="grid"><article class="card"><h3>선정 모드</h3><p><code>__WINNER_MODE__</code><br>수준 40% + 5일 충격 35% + 5일 가속 25%</p></article><article class="card"><h3>문턱</h3><p>수준 분위수 __WINNER_LEVEL__<br>충격·가속 z 문턱 __WINNER_SHOCK__</p></article><article class="card"><h3>정책</h3><p>최대 이전 __WINNER_TRANSFER__<br>채권 몫 __WINNER_BOND_SHARE__<br>리밸런싱 밴드 __WINNER_BAND__</p></article></div>
<div class="formula">P = percentile_252, falling back to percentile_126
L = clip((P - 0.90) / 0.10, 0, 1)
S = clip((shock_5 - 1.00) / 2.50, 0, 1)
A = clip((acceleration_z5 - 1.00) / 2.50, 0, 1)

stress = clip(0.40 × L + 0.35 × S + 0.25 × A, 0, 1)
transfer_fraction = 0.35 × stress</div>
<div class="formula">removed_equity = w_KODEX200 × transfer_fraction
removed_oil    = w_USO      × transfer_fraction
w_KODEX200 -= removed_equity
w_USO      -= removed_oil

removed = removed_equity + removed_oil
w_BOND += removed × bond_share           # winner: 0%
w_GLD  += removed × (1 - bond_share)     # winner: 100%</div>
<p>따라서 스트레스가 1이어도 전체 포트폴리오의 35%를 무조건 옮기는 것은 아니다. <b>그 시점 KODEX200과 USO 비중의 각각 35%</b>를 줄인다. 승자의 <code>bond_share=0</code>이므로 줄인 금액은 모두 GLD로 간다.</p>
</section>

<section class="section" id="search"><h2>8. “반복적인 후보·파라미터 탐색”의 실제 의미</h2><p class="lead">매일 미래를 보며 파라미터를 바꾸는 것이 아니다. 사전에 정한 유한한 격자를 2007–2017과 그 안의 2013–2017 두 창에서 반복 실행하고, 두 창 모두에서 기존 동적 VKOSPI보다 CAGR·Sharpe·MDD가 좋아진 후보만 남긴 뒤 한 개를 고정했다.</p>
<div class="formula">5 stress modes
× 3 level thresholds     {0.70, 0.80, 0.90}
× 3 shock thresholds     {0.0, 0.5, 1.0}
× 3 max transfers        {0.15, 0.25, 0.35}
× 2 bond shares          {0.0, 0.5}
× 3 rebalance bands      {0.10, 0.15, 0.20}
= 810 candidates</div>
<ol><li>각 후보를 같은 일별 가격·같은 비용·같은 월간 기준 비중 위에서 실행한다.</li><li>2007–2017과 2013–2017에서 CAGR, Sharpe, MDD, Calmar의 백분위 순위를 구한다.</li><li>두 창 모두에서 CAGR&gt;0, Sharpe&gt;0, MDD≥0의 상대개선과 평균 스트레스&gt;0.002를 요구한다.</li><li>엄격 통과 후보는 810개 중 <b>2개</b>였다.</li><li>통과군에서 다목적 평균순위, Sharpe, CAGR 순으로 <code>acceleration / .90 / 1.0 / .35 / .0 / .20</code>을 선택했다.</li><li>2018년 이후 103개월은 이 선택에 쓰지 않고 성과 확인에만 사용했다.</li></ol>
<div class="callout"><b>이 탐색이 보장하지 않는 것:</b> 후보 수가 많으면 우연히 사전구간에 맞는 조합이 생길 수 있다. 그래서 2018년 이후 잠금, 6개월 블록 부트스트랩, 비용 2배, 성분 절삭실험을 추가했지만 과적합이 없다는 증명은 아니다.</div>
</section>

<section class="section" id="execution"><h2>9. 일별 체결, 비용, 리밸런싱, 월간 조정</h2>
<div class="grid"><article class="card half"><h3>신호 시점</h3><p>__REF_SIM__은 수익을 계산할 날짜보다 앞선 마지막 VKOSPI 관측치만 사용한다. 테스트도 <code>signal_date &lt; return_date</code>를 강제한다.</p></article><article class="card half"><h3>20% 밴드</h3><p>월초에는 항상 리밸런싱한다. 월중에는 목표 비중과 현재 비중 사이의 반회전율이 20% 이상일 때만 거래한다. 신호가 조금 움직일 때마다 사고파는 것을 막는다.</p></article><article class="card half"><h3>비용</h3><p>총 비중변화에 15bp, GLD·USO 합산 비중변화에 추가 5bp를 부과한다. 레버리지의 현금/차입 부분은 연 4%를 일단위로 환산한다.</p></article><article class="card half"><h3>월간 상대수익 조정</h3><p>__REF_RECON__은 일별 오버레이 수익을 그대로 월간 알파로 붙이지 않는다. 같은 일별 엔진의 오버레이/중립 상대배수만 검증된 월간 기준수익에 곱한다.</p></article></div>
<div class="formula">relative_factor_m = (1 + daily_overlay_return_m) / (1 + daily_neutral_return_m)
final_return_m = (1 + validated_monthly_reference_return_m) × relative_factor_m - 1</div>
<p>이 조정은 월간 엔진과 일간 엔진의 조달비용·비용 복리 차이가 마치 오버레이의 성과인 것처럼 섞이는 문제를 줄인다.</p>
</section>

<section class="section" id="effects"><h2>10. 어떤 아이디어가 실제 효과를 냈나</h2><p class="lead">결과를 보면 “VKOSPI 특징을 더 복잡하게 만들었다”만으로 성과가 좋아진 것이 아니다. Robust 신호만 넣으면 오히려 잠금 Sharpe가 낮아졌고, <b>월중 20% 리밸런싱 밴드</b>까지 적용했을 때 최종 개선이 나타났다.</p>
<div class="table-wrap"><table><thead><tr><th>단계</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>월평균 turnover</th></tr></thead><tbody>__STEP_ROWS__</tbody></table></div>
<p><span class="kicker">핵심 해석:</span> 신호는 방어 필요도를 연속값으로 만들었고, 35% 최대 이전은 극단구간의 방어 폭을 열어 두었다. 하지만 작은 신호 변화에 반응하지 않는 20% 밴드가 불필요한 월중 뒤집기와 비용을 줄이며 위험조정 성과를 살렸다.</p>
<div class="table-wrap"><table><thead><tr><th>성분</th><th>2007–2017: CAGR / Sharpe / MDD</th><th>2018–2026: CAGR / Sharpe / MDD</th></tr></thead><tbody>__COMP_ROWS__</tbody></table></div>
<p class="subtle">2018년 이후만 보면 수준+충격 조합의 숫자가 선정식보다 더 좋아 보인다. 그러나 그 조합은 잠금구간 결과를 본 뒤 바꾸지 않았다. 선정식은 사전구간 규칙으로 고정됐고, 잠금구간은 선택을 뒤집는 데 쓰지 않았다.</p>
</section>

<section class="section" id="option"><h2>11. 옵션 자산 매수·매도 결정</h2><p class="lead">옵션 확장은 <b>풋을 매수한 뒤 청산하는 long-put 슬리브</b>다. 여기서 “매도”는 신규 풋 매도(short put)가 아니라 보유한 풋을 파는 청산이다. 의사결정은 __REF_OPTION_SIGNAL__, 종목 선택은 __REF_OPTION_SELECT__, 최종 비중은 __REF_OPTION_ALLOC__에 구현돼 있다.</p>
<h3>11-1. 진입 점수와 매수 조건</h3>
<div class="formula">ramp(x; θ, w) = clip((x - θ) / w, 0, 1)

left  = ramp(left_tail,      0.25, 1.75)
asym  = ramp(asymmetry,      0.25, 1.75)
imp   = ramp(left_impulse_z, 0.25, 1.75)
bread = ramp(breadth_z,      0.00, 2.00)

entry_score = max(left, asym, imp) × (0.50 + 0.50 × bread)
BUY long KOSPI200 put if entry_score ≥ 0.20</div>
<ul><li>모든 VIX6 특징은 <code>shift(1)</code> 후 사용한다. 신호일은 체결일보다 엄격히 앞선다.</li><li>그 달 조건을 처음 만족한 유동성 있는 풋 중 절대델타 5–15%, DTE 30–60일을 남긴다.</li><li>목표 10델타, 목표 DTE 45일에 가장 가깝고, 동률이면 거래량이 큰 종목을 한 개 고른다.</li><li>실제 완결 거래는 <b>__OPTION_TRADES__건</b>, 진입 델타는 __OPTION_DELTA_RANGE__, DTE는 __OPTION_DTE_RANGE__였다.</li></ul>
<h3>11-2. 회복 점수와 매도/청산 조건</h3>
<div class="formula">recovery_score =
    0.50 × clip((-left_change_5) / 1.50, 0, 1)
  + 0.30 × clip((-left_impulse_z) / 1.50, 0, 1)
  + 0.20 × clip((right_tail - left_tail) / 1.50, 0, 1)

SELL/CLOSE held put at the first liquid quote if recovery_score ≥ 0.35
otherwise CLOSE at the last liquid same-contract quote of the month</div>
<p>회복 신호 청산은 <b>__OPTION_RECOVERY_EXITS__건</b>, 월말 롤 청산은 <b>__OPTION_ROLL_EXITS__건</b>이었다. 진입과 청산 모두 거래량이 0보다 큰 관측치만 허용한다.</p>
<h3>11-3. 옵션 비중</h3>
<div class="formula">g(score) ∈ {score, sqrt(score), score², max(score, logistic_severity)}
w_option = w_max × g(entry_score),  w_max ∈ {0.5%, 1%, 2%, 3%}
w_existing_four_asset_sleeve = 1 - w_option

mixed_growth = (1 - w_option) × four_asset_growth
             + w_option × option_growth
             - 2 × 15bp × w_option</div>
<p>옵션 프리미엄을 공짜 레버리지로 더하지 않는다. 옵션 비중만큼 기존 네 자산 슬리브를 줄여 합이 100%가 되게 한다. 후보의 물리적 매수가는 종가×(1+슬리피지), 매도가는 종가×(1−슬리피지)다. VKOSPI가 80·90·97 분위수 또는 5일 충격 2.5 이상이면 슬리피지 배수를 1.25·1.5·2배로 올린다.</p>
<div class="decision"><div class="branch yes"><b>승격 조건</b><p>Base와 Conservative 슬리피지 모두에서 전체기간 CAGR↑, Sharpe↑, MDD 개선.</p></div><div class="gate">→</div><div class="branch no"><b>현재 결과</b><p>강건 후보 0개. 최종 옵션 비중 0%, 기존 Robust VKOSPI 유지.</p></div></div>
</section>

<section class="section" id="results"><h2>12. 성과 결과를 어떻게 읽을 것인가</h2>
<h3>12-1. 현재 전체 경로</h3><div class="table-wrap"><table><thead><tr><th>경로</th><th>개월</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th></tr></thead><tbody>__CURRENT_ROWS__</tbody></table></div>
<p class="subtle">무SJM 변형은 2007–2017 전체 사전구간에서 구 SJM10보다 낮았다(CAGR __NO_SJM_CAL_CAGR__ vs __SJM_CAL_CAGR__, Sharpe __NO_SJM_CAL_SHARPE__ vs __SJM_CAL_SHARPE__). 그러므로 잠금 이후 숫자가 좋아졌다는 이유만으로 “사전 승격 게이트를 통과한 새 배포판”이라고 표현하지 않는다. 현재 작업공간의 비교 기준이라는 뜻이다.</p>
<h3>12-2. 오버레이만 바꾼 순수 비교 · 2018–2026</h3><div class="table-wrap"><table><thead><tr><th>전략</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th></tr></thead><tbody>__OVERLAY_ROWS__</tbody></table></div>
<p>Robust 오버레이의 잠금 개선폭은 CAGR 약 +0.14%p, Sharpe +0.0341, MDD +0.16%p다. 방향은 세 지표 모두 좋지만 크기는 작다. 6개월 블록 부트스트랩에서 CAGR 개선 확률 __BOOT_CAGR__, Sharpe __BOOT_SHARPE__, MDD __BOOT_MDD__, 세 지표 동시개선 확률은 <b>__BOOT_ALL__</b>였다.</p>
<h3>12-3. 비용 2배</h3><div class="table-wrap"><table><thead><tr><th>비용</th><th>전략</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>월평균 turnover</th></tr></thead><tbody>__COST_ROWS__</tbody></table></div>
<h3>12-4. 옵션 자산 후보</h3><div class="table-wrap"><table><thead><tr><th>전략</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>판정</th></tr></thead><tbody>__OPTION_ROWS__</tbody></table></div>
<p>Base 비용의 최선도 옵션 없는 기준보다 CAGR과 Sharpe가 낮았다. MDD는 사실상 같았다. “옵션이 위기 때 오를 수 있다”는 아이디어와 “장기간 프리미엄·슬리피지까지 포함해 포트폴리오 성과를 높인다”는 결과는 다르다.</p>
<h3>12-5. VIX6 입력·단독·혼합 오버레이 · 2018–2026</h3><div class="table-wrap"><table><thead><tr><th>대안</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>기준 대비 CAGR / Sharpe / MDD</th><th>3개 동시</th></tr></thead><tbody>__VIX6_ALT_ROWS__</tbody></table></div>
<p>VIX6 단독은 CAGR을 높였지만 Sharpe와 MDD가 악화했고, 혼합은 CAGR이 소폭 높지만 Sharpe가 아주 조금 낮았다. 정해 둔 3개 동시개선 규칙에 따라 기존 Robust VKOSPI가 유지됐다.</p>
</section>

<section class="section" id="validation"><h2>13. 검증, 강건성, 남은 한계</h2>
<div class="grid">
 <article class="card"><h3>인과적 정렬</h3><p>월간 신호월&lt;투자월, 로지스틱 2개월 embargo, VKOSPI·VIX6 signal_date&lt;체결일을 테스트가 강제한다.</p></article>
 <article class="card"><h3>재현성</h3><p>로지스틱 확률 최대 오차 7.8e−15, 포트폴리오 수익 최대 오차 1.1e−16으로 저장 결과를 재현했다.</p></article>
 <article class="card"><h3>강건성</h3><p>Robust 후보 810개, 로지스틱 후보 __LOGISTIC_CANDIDATES__개, 비용 2배, 6개월 블록 부트스트랩, 성분 절삭실험을 기록했다.</p></article>
</div>
<div class="callout red"><b>가장 큰 경고:</b> 로지스틱 후보의 사전구간 포트폴리오 Sharpe CSCV/PBO는 __LOGISTIC_PBO__로 높다. 선택된 균형 L2 후보가 2018년 이후 포트폴리오 순위 __LOGISTIC_RANK__위를 유지했어도, 여러 후보 가운데 성과가 좋은 것을 고르는 과정의 과적합 위험은 크다는 뜻이다.</div>
<ul><li><b>데이터 범위:</b> 요청은 2007–2026이지만 실제 공통 수익 경로는 2007-04–2026-07이다.</li><li><b>수정·생존편향:</b> 거시/OAP/ETF 데이터의 과거 가용성과 수정 이력은 실시간 데이터베이스 수준으로 완전히 재현하지 못할 수 있다.</li><li><b>옵션 체결:</b> bid/ask 원자료가 없어 종가와 델타 버킷·틱 하한으로 슬리피지를 모형화했다. fractional contract도 허용했다.</li><li><b>옵션 후보 선택:</b> 2007–2026 결과를 보며 추가한 사후 탐색이다. 그래서 옵션을 채택하지 않은 판정은 보수적이지만, 이 구간을 새 전략의 깨끗한 홀드아웃으로 부를 수 없다.</li><li><b>성과지표:</b> Sharpe는 월수익 기반 연율화이며 별도 무위험 초과수익을 빼지 않은 프로젝트 내 공통 정의다.</li></ul>
</section>

<section class="section" id="run"><h2>14. 재실행 순서</h2><p class="lead">캐시와 결과 파일이 이미 있는 현재 작업공간에서 아래 순서로 재현할 수 있다. 옵션 원시 표면 재생성은 시간이 오래 걸릴 수 있으므로 기본 실행은 캐시를 사용한다.</p>
<div class="formula">$py = "D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe"

# 1) Robust VKOSPI 오버레이 후보·잠금 검증
&amp; $py vkospi_robust_dynamic_experiment.py
&amp; $py vkospi_robust_dynamic_attribution.py

# 2) 현재 무SJM + 균형 로지스틱 + Robust VKOSPI 전체 경로
&amp; $py balanced_logistic_no_sjm_strategy.py

# 3) VIX6 기반 옵션 매수·청산·비중·슬리피지 비교
&amp; $py option_asset_slippage_experiment.py

# 4) 핵심 회귀 테스트
&amp; $py -m pytest -q test_vkospi_robust_dynamic_experiment.py `
    test_balanced_logistic_no_sjm_strategy.py `
    test_option_asset_slippage_experiment.py

# 5) 이 설명서 재생성
&amp; $py build_robust_vkospi_implementation_guide.py</div>
<p>전략을 코드에서 따라갈 때는 __REF_FIXED_OVERLAY__, __REF_STRESS__, __REF_SIM__, __REF_RECON__, __REF_OPTION_SIGNAL__, __REF_OPTION_SELECT__, __REF_OPTION_ALLOC__ 순으로 보면 된다.</p>
</section>

</main></div>
<footer><div class="shell"><b>Robust VKOSPI implementation guide</b><br><span>코드·CSV·JSON 산출물을 자동으로 읽어 만든 단일 HTML. 투자 권유가 아니라 연구 구현·백테스트 감사 문서다.</span></div></footer>
<script>
const links=[...document.querySelectorAll('.toc a')];
const sections=links.map(a=>document.querySelector(a.getAttribute('href'))).filter(Boolean);
const observer=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){links.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+e.target.id));}})},{rootMargin:'-15% 0px -75% 0px'});
sections.forEach(s=>observer.observe(s));
</script>
</body></html>'''

    for key, value in replacements.items():
        template = template.replace(key, value)
    unresolved = [token for token in template.split() if token.startswith("__")]
    if unresolved:
        raise RuntimeError(f"Unresolved template tokens: {unresolved[:5]}")
    OUTPUT.write_text(template, encoding="utf-8")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
