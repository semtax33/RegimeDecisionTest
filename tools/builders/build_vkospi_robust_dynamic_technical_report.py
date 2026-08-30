from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tools.builders.build_vkospi_dynamic_deliverables import chart_svg, percent


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "artifacts/reports/vkospi_robust_dynamic_technical_report.html"


def p(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}%"


def signed_p(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):+.{digits}f}%p"


def code_block(source: str) -> str:
    return f'<pre><code>{html.escape(source.strip())}</code></pre>'


def stepwise_rows(frame: pd.DataFrame) -> str:
    labels = {
        "1_ExistingDynamic": "① 기존 동적 전략",
        "2_RobustSignal_OldPolicy": "② 새 신호 + 기존 정책",
        "3_RobustSignal_Transfer35": "③ 최대 이전 35%",
        "4_RobustWinner_Band20": "④ 최종 · 밴드 20%",
    }
    locked = frame.loc[frame["Period"] == "locked_2018_2026"].set_index(
        "Experiment"
    )
    rows = []
    for name in labels:
        row = locked.loc[name]
        emphasis = ' class="winner-row"' if name.startswith("4_") else ""
        rows.append(
            f"<tr{emphasis}><td><b>{labels[name]}</b></td>"
            f"<td>{p(row['CAGR'])}</td><td>{row['Sharpe']:.3f}</td>"
            f"<td>{p(row['MDD'])}</td><td>{row['Calmar']:.3f}</td>"
            f"<td>{row['AvgTurnover']:.3f}</td></tr>"
        )
    return "".join(rows)


def component_rows(frame: pd.DataFrame) -> str:
    labels = {
        "LevelOnly": "수준만",
        "ShockOnly": "5일 충격만",
        "AccelerationOnly": "가속만",
        "LevelShockRenormalized": "수준 + 충격",
        "SelectedAccelerationBlend": "수준 + 충격 + 가속 · 선정식",
    }
    locked = frame.loc[frame["Period"] == "locked_2018_2026"].set_index(
        "Experiment"
    )
    calibration = frame.loc[
        frame["Period"] == "calibration_2007_2017"
    ].set_index("Experiment")
    rows = []
    for name, label in labels.items():
        cal, test = calibration.loc[name], locked.loc[name]
        emphasis = ' class="winner-row"' if name == "SelectedAccelerationBlend" else ""
        rows.append(
            f"<tr{emphasis}><td><b>{label}</b></td>"
            f"<td>{p(cal['CAGR'])} / {cal['Sharpe']:.3f} / {p(cal['MDD'])}</td>"
            f"<td>{p(test['CAGR'])} / {test['Sharpe']:.3f} / {p(test['MDD'])}</td>"
            f"<td>{test['Calmar']:.3f}</td></tr>"
        )
    return "".join(rows)


def contribution_rows(frame: pd.DataFrame, positive: bool) -> str:
    ranked = frame.sort_values("ReturnDelta", ascending=not positive).head(5)
    rows = []
    for month, row in ranked.iterrows():
        rows.append(
            f"<tr><td><b>{month}</b></td><td>{p(row['ExistingReturn'])}</td>"
            f"<td>{p(row['RobustReturn'])}</td>"
            f"<td class={'up' if row['ReturnDelta'] >= 0 else 'down'}>"
            f"{signed_p(row['ReturnDelta'])}</td>"
            f"<td>{p(row['RobustAvgStress'])}</td></tr>"
        )
    return "".join(rows)


def relative_chart(contribution: pd.DataFrame) -> str:
    values = 100 * contribution["CumulativeRelative"].to_numpy(dtype=float)
    width, height, pad = 960, 280, 34
    low = min(float(np.nanmin(values)), 0.0)
    high = max(float(np.nanmax(values)), 0.0)
    if math.isclose(low, high):
        high = low + 1
    x = np.linspace(pad, width - pad, len(values))
    y = height - pad - (values - low) / (high - low) * (height - 2 * pad)
    path = "M " + " L ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in zip(x, y))
    zero_y = height - pad - (0 - low) / (high - low) * (height - 2 * pad)
    ticks = np.linspace(low, high, 5)
    grids = []
    for tick in ticks:
        ty = height - pad - (tick - low) / (high - low) * (height - 2 * pad)
        grids.append(
            f'<line x1="{pad}" x2="{width-pad}" y1="{ty:.2f}" y2="{ty:.2f}" class="grid"/>'
            f'<text x="{pad-6}" y="{ty+4:.2f}" text-anchor="end" class="axis">{tick:.1f}%</text>'
        )
    return f'''<figure class="chart-card"><figcaption><b>최종 전략 / 기존 전략의 누적 상대성과</b><span>2018-01–2026-07</span></figcaption>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="누적 상대성과">
    {''.join(grids)}<line x1="{pad}" x2="{width-pad}" y1="{zero_y:.2f}" y2="{zero_y:.2f}" class="zero"/>
    <path d="{path}" class="relative-line"/></svg></figure>'''


def signal_stats_cards(stats: pd.DataFrame) -> str:
    frame = stats.set_index("Statistic")
    specs = (
        (
            "평균 스트레스",
            p(frame.loc["AverageStress", "ExistingDynamic"]),
            p(frame.loc["AverageStress", "RobustWinner"]),
            "낮아짐",
        ),
        (
            "강한 스트레스일",
            f'{int(frame.loc["StressAbove025Days", "ExistingDynamic"])}일',
            f'{int(frame.loc["StressAbove025Days", "RobustWinner"])}일',
            "−211일",
        ),
        (
            "평균 위험 이전",
            p(frame.loc["AverageTransfer", "ExistingDynamic"]),
            p(frame.loc["AverageTransfer", "RobustWinner"]),
            "낮아짐",
        ),
        (
            "최대 위험 이전",
            p(frame.loc["MaximumTransfer", "ExistingDynamic"]),
            p(frame.loc["MaximumTransfer", "RobustWinner"]),
            "극단 시 확대",
        ),
    )
    return "".join(
        f'<article class="stat"><span>{name}</span><div><s>{old}</s><strong>{new}</strong></div><em>{note}</em></article>'
        for name, old, new, note in specs
    )


def report_href(path: str) -> str:
    if path.startswith(("http://", "https://", "#")):
        return path
    if "/" not in path and path.endswith(".ipynb"):
        return f"../notebooks/{path}"
    if "/" not in path and path.endswith(".zip"):
        return f"../bundles/{path}"
    if path.startswith(("strategies/", "tests/", "results/", "raw_data/", "cache/")):
        return f"../../{path}"
    return path


def file_rows() -> str:
    files = (
        ("raw_data/VKOSPIData.csv", "입력", "2003–2026 VKOSPI 일별 원자료"),
        ("strategies/core/regime_research.py", "기준", "거시 12개 변수, SparseJump2, 4개 국면, SLSQP 방어 경로"),
        ("strategies/stage03_tail_risk/build_crash_features.py", "특징", "국내 포트폴리오 상태·모멘텀·변동성·상관 입력 생성"),
        ("strategies/stage04_ml_feedback/final_blend_crash_meta_experiment.py", "모델", "중앙값 대치·표준화·균형 L2 로지스틱 파이프라인"),
        ("strategies/stage05_openassetpricing/openassetpricing_signal_experiment.py", "모델", "OAP 원시·합성 특징, 2개월 경로손실 라벨과 16개 후보 선택"),
        ("strategies/stage04_ml_feedback/market_structure_robustness.py", "배분", "hard/SLSQP 혼합, 꼬리위험 이동, 15% 변동성 목표"),
        ("strategies/stage06_vkospi/vkospi_feature_experiment.py", "게이트", "VKOSPI ML 후보의 잠금 승격과 기준 전략 유지"),
        ("strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py", "엔진", "데이터 정렬, 일별 시뮬레이터, 비용, 월별 재조정"),
        ("strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py", "전략", "robust 특징, 5개 스트레스 모드, 810개 사전 탐색"),
        ("strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py", "현재", "무SJM 거시확률, 균형 L2, Robust VKOSPI 현재 비교 경로"),
        ("strategies/stage06_vkospi/vkospi_robust_dynamic_attribution.py", "진단", "절삭실험, 성분별 비교, 월별 기여도 생성"),
        ("strategies/stage06_vkospi/vkospi_extended_diagnostics.py", "진단", "거시 상수 민감도, 꼬리분류 지표, 기간성과와 오버피팅 감사"),
        ("strategies/stage06_vkospi/vkospi_model_robustness.py", "감사", "로지스틱 28개·SJM 49개 시도, 잠금 순위·블록 부트스트랩·CSCV/PBO"),
        ("tests/test_vkospi_model_robustness.py", "테스트", "후보 수, 재현 오차, 실패 조합, HTML·Colab 반영 검증"),
        ("tests/test_vkospi_robust_dynamic_experiment.py", "테스트", "누수, 경계값, 재현성, 잠금 성과, 비용 2배 검증"),
        ("results/openassetpricing_medium_horizon_calibration.csv", "근거", "2·3개월 경로손실과 최대 이동 16개 비교"),
        ("results/openassetpricing_validation.json", "근거", "로지스틱 예측 성능과 OAP 선택값"),
        ("results/vkospi_validation.json", "근거", "VKOSPI ML 후보 탈락과 실제 배포 기준 경로"),
        ("results/vkospi_dynamic_validation.json", "근거", "기존 VKOSPI 동적 전략 836개 후보와 엄격 통과 96개"),
        ("results/vkospi_robust_dynamic_calibration.csv", "결과", "810개 후보의 2017년 이전 평가값"),
        ("results/vkospi_robust_dynamic_daily.csv", "결과", "최종 전략 일별 수익·비중·신호·비용"),
        ("results/vkospi_robust_dynamic_reconciled_monthly.csv", "결과", "권위 비교용 월별 재조정 경로"),
        ("results/vkospi_robust_dynamic_validation.json", "결과", "승자 설정, 잠금 성과, 부트스트랩"),
        ("results/balanced_logistic_no_sjm_validation.json", "현재", "현재 비교 기준의 설정·성과·사전 관문·재현 감사"),
        ("results/balanced_logistic_no_sjm_comparison.csv", "현재", "SJM 10%와 무SJM의 구간별 전체 성과 비교"),
        ("results/vkospi_macro_constant_sensitivity.csv", "감사", "0.20·0.55·0.10·0.85의 사후 1변수 민감도"),
        ("results/vkospi_tail_prediction_diagnostics.csv", "감사", "AUC·AP·Brier·LogLoss·ECE와 기준 Brier"),
        ("results/vkospi_tail_feature_diagnostics.csv", "감사", "16개 설명변수의 단변량 AUC와 순차 로지스틱 계수"),
        ("results/vkospi_extended_period_performance.csv", "감사", "2005 요청범위의 가용성 및 실제 전체·하위구간 성과"),
        ("results/vkospi_overfitting_diagnostics.json", "감사", "810개 탐색, 이웃 설정, 잠금·부트스트랩 위험 진단"),
        ("results/vkospi_model_robustness.json", "감사", "SJM·로지스틱 하이퍼파라미터 강건성 종합 결론"),
        ("results/vkospi_logistic_candidate_summary.csv", "감사", "로지스틱 28개 후보의 예측·성과·수렴·잠금 순위"),
        ("results/vkospi_sjm_candidate_summary.csv", "감사", "SJM 49개 시도와 12개 계산 실패를 포함한 후보 요약"),
        ("vkospi_robust_dynamic_strategy_colab.ipynb", "실행", "Google Colab 재현 노트북"),
        ("vkospi_robust_dynamic_colab_bundle.zip", "실행", "Colab용 코드·거시 원자료·성과·진단 결과 번들"),
    )
    return "".join(
        f'<tr><td><a href="{html.escape(report_href(path))}"><code>{html.escape(path)}</code></a></td>'
        f'<td><span class="tag">{kind}</span></td><td>{description}</td></tr>'
        for path, kind, description in files
    )


def medium_horizon_rows(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples():
        winner = (
            row.target == "oap_path_loss_2m_5"
            and int(row.horizon) == 2
            and math.isclose(float(row.max_shift), 0.20)
        )
        emphasis = ' class="winner-row"' if winner else ""
        target_label = {
            "oap_path_loss_2m_4": "2개월 중 −4%",
            "oap_path_loss_2m_5": "2개월 중 −5%",
            "oap_path_loss_3m_5": "3개월 중 −5%",
            "oap_path_loss_3m_6": "3개월 중 −6%",
        }[str(row.target)]
        rows.append(
            f"<tr{emphasis}><td><code>{row.target}</code><br>{target_label}</td>"
            f"<td>{int(row.events)}</td><td>{p(row.max_shift, 0)}</td>"
            f"<td>{p(row.CAGR)}</td><td>{row.Sharpe:.4f}</td>"
            f"<td>{p(row.MDD)}</td><td>{row.Calmar:.4f}</td></tr>"
        )
    return "".join(rows)


def oap_family_rows(frame: pd.DataFrame) -> str:
    labels = {
        "FinalBlend": "OAP 미적용 비교선",
        "oap_momentum_domestic": "모멘텀 + 국내 12개",
        "oap_reversal_domestic": "반전·쏠림 + 국내 12개",
        "oap_lowrisk_domestic": "저위험·꼬리 + 국내 12개",
        "oap_liquidity_domestic": "유동성·활동 + 국내 12개",
        "oap_all_domestic": "OAP 4개 + 국내 12개",
    }
    view = frame.loc[
        (frame["Period"] == "calibration_2007_2017")
        & frame["Strategy"].isin(labels)
    ].set_index("Strategy")
    rows = []
    for name, label in labels.items():
        row = view.loc[name]
        rows.append(
            f"<tr><td><b>{label}</b><br><code>{name}</code></td>"
            f"<td>{p(row['CAGR'])}</td><td>{row['Sharpe']:.4f}</td>"
            f"<td>{p(row['MDD'])}</td><td>{row['Calmar']:.4f}</td>"
            f"<td>{row['AvgTurnover']:.4f}</td></tr>"
        )
    return "".join(rows)


def formula_metric_rows(
    old_returns: pd.Series,
    new_returns: pd.Series,
) -> str:
    rows = []
    for label, values in (("기존 VKOSPI 동적", old_returns), ("최종 robust", new_returns)):
        r = pd.Series(values).dropna()
        wealth = (1 + r).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        trough = drawdown.idxmin()
        peak_month = wealth.loc[:trough].idxmax()
        mean = float(r.mean())
        std = float(r.std(ddof=1))
        downside = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * math.sqrt(12))
        rows.append(
            f"<tr><td><b>{label}</b></td><td>{len(r)} / {len(r)/12:.4f}년</td>"
            f"<td>{p(mean, 4)}</td><td>{p(std, 4)}</td>"
            f"<td>{p(wealth.iloc[-1] - 1)} / {wealth.iloc[-1]:.6f}배</td>"
            f"<td>{peak_month} → {trough}</td><td>{int((r > 0).sum())}개월</td>"
            f"<td>{p(downside, 4)}</td></tr>"
        )
    return "".join(rows)


def cost_sensitivity_rows(frame: pd.DataFrame) -> str:
    labels = {"ExistingDynamic": "기존 VKOSPI 동적", "RobustDynamic": "최종 robust"}
    rows = []
    for row in frame.itertuples():
        multiplier = "1배" if "1.0x" in str(row.Period) else "2배"
        emphasis = ' class="winner-row"' if row.Strategy == "RobustDynamic" else ""
        rows.append(
            f"<tr{emphasis}><td>{multiplier}</td><td><b>{labels[str(row.Strategy)]}</b></td>"
            f"<td>{p(row.CAGR)}</td><td>{row.Sharpe:.4f}</td>"
            f"<td>{p(row.MDD)}</td><td>{row.Calmar:.4f}</td>"
            f"<td>{row.AvgTurnover:.4f}</td></tr>"
        )
    return "".join(rows)


def extended_performance_rows(frame: pd.DataFrame, period: str) -> str:
    labels = {
        "ReferenceMediumHorizonOAPVol15": "월간 기준 전략",
        "ExistingVKOSPIDynamic": "기존 VKOSPI 동적",
        "RobustVKOSPIDynamic": "최종 robust 동적",
        "KODEX200ProxyBenchmark": "KODEX200 프록시 벤치마크",
    }
    rows = []
    for row in frame.loc[frame["period"] == period].itertuples():
        unavailable = str(row.status) == "unavailable_same_strategy"
        emphasis = ' class="winner-row"' if row.strategy == "RobustVKOSPIDynamic" else ""
        metric_cells = (
            '<td colspan="6">측정 불가 · 동일 전략의 최초 거래월은 2007-04</td>'
            if unavailable
            else (
                f"<td>{int(row.Months)}개월</td><td>{p(row.CAGR)}</td>"
                f"<td>{row.Sharpe:.3f}</td><td>{p(row.MDD)}</td>"
                f"<td>{row.Calmar:.3f}</td><td>{row.FinalMultiple:.4f}배</td>"
            )
        )
        rows.append(
            f"<tr{emphasis}><td><b>{labels.get(str(row.strategy), str(row.strategy))}</b>"
            f"<br><code>{row.strategy}</code></td><td>{row.start}~{row.end}</td>"
            f"{metric_cells}</tr>"
        )
    return "".join(rows)


def subperiod_performance_rows(frame: pd.DataFrame) -> str:
    periods = {
        "early_calibration_2007_04_2012_12": "초기 사전구간 · 2007-04~2012-12",
        "validation_2013_01_2017_12": "내부검증 · 2013-01~2017-12",
        "locked_early_2018_01_2021_12": "잠금 전반 · 2018-01~2021-12",
        "locked_late_2022_01_2026_07": "잠금 후반 · 2022-01~2026-07",
    }
    view = frame.loc[
        frame["period"].isin(periods)
        & frame["strategy"].isin(["ExistingVKOSPIDynamic", "RobustVKOSPIDynamic"])
    ]
    indexed = view.set_index(["period", "strategy"])
    rows = []
    for period, label in periods.items():
        old = indexed.loc[(period, "ExistingVKOSPIDynamic")]
        new = indexed.loc[(period, "RobustVKOSPIDynamic")]
        rows.append(
            f"<tr><td><b>{label}</b></td>"
            f"<td>{p(old.CAGR)} → {p(new.CAGR)} ({signed_p(new.CAGR-old.CAGR, 3)})</td>"
            f"<td>{old.Sharpe:.4f} → {new.Sharpe:.4f} ({new.Sharpe-old.Sharpe:+.4f})</td>"
            f"<td>{p(old.MDD)} → {p(new.MDD)} ({signed_p(new.MDD-old.MDD, 3)})</td></tr>"
        )
    return "".join(rows)


def macro_sensitivity_rows(frame: pd.DataFrame) -> str:
    deployed = frame.loc[frame["parameter"] == "deployed"].set_index("period")
    candidates = [
        ("배포값", "deployed", np.nan),
        ("3개월 변화 가중", "d3_weight", 0.00),
        ("3개월 변화 가중", "d3_weight", 0.10),
        ("3개월 변화 가중", "d3_weight", 0.30),
        ("3개월 변화 가중", "d3_weight", 0.40),
        ("sigmoid scale", "sigmoid_scale", 0.35),
        ("sigmoid scale", "sigmoid_scale", 0.45),
        ("sigmoid scale", "sigmoid_scale", 0.65),
        ("sigmoid scale", "sigmoid_scale", 0.75),
        ("SJM 혼합", "sjm_weight", 0.00),
        ("SJM 혼합", "sjm_weight", 0.05),
        ("SJM 혼합", "sjm_weight", 0.15),
        ("SJM 혼합", "sjm_weight", 0.20),
        ("당월 반영", "current_weight", 0.70),
        ("당월 반영", "current_weight", 0.80),
        ("당월 반영", "current_weight", 0.90),
        ("당월 반영", "current_weight", 1.00),
    ]
    rows = []
    for label, parameter, value in candidates:
        if parameter == "deployed":
            cal = deployed.loc["calibration_through_2017"]
            locked = deployed.loc["locked_2018_2026"]
            setting = "0.20 / 0.55 / 0.10 / 0.85"
            emphasis = ' class="winner-row"'
        else:
            selected = frame.loc[
                (frame["parameter"] == parameter) & np.isclose(frame["value"], value)
            ].set_index("period")
            cal = selected.loc["calibration_through_2017"]
            locked = selected.loc["locked_2018_2026"]
            setting = f"{value:.2f}"
            emphasis = ""
        rows.append(
            f"<tr{emphasis}><td><b>{label}</b></td><td>{setting}</td>"
            f"<td>{cal.mean_brier:.4f} / {p(cal.quadrant_accuracy)}</td>"
            f"<td>{locked.mean_brier:.4f} / {p(locked.quadrant_accuracy)}</td></tr>"
        )
    return "".join(rows)


def tail_prediction_rows(frame: pd.DataFrame) -> str:
    labels = {
        "calibration_through_2017": "2017년 이전",
        "locked_2018_2026": "2018년 이후 잠금",
    }
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"<tr><td><b>{labels[str(row.period)]}</b></td>"
            f"<td>{int(row.observations)} / {int(row.events)} ({p(row.event_rate)})</td>"
            f"<td>{row.roc_auc:.4f}</td><td>{row.average_precision:.4f}</td>"
            f"<td>{row.brier_score:.4f}</td><td>{row.calibration_prevalence_brier:.4f}</td>"
            f"<td>{row.log_loss:.4f}</td><td>{row.ece_5bin:.4f}</td>"
            f"<td>{p(row.recall_at_top_20pct)} / {p(row.precision_at_top_20pct)}</td></tr>"
        )
    return "".join(rows)


def tail_feature_rows(frame: pd.DataFrame) -> str:
    view = frame.pivot(index="feature", columns="period")
    rows = []
    for feature in TAIL_FEATURE_ORDER:
        group = "국내" if feature in DOMESTIC_FEATURE_ORDER else "OAP"
        cal = "calibration_through_2017"
        locked = "locked_2018_2026"
        rows.append(
            f"<tr><td>{group}</td><td><code>{feature}</code></td>"
            f"<td>{view.loc[feature, ('raw_univariate_auc', cal)]:.3f} / "
            f"{view.loc[feature, ('raw_univariate_auc', locked)]:.3f}</td>"
            f"<td>{view.loc[feature, ('median_standardized_logit_coefficient', cal)]:+.3f} / "
            f"{view.loc[feature, ('median_standardized_logit_coefficient', locked)]:+.3f}</td>"
            f"<td>{p(view.loc[feature, ('coefficient_sign_stability', cal)])} / "
            f"{p(view.loc[feature, ('coefficient_sign_stability', locked)])}</td></tr>"
        )
    return "".join(rows)


def grid_neighborhood_rows(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples():
        emphasis = ' class="winner-row"' if bool(row.Selected) else ""
        rows.append(
            f"<tr{emphasis}><td><code>{row.Config}</code></td>"
            f"<td>{int(row.grid_distance_from_winner)}</td><td>{'통과' if row.strict_pass else '탈락'}</td>"
            f"<td>{row.Cal_CAGRDelta:+.4%} / {row.Cal_SharpeDelta:+.4f} / {row.Cal_MDDDelta:+.4%}</td>"
            f"<td>{row.Validation_CAGRDelta:+.4%} / {row.Validation_SharpeDelta:+.4f} / {row.Validation_MDDDelta:+.4%}</td>"
            f"<td>{row.MultiObjectiveScore:.6f}</td></tr>"
        )
    return "".join(rows)


def logistic_robustness_rows(frame: pd.DataFrame) -> str:
    labels = {
        "l2_liblinear_c0.1_balanced": "배포값",
        "l2_liblinear_c0.1_none": "사전 예측 1위",
        "l2_lbfgs_c0.1_balanced": "solver 이웃",
        "l1_liblinear_c0.03_balanced": "L1 순위 역전",
        "l2_liblinear_c0.3_none": "잠금 진단 전용",
    }
    rows = []
    for candidate, label in labels.items():
        row = frame.loc[candidate]
        emphasis = ' class="winner-row"' if label == "배포값" else ""
        warning_count = int(row["convergence_warning_count"])
        rows.append(
            f"<tr{emphasis}><td><b>{label}</b><br><code>{candidate}</code></td>"
            f"<td>{row['cal_roc_auc']:.4f} / {row['cal_brier_score']:.4f} / {row['cal_Sharpe']:.4f}</td>"
            f"<td>{row['locked_roc_auc']:.4f} / {row['locked_brier_score']:.4f} / {row['locked_Sharpe']:.4f}</td>"
            f"<td>{int(row['locked_prediction_rank'])} / {int(row['locked_portfolio_rank'])}</td>"
            f"<td>{warning_count}</td></tr>"
        )
    return "".join(rows)


def sjm_robustness_rows(frame: pd.DataFrame) -> str:
    labels = {
        "no_sjm": "SJM 미사용",
        "sjm_j3_k4_w0.1": "배포값",
        "sjm_j0_k4_w0.1": "사전 예측 1위",
        "sjm_j0_k4_w0.2": "사전 soft 성과 1위",
        "sjm_j3_k2_w0.1": "한 축 이웃",
    }

    def number(value: float) -> str:
        return "—" if not np.isfinite(value) else f"{value:.4f}"

    rows = []
    for candidate, label in labels.items():
        row = frame.loc[candidate]
        emphasis = ' class="winner-row"' if label == "배포값" else ""
        rows.append(
            f"<tr{emphasis}><td><b>{label}</b><br><code>{candidate}</code></td>"
            f"<td>{row['cal_mean_brier']:.4f} / {row['cal_Soft_Sharpe']:.4f}</td>"
            f"<td>{row['locked_mean_brier']:.4f} / {row['locked_Soft_Sharpe']:.4f}</td>"
            f"<td>{number(float(row['locked_Proposed_Sharpe']))}</td>"
            f"<td>{int(row['locked_prediction_rank'])} / {int(row['locked_soft_rank'])}</td></tr>"
        )
    return "".join(rows)


def model_subperiod_rows(logistic: pd.DataFrame, sjm: pd.DataFrame) -> str:
    specs = (
        (logistic, "l2_liblinear_c0.1_balanced", "로지스틱 배포", "brier_score", "Sharpe"),
        (logistic, "l2_liblinear_c0.1_none", "로지스틱 무가중", "brier_score", "Sharpe"),
        (sjm, "sjm_j3_k4_w0.1", "SJM 배포", "mean_brier", "Soft_Sharpe"),
        (sjm, "no_sjm", "SJM 미사용", "mean_brier", "Soft_Sharpe"),
    )
    rows = []
    for frame, candidate, label, loss_column, sharpe_column in specs:
        view = frame.loc[frame["candidate"].eq(candidate)].set_index("period")
        early = view.loc["locked_early_2018_2021"]
        late = view.loc["locked_late_2022_2026"]
        rows.append(
            f"<tr><td><b>{label}</b><br><code>{candidate}</code></td>"
            f"<td>{early[loss_column]:.4f} / {early[sharpe_column]:.4f}</td>"
            f"<td>{late[loss_column]:.4f} / {late[sharpe_column]:.4f}</td></tr>"
        )
    return "".join(rows)


DOMESTIC_FEATURE_ORDER = [
    "base_USO", "base_GLD", "base_KODEX200", "p_inflation_high", "proxy_mom1",
    "proxy_mom6", "proxy_vol6", "daily_mom21", "daily_mom252", "daily_vol21",
    "daily_downvol21", "daily_mean_corr63",
]
TAIL_FEATURE_ORDER = DOMESTIC_FEATURE_ORDER + [
    "oap_momentum_trend_stress", "oap_reversal_crowding_stress",
    "oap_low_risk_tail_stress", "oap_liquidity_activity_stress",
]


SOURCE_FILES = (
    ("strategies/core/regime_research.py", "거시 12개 변수, SparseJump2, 국면확률, SLSQP 방어 비중"),
    ("strategies/stage03_tail_risk/build_crash_features.py", "국내 포트폴리오 상태 12개 생성"),
    ("strategies/stage04_ml_feedback/final_blend_crash_meta_experiment.py", "균형 L2 로지스틱 파이프라인과 walk-forward 적합"),
    ("strategies/stage05_openassetpricing/openassetpricing_signal_experiment.py", "OAP 원시·합성 변수와 중기 목표 선택"),
    ("strategies/stage04_ml_feedback/market_structure_robustness.py", "중기 위험 이동과 15% 변동성 목표"),
    ("strategies/stage06_vkospi/vkospi_feature_experiment.py", "VKOSPI ML 후보 승격 게이트"),
    ("strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py", "일별 오버레이, 체결비용, 월별 재조정"),
    ("strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py", "robust 특징, 5개 모드, 810개 탐색"),
    ("strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py", "현재 무SJM 거시확률, 균형 L2, Robust VKOSPI 비교 경로"),
    ("strategies/stage06_vkospi/vkospi_robust_dynamic_attribution.py", "사후 절삭·기여도·비용 민감도"),
    ("strategies/stage06_vkospi/vkospi_extended_diagnostics.py", "상수 민감도·분류 진단·기간성과·오버피팅 감사"),
    ("strategies/stage06_vkospi/vkospi_model_robustness.py", "SJM·로지스틱 하이퍼파라미터 강건성, 잠금 순위, CSCV/PBO"),
    ("tests/test_vkospi_robust_dynamic_experiment.py", "누수·경계값·재현성·잠금 성과 테스트"),
    ("tests/test_vkospi_model_robustness.py", "강건성 산출물·재현 오차·실패 조합 테스트"),
    ("tests/test_balanced_logistic_no_sjm_strategy.py", "현재 비교 경로의 설정·인과성·성과·사전 관문 테스트"),
)


def full_source_details() -> str:
    """Embed the complete strategy source so the report remains self-contained."""
    blocks = []
    for relative_path, description in SOURCE_FILES:
        path = ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        numbered = "\n".join(
            f"{line_no:04d}  {line}"
            for line_no, line in enumerate(source.splitlines(), start=1)
        )
        blocks.append(
            f'<details class="source"><summary><code>{html.escape(relative_path)}</code> · '
            f'{description} · {len(source.splitlines()):,}줄</summary>'
            f'<pre><code>{html.escape(numbered)}</code></pre></details>'
        )
    return "".join(blocks)


def comment_index_rows() -> str:
    """List every full-line source comment and classify the guardrail it documents."""
    rows = []
    rules = (
        (
            "시점·누수",
            re.compile(
                r"causal|future|embargo|cutoff|timing|date|locked|calibration|lookahead",
                re.I,
            ),
            "미래 정보가 특징·선정·검증으로 새어 들어가지 않게 시점을 고정하는 주석",
        ),
        (
            "선택·고정",
            re.compile(r"grid|candidate|winner|selected|strict|fixed|legacy|gate", re.I),
            "탐색한 값과 미리 고정한 값을 구분하고 잠금 뒤 재선택을 막는 주석",
        ),
        (
            "회계·비용",
            re.compile(r"reconcil|neutral|relative|cost|debt|finance|turnover", re.I),
            "빈도 차이·체결비용·차입비용을 성과로 잘못 세지 않게 하는 주석",
        ),
        (
            "실패·대체",
            re.compile(r"fallback|fail|missing|empty|retain|keep|guard", re.I),
            "자료나 최적화가 불완전할 때 기존 경로로 안전하게 돌아가기 위한 주석",
        ),
    )
    for relative_path, _ in SOURCE_FILES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        for line_no, line in enumerate(source.splitlines(), start=1):
            match = re.match(r"^\s*#\s?(.*)$", line)
            if not match:
                continue
            comment = match.group(1).strip()
            category = "구현 메모"
            explanation = "바로 아래 계산의 의도나 단계 경계를 기록한 주석"
            for name, pattern, note in rules:
                if pattern.search(comment):
                    category, explanation = name, note
                    break
            rows.append(
                f"<tr><td><code>{html.escape(relative_path)}:{line_no}</code></td>"
                f"<td>{html.escape(comment)}</td><td>{category}</td><td>{explanation}</td></tr>"
            )
    return "".join(rows)


def source_comment_count() -> int:
    return sum(
        1
        for relative_path, _ in SOURCE_FILES
        for line in (ROOT / relative_path).read_text(encoding="utf-8").splitlines()
        if re.match(r"^\s*#", line)
    )


def current_reference_rows(frame: pd.DataFrame) -> str:
    periods = {
        "calibration_2007_2017": "사전 전체 · 2007–2017",
        "validation_2013_2017": "사전 내부 · 2013–2017",
        "locked_2018_2026": "잠금 · 2018–2026",
        "full_2007_2026": "전체 · 2007–2026",
    }
    strategies = {
        "Deployed_SJM10_BalancedLogistic_RobustVKOSPI": "사전선정 SJM 10%",
        "NoSJM_BalancedLogistic_RobustVKOSPI": "현재 비교기준 · 무SJM",
    }
    rows: list[str] = []
    for period, period_label in periods.items():
        view = frame.loc[frame["Period"].eq(period)].set_index("Strategy")
        for strategy, strategy_label in strategies.items():
            row = view.loc[strategy]
            emphasis = ' class="winner-row"' if strategy.startswith("NoSJM") else ""
            rows.append(
                f"<tr{emphasis}><td>{period_label}</td><td><b>{strategy_label}</b></td>"
                f"<td>{int(row['Months'])}</td><td>{p(row['CAGR'])}</td>"
                f"<td>{row['Sharpe']:.3f}</td><td>{p(row['MDD'])}</td>"
                f"<td>{p(row['Volatility'])}</td><td>{row['Calmar']:.3f}</td>"
                f"<td>{p(row['AvgTurnover'])}</td></tr>"
            )
    return "".join(rows)


def current_prediction_rows(report: dict[str, object]) -> str:
    labels = {
        "calibration_2007_2017": "사전 전체 · 2007–2017",
        "validation_2013_2017": "사전 내부 · 2013–2017",
        "locked_2018_2026": "잠금 · 2018–2026",
    }
    rows: list[str] = []
    prediction = report["prediction"]
    for key, label in labels.items():
        row = prediction[key]
        rows.append(
            f"<tr><td>{label}</td><td>{int(row['observations'])}</td>"
            f"<td>{int(row['events'])} ({p(row['event_rate'])})</td>"
            f"<td>{row['roc_auc']:.3f}</td><td>{row['average_precision']:.3f}</td>"
            f"<td>{row['brier_score']:.3f}</td></tr>"
        )
    return "".join(rows)


def recent_regime_rows(frame: pd.DataFrame, count: int = 8) -> str:
    rows: list[str] = []
    for month, row in frame.tail(count).iterrows():
        rows.append(
            f"<tr><td>{month}</td><td>{row['signal_month']}</td>"
            f"<td>{p(row['p_growth_high'], 1)}</td><td>{p(row['p_inflation_high'], 1)}</td>"
            f"<td>{p(row['p_Goldilocks'], 1)}</td><td>{p(row['p_Overheating'], 1)}</td>"
            f"<td>{p(row['p_Slowdown'], 1)}</td><td>{p(row['p_Stagflation'], 1)}</td>"
            f"<td><b>{row['regime']}</b></td></tr>"
        )
    return "".join(rows)


def recent_tail_rows(frame: pd.DataFrame, count: int = 8) -> str:
    rows: list[str] = []
    for month, row in frame.tail(count).iterrows():
        event = "미확정" if pd.isna(row["tail_event"]) else ("사건" if row["tail_event"] else "정상")
        rows.append(
            f"<tr><td>{month}</td><td>{row['p_tail_raw']:.3f}</td>"
            f"<td>{p(row['risk_percentile'], 1)}</td><td>{p(row['risk_severity'], 1)}</td>"
            f"<td>{row['p_up']:.3f}</td><td>{event}</td></tr>"
        )
    return "".join(rows)


def build() -> str:
    report = json.loads(
        (RESULTS / "vkospi_robust_dynamic_validation.json").read_text(
            encoding="utf-8"
        )
    )
    old_path = pd.read_csv(
        RESULTS / "vkospi_dynamic_reconciled_monthly.csv", index_col=0
    )
    new_path = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_reconciled_monthly.csv", index_col=0
    )
    old_path.index = pd.PeriodIndex(old_path.index, freq="M")
    new_path.index = pd.PeriodIndex(new_path.index, freq="M")
    nav = chart_svg(old_path, new_path, "nav", "누적 NAV")
    drawdown = chart_svg(old_path, new_path, "drawdown", "드로다운")
    for before, after in (
        ("기준 전략", "기존 VKOSPI 동적"),
        ("VKOSPI 동적 전략", "최종 robust 전략"),
    ):
        nav = nav.replace(before, after)
        drawdown = drawdown.replace(before, after)

    stepwise = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_stepwise_attribution.csv"
    )
    components = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_component_ablation.csv"
    )
    stats = pd.read_csv(RESULTS / "vkospi_robust_dynamic_signal_statistics.csv")
    contribution = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_monthly_contribution.csv", index_col=0
    )
    costs = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_cost_sensitivity.csv"
    )
    winner = report["winner"]
    locked = report["locked"]
    boot = locked["bootstrap"]
    strict = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_calibration.csv"
    ).loc[
        lambda frame: (frame["Cal_CAGRDelta"] > 0)
        & (frame["Cal_SharpeDelta"] > 0)
        & (frame["Cal_MDDDelta"] >= 0)
        & (frame["Validation_CAGRDelta"] > 0)
        & (frame["Validation_SharpeDelta"] > 0)
        & (frame["Validation_MDDDelta"] >= 0)
        & (frame["AvgStress"] > 0.002)
    ]
    cost_2x = costs.loc[
        (costs["Period"] == "cost_2.0x_locked")
        & (costs["Strategy"].isin(["ExistingDynamic", "RobustDynamic"]))
    ].set_index("Strategy")
    oap_report = json.loads(
        (RESULTS / "openassetpricing_validation.json").read_text(encoding="utf-8")
    )
    vkospi_feature_report = json.loads(
        (RESULTS / "vkospi_validation.json").read_text(encoding="utf-8")
    )
    medium_grid = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_calibration.csv"
    )
    oap_comparison = pd.read_csv(RESULTS / "openassetpricing_comparison.csv")
    existing_structure_cal = oap_comparison.loc[
        (oap_comparison["Period"] == "calibration_2007_2017")
        & (oap_comparison["Strategy"] == "ExistingStructureVol15")
    ].iloc[0]
    medium_eligible = medium_grid.loc[
        (medium_grid["CAGR"] >= float(existing_structure_cal["CAGR"]))
        & (medium_grid["Sharpe"] >= float(existing_structure_cal["Sharpe"]))
        & (medium_grid["MDD"] >= float(existing_structure_cal["MDD"]) - 1e-8)
    ]
    macro_sensitivity = pd.read_csv(
        RESULTS / "vkospi_macro_constant_sensitivity.csv"
    )
    tail_prediction = pd.read_csv(
        RESULTS / "vkospi_tail_prediction_diagnostics.csv"
    )
    tail_features = pd.read_csv(
        RESULTS / "vkospi_tail_feature_diagnostics.csv"
    )
    extended_period = pd.read_csv(
        RESULTS / "vkospi_extended_period_performance.csv"
    )
    grid_neighborhood = pd.read_csv(
        RESULTS / "vkospi_robust_grid_neighborhood.csv"
    )
    overfit = json.loads(
        (RESULTS / "vkospi_overfitting_diagnostics.json").read_text(encoding="utf-8")
    )
    model_robustness = json.loads(
        (RESULTS / "vkospi_model_robustness.json").read_text(encoding="utf-8")
    )
    logistic_summary = pd.read_csv(
        RESULTS / "vkospi_logistic_candidate_summary.csv"
    ).set_index("candidate")
    sjm_summary = pd.read_csv(
        RESULTS / "vkospi_sjm_candidate_summary.csv"
    ).set_index("candidate")
    logistic_robustness_long = pd.read_csv(
        RESULTS / "vkospi_logistic_hyperparameter_robustness.csv"
    )
    sjm_robustness_long = pd.read_csv(RESULTS / "vkospi_sjm_robustness.csv")
    current_comparison = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_comparison.csv"
    )
    current_validation = json.loads(
        (RESULTS / "balanced_logistic_no_sjm_validation.json").read_text(
            encoding="utf-8"
        )
    )
    current_signals = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_signals.csv", index_col=0
    )
    current_factor = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_factor.csv", index_col=0
    )
    locked_start = pd.Period("2018-01", freq="M")
    old_locked_returns = old_path.loc[locked_start:, "return"]
    new_locked_returns = new_path.loc[locked_start:, "return"]

    formula_code = r"""
L_t = clip((percentile_252(t) - 0.90) / 0.10, 0, 1)
S_t = clip((shock_5(t)         - 1.00) / 2.50, 0, 1)
A_t = clip((acceleration_z5(t) - 1.00) / 2.50, 0, 1)

stress_t = 0.40 * L_t + 0.35 * S_t + 0.25 * A_t
transfer_t = 0.35 * stress_t
"""
    feature_code = r"""
scale_w = std(log_return, trailing=63) * sqrt(w)
shock_w = (log(VKOSPI_t) - log(VKOSPI_{t-w})) / scale_w

acceleration_5 = Δ5 log(VKOSPI_t) - Δ5 log(VKOSPI_{t-5})
acceleration_z5 = acceleration_5 / (std_63 * sqrt(5))

robust_z = (log(VKOSPI) - rolling_median) / (1.4826 * rolling_MAD)
"""
    allocation_code = r"""
removed_equity = desired[KODEX200] * transfer_fraction
removed_oil    = desired[USO]      * transfer_fraction

desired[KODEX200] -= removed_equity
desired[USO]      -= removed_oil
desired[GLD]      += removed_equity + removed_oil  # bond_share = 0

rebalance = month_boundary or desired_turnover >= 0.20
"""
    reconciliation_code = r"""
relative_factor_m = (1 + overlay_daily_return_m) / (1 + baseline_daily_return_m)
reconciled_return_m = (1 + validated_monthly_return_m) * relative_factor_m - 1
"""
    pipeline_code = r"""
macro_12 -> SparseJump2 + transparent composite -> 4 regime probabilities
regime probabilities + 4-asset history -> SLSQP defensive weights

domestic_12 + OAP_composites_4
    -> median impute -> standardize -> balanced L2 logistic
    -> 2-month path-loss probability -> causal top-20% severity

0.40 * hard_regime + 0.60 * SLSQP_defensive
    -> max 20% tail-risk shift -> 15% volatility target
    -> validated monthly reference weights

VKOSPI percentile + normalized shock + acceleration
    -> daily stress -> max 35% KODEX200/USO transfer to GLD
    -> 20% no-trade band -> costs -> relative reconciliation
"""
    current_full = current_comparison.loc[
        current_comparison["Period"].eq("full_2007_2026")
        & current_comparison["Strategy"].eq(
            "NoSJM_BalancedLogistic_RobustVKOSPI"
        )
    ].iloc[0]
    current_locked = current_comparison.loc[
        current_comparison["Period"].eq("locked_2018_2026")
        & current_comparison["Strategy"].eq(
            "NoSJM_BalancedLogistic_RobustVKOSPI"
        )
    ].iloc[0]
    deployed_full = current_comparison.loc[
        current_comparison["Period"].eq("full_2007_2026")
        & current_comparison["Strategy"].eq(
            "Deployed_SJM10_BalancedLogistic_RobustVKOSPI"
        )
    ].iloc[0]
    prelock_gate = current_validation["prelock_gate"]
    regime_transitions = int(
        (current_signals["regime"] != current_signals["regime"].shift())
        .iloc[1:]
        .sum()
    )
    current_reference_section = f"""
<section id="current-reference"><div class="head"><small>01A · CURRENT REFERENCE</small><div><h2>먼저 ‘Robust VKOSPI 전략’의 두 버전을 구분한다</h2><p class="lede">같은 일별 VKOSPI 오버레이를 쓰지만, 월간 거시 확률의 SJM 포함 여부에 따라 증거 수준이 다릅니다. 이 보고서는 사용자가 최근 비교해 온 <b>무SJM + 균형 L2 + Robust VKOSPI</b>를 중심으로 설명하되, 사전선정 경로를 나란히 남깁니다.</p></div></div>
<div class="grid2"><div class="panel"><h3>사전선정 경로 · SJM 10%</h3><p>2017년 이전 자료로 고정한 월간 기준 경로 위에서 Robust VKOSPI 810개를 탐색했습니다. 이 오버레이 자체는 두 사전 창의 CAGR·Sharpe·MDD 관문을 통과했습니다.</p><p><b>전체:</b> CAGR {p(deployed_full['CAGR'])}, Sharpe {deployed_full['Sharpe']:.3f}, MDD {p(deployed_full['MDD'])}</p></div><div class="panel"><h3>현재 비교 기준 · 무SJM</h3><p>SJM 기여를 0으로 놓고 같은 균형 L2 꼬리분류기와 Robust VKOSPI를 결합한 사후 ablation입니다. 잠금에서 좋아 보였지만 사전 두 창 동시 관문은 <b>통과하지 못했습니다</b>.</p><p><b>전체:</b> CAGR {p(current_full['CAGR'])}, Sharpe {current_full['Sharpe']:.3f}, MDD {p(current_full['MDD'])}</p></div></div>
<div class="warning"><b>반드시 붙여야 할 단서.</b> 현재 비교 기준의 2018–2026 CAGR {p(current_locked['CAGR'])}, Sharpe {current_locked['Sharpe']:.3f}, MDD {p(current_locked['MDD'])}는 계산된 사실입니다. 그러나 무SJM 변경은 2007–2017에서 SJM 10% 경로보다 CAGR {signed_p(prelock_gate['deltas_variant_minus_deployed']['calibration_2007_2017']['CAGR'])}, Sharpe {prelock_gate['deltas_variant_minus_deployed']['calibration_2007_2017']['Sharpe']:+.3f}였고 사전 관문이 실패했습니다. 따라서 ‘현재 연구 비교선’이지 깨끗한 신규 홀드아웃 승자라고 부르면 안 됩니다.</div>

<h3>같은 구간으로 본 전체 성과</h3><div class="table"><table><thead><tr><th>구간</th><th>경로</th><th>개월</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>변동성</th><th>Calmar</th><th>회전율</th></tr></thead><tbody>{current_reference_rows(current_comparison)}</tbody></table></div>
<div class="note">Sharpe는 월수익 평균 ÷ 월수익 표준편차 × √12이며 무위험수익률을 빼지 않습니다. MDD는 월말 누적 NAV 기준입니다. 일중 낙폭이나 세후 성과가 아닙니다.</div>

<h3>현재 비교 기준의 전체 파이프라인</h3>{code_block(pipeline_code.replace('SparseJump2 + transparent composite', 'transparent composite (SJM weight = 0)'))}
<div class="grid2"><div class="panel"><h3>느린 상태 추정</h3><p>월별 거시 12개로 성장·물가 확률을 만들고, 네 국면의 결합확률을 계산합니다. 이 확률은 SLSQP의 soft anchor에 연속값으로 들어갑니다.</p></div><div class="panel"><h3>빠른 위험 조절</h3><p>월별 로지스틱은 향후 두 달 꼬리손실 위험을 순위화하고, 일별 VKOSPI는 공포의 수준·충격·가속에 즉시 반응합니다. 세 시간축이 서로 다른 역할을 맡습니다.</p></div></div>

<h3>국면 확률은 이렇게 전환된다</h3><div class="equation">g* = mean(GDP, Export, BSI levels) + 0.20 × mean(their 3-month changes)<br>i* = mean(CPI, PPI, ImportPrice levels) + 0.20 × mean(their 3-month changes)<br>pg_raw = sigmoid(g* / 0.55), &nbsp; pi_raw = sigmoid(i* / 0.55)<br>pg_t = 0.85 × pg_raw,t + 0.15 × pg_(t−1)<br>pi_t = 0.85 × pi_raw,t + 0.15 × pi_(t−1)</div>
<p>현재 무SJM 경로에서는 SJM 확률이 식에 들어가지 않습니다. 15% 전월값은 한 달짜리 잡음을 완화하지만 상태를 강하게 고정하는 Markov 전이확률은 아닙니다. 네 국면확률은 <code>pg×(1−pi)</code>, <code>pg×pi</code>, <code>(1−pg)×(1−pi)</code>, <code>(1−pg)×pi</code>로 만들며 합은 항상 1입니다. 최댓값만 hard 국면 이름이 되고, 네 확률 전체는 soft anchor에 그대로 남습니다.</p>
<div class="table"><table><thead><tr><th>목표월</th><th>신호월</th><th>성장↑</th><th>물가↑</th><th>Goldilocks</th><th>Overheating</th><th>Slowdown</th><th>Stagflation</th><th>hard 국면</th></tr></thead><tbody>{recent_regime_rows(current_signals)}</tbody></table></div>
<div class="note">232개월 중 hard 국면 이름은 {regime_transitions}번 바뀌었습니다. 이름이 바뀌어도 포트폴리오 전체가 한 번에 갈아타는 것은 아닙니다. hard 경로는 기준점의 40%뿐이고, 나머지 60%는 확률가중 anchor를 추적하는 SLSQP 경로입니다. 이후 꼬리위험 이동·변동성 목표·20% 월중 무거래 밴드가 추가로 변화를 제한합니다.</div>

<h3>단기 예측은 사실 두 종류다</h3><div class="table"><table><thead><tr><th>층</th><th>시간축</th><th>예측/규칙</th><th>포트폴리오 반영</th></tr></thead><tbody>
<tr><td><b>중기 꼬리분류</b></td><td>월별 · 향후 2개월</td><td>균형 L2 로지스틱이 기준 포트폴리오의 2개월 누적경로 중 −5% 미만 도달 여부를 분류</td><td>과거 60개월 내 확률 상위 20%부터 위험자산을 최대 20% 방어자산으로 이동</td></tr>
<tr><td><b>일별 VKOSPI</b></td><td>일별 · t→t+1</td><td>학습모형이 아니라 252일 수준, 5일 shock, 5일 acceleration의 결정식</td><td>다음 거래일 오픈부터 KODEX200·USO의 최대 35%를 GLD로 이동</td></tr>
</tbody></table></div>
<h3>월별 꼬리분류기의 실제 진단</h3><div class="table"><table><thead><tr><th>구간</th><th>관측</th><th>사건</th><th>AUC</th><th>Average precision</th><th>Brier</th></tr></thead><tbody>{current_prediction_rows(current_validation)}</tbody></table></div>
<div class="warning"><b>raw 확률을 문자 그대로 읽지 마십시오.</b> 클래스 균형가중은 희귀사건을 놓치지 않기 위한 장치라 확률 보정을 희생할 수 있습니다. 사전 AUC는 낮고 사건도 5개뿐입니다. 전략은 <code>p_tail_raw=70%</code>를 실제 발생확률 70%로 쓰지 않고, 과거 60개월 내 인과적 순위로 바꿔 상위 20% 여부만 사용합니다. 잠금 AUC가 높다고 모델이 일반화됐다고 단정할 수도 없습니다.</div>
<div class="table"><table><thead><tr><th>월</th><th>raw 분류확률</th><th>과거순위</th><th>위험강도</th><th>p_up 제어값</th><th>사후 라벨</th></tr></thead><tbody>{recent_tail_rows(current_factor)}</tbody></table></div>
<div class="note"><code>p_up</code>은 다음 달 상승확률을 새로 추정한 값이 아니라 기존 비중이동 함수에 위험강도를 전달하기 위한 제어값입니다. 위험이 없으면 0.50, 위험강도 1이면 0.35가 되어 <code>score=(p_up−0.5)/0.15</code>가 0에서 −1로 움직입니다.</div>

<h3>핵심 하이퍼파라미터와 결정 근거</h3><div class="table"><table><thead><tr><th>계층</th><th>설정</th><th>값</th><th>결정 근거·증거 수준</th></tr></thead><tbody>
<tr><td>거시</td><td>최소 이력 / d3 / sigmoid / 평활</td><td>24개월 / 0.20 / 0.55 / 현재 0.85</td><td>작은 표본에서 수준을 주축으로 전환정보를 약하게 섞는 고정 설계. 한 축 민감도는 검사했지만 공동 최적화값은 아님.</td></tr>
<tr><td>거시</td><td>SJM 기여</td><td>현재 0%, 사전선정 10%</td><td>무SJM은 Brier가 좋아진 ablation이지만 사전 포트폴리오 관문 실패. 두 경로를 구분해 보고.</td></tr>
<tr><td>SLSQP</td><td>EWMA / 목표변동성 / CDaR</td><td>반감기 12개월 / 8%×국면조정 / 최대 16%</td><td>기준 전략에서 상속한 방어 최적화 설정. Robust VKOSPI 810개 탐색 대상이 아님.</td></tr>
<tr><td>기준 혼합</td><td>hard / SLSQP</td><td>40% / 60%</td><td>hard 국면의 방향성과 연속형 위험제어를 동시에 남긴 고정 구조. 별도 최신 최적화값이 아님.</td></tr>
<tr><td>꼬리 로짓</td><td>모형</td><td>L2, liblinear, C=.1, class_weight=balanced</td><td>후속 28개 강건성 감사에서 사전 포트폴리오 관문을 통과한 유일 후보였지만, 성과선택 PBO가 높아 ‘유일한 최적값’ 증거는 없음.</td></tr>
<tr><td>꼬리 로짓</td><td>학습 경계</td><td>36개월, 2개월 embargo, 양성 4·음성 12</td><td>2개월 미래 라벨의 누수 차단과 희귀사건 적합 가능성 확보를 위한 안전장치.</td></tr>
<tr><td>꼬리 이동</td><td>순위 / 최대 이동 / 목표변동성</td><td>60개월, 상위 20%, 20%, 15%</td><td>절대 확률 보정에 덜 의존하도록 인과 순위 사용. OAP 중기 경로에서 고정된 설정.</td></tr>
<tr><td>레버리지</td><td>범위 / 초기 / 금융비용</td><td>0.50–1.50 / 1.20 / 연 4%</td><td>24개월 가중 변동성으로 15%를 맞추되 폭주를 제한. 차입을 무비용으로 보지 않음.</td></tr>
<tr><td>VKOSPI</td><td>모드 / 수준 / shock</td><td>acceleration / 90% / 1σ</td><td>5×3×3×3×2×3=810개를 2017년 이전 두 창에서 평가해 엄격 관문 2개 중 선택.</td></tr>
<tr><td>VKOSPI</td><td>최대 이전 / 수령자 / 밴드</td><td>35% / GLD / 20%</td><td>사전 두 창의 세 지표 동시 개선. 잠금 귀속에서는 밴드 20%가 성과 개선의 핵심 실행조건으로 확인.</td></tr>
</tbody></table></div>
<div class="finding"><b>한 문장 요약:</b> 거시 확률은 ‘어느 방향의 자산을 보유할지’, 로지스틱은 ‘앞으로 두 달 위험예산을 얼마나 줄일지’, VKOSPI는 ‘오늘 공포가 빨라졌을 때 다음 거래일부터 얼마나 방어할지’를 결정합니다. 세 층을 하나의 예측모형으로 부르면 작동방식을 잘못 설명하는 것입니다.</div>
</section>
"""
    model_robustness_section = r"""
<section id="model-robustness"><div class="head"><small>15 · MODEL ROBUSTNESS</small><div><h2>SJM과 로지스틱 설정을 바꿔도 결론이 남는가</h2><p class="lede">배포값 주변만 조금 흔든 것이 아니라 규제 종류·강도·해법·클래스 가중과 SJM의 점프 벌점·희소 변수 수·혼합비를 함께 바꿨습니다. 결론부터 말하면 예측 순위에는 일부 안정성이 있지만, 포트폴리오 성과로 하이퍼파라미터를 고르는 과정은 강건하다고 보기 어렵습니다.</p></div></div>

<div class="warning"><b>이 검증은 “오버피팅이 없다”는 증명에 실패했습니다.</b> 로지스틱 포트폴리오 Sharpe의 사전 CSCV/PBO는 __LOGIT_PORT_PBO__였고, 수렴 경고 후보를 빼도 __LOGIT_PORT_PBO_STABLE__였습니다. SJM도 거시 Brier 선택 PBO __SJM_PRED_PBO__, soft 포트폴리오 Sharpe 선택 PBO __SJM_SOFT_PBO__로 높았습니다. 잠금 결과를 보고 더 좋아 보이는 설정으로 갈아타면 이 감사 자체가 새 튜닝이 되므로, 현재 배포값은 바꾸지 않았습니다.</div>

<h3>무엇을 고정하고 무엇을 바꿨나</h3><div class="grid2"><div class="panel"><h3>로지스틱 · 28개</h3><ul><li><code>C=0.01, .03, .1, .3, 1, 3</code></li><li>균형 L2·L1 <code>liblinear</code>, 균형 L2 <code>lbfgs</code>, 무가중 L2 <code>liblinear</code></li><li>균형 ElasticNet <code>saga</code>, <code>C=.03,.1,.3,1</code>, <code>l1_ratio=.5</code></li><li>입력 16개, 최소 36개월, 양성 4·음성 12, 2개월 embargo는 동일</li></ul></div><div class="panel"><h3>SJM · 49개 시도</h3><ul><li>점프 벌점 <code>0, 1.5, 3, 6</code></li><li>매 시점 유지 변수 <code>2, 4, 6</code></li><li>SJM 혼합비 <code>5%, 10%, 20%, 30%</code>와 SJM 미사용 비교선</li><li>49개 중 37개가 유효했고 12개는 비유한 상태확률로 계산이 중단됨</li></ul></div></div>
<p>로지스틱 후보에는 모두 같은 <code>max_shift=.20</code>·<code>target_vol=.15</code>를 적용했습니다. SJM 후보도 동일한 soft 배분을 사용했고, SLSQP proposed 경로는 배포값에서 한 축만 바뀐 9개에 한해 추가 계산했습니다. 마지막 일별 VKOSPI 오버레이는 이번 감사의 바깥에 고정했습니다. 따라서 아래 결과는 <b>중기 꼬리분류 계층과 거시 국면 계층</b>의 강건성이지, 최종 810개 VKOSPI 정책을 다시 고른 결과가 아닙니다.</p>

<h3>시간 경계와 진단법</h3><div class="table"><table><thead><tr><th>진단</th><th>계산 방식</th><th>읽는 법</th></tr></thead><tbody>
<tr><td>두 사전 창 엄격 관문</td><td>2007–2017과 그 안의 2013–2017에서 예측·Sharpe·MDD가 배포값 이상인지 동시 확인</td><td>로지스틱은 배포값 1개뿐, SJM은 5개가 통과. 두 창은 겹치므로 독립 검증은 아님</td></tr>
<tr><td>잠금 순위</td><td>2018-01 이후는 후보 선정이 아니라 사전순위 유지 여부만 감사</td><td>잠금 1위를 새 배포값으로 채택하지 않음</td></tr>
<tr><td>짝지은 블록 부트스트랩</td><td>잠금 월수익과 손실을 같은 6개월 블록으로 2,000회 재표본</td><td>후보와 배포값의 차이가 표본 경로에 얼마나 민감한지 측정</td></tr>
<tr><td>CSCV/PBO</td><td>2017년 이전을 8개 연속 블록으로 나누고 4개 선택·4개 평가의 70가지 분할을 모두 계산</td><td>사전 승자가 반대편에서 후보 중앙값 아래로 떨어진 비율. 높을수록 선택 과적합 위험이 큼</td></tr>
<tr><td>순위 상관</td><td>사전 종합순위와 잠금 종합순위의 Spearman 상관</td><td>1은 순위 유지, 0은 무관, 음수는 역전 경향</td></tr>
</tbody></table></div>

<h3>로지스틱: 예측값과 포트폴리오값을 따로 봐야 한다</h3><div class="table"><table><thead><tr><th>후보</th><th>사전 AUC / Brier / Sharpe</th><th>잠금 AUC / Brier / Sharpe</th><th>잠금 예측 / 성과 순위</th><th>수렴 경고</th></tr></thead><tbody>__LOGISTIC_ROBUSTNESS_ROWS__</tbody></table></div>
<div class="grid2"><div class="panel"><h3>상대 예측 선택은 비교적 안정</h3><ul><li>사전·잠금 예측 점수 순위상관은 <b>__LOGIT_PRED_RANK_CORR__</b>.</li><li>Brier 기준 CSCV/PBO는 <b>__LOGIT_PRED_PBO__</b>.</li><li>사전 예측 1위인 무가중 L2 <code>C=.1</code>은 잠금 예측 4위.</li><li>다만 이 PBO는 후보끼리의 상대순위일 뿐, 배포 확률의 절대 Brier가 좋다는 뜻은 아님.</li></ul></div><div class="panel"><h3>성과 선택은 불안정</h3><ul><li>배포값의 포트폴리오 순위는 사전 1위·잠금 1위였지만, 후보 전체 순위상관은 <b>__LOGIT_PORT_RANK_CORR__</b>에 불과.</li><li>CSCV/PBO는 <b>__LOGIT_PORT_PBO__</b>, 경고 후보 제외 후에도 <b>__LOGIT_PORT_PBO_STABLE__</b>.</li><li>L1 <code>C=.03</code>은 사전 성과 3위에서 잠금 24위로 밀려 순위 역전을 보임.</li><li>“배포값이 잠금에서도 1위”와 “사전 선택 규칙이 안정적”은 같은 말이 아님.</li></ul></div></div>
<div class="warning"><b>수치 안정성 경고도 별도입니다.</b> ElasticNet <code>saga</code> 3개 후보에서 모두 __LOGIT_WARNING_TOTAL__회의 수렴 경고가 났습니다. 또한 L1 <code>C=.01,.03</code>은 중앙 적합에서 16개 계수가 모두 0이었습니다. 강한 희소화가 작은 사건 표본에서 실질적으로 상수모형을 만들 수 있다는 뜻입니다.</div>

<h3>SJM: 상태 예측 순위는 남지만, 포트폴리오 증분효과는 약하다</h3><div class="table"><table><thead><tr><th>후보</th><th>사전 Brier / soft Sharpe</th><th>잠금 Brier / soft Sharpe</th><th>잠금 proposed Sharpe</th><th>잠금 예측 / soft 순위</th></tr></thead><tbody>__SJM_ROBUSTNESS_ROWS__</tbody></table></div>
<div class="grid2"><div class="panel"><h3>상태확률</h3><ul><li>사전·잠금 예측순위 상관은 <b>__SJM_PRED_RANK_CORR__</b>으로 높았음.</li><li>그런데 거시 Brier CSCV/PBO는 <b>__SJM_PRED_PBO__</b>. 분할별 승자가 자주 바뀌었음.</li><li>SJM 미사용은 잠금 Brier가 배포값보다 0.0051 낮았고, 2,000회 부트스트랩에서 더 나을 확률은 <b>__NOSJM_BOOT_BRIER__</b>.</li></ul></div><div class="panel"><h3>포트폴리오</h3><ul><li>soft 성과순위 상관은 <b>__SJM_SOFT_RANK_CORR__</b>으로 사실상 역전.</li><li>soft Sharpe CSCV/PBO는 <b>__SJM_SOFT_PBO__</b>.</li><li>SJM 미사용의 잠금 soft Sharpe 우위 확률은 <b>__NOSJM_BOOT_SHARPE__</b>로 동전 던지기에 가까움.</li><li>따라서 SJM 10% 혼합이 포트폴리오 성과를 일관되게 높였다는 증거는 없음.</li></ul></div></div>
<div class="warning"><b>점프 벌점 6은 안정성 경계를 넘었습니다.</b> 변수 수 2·4·6과 혼합비 4개를 조합한 12개 전부에서 SJM 상태 하나가 비거나 퇴화해 비유한 확률이 생겼습니다. 첫 실패는 변수 수에 따라 2007-05·06·09였습니다. 이 후보들을 0.5로 대체해 성과표에 억지로 남기지 않고 “계산 불능”으로 기록했습니다.</div>

<h3>잠금 전반과 후반으로 다시 쪼갠 결과</h3><p>아래의 손실은 로지스틱은 Brier, SJM은 성장·물가 평균 Brier입니다. 두 값 모두 낮을수록 좋고 Sharpe는 높을수록 좋습니다. 후반 결과는 새 선택 기준이 아니라 시간대별 약점 확인용입니다.</p><div class="table"><table><thead><tr><th>비교</th><th>2018–2021 손실 / Sharpe</th><th>2022–2026 손실 / Sharpe</th></tr></thead><tbody>__MODEL_SUBPERIOD_ROWS__</tbody></table></div>

<h3>이번 감사에서 내릴 수 있는 결론</h3><div class="grid2"><div class="panel"><h3>확인된 것</h3><ul><li>배포 경로 재계산 오차는 확률 __LOGIT_REPRO_PROB__ 이하, 월수익 __LOGIT_REPRO_RETURN__ 이하, SJM 국면확률 __SJM_REPRO_PROB__ 이하로 원본과 일치.</li><li>solver만 바꾼 L2 <code>C=.1</code> 이웃은 성과가 가까워 국소적인 구현 강건성은 있음.</li><li>무가중 로지스틱은 균형가중보다 확률보정이 낫고, SJM 미사용도 상태확률 Brier가 더 낮음.</li></ul></div><div class="panel"><h3>확인되지 않은 것</h3><ul><li>Sharpe가 좋은 하이퍼파라미터를 사전표본에서 안정적으로 고를 수 있다는 증거.</li><li>SJM 10%가 단순 합성확률보다 꾸준히 나은 증분효과.</li><li>잠금에서 좋아 보인 무가중 <code>C=.3</code> 또는 SJM 미사용을 지금 채택해도 된다는 근거.</li></ul></div></div>
<div class="finding"><b>운용 판단:</b> 현 배포값은 그대로 둡니다. 다음 변경은 ① 무가중 로지스틱·확률보정 여부, ② SJM 미사용 비교선, ③ 선택 목적함수를 Brier와 포트폴리오 성과 중 무엇으로 둘지 먼저 문서화한 뒤, 2026-08 이후 새 자료 또는 별도의 외부시장에 사전등록 방식으로 검증하는 편이 맞습니다. 잠금 2018–2026을 다시 학습·선택에 넣으면 이번 검증의 의미가 사라집니다.</div>
<details><summary>재현 파일과 전체 후보표</summary><ul><li><code>vkospi_model_robustness.py</code> — 실험·부트스트랩·CSCV/PBO·재현 단언</li><li><code>results/vkospi_logistic_hyperparameter_robustness.csv</code> — 28개 × 5기간 원표</li><li><code>results/vkospi_logistic_candidate_summary.csv</code> — 후보별 순위·수렴·부트스트랩</li><li><code>results/vkospi_sjm_robustness.csv</code> — 49개 시도 × 5기간, 무효 사유 포함</li><li><code>results/vkospi_sjm_candidate_summary.csv</code> — 37개 유효 후보와 12개 무효 후보 요약</li><li><code>results/vkospi_model_robustness.json</code> — 경계·PBO·순위·재현 오차 종합</li></ul></details>
</section>
"""
    non_vkospi_section = r"""
<section id="non-vkospi"><div class="head"><small>04 · NON-VKOSPI</small><div><h2>VKOSPI 밖에서 들어오는 변수와 분류 방식</h2><p class="lede">최종 전략에서 VKOSPI는 마지막 월중 방어막입니다. 그 전에 거시 지표로 월별 국면을 정하고, 시장 가격·거래량으로 앞으로 두 달의 꼬리손실 위험을 가늠해 기준 비중을 만듭니다.</p></div></div>
<div class="warning"><b>2018년 이후 잠금 구간이 가른 운영 경로</b> VKOSPI 머신러닝 보정 후보는 승격 기준을 넘지 못했습니다. <code>results/vkospi_validation.json</code>에 기록된 실제 기준 경로는 <code>ReferenceMediumHorizonOAPVol15</code>입니다. 아래에는 탈락한 후보가 아니라, 이 경로가 실제로 읽는 비VKOSPI 입력만 적었습니다.</div>
<h3>분류·점수·최적화는 서로 다른 계산이다</h3><div class="table"><table><thead><tr><th>계산</th><th>입력 수</th><th>알고리즘</th><th>출력</th></tr></thead><tbody>
<tr><td><b>거시 국면 분류</b></td><td>12개</td><td><code>SparseJump2</code>와 투명한 로짓 합성</td><td>성장·물가 확률과 4개 국면</td></tr>
<tr><td><b>중기 꼬리손실 분류</b></td><td>16개</td><td>중앙값 대치·표준화·균형 로지스틱 회귀</td><td>향후 2개월 경로손실이 −5% 미만일 확률</td></tr>
<tr><td><b>VKOSPI 일별 오버레이</b></td><td>3개, 백분위 대체값 1개</td><td>문턱값을 이용한 연속형 결정식</td><td>0~1 스트레스와 최대 35% 위험 이전</td></tr>
</tbody></table></div>
<div class="note">분류기는 앞의 두 줄뿐입니다. VKOSPI 스트레스는 학습 모델이 아니라 정해진 산식이며, SLSQP는 분류가 끝난 뒤 자산 비중을 푸는 최적화입니다. 이 운영 경로에는 LightGBM도 HMM도 들어가지 않습니다.</div>

<h3>거시 국면 분류에 넣은 12개 변수</h3><p>월별 거시 계열 6개에서 현재 수준과 3개월 변화를 하나씩 뽑았습니다. z-score는 그달까지 들어온 자료만으로 계산합니다.</p>
<div class="table"><table><thead><tr><th>원자료</th><th>수준 변수</th><th>전환 변수</th><th>전처리·역할</th></tr></thead><tbody>
<tr><td>GDP 전년동기비</td><td><code>GDP_level</code></td><td><code>GDP_level_d3</code></td><td>후행 72개월 z-score와 3개월 변화 · 성장</td></tr>
<tr><td>수출금액 전년동기비</td><td><code>Export_level</code></td><td><code>Export_level_d3</code></td><td>후행 36개월 z-score와 3개월 변화 · 성장</td></tr>
<tr><td>제조업 업황전망 BSI</td><td><code>BSI_level</code></td><td><code>BSI_level_d3</code></td><td>후행 24개월 z-score와 3개월 변화 · 성장</td></tr>
<tr><td>소비자물가 전년동기비</td><td><code>CPI_level</code></td><td><code>CPI_level_d3</code></td><td>후행 36개월 z-score와 3개월 변화 · 물가</td></tr>
<tr><td>생산자물가 전년동기비</td><td><code>PPI_level</code></td><td><code>PPI_level_d3</code></td><td>후행 36개월 z-score와 3개월 변화 · 물가</td></tr>
<tr><td>수입물가 전년동기비</td><td><code>ImportPrice_level</code></td><td><code>ImportPrice_level_d3</code></td><td>후행 36개월 z-score와 3개월 변화 · 물가</td></tr>
</tbody></table></div>
<details><summary>거시 원자료 파일</summary><div class="table"><table><thead><tr><th>파일</th><th>읽는 계열</th></tr></thead><tbody>
<tr><td><code>raw_data/GDP 성장률.xlsx</code></td><td>GDP YoY</td></tr>
<tr><td><code>raw_data/수출입 총괄_20260816.xlsx</code></td><td>수출 금액으로 계산한 <code>Export_YoY</code></td></tr>
<tr><td><code>raw_data/기업경기조사(전망).csv</code></td><td>제조업 업황전망BSI</td></tr>
<tr><td><code>raw_data/소비자물가 상승률.xlsx</code></td><td>CPI YoY</td></tr>
<tr><td><code>raw_data/생산자물가 상승률.xlsx</code></td><td>PPI YoY</td></tr>
<tr><td><code>raw_data/수출입물가 상승률.xlsx</code></td><td>Import Price YoY</td></tr>
</tbody></table></div></details>

<h3>거시 국면을 가르는 법</h3><div class="grid2"><div class="panel"><h3><code>SparseJump2</code></h3><p>성장 6개 열과 물가 6개 열은 따로 돌립니다. 먼저 중앙값과 IQR로 스케일을 맞추고 ±5를 벗어난 값은 잘라냅니다. 두 상태의 중심과 변수 가중치를 번갈아 추정하면서 상태 차이가 큰 변수 4개만 남깁니다. 마지막에는 동적계획법에 전환 벌점 3.0을 걸어, 한두 달 잡음만으로 국면이 뒤집히는 일을 줄였습니다.</p></div><div class="panel"><h3>확률 합성과 4개 국면</h3><div class="equation">p_composite = sigmoid((level_mean + 0.20 × d3_mean) / 0.55)<br>p_raw = 0.10 × p_SparseJump2 + 0.90 × p_composite<br>p_t = 0.85 × p_raw + 0.15 × p_{t−1}</div><p>성장 확률 <code>pg</code>와 물가 확률 <code>pi</code>를 교차하면 Goldilocks, Overheating, Slowdown, Stagflation의 네 확률이 나옵니다. 그중 가장 큰 확률을 그달 국면으로 삼습니다.</p></div></div>
<div class="table"><table><thead><tr><th>국면</th><th>확률식</th><th>KODEX200 / BOND / GLD / USO 기준 앵커</th></tr></thead><tbody>
<tr><td>Goldilocks</td><td><code>pg × (1 − pi)</code></td><td>58% / 22% / 15% / 5%</td></tr>
<tr><td>Overheating</td><td><code>pg × pi</code></td><td>30% / 12% / 23% / 35%</td></tr>
<tr><td>Slowdown</td><td><code>(1 − pg) × (1 − pi)</code></td><td>12% / 66% / 20% / 2%</td></tr>
<tr><td>Stagflation</td><td><code>(1 − pg) × pi</code></td><td>8% / 24% / 50% / 18%</td></tr>
</tbody></table></div>
<div class="note">위 표는 SLSQP 방어 경로가 확률가중해 쓰는 <code>soft anchor</code>입니다. 별도의 <code>hard regime</code> 경로는 Goldilocks에서 KODEX200 100%, Overheating에서 USO 100%, Slowdown에서 KODEX200 60%·BOND 40%, Stagflation에서 GLD 100%로 둡니다. 최종 월별 기준점은 이 hard 경로 40%와 SLSQP 방어 경로 60%를 섞어 만듭니다.</div>

<h3>중기 꼬리손실 분류기의 국내 변수 12개</h3><p>기준 포트폴리오의 과거 경로는 그달 거시 국면 하나를 고른 <code>hard anchor</code>로 계산합니다. 자료는 목표 월이 시작되기 이틀 전에서 끊습니다. 미국 종가가 한국의 첫 거래보다 뒤늦게 확정되는 시차까지 감안한 장치입니다.</p>
<div class="table"><table><thead><tr><th>변수명</th><th>정의</th><th>읽는 위험</th></tr></thead><tbody>
<tr><td><code>base_USO</code></td><td>hard regime의 USO 기준 비중</td><td>인플레이션·원자재 국면 노출</td></tr>
<tr><td><code>base_GLD</code></td><td>hard regime의 GLD 기준 비중</td><td>스태그플레이션·방어 노출</td></tr>
<tr><td><code>base_KODEX200</code></td><td>hard regime의 KODEX200 기준 비중</td><td>국내 주식 위험 노출</td></tr>
<tr><td><code>p_inflation_high</code></td><td>거시 분류기가 계산한 고물가 확률</td><td>물가 국면의 연속값</td></tr>
<tr><td><code>proxy_mom1</code></td><td>hard-anchor 포트폴리오의 과거 1개월 복리수익</td><td>단기 모멘텀</td></tr>
<tr><td><code>proxy_mom6</code></td><td>같은 포트폴리오의 과거 6개월 복리수익</td><td>중기 모멘텀</td></tr>
<tr><td><code>proxy_vol6</code></td><td>과거 6개월 월수익 표준편차 × √12</td><td>월간 변동성</td></tr>
<tr><td><code>daily_mom21</code></td><td>일별 hard-anchor 수익의 21거래일 복리수익</td><td>약 1개월 가격 경로</td></tr>
<tr><td><code>daily_mom252</code></td><td>일별 hard-anchor 수익의 252거래일 복리수익</td><td>약 1년 가격 경로</td></tr>
<tr><td><code>daily_vol21</code></td><td>21거래일 표준편차 × √252</td><td>단기 실현변동성</td></tr>
<tr><td><code>daily_downvol21</code></td><td>21거래일 음수 수익의 제곱평균제곱근 × √252</td><td>하방 변동성</td></tr>
<tr><td><code>daily_mean_corr63</code></td><td>KODEX200·GLD·USO의 63거래일 쌍별 상관계수 평균</td><td>분산효과 약화</td></tr>
</tbody></table></div>
<div class="note"><code>base_BOND</code>도 기준 비중 계산 과정에서는 존재하지만 분류기의 열에는 넣지 않습니다. 네 자산의 비중 합이 1이므로 나머지 세 비중으로 값이 결정돼 중복 정보를 피한 것입니다.</div>

<h3>Open Asset Pricing에서 가져온 4개 합성변수</h3><p><a href="https://openassetpricing.com/SignalDoc-Browser.html">Open Asset Pricing Signal Browser</a>에서 모멘텀, 반전, 저위험·꼬리, 유동성 아이디어를 빌렸습니다. 다만 원래의 개별 종목 횡단면 신호를 그대로 옮기지는 않았습니다. KOSPI200과 16개 KRX 업종지수의 가격·거래량으로 한국 시장용 시계열 유사 변수를 만들고, 각 시점까지의 자료로 구한 z-score를 묶었습니다.</p>
<div class="table"><table><thead><tr><th>분류기 입력</th><th>안에 들어간 원시 변수</th><th>합성 방향</th></tr></thead><tbody>
<tr><td><code>oap_momentum_trend_stress</code></td><td><code>oap_sector_mom12_median</code><br><code>oap_sector_mom6_median</code><br><code>oap_sector_intmom_median</code><br><code>oap_sector_high52_median</code><br><code>oap_sector_mom6_breadth</code><br><code>oap_sector_mom6_dolvol_weighted</code></td><td>12개월·6개월·중간 모멘텀, 52주 고점 근접도, 상승 업종 비율, 거래대금 가중 모멘텀의 z-score에 음수를 붙여 평균. 추세가 약할수록 스트레스가 커짐.</td></tr>
<tr><td><code>oap_reversal_crowding_stress</code></td><td><code>oap_sector_streversal_median</code><br><code>oap_kospi_streversal</code><br><code>oap_sector_streversal_dispersion</code><br><code>oap_sector_mom12_dispersion</code></td><td>업종·KOSPI200의 21일 수익은 음의 방향, 단기수익과 12개월 모멘텀의 업종 간 분산은 양의 방향으로 합성.</td></tr>
<tr><td><code>oap_low_risk_tail_stress</code></td><td><code>oap_sector_realizedvol21_median</code><br><code>oap_sector_idiovol252_median</code><br><code>oap_sector_maxret21_median</code><br><code>oap_sector_returnskew21_median</code><br><code>oap_sector_beta252_mean</code><br><code>oap_sector_beta252_dispersion</code></td><td>21일 실현변동성·최대 일수익, 252일 CAPM 잔차변동성·베타 수준·베타 분산은 양의 방향, 수익 왜도는 음의 방향으로 평균.</td></tr>
<tr><td><code>oap_liquidity_activity_stress</code></td><td><code>oap_sector_log_illiquidity63_median</code><br><code>oap_sector_illiquidity_ratio21_252_median</code><br><code>oap_kospi_illiquidity_ratio21_252</code><br><code>oap_sector_dolvol_z63_median</code><br><code>oap_sector_volume_trend60_median</code><br><code>oap_sector_volume_cv36_median</code></td><td>Amihud 비유동성과 단기/장기 비유동성 비율은 양의 방향, 거래대금 z-score와 60개월 거래량 추세는 음의 방향, 36개월 거래량 변동계수는 양의 방향. 6개 중 최소 4개가 있어야 계산.</td></tr>
</tbody></table></div>
<details><summary>OAP 원시 성분은 이렇게 계산했다</summary><ul>
<li><code>mom12</code>는 21~252거래일, <code>mom6</code>는 21~126거래일, <code>intmom</code>은 126~252거래일 수익입니다.</li>
<li><code>high52</code>는 현재 지수 ÷ 252거래일 고가, <code>breadth</code>는 6개월 모멘텀이 양수인 업종 비율입니다.</li>
<li><code>streversal</code>은 21거래일 수익, <code>dispersion</code>은 같은 시점 업종 횡단면의 표준편차입니다.</li>
<li><code>idiovol252</code>는 KOSPI200을 시장으로 둔 252거래일 CAPM 잔차의 연율화 표준편차입니다.</li>
<li><code>log_illiquidity63</code>는 |수익률| ÷ 거래대금의 63일 평균을 로그로 바꾼 값이며, <code>ratio21_252</code>는 21일 평균 ÷ 252일 평균입니다.</li>
</ul></details>
<details><summary>계산했지만 최종 기준 분류기에 넣지 않은 OAP 열</summary><p><code>oap_sector_mom6_dolvol_corr</code>, <code>oap_kospi_mom12</code>, <code>oap_kospi_high52</code>, <code>oap_sector_mrreversal_median</code>, <code>oap_sector_lrreversal_median</code>, <code>oap_kospi_volume_trend60</code>, <code>oap_kospi_volume_cv36</code>은 연구용 원시 테이블에는 생성됩니다. 그러나 선정된 4개 합성변수의 구성에는 포함되지 않아 ‘사용한 입력변수’ 수에는 세지 않았습니다.</p></details>
<details><summary>OAP 계산에 사용한 KRX 지수 17개</summary><p><code>1028</code> KOSPI200과 업종지수 <code>5043</code> 자동차, <code>5044</code> 반도체, <code>5045</code> 헬스케어, <code>5046</code> 은행, <code>5048</code> 에너지·화학, <code>5049</code> 철강, <code>5052</code> 건설, <code>5054</code> 증권, <code>5055</code> 기계, <code>5056</code> 보험, <code>5057</code> 운송, <code>5061</code> 경기소비재, <code>5062</code> 필수소비재, <code>5063</code> 미디어·엔터테인먼트, <code>5064</code> 정보기술, <code>5065</code> 유틸리티를 사용합니다. 가격과 거래량은 <code>raw_data/compass.db</code>에서 읽습니다.</p></details>

<h3>두 달 안에 −5%가 닿을지를 매달 다시 묻는다</h3><div class="grid2"><div class="panel"><h3>학습 목표와 전처리</h3><ul><li>목표값: 기준 포트폴리오의 향후 2개월 누적 경로 중 최저값이 <b>−5% 미만</b>이면 1, 아니면 0.</li><li>입력: 위 국내 변수 12개와 OAP 합성변수 4개, 합계 16개.</li><li>결측치는 학습표본 중앙값으로 대치하고, <code>StandardScaler</code>로 표준화.</li><li><code>LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear")</code>. 기본 L2 규제를 사용.</li></ul></div><div class="panel"><h3>시계열 누수 차단</h3><ul><li>최소 36개월이 쌓인 뒤 expanding walk-forward로 매달 다시 적합.</li><li>2개월 미래 경로를 쓰는 라벨이므로 예측 직전 2개월을 학습에서 제외.</li><li>양성 4건, 음성 12건 이상일 때만 적합.</li><li>특징·목표·최대 비중 이동 20%는 2017년 이전 자료로 선정하고 2018년 이후는 잠금.</li></ul></div></div>
<div class="equation">p_tail → causal_percentile<br>severity = clip((risk_percentile − 0.80) / 0.20, 0, 1)<br>p_up = 0.50 − 0.15 × severity</div>
<p>원시 확률 자체보다 과거 예측분포에서의 위치를 봅니다. 위험도가 과거 예측의 상위 20%에 들 때부터 방어하고, 가장 높아지면 <code>p_up</code>을 0.35까지 낮춥니다.</p>

<h3>분류 결과가 실제 비중이 되기까지</h3><div class="flow"><article><h3>거시 분류</h3><p>12개 변수로 성장·물가 확률과 4개 국면을 계산.</p></article><article><h3>기준 비중</h3><p>hard 국면 경로 40%와 SLSQP 방어 경로 60%를 혼합.</p></article><article><h3>꼬리 분류</h3><p>16개 변수로 2개월 −5% 경로손실 확률을 예측.</p></article><article><h3>위험 이동</h3><p>위험 점수만큼 주식·원유에서 채권·금으로 최대 20% 이동.</p></article><article><h3>변동성 조절</h3><p>최근 24개월 지수가중 변동성으로 연 15%를 목표.</p></article><article><h3>일별 오버레이</h3><p>VKOSPI 스트레스로 월중 주식·원유 일부를 GLD로 추가 이전.</p></article></div>
<div class="grid2" style="margin-top:18px"><div class="panel"><h3>SLSQP에 들어가는 비분류 입력</h3><p>거시 국면 확률, KODEX200·BOND·GLD·USO 과거 월수익, 84개월 EWMA 공분산, 장기·최근 기대수익, CDaR, 직전 비중, 현재 드로다운을 사용합니다. 수익·변동성·꼬리손실·회전율·기준비중 이탈을 함께 평가하며 합계 100%, 자산별 상하한, 목표 변동성, 최대 CDaR 제약을 둡니다.</p></div><div class="panel"><h3>자산 수익과 체결 입력</h3><p><code>cache/market_daily.csv</code>의 KODEX200·GLD·USO 시가와 USDKRW 종가, <code>raw_data/krx_bond_index.csv</code>의 채권지수, 2009년 4월 이전 KODEX200 연결용 <code>compass.db</code> symbol <code>1028</code>을 읽습니다. GLD·USO는 USDKRW를 곱해 원화 수익으로 바꾸며 연 4% 금융비용과 거래비용을 차감합니다.</p></div></div>
</section>
"""
    evidence_appendix = r"""
<section id="evidence"><div class="head"><small>13 · NUMBER LEDGER</small><div><h2>보고서의 숫자를 원자료까지 거슬러 올라가기</h2><p class="lede">같은 전략을 말하더라도 월간 기준 경로, 일별 재구성 경로, 잠금 부트스트랩은 서로 다른 표본과 계산 단위를 씁니다. 아래 표는 화면에 보이는 숫자가 어느 파일과 산식에서 나왔는지 한 줄씩 연결합니다.</p></div></div>
<div class="warning"><b>세 경로를 섞어 읽으면 숫자가 어긋납니다.</b> 결과 카드와 귀속표는 검증된 월간 기준에 일별 오버레이의 상대효과를 결합한 <code>reconciled</code> 경로입니다. 신호 일수와 비용 민감도는 실제 일별 재구성 경로입니다. 로지스틱 분류 성능은 포트폴리오 수익률과 별개의 예측 표본에서 계산합니다.</div>

<h3>기준 전략에서 최종 전략까지</h3><div class="table"><table><thead><tr><th>잠금 경로 · 2018-01~2026-07</th><th>CAGR</th><th>연율 변동성</th><th>Sharpe</th><th>MDD</th><th>Calmar</th><th>최종 배수</th></tr></thead><tbody>
<tr><td><b>월간 기준</b><br><code>ReferenceMediumHorizonOAPVol15</code></td><td>__REF_CAGR__</td><td>__REF_VOL__</td><td>__REF_SHARPE__</td><td>__REF_MDD__</td><td>__REF_CALMAR__</td><td>__REF_MULTIPLE__배</td></tr>
<tr><td><b>기존 VKOSPI 동적</b><br><code>vkospi_dynamic_reconciled_monthly.csv</code></td><td>__OLD_CAGR__</td><td>__OLD_VOL__</td><td>__OLD_SHARPE__</td><td>__OLD_MDD__</td><td>__OLD_CALMAR__</td><td>__OLD_MULTIPLE__배</td></tr>
<tr class="winner-row"><td><b>최종 robust 동적</b><br><code>vkospi_robust_dynamic_reconciled_monthly.csv</code></td><td>__NEW_CAGR__</td><td>__NEW_VOL__</td><td>__NEW_SHARPE__</td><td>__NEW_MDD__</td><td>__NEW_CALMAR__</td><td>__NEW_MULTIPLE__배</td></tr>
</tbody></table></div>
<p>월간 기준에 기존 VKOSPI 오버레이를 얹으면서 Sharpe가 __REF_SHARPE__에서 __OLD_SHARPE__로 높아지고 MDD는 __REF_MDD__에서 __OLD_MDD__로 줄었습니다. robust 재가공과 20% 밴드를 적용한 뒤에는 Sharpe __NEW_SHARPE__, MDD __NEW_MDD__가 됐습니다. 따라서 이 보고서가 말하는 개선분 <b>CAGR __D_CAGR__, Sharpe __D_SHARPE__, MDD __D_MDD__</b>는 월간 기준 대비가 아니라 기존 VKOSPI 동적 전략 대비입니다.</p>

<h3>성과지표를 그대로 다시 계산하면</h3><div class="equation">wealth_t = ∏(1 + r_m)<br>CAGR = wealth_T^(12 / Months) − 1<br>Volatility = sample_std(r_m, ddof=1) × √12<br>Sharpe = mean(r_m) / sample_std(r_m, ddof=1) × √12<br>Downside = √mean(min(r_m, 0)²) × √12<br>Sortino = mean(r_m) × 12 / Downside<br>MDD = min(wealth_t / cumulative_max(wealth_t) − 1)<br>Calmar = CAGR / |MDD|</div>
<div class="table"><table><thead><tr><th>경로</th><th>개월 / 연수</th><th>월평균</th><th>월 표준편차</th><th>누적수익 / 배수</th><th>MDD 고점→저점</th><th>양수 월</th><th>연율 하방편차</th></tr></thead><tbody>__FORMULA_METRIC_ROWS__</tbody></table></div>
<div class="note">이 Sharpe는 무위험수익률을 뺀 초과수익 Sharpe가 아닙니다. 코드가 월평균을 월 표준편차로 나눈 뒤 √12를 곱합니다. Sortino의 하방편차도 음수였던 달만 따로 표본화하지 않고, 모든 달에 <code>min(r, 0)</code>을 적용해 계산합니다.</div>

<h3>2005~2026을 요청했지만 동일 전략은 2007-04부터다</h3>
<div class="warning"><b>2005년 성과를 임의로 이어 붙이지 않았습니다.</b> KODEX200 프록시는 2000년부터 있지만 GLD는 2004-11, 채권지수는 2006-03, USO는 2006-04부터입니다. 네 자산의 공통 월수익은 2006-04에 시작하고, 거시 국면 모델이 요구하는 24개월 워밍업을 채우면 첫 거래월은 2007-04가 됩니다. 2005~2007-03에 채권·원유를 현금이나 다른 지수로 바꾸면 현재 전략이 아닌 별도 프록시 전략이므로 성과를 만들지 않았습니다.</div>
<div class="table"><table><thead><tr><th>경로</th><th>측정 구간</th><th>개월</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th><th>최종 배수</th></tr></thead><tbody>
__REQUESTED_PERIOD_ROWS__
__AVAILABLE_PERIOD_ROWS__
</tbody></table></div>
<p>KODEX200 프록시만 놓고 보면 2005-01~2026-07 CAGR 12.13%, Sharpe 0.606, MDD −48.52%입니다. 이는 시장 맥락을 보여 주는 무비용 단일자산 벤치마크일 뿐, 네 자산 전략의 2005년 성과가 아닙니다. 동일 전략의 실제 전체 232개월에서 최종 robust 경로는 CAGR 15.34%, Sharpe 1.128, MDD −12.98%였습니다.</p>
<details><summary>초기·내부검증·잠금 전반·잠금 후반 성과</summary><div class="table"><table><thead><tr><th>구간</th><th>CAGR · 기존→최종</th><th>Sharpe · 기존→최종</th><th>MDD · 기존→최종</th></tr></thead><tbody>__SUBPERIOD_PERFORMANCE_ROWS__</tbody></table></div></details>

<h3>숫자에 맞는 결과 파일</h3><div class="table"><table><thead><tr><th>보고서 숫자</th><th>권위 파일</th><th>계산 단위</th><th>주의할 차이</th></tr></thead><tbody>
<tr><td>상단 CAGR·Sharpe·MDD·Calmar</td><td><code>vkospi_robust_dynamic_validation.json</code></td><td>103개월 reconciled 수익</td><td>일별 엔진의 절대 수익이 아니라 월간 기준에 상대효과만 결합</td></tr>
<tr><td>순차 적용 ①~④</td><td><code>vkospi_robust_dynamic_stepwise_attribution.csv</code></td><td>각 정책의 reconciled 수익</td><td>사후 설명용이며 후보 재선정에는 사용하지 않음</td></tr>
<tr><td>수준·충격·가속 절삭표</td><td><code>vkospi_robust_dynamic_component_ablation.csv</code></td><td>같은 최종 실행정책에 성분만 교체</td><td>잠금 결과가 좋아도 사후 승격 금지</td></tr>
<tr><td>791일·837일·255일</td><td><code>vkospi_robust_dynamic_signal_statistics.csv</code></td><td>2,240 거래일 actual daily</td><td>월별 reconciled 표본과 단위가 다름</td></tr>
<tr><td>비용 1배·2배</td><td><code>vkospi_robust_dynamic_cost_sensitivity.csv</code></td><td>actual daily를 월 복리화</td><td>1배 CAGR도 상단 카드와 일치하지 않는 것이 정상</td></tr>
<tr><td>상·하위 기여 월</td><td><code>vkospi_robust_dynamic_monthly_contribution.csv</code></td><td>robust 월수익 − 기존 월수익</td><td>독립 알파가 아니라 두 오버레이의 상대 차이</td></tr>
</tbody></table></div>

<h3>비용을 2배로 올려도 남은 차이</h3><div class="table"><table><thead><tr><th>비용 배수</th><th>전략</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th><th>월평균 회전율</th></tr></thead><tbody>__COST_SENSITIVITY_ROWS__</tbody></table></div>
<p>이 표의 1배 CAGR 17.7447%·17.8863%가 상단 18.2984%·18.4408%보다 낮은 이유는 오류가 아닙니다. 비용표는 일별 엔진을 그대로 월 복리화한 <code>actual daily</code> 경로이고, 상단은 검증된 월간 기준에 상대효과를 결합한 경로입니다. 2배 비용에서도 최종 전략은 Sharpe 1.2760 대 1.2441, MDD −11.8346% 대 −11.9849%로 앞섰습니다.</p>

<h3>부트스트랩 확률 57.48%·86.80%·64.38%·40.40%</h3><p>103개월의 순서를 완전히 섞지 않고 6개월 연속 블록을 복원추출합니다. 매 회 기존·최종 경로에 같은 월 인덱스를 적용해 짝을 유지하고, 5,000회마다 CAGR·Sharpe·MDD의 차이를 다시 계산했습니다. 난수 시드는 <code>20260827 + 6</code>입니다.</p>
<div class="table"><table><thead><tr><th>지표 차이 · 최종 − 기존</th><th>개선 확률</th><th>5% 분위</th><th>중앙값</th><th>95% 분위</th></tr></thead><tbody>
<tr><td>CAGR</td><td>__P_CAGR__</td><td>__BOOT_CAGR_P05__</td><td>__BOOT_CAGR_MED__</td><td>__BOOT_CAGR_P95__</td></tr>
<tr><td>Sharpe</td><td>__P_SHARPE__</td><td>__BOOT_SHARPE_P05__</td><td>__BOOT_SHARPE_MED__</td><td>__BOOT_SHARPE_P95__</td></tr>
<tr><td>MDD</td><td>__P_MDD__</td><td>__BOOT_MDD_P05__</td><td>__BOOT_MDD_MED__</td><td>__BOOT_MDD_P95__</td></tr>
<tr class="winner-row"><td>세 지표가 동시에 양수</td><td>__P_ALL__</td><td colspan="3">각 회차에서 세 조건을 한꺼번에 확인</td></tr>
</tbody></table></div>
<p>Sharpe 개선 확률은 86.80%지만 세 지표 동시 개선은 40.40%입니다. CAGR과 MDD의 5% 분위가 각각 음수이므로, 잠금 표본에서 관측된 개선이 어떤 재표본에서도 유지된다고 말할 수는 없습니다.</p>
</section>

<section id="hyperparameters"><div class="head"><small>14 · LOGISTIC</small><div><h2>로지스틱 회귀 설정값의 출처와 한계</h2><p class="lede">이 모델에는 실증적으로 고른 숫자와 보수적으로 고정한 숫자가 함께 들어 있습니다. 어느 쪽인지 구분하지 않으면 <code>C=0.1</code>까지 최적화한 것처럼 오해하기 쉽습니다.</p></div></div>
<div class="warning"><b><code>C=0.1</code>은 이 전략에서 탐색한 최적값이 아닙니다.</b> <code>final_blend_crash_meta_experiment.py</code>의 소표본 꼬리손실 파이프라인에서 모델·전처리·출력 골격을 재사용하고, 라벨과 embargo는 현재 2개월 목표에 맞게 바꿨습니다. 현재 결과 파일에는 C·penalty·solver를 비교한 사전 그리드가 없습니다. 따라서 아래 근거는 설계상 이유이며, 수익률로 최적성이 입증됐다는 뜻은 아닙니다.</div>

<h3>먼저, ‘꼬리손실 파이프라인’이 무엇인지</h3>
<div class="note"><b>여기서 꼬리손실은 로지스틱 회귀의 손실함수 이름이 아닙니다.</b> 포트폴리오 수익경로가 미리 정한 큰 손실선에 닿았는지를 0·1로 표시한 예측 목표입니다. ‘파이프라인’은 그 라벨을 만들고, 과거 자료로 확률을 예측하고, 순위형 위험도로 바꿔 방어 비중까지 연결하는 전체 절차를 뜻합니다.</div>
<div class="flow">
<article><h3>사건 라벨</h3><p>현재 모델은 앞으로 2개월의 누적 경로 중 최저점이 −5% 아래면 <code>tail_event=1</code>로 둡니다.</p></article>
<article><h3>월별 입력</h3><p>그달까지 관측된 국내 상태 12개와 OAP 합성 4개, 모두 16개를 한 행으로 묶습니다.</p></article>
<article><h3>순차 학습</h3><p>최소 36개월 뒤부터 expanding 방식으로 매달 다시 적합하고 마지막 2개월은 학습에서 뺍니다.</p></article>
<article><h3>희귀사건 로짓</h3><p>중앙값 대치·표준화 뒤 균형 L2 로지스틱으로 <code>p_tail_raw</code>를 계산합니다.</p></article>
<article><h3>인과 순위</h3><p>원시확률을 그 시점 이전 예측 안의 백분위로 바꾸고 상위 20%에서만 severity를 올립니다.</p></article>
<article><h3>방어 비중</h3><p>severity에 따라 위험자산 비중을 최대 20% 옮기고 마지막에 연 15% 변동성 목표를 적용합니다.</p></article>
</div>
<p>예를 들어 첫 달 수익이 −2%, 다음 달이 −4%라면 두 달 누적경로는 −2%에서 약 −5.92%로 내려갑니다. 만기 누적수익만 보는 것이 아니라 중간 경로의 최저점을 보므로 이 달은 꼬리손실 사건 1입니다. 반대로 두 달 내내 −5% 선을 건드리지 않으면 0입니다.</p>

<h3>왜 새 분류기를 고르지 않고 기존 파이프라인을 재사용했나</h3>
<div class="grid2"><div class="panel"><h3>코드에서 직접 확인되는 이유</h3><ul><li><code>openassetpricing_signal_experiment.py</code>가 기존 파일의 <code>make_model</code>·<code>make_factor</code>·<code>predictive_metrics</code>를 직접 가져옵니다.</li><li>두 실험 모두 월별 소표본에서 드문 하락사건을 예측하고, 결과를 <code>p_tail_raw → risk_percentile → risk_severity</code> 형식으로 배분 엔진에 넘깁니다.</li><li>학습기를 고정하면 OAP 입력과 목표기간을 바꾼 효과를 기존 모델 구조 변경 효과와 섞지 않고 비교할 수 있습니다.</li><li>C·penalty·solver까지 추가 탐색하지 않아, 이미 작은 2017년 이전 표본에서 후보 수와 연구자 자유도가 더 늘어나는 것을 피했습니다.</li></ul></div><div class="panel"><h3>그 선택이 실무적으로 맞아 보였던 까닭</h3><ul><li>캘리브레이션 예측은 91건, 사건은 5건뿐이라 복잡한 비선형 모델보다 강하게 축소한 선형모델이 수치적으로 다루기 쉽습니다.</li><li>중앙값 대치는 시작일이 다른 업종 자료를 보존하고, 표준화는 단위가 다른 16개 열에 같은 L2 벌점을 적용하게 합니다.</li><li>균형 클래스 가중과 최소 사건 수 가드는 모두 0만 예측하거나 완전분리되는 문제를 줄입니다.</li><li>기존 출력 형식을 유지해 하위 비중 엔진과 성과 비교 코드를 바꾸지 않아도 됩니다.</li></ul></div></div>
<div class="note"><b>근거 수준을 구분해야 합니다.</b> 직접 import해 재사용했다는 사실은 코드로 확인됩니다. 반면 위의 ‘비교 가능성·자유도 억제·인터페이스 유지’는 구현 구조에서 읽히는 합리적 이유이며, 당시 개발자가 남긴 별도 의사결정 문서나 사전등록 기록은 없습니다.</div>

<h3>그대로 가져온 부분과 현재 목표에 맞게 바꾼 부분</h3>
<div class="table"><table><thead><tr><th>구분</th><th>기존 <code>final_blend</code> 파이프라인</th><th>현재 2개월 꼬리손실 모델</th><th>의미</th></tr></thead><tbody>
<tr><td>예측기</td><td colspan="2"><code>median imputer → StandardScaler → balanced L2 LogisticRegression(C=0.1)</code></td><td>모델과 전처리는 그대로 재사용</td></tr>
<tr><td>학습 가드</td><td colspan="2">최소 36개월, 양성 ≥ 4, 음성 ≥ 12, expanding walk-forward</td><td>소표본 적합 조건도 유지</td></tr>
<tr><td>원래 라벨</td><td>다음 1개월 FinalBlend 수익이 −3%·−4% 또는 후행 1σ 아래인지</td><td>사용하지 않음</td><td>기존 파이프라인이 처음 풀던 문제</td></tr>
<tr><td>현재 라벨</td><td>사용하지 않음</td><td>앞으로 2개월 누적경로의 최저점이 −5% 아래인지</td><td>한 달 손실이 아니라 짧은 경로 붕괴를 예측</td></tr>
<tr><td>입력</td><td>국내 12개 또는 VIX·신용·금융스트레스 보조열을 포함한 사양</td><td>국내 12개 + OAP 합성 4개</td><td>학습기는 고정하고 설명변수 묶음은 교체</td></tr>
<tr><td>embargo</td><td>1개월</td><td>2개월</td><td>라벨이 소비하는 미래수익 길이에 맞춰 누수 차단을 늘림</td></tr>
<tr><td>최대 위험 이전</td><td>0~20% 후보</td><td>5%·10%·15%·20% 후보 중 20% 선택</td><td>모델 C가 아니라 배분 강도를 사전구간에서 비교</td></tr>
</tbody></table></div>
<div class="warning"><b>재사용은 최적성의 증거가 아니라 비교를 통제한 설계 선택입니다.</b> 같은 파이프라인을 썼다고 해서 <code>C=0.1</code>이 2개월 −5% 라벨에도 가장 좋다는 뜻은 아닙니다. 실제로 균형 클래스 가중의 원시확률은 Brier가 나쁩니다. 이번 사후 감사에서 C·L1/L2·해법을 비교했지만, 실제 변경 여부를 판단하려면 2018년 이후 잠금자료를 다시 튜닝에 쓰지 말고 새로운 미래 구간의 규칙을 먼저 고정해야 합니다.</div>

<h3>모델 파이프라인과 숫자의 성격</h3><div class="table"><table><thead><tr><th>항목</th><th>코드값</th><th>구현상 역할</th><th>정한 근거와 한계</th></tr></thead><tbody>
<tr><td>결측치</td><td><code>SimpleImputer(strategy="median")</code></td><td>각 적합 시점의 학습표본 중앙값으로 빈칸 대치</td><td>OAP 업종 자료의 시작 시점과 결측이 서로 달라 행을 버리지 않기 위한 고정 설계</td></tr>
<tr><td>스케일</td><td><code>StandardScaler()</code></td><td>평균 0·표준편차 1로 변환</td><td>비중·수익률·변동성·z-score의 단위가 다르고 L2 벌점은 스케일에 민감하기 때문</td></tr>
<tr><td>규제</td><td>기본 <code>penalty="l2"</code></td><td>계수를 0 쪽으로 줄여 분산 억제</td><td>16개 열에 비해 사건 수가 적은 월별 표본에서 완전분리와 큰 계수를 막는 보수적 선택. L1과 비교하지는 않음</td></tr>
<tr><td>역규제 강도</td><td><code>C=0.1</code></td><td>C가 작을수록 규제가 강함</td><td>기본값 1보다 강한 축소를 거는 레거시 고정값. 이 전략 안에서 C 그리드를 돌린 근거는 없음</td></tr>
<tr><td>희귀사건 가중</td><td><code>class_weight="balanced"</code></td><td>클래스 빈도의 역수에 비례해 양성 손실을 확대</td><td>캘리브레이션 예측 91건 중 사건이 5건(5.49%)뿐이라 모두 0으로 치우치는 적합을 막기 위한 설계</td></tr>
<tr><td>해법</td><td><code>solver="liblinear"</code></td><td>작은 이진 분류를 결정적으로 적합</td><td>월별 소표본·L2 로지스틱에 맞춘 구현 선택. 다른 solver와 속도·성능을 비교한 기록은 없음</td></tr>
<tr><td>반복 상한</td><td><code>max_iter=2000</code></td><td>수렴 전에 강제로 멈추지 않도록 넉넉한 상한</td><td>학습된 하이퍼파라미터가 아니라 안전 한도. 실제 반복 횟수는 결과 파일에 저장하지 않음</td></tr>
<tr><td>최소 학습 길이</td><td><code>min_train=36</code></td><td>최소 3년이 쌓인 뒤 첫 예측</td><td>월별 계절·국면을 어느 정도 포함하기 위한 고정 하한. 36·48·60개월 비교는 없음</td></tr>
<tr><td>사건 수 가드</td><td>양성 ≥ 4, 음성 ≥ 12</td><td>한 클래스가 거의 없는 적합을 건너뜀</td><td>수치적 실패를 막는 최소 조건. 통계적 충분성을 보장하는 기준은 아님</td></tr>
<tr><td>미래 라벨 차단</td><td><code>embargo_months=2</code></td><td>예측 직전 2개월을 학습 끝에서 제외</td><td>2개월 경로손실 라벨끼리 미래 수익 구간이 겹치지 않도록 목표 horizon과 같은 길이를 사용</td></tr>
<tr><td>확률 트리거</td><td>과거 백분위 0.80</td><td>위험 예측 상위 20%에서만 방어</td><td>레거시 정책 고정값. 중기 목표 탐색표에서는 0.80을 따로 비교하지 않음</td></tr>
</tbody></table></div>

<h3>분류 성능 숫자</h3><div class="table"><table><thead><tr><th>표본</th><th>예측 가능 월</th><th>사건</th><th>사건률</th><th>ROC AUC</th><th>평균정밀도</th><th>Brier</th><th>상위 20% 재현율 / 정밀도</th></tr></thead><tbody>
<tr><td>2017년 이전</td><td>__LOGIT_CAL_N__</td><td>__LOGIT_CAL_EVENTS__</td><td>__LOGIT_CAL_RATE__</td><td>__LOGIT_CAL_AUC__</td><td>__LOGIT_CAL_AP__</td><td>__LOGIT_CAL_BRIER__</td><td>__LOGIT_CAL_RECALL__ / __LOGIT_CAL_PRECISION__</td></tr>
<tr><td>2018년 이후 잠금</td><td>__LOGIT_LOCK_N__</td><td>__LOGIT_LOCK_EVENTS__</td><td>__LOGIT_LOCK_RATE__</td><td>__LOGIT_LOCK_AUC__</td><td>__LOGIT_LOCK_AP__</td><td>__LOGIT_LOCK_BRIER__</td><td>__LOGIT_LOCK_RECALL__ / __LOGIT_LOCK_PRECISION__</td></tr>
</tbody></table></div>
<div class="note">2017년 이전 ROC AUC는 __LOGIT_CAL_AUC__로 0.5보다 낮습니다. 잠금 구간에서는 __LOGIT_LOCK_AUC__로 높아졌지만, 이 사후 차이를 근거로 모델을 다시 고르지는 않았습니다. 포트폴리오 캘리브레이션을 통과했다는 사실과 확률 분류기가 안정적으로 우수하다는 주장은 서로 다릅니다.</div>

<h3>AUC와 Brier는 입력변수가 아니라 검증지표다</h3>
<div class="table"><table><thead><tr><th>지표</th><th>무엇을 재나</th><th>좋은 방향</th><th>이 표본에서 읽는 법</th></tr></thead><tbody>
<tr><td><b>ROC AUC</b></td><td>무작위 사건 1건의 점수가 무작위 비사건 1건보다 높을 확률에 해당하는 순위 판별력</td><td>1에 가까울수록 좋고 0.5는 무작위</td><td>확률의 절대 크기가 맞는지는 말해 주지 않음</td></tr>
<tr><td><b>Average Precision</b></td><td>정밀도-재현율 곡선의 요약. 희귀사건을 상위에 모으는 능력</td><td>1에 가까울수록 좋음</td><td>무정보 기준은 대략 사건률 5.49%·9.80%</td></tr>
<tr><td><b>Brier</b></td><td><code>mean((p_tail − y)²)</code>. 사건 확률과 실제 0/1의 제곱오차</td><td>0에 가까울수록 좋음</td><td>AUC가 좋아도 확률이 지나치게 높거나 낮으면 나빠짐</td></tr>
<tr><td><b>LogLoss</b></td><td>실제 사건에 낮은 확률을 주거나 비사건에 높은 확률을 줄 때 큰 벌점</td><td>0에 가까울수록 좋음</td><td>극단적으로 자신한 오답에 Brier보다 민감</td></tr>
<tr><td><b>ECE · 5구간</b></td><td>예측확률 구간별 평균확률과 실제 사건률의 절대 차이를 표본수로 가중</td><td>0에 가까울수록 좋음</td><td>표본이 91·102건으로 작아 구간별 오차가 불안정함</td></tr>
<tr><td><b>상위 20% Recall / Precision</b></td><td>위험점수 상위 20%가 전체 사건을 얼마나 잡았는지 / 그 상위군 중 실제 사건 비율</td><td>둘 다 높을수록 좋음</td><td>실제 운용 트리거 0.80과 직접 연결</td></tr>
</tbody></table></div>
<div class="table" style="margin-top:14px"><table><thead><tr><th>표본</th><th>관측 / 사건</th><th>AUC</th><th>AP</th><th>Brier</th><th>사건률 고정예측 Brier</th><th>LogLoss</th><th>ECE</th><th>상위20% Recall / Precision</th></tr></thead><tbody>__TAIL_PREDICTION_ROWS__</tbody></table></div>
<div class="warning"><b>잠금 AUC 0.7565만 보면 안 됩니다.</b> 2017년 이전 AUC는 0.3791이고, Brier는 사전구간 0.1962 대 단순 사건률 예측 0.0519, 잠금구간 0.2193 대 0.0903으로 더 나쁩니다. <code>class_weight="balanced"</code>가 희귀사건의 순위 판별을 돕는 대신 원시 확률을 실제 사건률보다 높게 만들었기 때문입니다. 그래서 이 전략은 <code>p_tail</code>의 절대값을 비중으로 쓰지 않고 과거 예측 안에서의 백분위로 다시 바꿉니다.</div>

<details><summary>16개 설명변수의 단변량 AUC와 순차 로지스틱 계수</summary><p>AUC는 각 변수 하나만 봤을 때의 방향 있는 판별력이고, 계수는 매달 그 시점까지의 자료로 다시 적합한 표준화 로지스틱 계수의 중앙값입니다. 상관된 변수를 함께 넣으므로 단변량 AUC 방향과 다변량 계수 부호가 다를 수 있습니다. 잠금 열은 사후 설명용이며 변수 재선정에 쓰지 않았습니다.</p><div class="table"><table><thead><tr><th>묶음</th><th>설명변수</th><th>단변량 AUC · 사전 / 잠금</th><th>표준화 계수 중앙값 · 사전 / 잠금</th><th>계수 부호 안정률 · 사전 / 잠금</th></tr></thead><tbody>__TAIL_FEATURE_ROWS__</tbody></table></div></details>

<h3>실제로 탐색한 16개 조합</h3><p>탐색한 것은 C가 아니라 목표 라벨 4개와 최대 비중 이동 4개입니다. 비교선 <code>ExistingStructureVol15</code>의 2017년 이전 성과는 CAGR __OAP_BASE_CAGR__, Sharpe __OAP_BASE_SHARPE__, MDD __OAP_BASE_MDD__, Calmar __OAP_BASE_CALMAR__였습니다. 이 세 지표를 모두 넘은 후보는 __MEDIUM_ELIGIBLE__개였고, 그중 Calmar·Sharpe·CAGR 순으로 정렬해 2개월 −5%·20% 이동을 골랐습니다.</p>
<details open><summary>중기 목표·비중 이동 전 후보</summary><div class="table"><table><thead><tr><th>목표</th><th>2017년 이전 사건</th><th>최대 이동</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th></tr></thead><tbody>__MEDIUM_ROWS__</tbody></table></div></details>
<div class="finding">선정 후보의 2017년 이전 수치는 CAGR 12.6745%, Sharpe 0.9312, MDD −12.9823%, Calmar 0.9763입니다. 2개월 −4%·20% 후보의 Calmar 0.9762와 차이가 매우 작습니다. −5%가 압도적인 경계라서가 아니라, 미리 정한 정렬 규칙에서 근소하게 앞섰기 때문에 남았습니다.</div>

<h3>확률을 비중으로 바꾸는 숫자</h3><div class="equation">risk_percentile_t = rank(p_tail_t among past predictions)<br>severity_t = clip((risk_percentile_t − 0.80) / 0.20, 0, 1)<br>p_up_t = 0.50 − 0.15 × severity_t<br>score_t = clip((p_up_t − 0.50) / 0.15, −1, 1) = −severity_t<br>shift_t = 0.20 × severity_t</div>
<div class="table"><table><thead><tr><th>단계</th><th>왜 필요한가</th><th>숫자가 나온 이유</th></tr></thead><tbody>
<tr><td><code>p_tail</code></td><td>16개 입력으로 예측한 2개월 −5% 경로손실의 로지스틱 원시 확률</td><td>균형 클래스 가중 때문에 실제 사건률로 곧바로 읽기 어려움</td></tr>
<tr><td><code>causal_percentile</code></td><td>현재 확률을 오직 과거 예측들과 비교해 0~1 순위로 변환</td><td>확률 캘리브레이션과 시기별 분포 이동에 덜 민감하게 만들고 미래 분포는 보지 않음</td></tr>
<tr><td><code>0.80</code></td><td>과거 예측의 상위 20%에서만 방어 시작</td><td>레거시 꼬리위험 정책의 고정 트리거. 이번 16개 중기 후보 그리드에서는 따로 탐색하지 않았으므로 최적값 근거는 없음</td></tr>
<tr><td><code>/ 0.20</code></td><td>80백분위에서 0, 100백분위에서 1이 되도록 선형 확대</td><td><code>1 − 0.80 = 0.20</code>에서 자동으로 나온 분모이지 별도 하이퍼파라미터가 아님</td></tr>
<tr><td><code>0.50 − 0.15×severity</code></td><td>기존 팩터 엔진이 기대하는 중립 <code>p_up=0.50</code> 좌표로 위험도를 포장</td><td>severity 0~1을 p_up 0.50~0.35로 옮기는 호환 스케일. 경제적으로 추정한 상승확률이 아님</td></tr>
<tr><td><code>/ 0.15</code></td><td><code>score=(p_up−0.50)/0.15</code>로 다시 −1~0 점수화</td><td>앞의 0.15와 정확히 소거돼 <code>score=−severity</code>. 최종 비중에 독립적인 영향이 없음</td></tr>
<tr><td><code>shift=0.20×severity</code></td><td>주식·원유에서 채권·금으로 옮길 최대 비중 결정</td><td>5%·10%·15%·20% 사전 후보 중 20%가 세 성과지표를 모두 넘긴 후보로 선택됨</td></tr>
</tbody></table></div>
<p>결국 이 식의 실질적인 자유도는 <b>상위 20% 트리거</b>와 <b>최대 이동 20%</b>입니다. <code>p_up</code>의 0.15는 중간 인터페이스에서만 나타났다가 사라집니다. 따라서 이 값을 예측확률의 근거로 설명하면 잘못입니다.</p>
</section>

<section id="formula-basis"><div class="head"><small>15 · FACTORS & FORMULAS</small><div><h2>팩터를 남긴 근거와 수식을 이렇게 쓴 이유</h2><p class="lede">최종 경로에는 자동 선택된 변수, 사람이 설계한 합성변수, 그달마다 희소 가중치가 달라지는 변수가 함께 있습니다. ‘선택’이라는 말을 하나로 뭉뚱그리지 않고 각 층을 나눠 봅니다.</p></div></div>

<h3>팩터마다 선택 근거가 다르다</h3><div class="table"><table><thead><tr><th>팩터 묶음</th><th>남긴 이유</th><th>실제 선택 방식</th><th>해석할 때의 한계</th></tr></thead><tbody>
<tr><td>거시 12개</td><td>성장과 물가의 수준·전환점을 각각 세 계열로 관측</td><td>변수 목록은 설계 고정. <code>SparseJump2</code>가 매 시점 상태 분리가 큰 4개에만 희소 가중</td><td>12개를 전부 후보 그리드로 비교해 최적 조합을 찾았다고 말할 수 없음</td></tr>
<tr><td>국내 포트폴리오 상태 12개</td><td>현재 국면 비중, 모멘텀, 변동성, 하방위험, 자산 상관을 작은 열 수로 요약</td><td><code>DOMESTIC_FEATURES</code>에 코드로 선언한 고정 목록</td><td>최종 중기 모델 안에서 개별 열을 자동 제거하거나 중요도 순으로 뽑지 않음</td></tr>
<tr><td>OAP 합성 4개</td><td>22개 원시 열을 모멘텀·반전·꼬리·유동성의 네 경제적 축으로 압축</td><td>부호와 동일가중 평균은 사전 설계. 중기 분류기는 네 합성값을 모두 입력</td><td>회귀계수가 아니라 사람이 정한 합성식이므로 각 1/4 비중이 통계적으로 최적이라는 근거는 없음</td></tr>
<tr><td>VKOSPI 최종 3개</td><td>높은 수준, 단기 충격, 공포 가속을 서로 다른 위험 상태로 측정</td><td>5개 모드와 정책을 합친 810개 사전 탐색에서 <code>acceleration</code> 모드가 엄격 조건 통과</td><td>백분위·충격·가속의 40/35/25 가중치를 따로 연속 최적화하지 않음</td></tr>
</tbody></table></div>

<h3>0.55·0.20·0.10·0.85·0.15는 어떻게 읽어야 하나</h3>
<div class="warning"><b>이 다섯 값은 최종 수익률 탐색에서 나온 최적값이 아닙니다.</b> 거시 국면 엔진을 처음 만들 때 해석 가능성과 월별 잡음 억제를 위해 고정한 휴리스틱입니다. 새로 수행한 1변수 민감도 감사에서도 다른 값이 더 좋은 지표를 보이는 경우가 있어, 배포값을 통계적으로 최적이라고 주장할 수 없습니다.</div>
<div class="table"><table><thead><tr><th>상수</th><th>수학적으로 하는 일</th><th>설계 당시 의도</th><th>지금 확인된 한계</th></tr></thead><tbody>
<tr><td><code>0.20 × d3_mean</code></td><td>현재 수준 계수 1에 대해 3개월 변화 계수를 0.20으로 둠. 같은 크기라면 수준:변화가 5:1</td><td>GDP·수출·BSI 또는 물가 수준을 주 신호로 두면서 전환점만 약하게 앞당김</td><td>사전·잠금 모두 0.30~0.40에서 평균 Brier가 더 낮았습니다. 0.20의 최적 근거는 없음</td></tr>
<tr><td><code>/ 0.55</code></td><td>sigmoid의 온도. 합성 z가 +0.55면 0.731, +1이면 약 0.860</td><td>0 근처 차이를 확률 0.5 주변에서 비교적 선명하게 벌리되 즉시 0·1로 포화시키지 않음</td><td>0.35는 Brier가 더 낮지만 4분면 정확도는 엇갈립니다. 0.55는 절충값이지 추정치가 아님</td></tr>
<tr><td><code>0.10 SJM + 0.90 composite</code></td><td>희소 점프모델 확률은 10%, 투명 합성확률은 90%</td><td>월별 표본이 작아 복잡한 상태모델의 영향은 작게 두고 변수 선택·상태 지속성만 보탬</td><td>새 감사에서는 <code>sjm_weight=0</code>이 사전·잠금 평균 Brier와 4분면 정확도 모두 더 좋았습니다. 10% 혼합의 증분효과는 확인되지 않음</td></tr>
<tr><td><code>0.85 raw + 0.15 previous</code></td><td>당월 원시확률 85%, 직전 평활확률 15%. 과거 원시확률 가중은 12.75%, 1.91%처럼 빠르게 감소</td><td>한 달치 데이터가 국면을 완전히 뒤집는 것을 조금 누르되 전환 반응은 대부분 유지</td><td>0.90~1.00이 Brier에서 더 나은 구간도 있어 15% 평활의 최적 근거는 없음</td></tr>
<tr><td><code>p_{t−1}</code>의 <code>0.15</code></td><td>별도 새 상수가 아니라 위 식에서 <code>1 − 0.85</code></td><td>두 가중치 합을 1로 유지</td><td>0.85를 바꾸면 자동으로 함께 바뀌어야 함</td></tr>
</tbody></table></div>

<details open><summary>고정값을 바꾸지 않고 수행한 사후 1변수 민감도</summary><p>한 번에 한 상수만 바꾸고 다른 값은 배포 설정에 고정했습니다. 낮은 평균 Brier와 높은 4분면 정확도가 좋습니다. 잠금 결과는 설정을 다시 고르는 데 쓰지 않는 감사값입니다.</p><div class="table"><table><thead><tr><th>바꾼 축</th><th>값</th><th>사전 평균 Brier / 4분면 정확도</th><th>잠금 평균 Brier / 4분면 정확도</th></tr></thead><tbody>__MACRO_SENSITIVITY_ROWS__</tbody></table></div></details>
<p>배포값의 사전 평균 Brier는 0.0962, 잠금은 0.0948입니다. 예를 들어 SJM을 빼면 각각 0.0945·0.0896으로 낮아지고 4분면 정확도도 79.07%·80.39%로 높아졌습니다. 이 결과를 보고 지금 값을 바꾸면 잠금자료를 이용한 재튜닝이 되므로 보고서에서는 <b>약점으로 기록만 하고 전략은 유지</b>합니다.</p>

<h3>OAP 팩터 가족의 2017년 이전 비교</h3><div class="table"><table><thead><tr><th>입력 가족</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th><th>월평균 회전율</th></tr></thead><tbody>__OAP_FAMILY_ROWS__</tbody></table></div>
<p>단일 OAP 계열 비교에서는 모멘텀의 Calmar 0.8921이 반전 0.8921보다 아주 조금 높아 <code>standalone_oap_winner</code>가 됐습니다. 하지만 최종 중기 기준 경로는 이 단일 승자를 쓰지 않고 OAP 네 합성변수를 모두 넣은 별도 2개월 꼬리손실 모델입니다. 두 실험의 목표와 선택 단위를 구분해야 합니다.</p>

<h3>수식과 상수는 이렇게 정했다</h3><div class="table"><table><thead><tr><th>수식·숫자</th><th>코드에서 하는 일</th><th>왜 이렇게 구현했나</th><th>근거 성격</th></tr></thead><tbody>
<tr><td><code>rolling z-score 72/36/24개월</code></td><td>GDP는 72개월, 수출·CPI·PPI·수입물가는 36개월, BSI는 24개월 기준화</td><td>발표 주기와 계열 길이가 다른 거시 자료를 각자의 후행 역사 안에서 비교</td><td>설계 고정, 이 보고서에서 window 탐색 안 함</td></tr>
<tr><td><code>level + 0.20 × d3</code>, <code>/ 0.55</code></td><td>현재 수준을 주로 보고 3개월 변화는 보조로 넣은 뒤 sigmoid 확률화</td><td>짧은 반전을 반영하되 수준 신호를 압도하지 않게 함</td><td>설계 고정. 0.20·0.55 그리드 없음</td></tr>
<tr><td><code>0.10 SJM + 0.90 composite</code></td><td>희소 점프모델과 투명 합성확률 혼합</td><td>코드 주석대로 월별 표본이 작아 단순 합성값을 주 예측으로 두고 SJM은 선택·지속성만 보탬</td><td>설계 고정</td></tr>
<tr><td><code>0.85 raw + 0.15 previous</code></td><td>직전 확률을 15% 남겨 월별 출렁임 완화</td><td>한 달치 잡음이 국면확률을 급히 뒤집는 현상을 줄임</td><td>설계 고정</td></tr>
<tr><td><code>jump_penalty=3.0</code>, <code>keep_features=4</code></td><td>상태 전환 비용과 희소 변수 수</td><td>잦은 점프를 억제하고 6개 열 중 분리력이 큰 소수만 사용</td><td>설계 고정, 민감도표 없음</td></tr>
<tr><td><code>causal z-score</code> 뒤 OAP 동일가중</td><td>서로 다른 단위의 22개 원시 열을 네 합성값으로 축약</td><td>미래 평균·표준편차를 쓰지 않으면서 부호 방향을 맞추고 차원을 줄임</td><td>OAP 아이디어 기반 설계</td></tr>
<tr><td><code>1.4826 × MAD</code></td><td>robust z-score의 분모</td><td>정규분포에서 MAD를 표준편차와 같은 척도로 맞추는 일관성 상수</td><td>통계 관례, 최종 스트레스에는 직접 미사용</td></tr>
<tr><td><code>shock = Δlog / (σ63 × √w)</code></td><td>5·10·21일 VKOSPI 변화를 당시 63일 변동성으로 나눔</td><td>같은 절대 상승률도 평온기와 위기 중 의미가 다르기 때문</td><td>window 일부는 후보 생성, 최종은 5일</td></tr>
<tr><td><code>(signal − threshold) / 2.5</code></td><td>문턱부터 문턱+2.5σ까지를 0~1로 선형 변환</td><td>극단치가 비중을 무한히 키우지 않게 포화</td><td>2.5는 설계 고정</td></tr>
<tr><td><code>0.40 level + 0.35 shock + 0.25 acceleration</code></td><td>선정된 acceleration 모드의 스트레스</td><td>수준을 가장 크게 두고 충격과 가속을 확인항으로 더함</td><td>5개 모드 중 선택됐지만 세 가중치 자체는 모드 정의로 고정</td></tr>
<tr><td><code>transfer = 0.35 × stress</code></td><td>KODEX200·USO 각 비중의 최대 35%를 줄임</td><td>전량 청산을 막으면서 극단 신호에서는 기존 25%보다 더 방어</td><td>15/25/35% 그리드에서 선택</td></tr>
<tr><td><code>bond_share=0</code></td><td>줄인 주식·원유 비중을 전부 GLD로 이동</td><td>잠금 이전 양쪽 창에서 채권 혼합보다 세 지표를 함께 개선한 후보 구조</td><td>0/50% 그리드에서 선택</td></tr>
<tr><td><code>rebalance_band=0.20</code></td><td>목표와 현재 비중의 half-L1 차이가 20% 미만이면 월중 거래 보류</td><td>작은 신호 변화를 거래하지 않아 비용과 반전 손실을 줄임</td><td>10/15/20% 그리드에서 선택</td></tr>
<tr><td><code>target_vol=0.15</code>, 24개월 지수가중</td><td>예상 연변동성에 맞춰 레버리지를 0.50~1.50으로 제한</td><td>최근 관측에 더 큰 가중을 주되 레버리지 폭주와 과도한 축소를 막음</td><td>중기 경로에서 15% 고정, 별도 C와 마찬가지로 이 표에서 최적화 안 함</td></tr>
<tr><td><code>trade 0.15%</code>, <code>FX 0.05%</code>, <code>finance 4%</code></td><td>절대 비중 변화, 해외자산 비중 변화, 차입 비중에 비용 부과</td><td>회전과 환전·레버리지를 무비용으로 두지 않기 위한 가정</td><td>비용 가정. 실제 계좌 체결자료로 추정하지 않음</td></tr>
<tr><td><code>relative_factor</code> 재조정</td><td>일별 오버레이/일별 중립 경로의 상대배수만 검증된 월간 수익에 곱함</td><td>금융비용·거래비용의 일·월 복리 시점 차이를 새 알파로 잘못 세지 않기 위해</td><td>회계 정합성 설계</td></tr>
</tbody></table></div>

<details><summary>비교선인 기존 VKOSPI 동적 전략도 어떻게 골랐나</summary><p>기존 전략은 <code>mean</code> 모드, 수준 백분위 0.80, 5일 모멘텀, 15% 급등 문턱, 최대 위험 이전 25%, GLD 100%, 월중 밴드 15%, 금융비용 4%입니다. 528개 coarse 후보를 먼저 시험하고, 양쪽 사전 창에서 CAGR·Sharpe·MDD를 모두 개선한 구조 주변만 세부 탐색했습니다. 중복 제거 뒤 총 836개, 엄격 통과 96개였으며 다목적 백분위 순위로 이 설정을 골랐습니다. 이 기존 전략이 robust 810개 탐색의 비교선입니다.</p></details>

<h3>810개 후보의 범위가 나온 이유</h3>
<p>무작정 연속 숫자를 촘촘히 훑은 것이 아니라, 기존 전략의 가운데 값과 그 양옆을 둔 거친 구조 실험입니다. 기존 수준 0.80·이전 25%·밴드 15%를 중심으로 한 단계 완화와 한 단계 강화를 붙였고, 정규화 충격은 없음·보통·강함의 세 단계로 나눴습니다.</p>
<div class="table"><table><thead><tr><th>탐색 축</th><th>후보</th><th>범위를 정한 논리</th><th>자유도를 줄인 장치</th></tr></thead><tbody>
<tr><td>스트레스 구조 · 5개</td><td>mean, max, confirmation, acceleration, exhaustion-adjusted</td><td>두 신호 평균, 어느 하나의 극단, 가격위치 확인, 공포 가속, 고점 후 진정이라는 서로 다른 경제적 가설</td><td>식 안의 50/50·45/55·40/35/25 등은 고정하고 구조 전체만 후보로 비교</td></tr>
<tr><td>수준 문턱 · 3개</td><td>0.70 / 0.80 / 0.90</td><td>VKOSPI가 후행 분포의 상위 30%·20%·10%일 때 시작. 기존 0.80을 중심으로 넓고 좁은 발동구간 확인</td><td>인과적 백분위를 써 시대별 절대 VKOSPI 수준을 따로 최적화하지 않음</td></tr>
<tr><td>충격 문턱 · 3개</td><td>0 / 0.5 / 1.0σ</td><td>5일 상승이 양수면 반응, 반 표준편차부터 반응, 한 표준편차 이상만 반응하는 없음·중간·강한 확인</td><td>후행 63일 변동성으로 표준화하고 0.5σ 간격의 세 점만 사용</td></tr>
<tr><td>최대 이전 · 3개</td><td>15% / 25% / 35%</td><td>기존 25%를 중심으로 보수·기준·강한 방어. 35%에서도 위험자산 전량 청산은 하지 않음</td><td>5% 단위 촘촘한 탐색이나 50~100% 극단 이전을 제외</td></tr>
<tr><td>방어 배분 · 2개</td><td>GLD 100% / GLD 50%+BOND 50%</td><td>공포 헤지 자산에 집중할지, 금리 방어와 반씩 나눌지 비교</td><td>연속 배분비 대신 두 경제적으로 해석 가능한 끝점만 사용</td></tr>
<tr><td>무거래 밴드 · 3개</td><td>10% / 15% / 20%</td><td>기존 15% 주위에서 민감·기준·둔감 체결정책을 비교</td><td>5% 단위 미세 튜닝을 하지 않고 half-L1 회전율 기준을 고정</td></tr>
</tbody></table></div>
<div class="equation">5 structures × 3 levels × 3 shocks × 3 transfer caps × 2 defensive splits × 3 bands = 810</div>
<div class="note">이 설명은 후보 범위가 경제적으로 해석 가능하다는 뜻이지, 0.70·0.80·0.90이나 15·25·35%가 이론에서 유도됐다는 뜻은 아닙니다. 특히 기존 승자를 중심으로 새 격자를 만들었으므로 연구과정 전체의 선택 편향은 남습니다.</div>

<details><summary>후보였던 다섯 스트레스 모드의 정확한 식</summary><div class="equation">robust_mean = 0.50L + 0.50S<br>robust_max = max(L, S)<br>confirmed = (0.45L + 0.55S) × (0.35 + 0.65 × close_location21)<br>acceleration = 0.40L + 0.35S + 0.25A<br>exhaustion_adjusted = (0.50L + 0.50S) × (1 − 0.75 × exhaustion)<br><br>falling = clip(−shock_raw / 2.5, 0, 1)<br>exhaustion = clip(L × clip(−distance_high21, 0, 0.5) / 0.5 × falling, 0, 1)</div><p>여기서 <code>L</code>은 백분위 수준, <code>S</code>는 5일 정규화 충격, <code>A</code>는 5일 가속입니다. 다섯 식 자체가 후보 단위였으며 식 안의 50/50, 45/55, 35/65, 40/35/25, 75%를 따로 조정하지는 않았습니다.</p></details>

<h3>810개 탐색에서 실제로 남은 두 후보</h3><div class="table"><table><thead><tr><th>수준 문턱</th><th>2007–2017 ΔCAGR / ΔSharpe / ΔMDD</th><th>2013–2017 ΔCAGR / ΔSharpe / ΔMDD</th><th>다목적 순위</th><th>선정</th></tr></thead><tbody>
<tr><td>0.80</td><td>+0.0183%p / +0.002244 / +0.0057%p</td><td>+0.0264%p / +0.004302 / +0.0149%p</td><td>0.883256</td><td>아니오</td></tr>
<tr class="winner-row"><td>0.90</td><td>+0.0624%p / +0.003549 / +0.0057%p</td><td>+0.0178%p / +0.003424 / +0.0149%p</td><td>0.889506</td><td>예</td></tr>
</tbody></table></div>
<p>두 후보는 acceleration, 충격 문턱 1.0σ, 최대 이전 35%, GLD 100%, 밴드 20%가 같았습니다. 0.90 문턱의 다목적 백분위 평균이 0.889506으로 0.80의 0.883256보다 높아 최종식이 됐습니다. 차이가 작으므로 0.90을 보편적인 최적 문턱으로 해석해서는 안 됩니다.</p>

<h3>오버피팅은 없다고 결론낼 수 없다</h3>
<div class="grid2"><div class="panel"><h3>코드에 들어 있는 방어장치</h3><ul><li>후보 선택은 2017-12에서 끝나고 2018-01 이후 103개월은 잠금 평가에만 사용.</li><li>2007-2017 전체와 2013-2017에서 CAGR·Sharpe·MDD가 모두 개선돼야 통과.</li><li>810개 중 2개, __OVERFIT_PASS_RATE__만 엄격 관문을 통과.</li><li>6개월 짝지은 블록 부트스트랩과 비용 2배 결과를 별도 보고.</li><li>신호일이 수익 발생일보다 빠른지 테스트로 확인.</li></ul></div><div class="panel"><h3>그래도 남는 선택 편향</h3><ul><li>810개를 비교했으므로 우연히 좋아 보이는 후보를 고를 가능성이 큼.</li><li>2013-2017은 2007-2017 안에 포함된 구간이라 두 관문이 독립 표본이 아님.</li><li>승자와 차점자의 점수 차이는 __OVERFIT_SCORE_GAP__뿐.</li><li>거시 상수와 모드 내부 가중치는 사전등록된 추정치가 아니라 개발과정의 판단값.</li><li>개정 거시자료와 여러 선행 실험을 거쳐 연구자 자유도가 누적됨.</li><li>Deflated Sharpe, White Reality Check, CSCV/PBO, 외부시장 재현은 아직 수행하지 않음.</li></ul></div></div>
<div class="warning"><b>결론은 ‘누수 통제는 있으나 오버피팅을 배제할 수 없다’입니다.</b> 잠금 Sharpe 개선 확률은 86.80%지만 세 지표 동시개선 확률은 40.40%입니다. 잠금 9개 연도 중 최종 전략이 기존 전략보다 연수익이 높았던 해는 __LOCKED_OUTPERFORM_YEARS__개였습니다. 잠금 후반 2022-01~2026-07에는 CAGR이 오히려 −0.362%p 낮았고 Sharpe 차이는 +0.0008에 그쳤습니다.</div>
<details><summary>승자에서 한 축만 바꾼 13개 이웃과 승자</summary><div class="table"><table><thead><tr><th>설정</th><th>격자 거리</th><th>엄격 관문</th><th>2007-2017 ΔCAGR / ΔSharpe / ΔMDD</th><th>2013-2017 ΔCAGR / ΔSharpe / ΔMDD</th><th>다목적 점수</th></tr></thead><tbody>__GRID_NEIGHBORHOOD_ROWS__</tbody></table></div></details>

<details><summary>SLSQP 방어 경로에 고정된 숫자</summary><div class="table"><table><thead><tr><th>구분</th><th>값</th><th>쓰임</th></tr></thead><tbody>
<tr><td>기준 설정</td><td><code>target_vol=.08</code>, <code>half_life=12</code>, <code>invvol_tilt=.35</code></td><td>방어 경로의 변동성 목표·공분산 감쇠·역변동성 기울기</td></tr>
<tr><td>앵커 혼합</td><td><code>regime_strength=.75</code>, prior는 앵커 55% + 역변동성 45%</td><td>전략 앵커를 유지하면서 자산별 위험 차이를 일부 반영</td></tr>
<tr><td>공분산·목표</td><td>최근 84개월, 목표 <code>.08 × (.86 + .20 × p_growth_high)</code></td><td>성장확률에 따라 약 6.88~8.48% 사이에서 방어 경로의 목표변동성 조절</td></tr>
<tr><td>목적함수</td><td><code>return 1.15</code>, <code>vol .18</code>, <code>CDaR .25</code>, <code>turnover .05</code>, <code>tracking .32</code></td><td>기대수익 보상과 네 종류 벌점의 상대계수</td></tr>
<tr><td>제약</td><td><code>max_cdar=.16</code>, 합계 1</td><td>84개월 경로의 꼬리 드로다운과 총 비중 제한</td></tr>
<tr><td>자산 bounds</td><td>KODEX200 2~68%, BOND 5~88%, GLD 2~62%, USO 0~38%</td><td>한 자산 집중과 음수 비중 차단</td></tr>
<tr><td>기대수익</td><td>장기 80% + 최근 20%, 월 −0.6~1.5% 절삭</td><td>최근값만 좇지 않으면서 극단 평균 제한</td></tr>
<tr><td>드로다운 가드</td><td>−5%부터, 12% 구간, 강도 .75</td><td>실현 손실이 커질수록 방어 벡터 5/72/23/0% 쪽으로 이동</td></tr>
<tr><td>최적화</td><td><code>SLSQP maxiter=80, ftol=1e-8</code></td><td>제약식 아래 목적함수를 수치적으로 풂</td></tr>
</tbody></table></div><p>이 숫자들은 현재 robust VKOSPI 810개 탐색 대상이 아닙니다. 기존 기준 전략에서 고정된 배분 정책이며, 이 보고서만으로 각각의 최적성을 주장할 수 없습니다.</p></details>
</section>

<section id="code-trace"><div class="head"><small>16 · CODE TRACE</small><div><h2>코드와 주석을 실행 순서대로 읽기</h2><p class="lede">파일명만 나열하지 않고, 각 함수가 어떤 입력을 받고 어떤 숫자를 만들며 주석이 무엇을 막으려는지 정리했습니다.</p></div></div>
<div class="table"><table><thead><tr><th>순서</th><th>파일·함수</th><th>하는 일</th><th>주석과 방어장치가 뜻하는 것</th></tr></thead><tbody>
<tr><td>1</td><td><code>regime_research.py::load_macro_data</code></td><td>6개 거시 원자료를 월말에 맞추고 12개 수준·3개월 변화 열 생성</td><td>“수준은 국면, 3개월 변화는 전환점”이라는 주석이 변수 역할을 분리</td></tr>
<tr><td>2</td><td><code>regime_research.py::SparseJump2.fit_predict_high</code></td><td>중앙값/IQR 정규화, 상위 4개 변수, 전환벌점 DP로 성장·물가 고/저 확률 계산</td><td>모델 이름과 달리 확률 전체를 맡기지 않고 희소 선택과 상태 지속성만 보탬</td></tr>
<tr><td>3</td><td><code>compute_regime_signals</code></td><td>SJM 10%·합성확률 90%·직전확률 15%를 섞어 4개 국면 확률 생성</td><td>“표본이 작아서 투명 합성을 주 예측으로 둔다”는 코드 주석이 10/90의 설계 의도를 밝힘</td></tr>
<tr><td>4</td><td><code>controlled_weights</code></td><td>EWMA 공분산·CDaR·회전율·tracking 아래 SLSQP 방어 비중 계산</td><td>최적화 실패 시 <code>prior</code>로 돌아가고, 드로다운 중에도 위험자산 바닥을 남겨 반등 참여를 보존</td></tr>
<tr><td>5</td><td><code>build_crash_features.py</code></td><td>hard 국면 비중과 과거 월·일 수익으로 국내 12개 입력 생성</td><td>목표 월 시작 이틀 전 cutoff는 미국 종가가 한국 첫 거래 뒤 확정되는 시차를 차단</td></tr>
<tr><td>6</td><td><code>openassetpricing_signal_experiment.py::build_oap_features</code></td><td>KOSPI200·16개 업종의 가격·거래량으로 29개 연구용 OAP 유사 열 생성</td><td>최소 756일, 업종 요약 최소 4개, CAPM 최소 40일로 불완전한 횡단면 계산을 억제</td></tr>
<tr><td>7</td><td><code>build_oap_composites</code></td><td>실제 사용 원시 열 22개를 인과 z-score 뒤 네 합성변수로 압축</td><td>유동성 합성은 6개 중 4개 이상이 있을 때만 값 생성</td></tr>
<tr><td>8</td><td><code>final_blend_crash_meta_experiment.py::make_model</code></td><td>중앙값 대치 → 표준화 → 균형 L2 로지스틱 파이프라인 생성</td><td>하이퍼파라미터 그리드는 이 함수에 없으며 고정값을 반환</td></tr>
<tr><td>9</td><td><code>walk_forward_probability_embargo</code></td><td>매달 과거만으로 다시 적합하고 2개월 경로손실 확률 출력</td><td>라벨이 소비하는 미래 2개월과 같은 수의 마지막 행을 빼 학습·예측의 미래수익 중첩 차단</td></tr>
<tr><td>10</td><td><code>choose_medium_horizon_oap</code></td><td>4개 목표 × 4개 최대 이동을 2017년 이전 성과로 비교</td><td>CAGR·Sharpe·MDD를 모두 넘는 후보만 남기고 Calmar→Sharpe→CAGR 순으로 결정</td></tr>
<tr><td>11</td><td><code>market_structure_robustness.py::run_factor_vol_target</code></td><td>hard 40%·SLSQP 60% 기준, 꼬리점수 이동, 15% 변동성 목표, 비용 차감</td><td>최근 24개월에 <code>exp(-2…0)</code> 가중을 주고 레버리지를 0.50~1.50으로 절삭</td></tr>
<tr><td>12</td><td><code>vkospi_feature_experiment.py</code></td><td>VKOSPI 머신러닝 후보를 기준 OAP 경로와 비교</td><td>잠금 승격 실패 시 이전 기준을 그대로 둔다는 주석대로 <code>ReferenceMediumHorizonOAPVol15</code> 유지</td></tr>
<tr><td>13</td><td><code>vkospi_dynamic_risk_experiment.py::prepare_arrays</code></td><td>t일 이전 마지막 VKOSPI 신호를 t일 오픈-투-오픈 자산수익에 정렬</td><td><code>cutoff = date − 1일</code>과 테스트가 같은 날·미래 신호 사용을 금지</td></tr>
<tr><td>14</td><td><code>build_robust_daily_features</code></td><td>백분위·MAD z·정규화 충격·가속·고점거리 등 15개 열 생성</td><td>종가가 양수일 때만 로그를 쓰고 0분모·무한대·극단치를 NaN 또는 절삭으로 처리</td></tr>
<tr><td>15</td><td><code>stress_from_features</code></td><td>5개 연속 스트레스 모드와 0~1 절삭 구현</td><td>결측은 방어 0으로 돌리고, 252일 백분위가 없으면 126일로 대체</td></tr>
<tr><td>16</td><td><code>vkospi_robust_dynamic_experiment.py::main</code></td><td>810개 후보, 두 사전 창, 엄격 게이트, 다목적 순위, 잠금 평가 실행</td><td>2018년 이후는 winner를 고르는 코드에 들어가지 않고 마지막 평가에서만 읽음</td></tr>
<tr><td>17</td><td><code>simulate</code></td><td>월초 강제 재조정, 월중 밴드, 비용, 금융비용, 비중 drift를 일별 반영</td><td>첫 거래는 총 절대변화, 이후는 half-L1 회전율을 써 초기 포트폴리오 구축을 따로 처리</td></tr>
<tr><td>18</td><td><code>reconcile_to_monthly_reference</code></td><td>일별 오버레이의 상대배수만 월간 검증 경로에 적용</td><td>함수 docstring이 빈도 변환 차이를 알파로 세지 말라고 명시</td></tr>
<tr><td>19</td><td><code>paired_multiobjective_bootstrap</code></td><td>같은 6개월 블록으로 두 전략을 5,000회 함께 재표본</td><td>paired 표본이라 시장 국면 차이를 유지하고, CAGR·Sharpe·MDD 동시개선도 따로 셈</td></tr>
<tr><td>20</td><td><code>vkospi_robust_dynamic_attribution.py</code></td><td>순차 정책·성분 절삭·신호 통계·월 기여도 작성</td><td>선정 뒤의 잠금 자료를 설명에만 쓰며 winner 설정에는 되먹이지 않음</td></tr>
<tr><td>21</td><td><code>vkospi_extended_diagnostics.py</code></td><td>거시 상수 1변수 민감도, 16개 꼬리 설명변수, 2005 요청구간 가용성과 810개 오버피팅 위험을 감사</td><td>잠금 민감도는 재선정이 아니라 약점을 드러내는 사후 진단으로만 저장</td></tr>
</tbody></table></div>

<h3>실행 흐름을 의사코드로 묶으면</h3><div class="grid2"><div>__PIPELINE_CODE__</div><div class="panel"><h3>숫자를 재현할 때 지켜야 할 경계</h3><ul><li>거시 신호는 목표 월의 직전 월까지만 읽습니다.</li><li>OAP·국내 일별 특징은 목표 월 시작 이틀 전에서 자릅니다.</li><li>2개월 라벨은 직전 2개 학습 행을 embargo합니다.</li><li>VKOSPI 신호일은 자산 수익 발생일보다 반드시 빠릅니다.</li><li>후보 선정은 2017-12에서 끝나며 2018-01 이후는 잠금입니다.</li><li>잠금 절삭실험은 설명 자료이지 새 후보 탐색이 아닙니다.</li></ul></div></div>

<h3>원래 코드 주석 __COMMENT_COUNT__개를 역할별로 읽기</h3><p>아래 표는 핵심 소스의 독립 행 주석을 빠짐없이 모은 것입니다. 원문을 그대로 보존하고, 주석이 지키려는 경계를 한국어로 덧붙였습니다. 함수 docstring은 21단계 실행표에서 따로 설명했으며, 전체 본문은 바로 다음 접이식 원문에서 확인할 수 있습니다.</p>
<details><summary>코드 주석 전체 색인</summary><div class="table"><table><thead><tr><th>파일·줄</th><th>원래 주석</th><th>성격</th><th>읽는 기준</th></tr></thead><tbody>__COMMENT_INDEX_ROWS__</tbody></table></div></details>

<h3>전략 핵심 파일 전체 소스</h3><p>아래 __SOURCE_FILE_COUNT__개 파일은 일부 발췌가 아니라 현재 실행본 전체입니다. 각 줄 앞에 보고서용 줄 번호만 붙였고, 코드와 주석 내용은 바꾸지 않았습니다. 긴 파일은 접혀 있으므로 필요한 파일만 열어 볼 수 있습니다.</p>
__FULL_SOURCE_DETAILS__

<h3>코드로 확인한 것과 아직 확인하지 못한 것</h3><div class="grid2"><div class="panel"><h3>코드로 다시 확인할 수 있는 내용</h3><p>데이터 시점 정렬, 후보 수 810개, 엄격 통과 2개, 선택 규칙, 비용 산식, 103개월 성과, 5,000회 부트스트랩은 코드와 결과 파일을 다시 계산해 확인할 수 있습니다.</p></div><div class="panel"><h3>아직 최적성까지 확인하지 못한 설정</h3><p>로지스틱 <code>C=0.1</code>, OAP 합성 동일가중, 거시 10/90 혼합, SLSQP 목적함수 계수는 현재 결과만으로 최적성을 입증할 수 없습니다. 이 값들은 재현 가능한 고정 설계이지 통계적으로 유일한 답이 아닙니다.</p></div></div>
<div class="note">설정값 근거는 14절, 팩터 선택은 15절 앞부분, 수식 설계는 15절 상수표, 실행 코드와 원본 주석은 16절에서 확인할 수 있습니다.</div>
</section>
"""

    rebuttal_section = f"""
<section id="rebuttal"><div class="head"><small>18 · CHALLENGE & RESPONSE</small><div><h2>예상 반박 질문과 코드에 근거한 답변</h2><p class="lede">아래 답변은 전략을 방어하기 위한 홍보문이 아니라, 어디까지 말할 수 있고 어디서 멈춰야 하는지를 정리한 질의응답입니다. 반박이 타당한 경우에는 그대로 인정합니다.</p></div></div>

<details open><summary>Q1. 결국 2018년 이후를 보고 무SJM으로 바꾼 것 아닌가?</summary><p><b>맞습니다. 그래서 무SJM 경로를 신규 홀드아웃 승자라고 부르지 않습니다.</b> 무SJM은 잠금 결과를 확인한 뒤 수행한 post-lock ablation이고, 2007–2017 및 2013–2017 동시 관문도 실패했습니다. 현재 저장소의 비교 기준일 뿐, 사전선정 근거가 가장 강한 경로는 SJM 10% 기준입니다.</p></details>
<details><summary>Q2. 그렇다면 CAGR {p(current_full['CAGR'])}, Sharpe {current_full['Sharpe']:.3f}, MDD {p(current_full['MDD'])}를 믿으면 안 되는가?</summary><p>계산값 자체는 회귀 테스트와 재현 오차 검증을 통과했습니다. 다만 <b>측정값의 재현성</b>과 <b>미래 일반화</b>는 별개입니다. 이 숫자는 현재 자료에서의 사후 비교치이며, 2026-08 이후 새 데이터나 외부시장 사전등록 검증이 있어야 증거 수준이 올라갑니다.</p></details>
<details><summary>Q3. 학습을 하지 않는다면서 로지스틱 회귀를 매달 쓰는 것은 모순 아닌가?</summary><p>‘고정모형을 한 번도 적합하지 않는다’는 뜻이 아닙니다. 하이퍼파라미터와 입력정의는 고정하고, 매월 그 시점까지 확정된 과거만으로 로지스틱을 다시 적합하는 <b>확장형 walk-forward</b>입니다. 목표월 기준 두 달 embargo가 있어 아직 완성되지 않은 미래 라벨은 학습에 들어가지 않습니다.</p></details>
<details><summary>Q4. 거시 국면확률은 실제 전이확률인가?</summary><p>아닙니다. HMM의 상태전이행렬에서 나온 확률이 아닙니다. 현재 경로는 거시 z-score 합성을 sigmoid로 0~1에 매핑하고 전월 확률 15%를 섞은 <b>상태 점수</b>입니다. 네 국면확률은 성장·물가 두 확률의 곱으로 만든 결합확률입니다.</p></details>
<details><summary>Q5. hard 국면이 바뀌면 포트폴리오도 한 번에 바뀌지 않는가?</summary><p>hard 경로만 보면 그렇지만 최종 기준 비중에서 hard는 40%입니다. 나머지 60%는 네 국면확률 전체를 쓰는 SLSQP 경로이고, 이후 꼬리위험 이동·변동성 목표·거래비용·VKOSPI 20% 무거래 밴드가 적용됩니다. 실제 232개월 동안 hard 이름은 {regime_transitions}번 바뀌었지만 전체 비중은 여러 완충층을 거칩니다.</p></details>
<details><summary>Q6. 로지스틱의 raw 확률이 Brier 기준으로 나쁜데 왜 사용하는가?</summary><p>타당한 지적입니다. class-balanced 로짓은 희귀사건 탐지에 초점을 두므로 절대확률 보정이 좋지 않을 수 있고, 사전 AUC도 낮았습니다. 그래서 raw 확률을 실제 발생확률로 쓰지 않고 과거 60개월 내 순위로 변환합니다. 그래도 사전 사건이 5개뿐이라는 약점은 사라지지 않으며, 확률보정 모델은 앞으로 별도 사전검증해야 합니다.</p></details>
<details><summary>Q7. 잠금 AUC가 0.836이면 예측력이 증명된 것 아닌가?</summary><p>아닙니다. 잠금 102개 관측과 8개 사건에서 나온 결과이고, 이미 여러 후속 연구를 거친 저장소입니다. 좋은 잠금 AUC는 긍정적 증거지만 다중시도와 작은 사건 수를 지우지 못합니다. 또한 예측 AUC가 좋아도 거래비용과 비중매핑 후 포트폴리오 성과가 자동으로 좋아지는 것은 아닙니다.</p></details>
<details><summary>Q8. C=.1과 balanced L2는 왜 최적인가?</summary><p>‘최적’이라고 말할 수 없습니다. 28개 후보 강건성 감사에서 이 후보만 두 사전 창의 포트폴리오 관문을 통과했고 사전·잠금 포트폴리오 순위가 모두 1위였지만, 사전 Sharpe 선택 CSCV/PBO는 {p(model_robustness['logistic']['prelock_portfolio_sharpe_pbo']['pbo'], 1)}로 높았습니다. 재현 가능한 고정값이지 유일한 통계적 해답은 아닙니다.</p></details>
<details><summary>Q9. 복잡한 LightGBM이 로지스틱보다 낫지 않은가?</summary><p>이 월간 표본은 실질 적합확률이 194개이고 사건 수도 적습니다. Stage 04에서 LightGBM을 비교했지만 선택된 위험이동은 <code>max_shift=0</code>이어서 승격되지 않았습니다. 복잡한 모델이 표현력은 높아도 이 표본에서는 분산과 선택편향이 더 큰 문제가 됐습니다.</p></details>
<details><summary>Q10. 810개 VKOSPI 후보는 전형적인 데이터 마이닝 아닌가?</summary><p>그 위험이 큽니다. 후보선정은 2017년 이전으로 제한했고 두 사전 창에서 CAGR·Sharpe·MDD 동시 개선을 요구했지만, 2013–2017은 2007–2017에 포함돼 독립 표본이 아닙니다. 엄격 통과가 2개뿐이어도 우연 가능성은 남습니다. 잠금 5,000회 블록 부트스트랩에서 세 지표 동시 개선 확률도 {p(boot['probability_all_three_improve'], 1)}에 그쳤습니다.</p></details>
<details><summary>Q11. Robust 오버레이의 개선 폭이 너무 작은 것 아닌가?</summary><p>그렇습니다. 사전선정 SJM 10% 경로에서 Robust 오버레이 자체의 잠금 개선은 CAGR 약 +0.14%p, Sharpe +0.034, MDD 약 +0.16%p입니다. 이는 독립 알파 전략이 아니라 기존 포트폴리오의 월중 방어 실행을 조금 다듬은 결과입니다. 과장해서는 안 됩니다.</p></details>
<details><summary>Q12. 그런데 현재 무SJM 경로의 잠금 개선은 왜 훨씬 큰가?</summary><p>그 차이는 Robust VKOSPI 오버레이만의 효과가 아닙니다. SJM 제거로 거시 확률·hard 기준 포트폴리오·로지스틱 입력과 라벨이 함께 바뀝니다. 따라서 무SJM 대비 SJM 10% 차이를 ‘VKOSPI 알파’로 귀속하면 잘못입니다.</p></details>
<details><summary>Q13. 왜 방어 이전분을 채권이 아니라 GLD에 전부 보내는가?</summary><p><code>bond_share=0</code>은 810개 사전 탐색의 승자 설정입니다. 위험자산 KODEX200·USO에서 줄인 비중을 GLD로 보냈을 때 두 사전 창의 관문을 통과했습니다. 다만 미래에도 금이 항상 주식·원유 스트레스의 헤지라는 보장은 없고, 금리·달러 국면에서는 실패할 수 있습니다.</p></details>
<details><summary>Q14. 왜 VKOSPI가 높은 것만 보지 않고 충격과 가속을 섞는가?</summary><p>높은 수준만 쓰면 위기가 이미 오래 진행된 때 계속 방어해 반등을 놓칠 수 있습니다. 5일 shock은 당시 변동성 대비 이례성을, acceleration은 최근 5일 공포상승이 직전 5일보다 빨라졌는지를 봅니다. 가속은 독립 알파보다는 방어 진입 확인항입니다.</p></details>
<details><summary>Q15. 20% 무거래 밴드가 왜 중요한가?</summary><p>같은 robust 신호를 기존 15% 정책에 넣었을 때 잠금 CAGR과 Sharpe가 오히려 낮아졌습니다. 최대 이전을 35%로 올리는 것만으로도 부족했고, 20% 밴드까지 적용했을 때 세 지표가 함께 개선됐습니다. 신호보다 <b>거래로 번역하는 빈도</b>가 성과를 갈랐다는 뜻입니다.</p></details>
<details><summary>Q16. 비용 가정이 너무 낙관적인 것 아닌가?</summary><p>매매 15bp, 달러자산 순비중 변화 환전 5bp, 차입 연 4%를 반영했습니다. 비용 2배에서도 Robust 오버레이의 상대 우위는 남았지만 세금, 시장충격, 상품 롤비용, 실시간 슬리피지, 대규모 운용 제약은 포함하지 않았습니다. 실거래 기대수익은 백테스트보다 낮게 보는 것이 맞습니다.</p></details>
<details><summary>Q17. 레버리지 1.5배는 MDD를 과소평가하지 않는가?</summary><p>레버리지는 24개월 가중 변동성으로 15% 목표를 맞추며 0.5–1.5로 제한되고 연 4% 금융비용을 냅니다. 그래도 월말 MDD는 일중 마진콜과 갭 위험을 잡지 못합니다. 따라서 −12.96%를 실거래 최대손실의 상한으로 해석하면 안 됩니다.</p></details>
<details><summary>Q18. 매크로 데이터 개정과 발표시차 누수는 없는가?</summary><p>코드는 GDP·수출·BSI를 한 달, 물가자료를 두 월말 밀고 목표월 전 신호만 사용합니다. 그러나 현재 파일은 당시 공개본을 보존한 point-in-time vintage 데이터베이스가 아닙니다. 발표시차는 모사했지만 사후 개정값 위험까지 제거했다고 말할 수 없습니다.</p></details>
<details><summary>Q19. VKOSPI 신호에 당일 종가를 쓰고 당일 수익을 먹는 누수는 없는가?</summary><p>배열 단언과 테스트에서 <code>signal_date &lt; return_date</code>를 확인합니다. t일 종가로 만든 신호는 t+1일 오픈부터 적용됩니다. 월별 가격 특징도 목표월 시작 이틀 전에서 끊어 미국장 종가 시차를 피합니다.</p></details>
<details><summary>Q20. 2005년부터 결과를 왜 보여주지 않는가?</summary><p>네 자산의 공통 월수익은 USO와 채권자료 때문에 2006-04부터 가능하고 거시 국면은 24개월 워밍업이 필요합니다. 같은 전략의 첫 거래월은 2007-04입니다. 2005년을 현금이나 다른 지수로 채우면 별도 프록시 전략이므로 이어 붙이지 않았습니다.</p></details>
<details><summary>Q21. VIX6 decomposition과 옵션은 왜 최종 비중에 없는가?</summary><p>VIX6와 KOSPI200 옵션을 신호·자산으로 비교했지만 사전 강건성 및 슬리피지 관문을 통과한 옵션 후보가 없었습니다. 그래서 현재 옵션 비중은 0%입니다. ‘아이디어가 틀렸다’가 아니라 현재 데이터·비용 가정에서 승격 증거가 부족했다는 결론입니다.</p></details>
<details><summary>Q22. 이 전략을 실전 배포해도 되는가?</summary><p>연구 프로토타입으로는 재현 가능하지만 즉시 실전 배포를 정당화하지는 않습니다. 필요한 다음 단계는 point-in-time 거시자료, 2026-08 이후 사전등록 검증, 주문·세금·롤비용을 포함한 페이퍼 트레이딩, 포지션·손실 한도, 데이터 결측 시 fallback, 모델 드리프트 경보입니다.</p></details>

<div class="finding"><b>답변 원칙:</b> 계산된 성과는 숨기지 않되, 사전선정과 사후분석을 구분하고, raw 확률을 보정확률로 과장하지 않으며, 작은 표본·다중탐색·데이터 개정·실거래 비용을 한계로 먼저 인정합니다. 이 원칙을 지켜야 보고서 수치가 반박 질문을 견딜 수 있습니다.</div>
</section>
"""

    template = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VKOSPI Robust Dynamic · 코드와 효과의 기술 해설</title><style>
:root{--ink:#17211d;--muted:#64716b;--paper:#f3f0e8;--card:#fffdf7;--line:#d8d3c7;--green:#087f5b;--green2:#22a77a;--mint:#e1f2ea;--orange:#d87931;--navy:#213940;--red:#b64d43;--code:#182726}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Apple SD Gothic Neo","Noto Sans KR",sans-serif;line-height:1.68}a{color:var(--green)}code{font-family:"Cascadia Code",Consolas,monospace}.wrap{width:min(1180px,calc(100% - 40px));margin:auto}
.top{display:flex;justify-content:space-between;align-items:center;padding:17px 0;font-size:13px}.top nav a{margin-left:18px;text-decoration:none;color:var(--ink)}header{border-block:1px solid var(--line);background:radial-gradient(circle at 85% 15%,rgba(8,127,91,.2),transparent 31%),linear-gradient(135deg,#fffaf0,#e7f2eb)}.hero{padding:78px 0 70px}.eyebrow{color:var(--green);font-weight:800;letter-spacing:.14em;text-transform:uppercase}.hero h1{font:clamp(43px,7vw,78px)/1.04 Georgia,"Noto Serif KR",serif;letter-spacing:-.055em;max-width:980px;margin:15px 0 22px}.hero p{font-size:19px;color:#3f5048;max-width:840px}.stamp{display:inline-block;border:1px solid #91c6b2;border-radius:999px;padding:9px 14px;background:#ffffff9c;color:var(--green);font-weight:800}
main{padding:64px 0 100px}section{margin-bottom:78px}.head{display:grid;grid-template-columns:160px 1fr;gap:22px;margin-bottom:28px}.head small{color:var(--green);font-weight:800;letter-spacing:.1em}.head h2{font:clamp(29px,4vw,43px)/1.16 Georgia,"Noto Serif KR",serif;letter-spacing:-.035em;margin:0}.lede{font-size:17px;color:#4e5d56;max-width:850px;margin-bottom:0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.metric,.stat{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:22px}.metric span,.stat span{font-size:13px;color:var(--muted);letter-spacing:.07em}.metric strong{display:block;font:34px Georgia,serif;margin:7px 0}.metric div,.stat div{display:flex;justify-content:space-between;align-items:baseline}.metric b,.stat strong,.up{color:var(--green)}.metric s,.stat s{color:#8d9692}.metric em,.stat em{font-style:normal;color:var(--green);font-weight:800;font-size:13px}.note,.finding,.warning{padding:17px 19px;border-radius:12px;margin-top:18px}.note{background:#fff8eb;border-left:3px solid var(--orange)}.finding{background:var(--mint);border-left:3px solid var(--green)}.warning{background:#fff3ee;border:1px dashed #c76e4b}.warning b{color:#a54a31}
.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;counter-reset:item}.flow article{background:var(--navy);color:white;padding:20px 15px;border-radius:13px;min-height:176px}.flow article:before{counter-increment:item;content:"0" counter(item);font:700 12px Georgia;color:#82d2b7}.flow h3{font-size:16px;margin:18px 0 7px}.flow p{font-size:13px;color:#d4dfdb;margin:0}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid2>*{min-width:0}.panel{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:24px}.panel h3{margin-top:0}.panel ul{padding-left:20px}.panel li{margin:8px 0}.equation{font-family:"Cascadia Code",Consolas,monospace;background:#eef5f1;border:1px solid #cdded6;padding:14px;border-radius:10px;overflow:auto}.callout{display:grid;grid-template-columns:40px minmax(0,1fr);gap:13px;align-items:start;margin:15px 0;min-width:0}.callout i{display:grid;place-items:center;width:36px;height:36px;border-radius:50%;background:var(--mint);color:var(--green);font-style:normal;font-weight:900}.callout p{margin:0;min-width:0}p code,li code,td code,.callout code{overflow-wrap:anywhere;word-break:break-word}
pre{background:var(--code);color:#d8ebe4;border-radius:13px;padding:19px;overflow:auto;font-size:13px;line-height:1.6;border:1px solid #314744}.table{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}table{width:100%;border-collapse:collapse;min-width:720px}th,td{text-align:left;padding:13px 15px;border-bottom:1px solid #e7e3d9;font-size:14px}th{background:#f8f5ed;color:var(--muted);font-size:12px;letter-spacing:.05em}tr:last-child td{border-bottom:0}.winner-row{background:#eaf6f0}.winner-row td:first-child{border-left:3px solid var(--green)}.down{color:var(--red)}.tag{display:inline-block;padding:3px 8px;background:var(--mint);border-radius:999px;color:var(--green);font-size:12px;font-weight:800}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:13px}.stat strong{font:27px Georgia,serif}.stat em{display:block;margin-top:9px}.chart-grid{display:grid;gap:18px}.chart-card{margin:0;background:var(--card);border:1px solid var(--line);padding:20px;border-radius:15px;overflow:hidden}.chart-card figcaption{display:flex;justify-content:space-between}.chart-card figcaption span{font-size:13px;color:var(--muted)}.chart-card svg{width:100%;height:auto;margin-top:12px}.grid{stroke:#e5e1d7;stroke-width:1}.axis{font-size:11px;fill:#7d8983}.zero{stroke:#aab2ae;stroke-dasharray:4 4}.line{fill:none;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}.reference{stroke:#9ba5a0}.dynamic,.relative-line{fill:none;stroke:var(--green);stroke-width:3;stroke-linejoin:round}.legend{display:flex;justify-content:flex-end;gap:18px;color:var(--muted);font-size:13px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.ref-dot{background:#9ba5a0}.dyn-dot{background:var(--green)}details{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:15px 18px;margin:11px 0;min-width:0}summary{cursor:pointer;font-weight:800;color:var(--green)}summary code{overflow-wrap:anywhere;word-break:break-word}.files td:first-child{min-width:310px}.download{display:grid;grid-template-columns:1fr auto;gap:22px;align-items:center;background:var(--ink);color:white;border-radius:19px;padding:31px}.download h2{font:32px Georgia,serif;margin:0}.download p{color:#cad5d0}.download a{display:block;background:#28c490;color:#06271c;padding:11px 16px;text-decoration:none;border-radius:9px;font-weight:800;text-align:center;margin:7px}footer{border-top:1px solid var(--line);padding:30px 0 50px;color:var(--muted);font-size:13px}
@media(max-width:980px){.flow{grid-template-columns:repeat(3,1fr)}.head{grid-template-columns:1fr}.metrics,.stats{grid-template-columns:repeat(2,1fr)}.grid2,.download{grid-template-columns:1fr}.top nav{display:none}}@media(max-width:560px){.wrap{width:calc(100% - 24px)}.hero{padding:52px 0}.metrics,.stats,.flow{grid-template-columns:1fr}.metric strong{font-size:29px}.files td:first-child{min-width:240px}}@media print{body{background:white}.top nav,.download{display:none}section{break-inside:avoid}.chart-card,pre{break-inside:avoid}}
</style></head><body>
<div class="wrap top"><b>REGIME DECISION TEST · TECHNICAL REPORT</b><nav><a href="#architecture">구조</a><a href="#features">VKOSPI</a><a href="#non-vkospi">분류</a><a href="#effect">효과</a><a href="#evidence">숫자</a><a href="#hyperparameters">로지스틱</a><a href="#code-trace">코드</a><a href="#files">파일</a></nav></div>
<header><div class="wrap hero"><span class="eyebrow">VKOSPI Robust Dynamic · Code & Attribution</span><h1>코드를 따라가며 살펴본<br>성과 개선의 이유</h1><p>소스 파일을 따라가며 VKOSPI 재가공 전략을 뜯어봤습니다. 데이터 정렬과 특징 산식부터 810개 후보 선택, 일별 체결, 월별 재조정, 절삭실험과 한계까지 차례로 짚습니다.</p><span class="stamp">2018–2026 잠금 · CAGR 18.44% · Sharpe 1.363 · MDD −11.13%</span></div></header>
<main class="wrap">
<section><div class="head"><small>01 · RESULT</small><div><h2>먼저 개선 폭부터 확인하기</h2><p class="lede">기존 VKOSPI 동적 전략을 기준으로 동일한 103개월과 동일한 월간 기준 경로를 사용했습니다.</p></div></div><div class="metrics">
<article class="metric"><span>CAGR</span><strong>__NEW_CAGR__</strong><div><s>__OLD_CAGR__</s><em>__D_CAGR__</em></div></article>
<article class="metric"><span>Sharpe</span><strong>__NEW_SHARPE__</strong><div><s>__OLD_SHARPE__</s><em>__D_SHARPE__</em></div></article>
<article class="metric"><span>MDD</span><strong>__NEW_MDD__</strong><div><s>__OLD_MDD__</s><em>__D_MDD__</em></div></article>
<article class="metric"><span>Calmar</span><strong>__NEW_CALMAR__</strong><div><s>__OLD_CALMAR__</s><em>__D_CALMAR__</em></div></article></div>
<div class="finding">성과는 가속 특징 하나만으로 설명되지 않습니다. VKOSPI 고점을 더 엄격하게 가려내고 충격을 변동성으로 나눠 강한 방어 신호를 줄였습니다. 극단 상황에서만 이전폭을 키웠고, 20% 무거래 밴드로 월중 소음도 눌렀습니다. 잠금 구간에서 확인한 개선은 이 <b>신호와 실행 정책의 조합</b>에서 나왔습니다.</div></section>

<section id="architecture"><div class="head"><small>02 · ARCHITECTURE</small><div><h2>원자료에서 최종 월수익까지, 여섯 단계</h2></div></div><div class="flow"><article><h3>VKOSPI 로드</h3><p>CSV 날짜·숫자 정제, 중복 제거, OHLC 보완.</p></article><article><h3>인과 특징</h3><p>t일까지의 126/252일 상대수준, 충격과 가속.</p></article><article><h3>신호 정렬</h3><p>t일 신호를 t+1일 오픈-투-오픈 수익에 연결.</p></article><article><h3>스트레스</h3><p>수준 40% + 충격 35% + 가속 25%.</p></article><article><h3>일별 체결</h3><p>주식·원유 일부를 GLD로 옮기고 비용 차감.</p></article><article><h3>월별 재조정</h3><p>검증된 월수익에 일별 오버레이 상대효과만 결합.</p></article></div>
<div class="grid2" style="margin-top:18px"><div class="panel"><h3>파일별 역할</h3><div class="callout"><i>A</i><p><code>vkospi_dynamic_risk_experiment.py</code>가 공통 데이터·시뮬레이션 엔진입니다.</p></div><div class="callout"><i>B</i><p><code>vkospi_robust_dynamic_experiment.py</code>가 새 특징과 탐색·선정을 담당합니다.</p></div><div class="callout"><i>C</i><p><code>vkospi_robust_dynamic_attribution.py</code>는 선정 결과를 그대로 둔 채 효과만 진단합니다.</p></div></div><div class="panel"><h3>시간축 규칙</h3><p>배열의 <code>signal_date</code>는 반드시 수익 발생일 <code>date</code>보다 작습니다. t일 종가로 만든 VKOSPI 신호가 t+1일 오픈 가격부터 적용되므로 당일 종가를 당일 수익에 사용하는 누수가 없습니다.</p><div class="equation">signal_date(t) &lt; return_date(t)<br>feature_t → weights_{t+1 open} → return_{t+1 open→t+2 open}</div></div></div></section>

<section id="features"><div class="head"><small>03 · FEATURES</small><div><h2>생성한 특징과 최종식에 반영된 특징</h2><p class="lede">특징을 만들었다고 해서 모두 최종식에 들어가는 것은 아닙니다. 둘을 구분하지 않으면 성과의 원인을 잘못 짚기 쉽습니다.</p></div></div><div class="grid2"><div class="panel"><h3>최종식에 반영된 특징</h3><ul><li><b>252일 인과적 백분위</b>: 현재 VKOSPI가 과거 1년 중 어디에 위치하는지.</li><li><b>5일 변동성 정규화 충격</b>: 같은 10% 상승도 평온기와 고변동기를 다르게 평가.</li><li><b>5일 가속</b>: 최근 5일 상승이 직전 5일보다 더 빨라졌는지.</li></ul></div><div class="panel"><h3>생성했지만 최종식에는 빠진 특징</h3><ul><li>63/252일 robust z-score</li><li>10/21일 정규화 충격</li><li>5/21일 상승일 비율과 fast-minus-slow</li></ul><p>이 열들은 후속 실험을 염두에 두고 만들었지만, 현재 5개 스트레스 모드에서 고른 최종식에는 들어가지 않습니다. 따라서 지금 성과를 설명하는 근거로 삼지 않습니다.</p></div></div>
<h3>전략이 실제로 읽는 데이터 입력</h3><div class="table"><table><thead><tr><th>입력 파일</th><th>읽는 열</th><th>쓰임</th></tr></thead><tbody>
<tr><td><code>raw_data/VKOSPIData.csv</code></td><td><code>date</code>, <code>close</code>, <code>change</code>, <code>return_pct</code>, <code>open</code>, <code>high</code>, <code>low</code></td><td><code>date</code>는 시점 정렬, <code>close</code>·<code>high</code>·<code>low</code>는 특징 계산에 사용. 나머지 세 가격 열은 정제만 하고 현재 산식에는 넣지 않음.</td></tr>
<tr><td><code>cache/market_daily.csv</code></td><td><code>date</code>, <code>symbol</code>, <code>open</code>, <code>close</code></td><td>KODEX200·GLD·USO의 시가와 USDKRW 종가를 읽어 원화 기준 일별 자산 수익률 계산.</td></tr>
<tr><td><code>raw_data/compass.db</code></td><td><code>symbol</code>, <code>date</code>, <code>open</code>, <code>close</code></td><td>실제 KODEX200 시세가 시작되기 전 구간을 symbol 1028 프록시로 연결.</td></tr>
<tr><td><code>raw_data/krx_bond_index.csv</code></td><td>날짜, 채권지수 수준</td><td>BOND 일별 수준과 오픈-투-오픈 수익률 계산.</td></tr>
<tr><td><code>results/vkospi_selected_backtest.csv</code></td><td><code>w_KODEX200</code>, <code>w_BOND</code>, <code>w_GLD</code>, <code>w_USO</code>, <code>return</code></td><td>기준 전략의 월별 자산 비중과 재조정 성과 경로.</td></tr>
</tbody></table></div>
<h3>사용한 지표(입력변수) 전체 목록</h3><p>아래 표는 <code>build_robust_daily_features()</code>가 생성한 15개 열을 빠짐없이 정리한 것입니다. 내부 계산에는 <code>log_close = ln(close)</code>와 <code>log_return = Δlog_close</code>도 사용합니다.</p>
<div class="table"><table><thead><tr><th>구분</th><th>변수명</th><th>산식·의미</th><th>최종식 사용</th></tr></thead><tbody>
<tr><td>원자료</td><td><code>close</code></td><td>VKOSPI 종가. 로그 수준과 모든 기간 변화의 출발점.</td><td>간접 사용</td></tr>
<tr><td>상대 수준</td><td><code>percentile_126</code></td><td>126일 인과적 백분위, 최소 84개 관측치. 252일 값이 없을 때 대체.</td><td>보조</td></tr>
<tr><td>상대 수준</td><td><code>percentile_252</code></td><td>252일 인과적 백분위, 최소 126개 관측치.</td><td><b>직접 사용</b></td></tr>
<tr><td>강건 수준</td><td><code>robust_z_63</code></td><td>로그 VKOSPI의 63일 median·MAD z-score, 최소 42개 관측치, ±6 절삭.</td><td>미사용</td></tr>
<tr><td>강건 수준</td><td><code>robust_z_252</code></td><td>로그 VKOSPI의 252일 median·MAD z-score, 최소 126개 관측치, ±6 절삭.</td><td>미사용</td></tr>
<tr><td>충격</td><td><code>shock_5</code></td><td>5일 로그 변화 ÷ (후행 63일 일간 변동성 × √5), ±8 절삭.</td><td><b>직접 사용</b></td></tr>
<tr><td>충격</td><td><code>shock_10</code></td><td>10일 로그 변화 ÷ (후행 63일 일간 변동성 × √10), ±8 절삭.</td><td>미사용</td></tr>
<tr><td>충격</td><td><code>shock_21</code></td><td>21일 로그 변화 ÷ (후행 63일 일간 변동성 × √21), ±8 절삭.</td><td>미사용</td></tr>
<tr><td>경로</td><td><code>acceleration_5</code></td><td>최근 5일 로그 변화 − 직전 5일 로그 변화.</td><td>중간 계산</td></tr>
<tr><td>경로</td><td><code>acceleration_z5</code></td><td><code>acceleration_5</code> ÷ (후행 63일 변동성 × √5), ±8 절삭.</td><td><b>직접 사용</b></td></tr>
<tr><td>경로</td><td><code>distance_high21</code></td><td>종가 ÷ 최근 21일 고가 − 1. 고점에서 얼마나 내려왔는지 측정.</td><td>다른 모드만 사용</td></tr>
<tr><td>경로</td><td><code>close_location21</code></td><td>최근 21일 고가·저가 범위 안에서 종가의 위치, 0~1.</td><td>다른 모드만 사용</td></tr>
<tr><td>방향 확인</td><td><code>positive_fraction5</code></td><td>최근 5일 중 로그수익률이 양수인 날의 비율.</td><td>현재 5개 모드 미사용</td></tr>
<tr><td>방향 확인</td><td><code>positive_fraction21</code></td><td>최근 21일 중 로그수익률이 양수인 날의 비율.</td><td>현재 5개 모드 미사용</td></tr>
<tr><td>속도 차이</td><td><code>fast_slow</code></td><td>5일 로그 변화 − (5/21 × 21일 로그 변화).</td><td>현재 5개 모드 미사용</td></tr>
</tbody></table></div>
<div class="note"><b>최종식과 보조 입력의 경계</b> 최종 <code>acceleration</code> 모드는 <code>percentile_252</code>(결측 시 <code>percentile_126</code>), <code>shock_5</code>, <code>acceleration_z5</code>만 사용합니다. 자산 수익 계산에는 KODEX200·BOND·GLD·USO의 일별 시가와 월별 기준 비중이 별도 입력으로 들어갑니다.</div>
<h3>시기마다 다른 변동성 수준을 맞추기</h3>__FEATURE_CODE__
<div class="note"><b>정규화가 필요한 이유</b> VKOSPI의 5일 +10%는 평온기에는 이례적이지만 위기 중에는 흔할 수 있습니다. 그래서 후행 63일 표준편차로 나눠, 절대 변화율이 아니라 당시 시장에서 얼마나 이례적인 움직임이었는지를 봅니다. median·MAD는 이상치의 영향을 덜 받지만 현재 최종식에는 직접 들어가지 않습니다.</div></section>

__NON_VKOSPI_SECTION__

<section><div class="head"><small>05 · STRESS</small><div><h2>선정된 스트레스 산식</h2></div></div><div class="grid2"><div>__FORMULA_CODE__</div><div class="panel"><h3>해석</h3><p><b>수준 문턱 0.90</b>은 최근 252일 상위 10%부터 방어 강도가 생긴다는 뜻입니다.</p><p><b>충격 문턱 1.0σ</b>는 5일 VKOSPI 상승이 후행 변동성 대비 1표준편차를 넘을 때만 반응합니다.</p><p><b>가속 25%</b>는 높은 수준만으로 방어하지 않고 공포가 더 빨라지는지를 확인합니다.</p><p>각 성분과 최종 스트레스는 <code>[0, 1]</code>로 잘라 극단치가 비중을 폭주시킬 수 없게 합니다.</p></div></div></section>

<section><div class="head"><small>05 · SEARCH</small><div><h2>2017년 이전에만 수행한 810개 탐색</h2></div></div><div class="grid2"><div class="panel"><h3>후보 공간</h3><div class="equation">5 modes × 3 level thresholds × 3 shock thresholds<br>× 3 transfer limits × 2 defensive allocations × 3 bands<br>= 810 candidates</div><ul><li>모드: mean, max, confirmation, acceleration, exhaustion-adjusted</li><li>수준: 0.70 / 0.80 / 0.90</li><li>충격: 0 / 0.5 / 1.0σ</li><li>최대 이전: 15% / 25% / 35%</li><li>방어: GLD 전부 또는 GLD·채권 혼합</li><li>밴드: 10% / 15% / 20%</li></ul></div><div class="panel"><h3>통과 규칙</h3><p>2007–2017 전체와 2013–2017 내부검증에서 기존 동적 전략보다 <b>CAGR &gt; 0, Sharpe &gt; 0, MDD ≥ 0</b>의 변화가 동시에 필요한 엄격 조건입니다.</p><p>810개 중 <b>__STRICT_COUNT__개</b>만 통과했고, 모두 <code>acceleration / shock 1σ / transfer 35% / GLD / band 20%</code> 구조였습니다. 수준 문턱 0.90 후보가 사전 다목적 순위에서 0.80보다 높아 선택됐습니다.</p><p><b>2018년 이후는 이 선택에 사용하지 않았습니다.</b></p></div></div>
<details><summary>선정 후보의 사전 개선 폭</summary><div class="table"><table><thead><tr><th>수준 문턱</th><th>2007–2017 ΔCAGR / ΔSharpe / ΔMDD</th><th>2013–2017 ΔCAGR / ΔSharpe / ΔMDD</th></tr></thead><tbody>__STRICT_ROWS__</tbody></table></div></details></section>

<section><div class="head"><small>06 · PORTFOLIO</small><div><h2>스트레스를 실제 자산 비중으로 바꾸는 법</h2></div></div><div class="grid2"><div>__ALLOCATION_CODE__</div><div class="panel"><h3>실제로 비중이 바뀌는 방식</h3><ul><li>기존 KODEX200과 USO 비중의 동일 비율을 줄입니다.</li><li><code>bond_share=0</code>이므로 줄인 비중은 전부 GLD로 갑니다.</li><li>최대 스트레스 1에서도 각 위험자산의 35%만 줄이며 전량 청산하지 않습니다.</li><li>월초에는 기준 전략의 새 월간 비중으로 반드시 맞춥니다.</li><li>월중에는 목표와 현재 비중 차이의 half-L1 turnover가 20% 이상일 때만 거래합니다.</li></ul></div></div>
<div class="grid2"><div class="panel"><h3>비용</h3><div class="equation">trade_cost = Σ|Δweight| × 0.15%<br>fx_cost = |Δ(GLD + USO)| × 0.05%</div><p>첫 거래 이후 회전율은 half-L1 방식으로 측정합니다. 비용 2배 잠금에서도 Sharpe <b>__COST2_OLD_SHARPE__ → __COST2_NEW_SHARPE__</b>, MDD <b>__COST2_OLD_MDD__ → __COST2_NEW_MDD__</b>로 우위가 유지됐습니다.</p></div><div class="panel"><h3>레버리지 금융비용</h3><p>기준 비중 합이 1을 넘으면 <code>debt_weight = 1 - Σweights</code>로 차입을 표현하고 연 4%를 일복리로 적용합니다. 따라서 높은 명목 비중을 무비용 레버리지로 취급하지 않습니다.</p></div></div></section>

<section><div class="head"><small>07 · RECONCILE</small><div><h2>빈도 차이를 성과로 오인하지 않기</h2></div></div><div class="grid2"><div>__RECONCILIATION_CODE__</div><div class="panel"><h3>재조정이 필요한 이유</h3><p>기존 월간 엔진을 일간으로 재구성하면 금융비용과 거래비용의 복리 시점 때문에 작은 차이가 생깁니다. 이 차이를 새 VKOSPI 알파로 세지 않도록, 일간 오버레이의 <b>중립 일간 경로 대비 상대수익 배수만</b> 검증된 월간 기준에 곱합니다.</p><p>동시에 실제 일별 경로 성과도 별도로 보고해 재조정 결과만 의존하지 않습니다.</p></div></div></section>

<section id="effect"><div class="head"><small>08 · ATTRIBUTION</small><div><h2>성과 개선은 어디에서 나왔나</h2><p class="lede">아래는 2018년 이후를 사용한 사후 설명용 절삭실험입니다. 선정 결과를 다시 고르는 용도가 아닙니다.</p></div></div><div class="table"><table><thead><tr><th>순차 적용</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th><th>월평균 회전율</th></tr></thead><tbody>__STEPWISE_ROWS__</tbody></table></div>
<div class="warning">표에서 눈여겨볼 부분은 ②와 ④의 차이입니다. 새 신호를 기존 15% 밴드에만 적용한 ②는 MDD를 줄였지만 CAGR과 Sharpe를 낮췄습니다. 최대 이전을 35%로 올린 ③도 충분하지 않았습니다. CAGR 18.44%, Sharpe 1.363, MDD −11.13%가 함께 개선된 것은 <b>20% 밴드까지 적용한 ④</b>뿐입니다. 신호를 얼마나 자주 거래로 옮길지 정한 20% 밴드가 결과를 갈랐습니다.</div>
<h3>신호 행동은 어떻게 달라졌나</h3><div class="stats">__SIGNAL_STATS__</div><div class="finding">새 신호는 양(+)의 스트레스가 발생한 날은 조금 많지만, 스트레스 0.25 이상인 날은 <b>466일 → 255일</b>로 크게 줄었습니다. 평균 위험 이전도 2.91% → 2.60%로 낮아졌고, 최대 35% 이전은 극단적 공포에만 남았습니다. 자주 살피되 약하게 반응하고, 공포가 뚜렷할 때만 크게 움직이는 구조입니다.</div></section>

<section><div class="head"><small>09 · COMPONENTS</small><div><h2>수준·충격·가속의 역할</h2></div></div><div class="table"><table><thead><tr><th>성분</th><th>2007–2017 CAGR / Sharpe / MDD</th><th>2018–2026 CAGR / Sharpe / MDD</th><th>잠금 Calmar</th></tr></thead><tbody>__COMPONENT_ROWS__</tbody></table></div>
<div class="note">잠금 구간만 놓고 보면 수준+충격 조합의 CAGR·Sharpe가 최종식보다 높았습니다. 그러나 이 결과는 잠금 구간을 확인한 뒤 얻은 사후 결과입니다. 이 조합으로 전략을 바꿔서는 안 됩니다. 2017년 이전의 엄격한 기준으로 고른 최종식은 잠금 구간에서 수준+충격 조합보다 MDD를 약 0.04%p 더 줄이는 대신 CAGR 약 0.18%p를 내줬습니다. 가속은 독립적인 알파라기보다 <b>방어 진입을 한 번 더 확인하는 항</b>에 가깝습니다.</div></section>

<section><div class="head"><small>10 · MONTHS</small><div><h2>기존 전략과 수익률이 크게 벌어진 달</h2></div></div>__RELATIVE_CHART__<div class="grid2"><div><h3>상대성과 상위 5개월</h3><div class="table"><table><thead><tr><th>월</th><th>기존</th><th>최종</th><th>차이</th><th>평균 스트레스</th></tr></thead><tbody>__TOP_MONTHS__</tbody></table></div></div><div><h3>상대성과 하위 5개월</h3><div class="table"><table><thead><tr><th>월</th><th>기존</th><th>최종</th><th>차이</th><th>평균 스트레스</th></tr></thead><tbody>__BOTTOM_MONTHS__</tbody></table></div></div></div>
<div class="finding">2020-03에는 최종 전략이 기존보다 월수익을 약 <b>+1.71%p</b> 높였습니다. 반대로 2025-10에는 방어 포지션 때문에 약 <b>−1.40%p</b> 뒤졌습니다. 방어 오버레이는 매달 수익을 높이는 예측기가 아닙니다. 큰 손실을 피하는 대신 강한 반등에서는 기회비용을 치릅니다.</div></section>

<section><div class="head"><small>11 · PATH & RISK</small><div><h2>전체 성과 경로와 남은 불확실성</h2></div></div><div class="chart-grid">__NAV_CHART____DD_CHART__</div><div class="grid2" style="margin-top:18px"><div class="panel"><h3>6개월 블록 부트스트랩 · 5,000회</h3><ul><li>CAGR 개선 확률: <b>__P_CAGR__</b></li><li>Sharpe 개선 확률: <b>__P_SHARPE__</b></li><li>MDD 개선 확률: <b>__P_MDD__</b></li><li>세 지표 동시 개선: <b>__P_ALL__</b></li></ul></div><div class="panel"><h3>결과를 해석할 때 남는 제약</h3><ul><li>개선 폭이 작고, 부트스트랩에서도 세 지표가 함께 개선될 확률은 높지 않았습니다.</li><li>선택 후보 810개에 대한 다중검정 위험이 남습니다.</li><li>GLD·USO 추적오차, 세금, 호가충격과 상품 교체는 포함하지 않았습니다.</li><li>VKOSPI 구조와 자산 상관관계는 미래에 바뀔 수 있습니다.</li></ul></div></div></section>

__EVIDENCE_APPENDIX__

<section id="files"><div class="head"><small>12 · FILE MAP</small><div><h2>관련 코드와 결과 파일</h2><p class="lede">파일명을 누르면 같은 폴더의 원본을 열 수 있습니다.</p></div></div><div class="table files"><table><thead><tr><th>파일</th><th>구분</th><th>역할</th></tr></thead><tbody>__FILE_ROWS__</tbody></table></div>
<details><summary>핵심 코드 위치</summary><ul><li><code>vkospi_dynamic_risk_experiment.py:83</code> — VKOSPI 로드</li><li><code>:187</code> — 신호일과 수익일 배열 정렬</li><li><code>:260</code> — 일별 시뮬레이션·비용·밴드</li><li><code>:386</code> — 월별 기준 재조정</li><li><code>vkospi_robust_dynamic_experiment.py:72</code> — robust 특징</li><li><code>:117</code> — 5개 스트레스 모드</li><li><code>:176</code> — 810개 탐색과 사전 선택</li></ul></details></section>

<section><div class="download"><div><h2>실행과 재현</h2><p>Colab 노트북은 번들 ZIP을 업로드하면 810개 후보 재탐색, 잠금 검증, 상수 민감도·AUC·Brier·기간성과·오버피팅 진단과 결과 저장까지 수행합니다.</p></div><div><a href="../notebooks/vkospi_robust_dynamic_strategy_colab.ipynb">Colab 노트북</a><a href="../bundles/vkospi_robust_dynamic_colab_bundle.zip">실행 번들 ZIP</a><a href="vkospi_robust_dynamic_strategy_explainer.html">간결한 전략 요약</a></div></div></section>
</main><footer><div class="wrap">Generated from robust experiment, attribution CSVs and source code · 사후 귀속 분석은 결과 설명에만 사용했으며 파라미터 선정에는 반영하지 않았습니다 · 투자 조언이 아닙니다.</div></footer></body></html>'''

    # The non-VKOSPI section was inserted after the first report was completed.
    # Renumber the following labels in reverse order to avoid replacement cascades.
    template = template.replace("__NON_VKOSPI_SECTION__", non_vkospi_section)
    template = template.replace("__EVIDENCE_APPENDIX__", evidence_appendix)
    template = template.replace(
        '<section id="architecture">',
        current_reference_section + '\n<section id="architecture">',
        1,
    )
    template = template.replace(
        '<section id="files">',
        rebuttal_section + '\n<section id="files">',
        1,
    )
    template = template.replace(
        '<section id="formula-basis">',
        model_robustness_section + '\n<section id="formula-basis">',
        1,
    )
    for before, after in (
        ("12 · FILE MAP", "13 · FILE MAP"),
        ("11 · PATH & RISK", "12 · PATH & RISK"),
        ("10 · MONTHS", "11 · MONTHS"),
        ("09 · COMPONENTS", "10 · COMPONENTS"),
        ("08 · ATTRIBUTION", "09 · ATTRIBUTION"),
        ("07 · RECONCILE", "08 · RECONCILE"),
        ("06 · PORTFOLIO", "07 · PORTFOLIO"),
        ("05 · SEARCH", "06 · SEARCH"),
    ):
        template = template.replace(before, after)
    template = template.replace("13 · FILE MAP", "19 · FILE MAP")
    template = template.replace("16 · CODE TRACE", "17 · CODE TRACE")
    template = template.replace("15 · FACTORS & FORMULAS", "16 · FACTORS & FORMULAS")
    template = template.replace(
        '<a href="#hyperparameters">로지스틱</a>',
        '<a href="#hyperparameters">로지스틱</a><a href="#model-robustness">강건성</a>',
    )
    template = template.replace(
        '<nav><a href="#architecture">구조</a>',
        '<nav><a href="#current-reference">현재기준</a><a href="#architecture">구조</a>',
    )
    template = template.replace(
        '<a href="#files">파일</a></nav>',
        '<a href="#rebuttal">반박 Q&A</a><a href="#files">파일</a></nav>',
    )
    template = template.replace(
        '<title>VKOSPI Robust Dynamic · 코드와 효과의 기술 해설</title>',
        '<title>VKOSPI Robust 전략 완전 해설 · 국면확률·단기예측·성과·반박 Q&A</title>',
    )
    template = template.replace(
        '<h1>코드를 따라가며 살펴본<br>성과 개선의 이유</h1>',
        '<h1>VKOSPI Robust 전략을<br>끝까지 설명하는 보고서</h1>',
    )
    template = template.replace(
        '<span class="stamp">2018–2026 잠금 · CAGR 18.44% · Sharpe 1.363 · MDD −11.13%</span>',
        f'<span class="stamp">현재 비교기준 2007–2026 · CAGR {p(current_full["CAGR"])} · Sharpe {current_full["Sharpe"]:.3f} · MDD {p(current_full["MDD"])}</span>',
    )
    template = template.replace(
        "현재 결과 파일에는 C·penalty·solver를 비교한 사전 그리드가 없습니다.",
        "기존 배포 당시에는 C·penalty·solver를 고르기 위한 사전 그리드가 없었습니다. 이번 보고서에서는 배포값을 바꾸지 않는 사후 강건성 감사로 28개 후보를 추가 비교했습니다.",
    )
    template = template.replace(
        "L1과 비교하지는 않음",
        "이번 사후 강건성 감사에서 L1과 비교했으며, 강한 규제에서는 계수가 모두 0이 되는 후보도 확인",
    )
    template = template.replace(
        "이 전략 안에서 C 그리드를 돌린 근거는 없음",
        "배포 당시 선택 근거는 없었고, 이번 사후 감사에서 C 여섯 값을 비교했으나 잠금 재선정에는 쓰지 않음",
    )
    template = template.replace(
        "다른 solver와 속도·성능을 비교한 기록은 없음",
        "이번 사후 감사에서 lbfgs·saga와 비교했고, saga 일부 후보는 수렴 경고를 보임",
    )
    template = template.replace(
        "Deflated Sharpe, White Reality Check, CSCV/PBO, 외부시장 재현은 아직 수행하지 않음.",
        "CSCV/PBO는 이번에 수행했지만 Deflated Sharpe, White Reality Check와 외부시장 재현은 아직 수행하지 않음.",
    )

    strict_rows = "".join(
        f"<tr><td>{row.level_threshold:.2f}</td>"
        f"<td>{signed_p(row.Cal_CAGRDelta, 3)} / {row.Cal_SharpeDelta:+.4f} / {signed_p(row.Cal_MDDDelta, 3)}</td>"
        f"<td>{signed_p(row.Validation_CAGRDelta, 3)} / {row.Validation_SharpeDelta:+.4f} / {signed_p(row.Validation_MDDDelta, 3)}</td></tr>"
        for row in strict.sort_values("MultiObjectiveScore", ascending=False).itertuples()
    )
    reference_locked = vkospi_feature_report["locked_test"]["deployed"]
    logit_cal = oap_report["medium_horizon_prediction"]["calibration"]
    logit_locked = oap_report["medium_horizon_prediction"]["locked"]
    replacements = {
        "__OLD_CAGR__": p(locked["existing"]["CAGR"]),
        "__NEW_CAGR__": p(locked["robust"]["CAGR"]),
        "__D_CAGR__": signed_p(locked["deltas"]["CAGR"]),
        "__OLD_SHARPE__": f'{locked["existing"]["Sharpe"]:.3f}',
        "__NEW_SHARPE__": f'{locked["robust"]["Sharpe"]:.3f}',
        "__D_SHARPE__": f'{locked["deltas"]["Sharpe"]:+.3f}',
        "__OLD_MDD__": p(locked["existing"]["MDD"]),
        "__NEW_MDD__": p(locked["robust"]["MDD"]),
        "__D_MDD__": signed_p(locked["deltas"]["MDD"]),
        "__OLD_CALMAR__": f'{locked["existing"]["Calmar"]:.3f}',
        "__NEW_CALMAR__": f'{locked["robust"]["Calmar"]:.3f}',
        "__D_CALMAR__": f'{locked["deltas"]["Calmar"]:+.3f}',
        "__OLD_VOL__": p(locked["existing"]["Volatility"]),
        "__NEW_VOL__": p(locked["robust"]["Volatility"]),
        "__OLD_MULTIPLE__": f'{locked["existing"]["FinalMultiple"]:.6f}',
        "__NEW_MULTIPLE__": f'{locked["robust"]["FinalMultiple"]:.6f}',
        "__REF_CAGR__": p(reference_locked["CAGR"]),
        "__REF_VOL__": p(reference_locked["Volatility"]),
        "__REF_SHARPE__": f'{reference_locked["Sharpe"]:.3f}',
        "__REF_MDD__": p(reference_locked["MDD"]),
        "__REF_CALMAR__": f'{reference_locked["Calmar"]:.3f}',
        "__REF_MULTIPLE__": f'{reference_locked["FinalMultiple"]:.6f}',
        "__FEATURE_CODE__": code_block(feature_code),
        "__FORMULA_CODE__": code_block(formula_code),
        "__ALLOCATION_CODE__": code_block(allocation_code),
        "__RECONCILIATION_CODE__": code_block(reconciliation_code),
        "__PIPELINE_CODE__": code_block(pipeline_code),
        "__COMMENT_INDEX_ROWS__": comment_index_rows(),
        "__COMMENT_COUNT__": str(source_comment_count()),
        "__SOURCE_FILE_COUNT__": str(len(SOURCE_FILES)),
        "__FULL_SOURCE_DETAILS__": full_source_details(),
        "__FORMULA_METRIC_ROWS__": formula_metric_rows(
            old_locked_returns, new_locked_returns
        ),
        "__MEDIUM_ROWS__": medium_horizon_rows(medium_grid),
        "__OAP_FAMILY_ROWS__": oap_family_rows(oap_comparison),
        "__COST_SENSITIVITY_ROWS__": cost_sensitivity_rows(costs),
        "__REQUESTED_PERIOD_ROWS__": extended_performance_rows(
            extended_period, "requested_2005_01_2026_07"
        ),
        "__AVAILABLE_PERIOD_ROWS__": extended_performance_rows(
            extended_period, "available_full_2007_04_2026_07"
        ),
        "__SUBPERIOD_PERFORMANCE_ROWS__": subperiod_performance_rows(
            extended_period
        ),
        "__MACRO_SENSITIVITY_ROWS__": macro_sensitivity_rows(
            macro_sensitivity
        ),
        "__TAIL_PREDICTION_ROWS__": tail_prediction_rows(tail_prediction),
        "__TAIL_FEATURE_ROWS__": tail_feature_rows(tail_features),
        "__GRID_NEIGHBORHOOD_ROWS__": grid_neighborhood_rows(grid_neighborhood),
        "__LOGISTIC_ROBUSTNESS_ROWS__": logistic_robustness_rows(logistic_summary),
        "__SJM_ROBUSTNESS_ROWS__": sjm_robustness_rows(sjm_summary),
        "__MODEL_SUBPERIOD_ROWS__": model_subperiod_rows(
            logistic_robustness_long, sjm_robustness_long
        ),
        "__LOGIT_PRED_RANK_CORR__": f'{model_robustness["logistic"]["prediction_rank_correlation_calibration_vs_locked"]:.3f}',
        "__LOGIT_PORT_RANK_CORR__": f'{model_robustness["logistic"]["portfolio_rank_correlation_calibration_vs_locked"]:.3f}',
        "__LOGIT_PRED_PBO__": p(model_robustness["logistic"]["prelock_prediction_brier_pbo"]["pbo"], 1),
        "__LOGIT_PORT_PBO__": p(model_robustness["logistic"]["prelock_portfolio_sharpe_pbo"]["pbo"], 1),
        "__LOGIT_PORT_PBO_STABLE__": p(model_robustness["logistic"]["prelock_portfolio_sharpe_pbo_excluding_warning_candidates"]["pbo"], 1),
        "__LOGIT_WARNING_TOTAL__": str(model_robustness["logistic"]["total_convergence_warnings"]),
        "__SJM_PRED_RANK_CORR__": f'{model_robustness["sjm"]["prediction_rank_correlation_calibration_vs_locked"]:.3f}',
        "__SJM_SOFT_RANK_CORR__": f'{model_robustness["sjm"]["soft_rank_correlation_calibration_vs_locked"]:.3f}',
        "__SJM_PRED_PBO__": p(model_robustness["sjm"]["prelock_macro_brier_pbo"]["pbo"], 1),
        "__SJM_SOFT_PBO__": p(model_robustness["sjm"]["prelock_soft_sharpe_pbo"]["pbo"], 1),
        "__NOSJM_BOOT_BRIER__": p(model_robustness["sjm"]["no_sjm_vs_deployed"]["bootstrap_probability_no_sjm_brier_better"], 1),
        "__NOSJM_BOOT_SHARPE__": p(model_robustness["sjm"]["no_sjm_vs_deployed"]["bootstrap_probability_no_sjm_soft_sharpe_better"], 1),
        "__LOGIT_REPRO_PROB__": f'{model_robustness["logistic"]["deployed_reproduction"]["max_absolute_probability_difference"]:.2e}',
        "__LOGIT_REPRO_RETURN__": f'{model_robustness["logistic"]["deployed_reproduction"]["max_absolute_portfolio_return_difference"]:.2e}',
        "__SJM_REPRO_PROB__": f'{max(model_robustness["sjm"]["deployed_reproduction"]["max_absolute_growth_probability_difference"], model_robustness["sjm"]["deployed_reproduction"]["max_absolute_inflation_probability_difference"]):.2e}',
        "__OVERFIT_PASS_RATE__": p(overfit["strict_pass_rate"]),
        "__OVERFIT_SCORE_GAP__": f'{overfit["winner_runner_score_gap"]:.6f}',
        "__LOCKED_OUTPERFORM_YEARS__": str(
            int(overfit["locked_years_robust_outperformed"])
        ),
        "__OAP_BASE_CAGR__": p(existing_structure_cal["CAGR"]),
        "__OAP_BASE_SHARPE__": f'{existing_structure_cal["Sharpe"]:.4f}',
        "__OAP_BASE_MDD__": p(existing_structure_cal["MDD"]),
        "__OAP_BASE_CALMAR__": f'{existing_structure_cal["Calmar"]:.4f}',
        "__MEDIUM_ELIGIBLE__": str(len(medium_eligible)),
        "__LOGIT_CAL_N__": str(int(logit_cal["observations"])),
        "__LOGIT_CAL_EVENTS__": str(int(logit_cal["events"])),
        "__LOGIT_CAL_RATE__": p(logit_cal["event_rate"]),
        "__LOGIT_CAL_AUC__": f'{logit_cal["roc_auc"]:.4f}',
        "__LOGIT_CAL_AP__": f'{logit_cal["average_precision"]:.4f}',
        "__LOGIT_CAL_BRIER__": f'{logit_cal["brier_score"]:.4f}',
        "__LOGIT_CAL_RECALL__": p(logit_cal["recall_at_top_20pct"]),
        "__LOGIT_CAL_PRECISION__": p(logit_cal["precision_at_top_20pct"]),
        "__LOGIT_LOCK_N__": str(int(logit_locked["observations"])),
        "__LOGIT_LOCK_EVENTS__": str(int(logit_locked["events"])),
        "__LOGIT_LOCK_RATE__": p(logit_locked["event_rate"]),
        "__LOGIT_LOCK_AUC__": f'{logit_locked["roc_auc"]:.4f}',
        "__LOGIT_LOCK_AP__": f'{logit_locked["average_precision"]:.4f}',
        "__LOGIT_LOCK_BRIER__": f'{logit_locked["brier_score"]:.4f}',
        "__LOGIT_LOCK_RECALL__": p(logit_locked["recall_at_top_20pct"]),
        "__LOGIT_LOCK_PRECISION__": p(logit_locked["precision_at_top_20pct"]),
        "__STRICT_COUNT__": str(len(strict)),
        "__STRICT_ROWS__": strict_rows,
        "__COST2_OLD_SHARPE__": f'{cost_2x.loc["ExistingDynamic", "Sharpe"]:.3f}',
        "__COST2_NEW_SHARPE__": f'{cost_2x.loc["RobustDynamic", "Sharpe"]:.3f}',
        "__COST2_OLD_MDD__": p(cost_2x.loc["ExistingDynamic", "MDD"]),
        "__COST2_NEW_MDD__": p(cost_2x.loc["RobustDynamic", "MDD"]),
        "__STEPWISE_ROWS__": stepwise_rows(stepwise),
        "__SIGNAL_STATS__": signal_stats_cards(stats),
        "__COMPONENT_ROWS__": component_rows(components),
        "__RELATIVE_CHART__": relative_chart(contribution),
        "__TOP_MONTHS__": contribution_rows(contribution, True),
        "__BOTTOM_MONTHS__": contribution_rows(contribution, False),
        "__NAV_CHART__": nav,
        "__DD_CHART__": drawdown,
        "__P_CAGR__": p(boot["probability_cagr_improves"], 2),
        "__P_SHARPE__": p(boot["probability_sharpe_improves"], 2),
        "__P_MDD__": p(boot["probability_mdd_improves"], 2),
        "__P_ALL__": p(boot["probability_all_three_improve"], 2),
        "__BOOT_CAGR_P05__": signed_p(boot["cagr_delta_p05"], 2),
        "__BOOT_CAGR_MED__": signed_p(boot["cagr_delta_median"], 2),
        "__BOOT_CAGR_P95__": signed_p(boot["cagr_delta_p95"], 2),
        "__BOOT_SHARPE_P05__": f'{boot["sharpe_delta_p05"]:+.4f}',
        "__BOOT_SHARPE_MED__": f'{boot["sharpe_delta_median"]:+.4f}',
        "__BOOT_SHARPE_P95__": f'{boot["sharpe_delta_p95"]:+.4f}',
        "__BOOT_MDD_P05__": signed_p(boot["mdd_delta_p05"], 2),
        "__BOOT_MDD_MED__": signed_p(boot["mdd_delta_median"], 2),
        "__BOOT_MDD_P95__": signed_p(boot["mdd_delta_p95"], 2),
        "__FILE_ROWS__": file_rows(),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    unresolved = sorted(set(re.findall(r"__[A-Z0-9_]+__", template)))
    if unresolved:
        raise ValueError(f"Unresolved placeholders: {unresolved}")
    return template


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"HTML: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
