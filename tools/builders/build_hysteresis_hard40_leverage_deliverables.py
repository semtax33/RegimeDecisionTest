from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path

import pandas as pd

from tools.builders.build_vkospi_dynamic_deliverables import chart_svg, code_cell, markdown_cell, percent
from strategies.core.regime_research import get_path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HTML_PATH = ROOT / "artifacts/reports/hysteresis_hard40_leverage_report.html"
NOTEBOOK_PATH = ROOT / "artifacts/notebooks/hysteresis_hard40_leverage_colab.ipynb"
BUNDLE_PATH = ROOT / "artifacts/bundles/hysteresis_hard40_leverage_colab_bundle.zip"


def _metric_cards(report: dict) -> str:
    current = report["full_audit_2007_2026"]["current"]
    selected = report["full_audit_2007_2026"]["selected"]
    delta = report["full_audit_2007_2026"]["delta_selected_minus_current"]
    specs = (
        ("CAGR", percent(current["CAGR"]), percent(selected["CAGR"]), f"{100 * delta['CAGR']:+.2f}%p"),
        ("Sharpe", f"{current['Sharpe']:.3f}", f"{selected['Sharpe']:.3f}", f"{delta['Sharpe']:+.3f}"),
        ("MDD", percent(current["MDD"]), percent(selected["MDD"]), f"{100 * delta['MDD']:+.2f}%p"),
        ("Calmar", f"{current['Calmar']:.3f}", f"{selected['Calmar']:.3f}", f"{delta['Calmar']:+.3f}"),
    )
    return "".join(
        f'<article class="metric"><span>{name}</span><strong>{after}</strong>'
        f'<div><s>{before}</s><b class="{"good" if (name in ("Sharpe", "MDD") and float(change.replace("%p", "")) > 0) else "bad"}">{change}</b></div></article>'
        for name, before, after, change in specs
    )


def _period_rows(comparison: pd.DataFrame, period: str) -> str:
    order = [
        "Current_NoHysteresis_LevCap1.5",
        "Hysteresis_Hard40_LevCap1.0",
        "Hysteresis_Hard40_LevCap1.1",
        "Hysteresis_Hard40_LevCap1.2",
        "Hysteresis_Hard40_LevCap1.3",
    ]
    labels = {
        order[0]: "현재 전략 · 비히스테리시스 · 상한 1.5",
        order[1]: "히스테리시스 · 상한 1.0 (사전선택)",
        order[2]: "히스테리시스 · 상한 1.1",
        order[3]: "히스테리시스 · 상한 1.2",
        order[4]: "히스테리시스 · 상한 1.3",
    }
    view = comparison.loc[comparison["Period"].eq(period)].set_index("Strategy")
    rows = []
    for strategy in order:
        row = view.loc[strategy]
        rows.append(
            f'<tr class="{"selected" if strategy == order[1] else ""}">'
            f"<td>{labels[strategy]}</td><td>{int(row['Months'])}</td>"
            f"<td>{percent(row['CAGR'])}</td><td>{row['Sharpe']:.3f}</td>"
            f"<td>{percent(row['MDD'])}</td><td>{percent(row['Volatility'])}</td>"
            f"<td>{row['Calmar']:.3f}</td></tr>"
        )
    return "".join(rows)


def _calibration_rows(calibration: pd.DataFrame) -> str:
    rows = []
    for _, row in calibration.sort_values("LeverageCap").iterrows():
        rows.append(
            f'<tr class="{"selected" if row["Selected"] else ""}">'
            f"<td>{row['LeverageCap']:.1f}</td>"
            f"<td>{percent(row['Cal_CAGR'])}</td><td>{row['Cal_Sharpe']:.3f}</td><td>{percent(row['Cal_MDD'])}</td>"
            f"<td>{percent(row['Validation_CAGR'])}</td><td>{row['Validation_Sharpe']:.3f}</td><td>{percent(row['Validation_MDD'])}</td>"
            f"<td>{row['MultiObjectiveScore']:.3f}</td><td>{'예' if row['Selected'] else '아니오'}</td></tr>"
        )
    return "".join(rows)


def make_html(report: dict, comparison: pd.DataFrame, calibration: pd.DataFrame) -> str:
    current = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv", index_col=0
    )
    selected = pd.read_csv(
        RESULTS / "hysteresis_hard40_leverage_selected_reconciled.csv", index_col=0
    )
    current.index = pd.PeriodIndex(current.index, freq="M")
    selected.index = pd.PeriodIndex(selected.index, freq="M")
    nav = chart_svg(current, selected, "nav", "누적 자산가치")
    drawdown = chart_svg(current, selected, "drawdown", "드로다운")
    for before, after in (
        ("기준 전략", "현재 전략 · 상한 1.5"),
        ("VKOSPI 동적 전략", "히스테리시스 · 상한 1.0"),
    ):
        nav = nav.replace(before, after)
        drawdown = drawdown.replace(before, after)

    audit = report["regime_audit"]
    locked = report["locked_audit_2018_2026"]
    boot = locked["bootstrap"]
    template = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>히스테리시스 + Hard 40% + 레버리지 상한 실험</title><style>
:root{--ink:#17211d;--muted:#66736d;--paper:#f3f1e9;--card:#fffdf7;--line:#d9d4c8;--green:#087f5b;--mint:#dff2e8;--red:#b54736;--gold:#d88934;--navy:#203a43}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Apple SD Gothic Neo","Noto Sans KR",sans-serif;line-height:1.68}a{color:var(--green)}.wrap{width:min(1160px,calc(100% - 40px));margin:auto}.top{padding:17px 0;display:flex;justify-content:space-between;font-size:13px}.top nav a{margin-left:20px;text-decoration:none;color:var(--ink)}header{border-block:1px solid var(--line);background:radial-gradient(circle at 84% 12%,rgba(8,127,91,.18),transparent 34%),linear-gradient(135deg,#fffaf0,#e7f2eb)}.hero{padding:72px 0}.eyebrow{color:var(--green);font-weight:800;letter-spacing:.12em}.hero h1{font:clamp(40px,6.5vw,74px)/1.05 Georgia,"Noto Serif KR",serif;letter-spacing:-.052em;margin:14px 0 22px;max-width:1000px}.hero p{font-size:19px;max-width:820px;color:#405049}.stamp{display:inline-block;margin-top:13px;border:1px solid #d5ab7d;border-radius:999px;padding:9px 15px;color:#8a4a13;font-weight:800;background:#fff7e9}main{padding:62px 0 96px}section{margin-bottom:68px}.head{display:grid;grid-template-columns:145px 1fr;gap:22px;margin-bottom:25px}.head small{color:var(--green);font-weight:800;letter-spacing:.1em}.head h2{font:clamp(28px,4vw,42px)/1.18 Georgia,"Noto Serif KR",serif;letter-spacing:-.03em;margin:0}.lede{font-size:17px;color:#4f5e57;max-width:850px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.metric{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:23px}.metric span{font-size:13px;color:var(--muted)}.metric strong{display:block;font:36px Georgia,serif;margin:7px 0}.metric div{display:flex;justify-content:space-between}.metric s{color:#8c9691}.good{color:var(--green)}.bad{color:var(--red)}.verdict{margin-top:18px;border-left:4px solid var(--gold);background:#fff8eb;padding:17px 20px}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;counter-reset:step}.step{background:var(--navy);color:white;border-radius:14px;padding:22px 18px;min-height:205px}.step:before{counter-increment:step;content:"0" counter(step);color:#83d4b8;font:700 12px Georgia}.step h3{margin:20px 0 8px}.step p{margin:0;color:#d5e0dc;font-size:14px}.formula{font-family:Consolas,"D2Coding",monospace;background:#102c31;color:#e5fff6;border-radius:13px;padding:20px;overflow:auto;white-space:pre-wrap}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px}.panel h3{margin-top:0}.table{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:15px}table{width:100%;border-collapse:collapse;min-width:790px}th,td{text-align:left;padding:13px 15px;border-bottom:1px solid #e8e4d9;font-size:14px}th{font-size:12px;color:var(--muted);background:#f8f5ed}.selected{background:#eaf6f0}.chart-grid{display:grid;gap:18px}.chart-card{margin:0;background:var(--card);border:1px solid var(--line);padding:20px;border-radius:16px;overflow:hidden}.chart-card figcaption{display:flex;justify-content:space-between}.chart-card svg{width:100%;height:auto;margin-top:12px}.grid{stroke:#e7e3d9}.axis{font-size:11px;fill:#818a86}.line{fill:none;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}.reference{stroke:#9ca6a1}.dynamic{stroke:var(--green)}.legend{display:flex;justify-content:flex-end;gap:18px;color:var(--muted);font-size:13px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.ref-dot{background:#9ca6a1}.dyn-dot{background:var(--green)}.tags{display:flex;flex-wrap:wrap;gap:8px}.tags code{background:var(--mint);color:#075f46;border-radius:999px;padding:7px 11px;font-size:12px}.prob{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.prob div{background:var(--mint);border-radius:13px;padding:18px}.prob strong{display:block;color:var(--green);font:30px Georgia,serif}.prob span{font-size:13px;color:#53645c}.files{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.files a{display:block;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;text-decoration:none;font-weight:700}.download{background:var(--ink);color:white;border-radius:20px;padding:30px;display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center}.download h2{margin:0;font:33px Georgia,serif}.download p{color:#c9d5d0}.download a{display:block;background:#24c38e;color:#07271d;text-decoration:none;font-weight:800;padding:11px 16px;border-radius:9px;margin:7px;text-align:center}footer{border-top:1px solid var(--line);padding:28px 0 48px;color:var(--muted);font-size:13px}@media(max-width:900px){.head{grid-template-columns:1fr}.metrics,.prob{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:1fr 1fr}.grid2,.download,.files{grid-template-columns:1fr}.top nav{display:none}}@media(max-width:560px){.wrap{width:calc(100% - 24px)}.metrics,.prob,.flow{grid-template-columns:1fr}}@media print{body{background:#fff}.download,.top nav{display:none}section{break-inside:avoid}}
</style></head><body><div class="wrap top"><b>REGIME DECISION TEST · RESEARCH NOTE</b><nav><a href="#result">결론</a><a href="#algorithm">알고리즘</a><a href="#inputs">입력변수</a><a href="#tables">성과표</a><a href="#files">파일</a></nav></div>
<header><div class="wrap hero"><span class="eyebrow">NOTEBOOK HYSTERESIS × CURRENT ROBUST PIPELINE</span><h1>국면은 덜 바뀌었지만,<br>수익 기회도 함께 줄었다.</h1><p>노트북의 ±0.2 히스테리시스를 현재 Hard 40% 구간에만 넣고, 최종 레버리지 상한을 1.0·1.1·1.2·1.3으로 제한해 비교했습니다. 확률·SLSQP·균형 L2 로지스틱·Robust VKOSPI는 그대로 유지했습니다.</p><span class="stamp">결론 · 상한 1.0은 방어형 연구 후보, 현재 전략은 유지</span></div></header>
<main class="wrap"><section id="result"><div class="head"><small>01 · VERDICT</small><div><h2>전체 위험은 낮아졌지만 세 목표를 함께 개선하지는 못했다</h2><p class="lede">아래는 2007-04–2026-07 전체 구간입니다. MDD 변화가 양수이면 최대 낙폭이 줄었다는 뜻입니다.</p></div></div><div class="metrics">__METRIC_CARDS__</div><div class="verdict"><b>교체하지 않은 이유.</b> 사전 구간에서 CAGR·Sharpe·MDD를 동시에 개선한 후보가 0개였습니다. 선택된 상한 1.0은 전체 Sharpe와 MDD는 좋아졌지만 CAGR이 2.62%p 낮아졌고, 2018년 이후에는 Sharpe도 1.497에서 1.429로 하락했습니다. 잠금 구간을 보고 파라미터를 다시 고르지 않았으며, 현재 결과 파일도 덮어쓰지 않았습니다.</div></section>
<section id="algorithm"><div class="head"><small>02 · ALGORITHM</small><div><h2>바뀐 곳은 Hard 국면과 레버리지 천장, 두 군데뿐</h2></div></div><div class="flow"><article class="step"><h3>거시 점수</h3><p>직전 월까지의 GDP·수출·BSI 성장 z-score와 CPI·PPI·수입물가 z-score를 각각 평균합니다.</p></article><article class="step"><h3>히스테리시스</h3><p>+0.2 위에서만 high, −0.2 아래에서만 low로 전환합니다. 그 사이는 직전 상태를 유지합니다.</p></article><article class="step"><h3>Hard 40%</h3><p>상태 조합을 4개 국면으로 바꿔 기존 Hard 자산배분에 40%만 반영합니다.</p></article><article class="step"><h3>기존 위험 조절</h3><p>나머지 60% SLSQP, 16변수 균형 L2 로지스틱, 15% 목표 변동성은 그대로입니다.</p></article><article class="step"><h3>상한·오버레이</h3><p>레버리지를 0.5–cap으로 제한한 뒤 기존 Robust VKOSPI 일간 오버레이를 적용합니다.</p></article></div>
<div class="formula">g_state(t) = +1 if g(t) &gt; +0.2; −1 if g(t) &lt; −0.2; otherwise g_state(t−1)
i_state(t) = +1 if i(t) &gt; +0.2; −1 if i(t) &lt; −0.2; otherwise i_state(t−1)

w_base = 0.40 × w_hard(hysteresis regime) + 0.60 × w_SLSQP
w_tilt = balanced_L2_logistic_tilt(w_base, max_shift=20%p)
leverage = clip(15% / forecast_vol_24m, 0.50, cap), cap ∈ {1.0, 1.1, 1.2, 1.3}
w_monthly = leverage × w_tilt
return_final = (1 + return_monthly) × (1 + robust_VKOSPI_relative_return) − 1</div></section>
<section><div class="head"><small>03 · STATE</small><div><h2>국면 전환은 44회에서 23회로 줄었다</h2><p class="lede">232개월 중 32개월에서 확률 argmax 국면과 히스테리시스 국면이 달랐습니다. 매매 신호는 항상 목표 투자월보다 앞선 월의 정보만 사용했습니다.</p></div></div><div class="grid2"><div class="panel"><h3>상태 조합과 Hard 자산</h3><p>성장 high · 물가 low → Goldilocks → KODEX200 100%</p><p>성장 high · 물가 high → Overheating → USO 100%</p><p>성장 low · 물가 low → Slowdown → KODEX200 60% + BOND 40%</p><p>성장 low · 물가 high → Stagflation → GLD 100%</p></div><div class="panel"><h3>왜 성과가 달라졌나</h3><p>히스테리시스는 경계 부근의 잦은 국면 변경을 줄였지만, 변화가 실제 추세 전환인 경우에도 반응을 늦춥니다.</p><p>레버리지 상한은 강한 상승기에 위험노출을 줄여 MDD와 변동성을 낮추는 대신 복리 수익을 희생했습니다.</p><p>특히 현재 전략은 232개월 중 209개월이 1배 초과, 146개월이 1.5배였으므로 상한 효과가 컸습니다.</p></div></div></section>
<section id="inputs"><div class="head"><small>04 · INPUTS</small><div><h2>이번 실험에서 실제로 사용한 입력변수</h2></div></div><h3>거시 원자료와 파생값</h3><div class="tags"><code>GDP YoY</code><code>수출 YoY</code><code>제조업 전망 BSI</code><code>CPI YoY</code><code>PPI YoY</code><code>수입물가 YoY</code><code>growth_level</code><code>growth_d3</code><code>inflation_level</code><code>inflation_d3</code><code>growth_state</code><code>inflation_state</code></div><h3>균형 L2 로지스틱 16개 변수</h3><div class="tags"><code>base_USO</code><code>base_GLD</code><code>base_KODEX200</code><code>p_inflation_high</code><code>proxy_mom1</code><code>proxy_mom6</code><code>proxy_vol6</code><code>daily_mom21</code><code>daily_mom252</code><code>daily_vol21</code><code>daily_downvol21</code><code>daily_mean_corr63</code><code>oap_momentum_trend_stress</code><code>oap_reversal_crowding_stress</code><code>oap_low_risk_tail_stress</code><code>oap_liquidity_activity_stress</code></div><h3>Robust VKOSPI 15개 변수</h3><div class="tags"><code>close</code><code>pct_126</code><code>pct_252</code><code>robust_z_63</code><code>robust_z_252</code><code>shock_5</code><code>shock_10</code><code>shock_21</code><code>acceleration_5</code><code>acceleration_z_5</code><code>distance_high_21</code><code>close_location_21</code><code>positive_fraction_5</code><code>positive_fraction_21</code><code>fast_slow</code></div></section>
<section id="tables"><div class="head"><small>05 · SELECTION</small><div><h2>2018년 이전 자료만으로 상한을 골랐다</h2><p class="lede">2007–2017과 2013–2017에서 CAGR·Sharpe·MDD의 백분위 순위를 동일 가중해 평균했습니다. 세 지표를 두 구간에서 모두 이긴 후보가 없어서 상한 1.0을 ‘연구 후보’로만 선택했습니다.</p></div></div><div class="table"><table><thead><tr><th rowspan="2">상한</th><th colspan="3">2007–2017</th><th colspan="3">2013–2017</th><th rowspan="2">종합점수</th><th rowspan="2">선택</th></tr><tr><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>CAGR</th><th>Sharpe</th><th>MDD</th></tr></thead><tbody>__CAL_ROWS__</tbody></table></div></section>
<section><div class="head"><small>06 · FULL</small><div><h2>2007–2026 전체 성과</h2></div></div><div class="table"><table><thead><tr><th>전략</th><th>개월</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>변동성</th><th>Calmar</th></tr></thead><tbody>__FULL_ROWS__</tbody></table></div></section>
<section><div class="head"><small>07 · LOCKED</small><div><h2>2018–2026 사후 잠금 검증</h2></div></div><div class="table"><table><thead><tr><th>전략</th><th>개월</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>변동성</th><th>Calmar</th></tr></thead><tbody>__LOCKED_ROWS__</tbody></table></div><div class="prob"><div><strong>__PCAGR__</strong><span>CAGR 개선 재표본 확률</span></div><div><strong>__PSHARPE__</strong><span>Sharpe 개선 재표본 확률</span></div><div><strong>__PMDD__</strong><span>MDD 개선 재표본 확률</span></div><div><strong>__PALL__</strong><span>세 지표 동시 개선 확률</span></div></div></section>
<section><div class="head"><small>08 · PATH</small><div><h2>누적수익과 낙폭 경로</h2></div></div><div class="chart-grid">__NAV____DRAWDOWN__</div></section>
<section><div class="head"><small>09 · INTERPRET</small><div><h2>이 결과를 어떻게 써야 하나</h2></div></div><div class="grid2"><div class="panel"><h3>수익 우선</h3><p>현재 비히스테리시스·상한 1.5 전략을 유지하는 편이 맞습니다. 전체와 잠금 구간 모두 CAGR이 높고, 잠금 Sharpe도 더 높았습니다.</p></div><div class="panel"><h3>무레버리지·낙폭 우선</h3><p>상한 1.0 후보는 전체 변동성 11.15%, MDD −11.17%로 방어력이 좋았습니다. 다만 별도 저위험 상품으로 관리해야 하며 현재 전략의 ‘개선판’이라고 부르기는 어렵습니다.</p></div></div><div class="verdict"><b>주의:</b> 백테스트는 월별 비용 15bp, 해외자산 환전비용 5bp, 차입금리 연 4%를 반영했지만 세금·시장충격·추적오차·상품 존속 위험은 완전히 포함하지 않습니다. 과거 성과는 미래 성과를 보장하지 않습니다.</div></section>
<section id="files"><div class="head"><small>10 · FILES</small><div><h2>코드·결과·재현 파일</h2></div></div><div class="files"><a href="../../strategies/stage09_hysteresis/hysteresis_hard40_leverage_experiment.py">실험 코드</a><a href="../../results/hysteresis_hard40_leverage_validation.json">검증 보고서 JSON</a><a href="../../results/hysteresis_hard40_leverage_calibration.csv">사전 선택표 CSV</a><a href="../../results/hysteresis_hard40_leverage_comparison.csv">전 기간 비교표 CSV</a><a href="../../results/hysteresis_hard40_leverage_selected_reconciled.csv">선택 후보 월수익 CSV</a><a href="robust_vkospi_implementation_guide.html">전체 Robust 전략 설명서</a></div></section>
<section><div class="download"><div><h2>Google Colab에서 바로 재현</h2><p>노트북과 번들 ZIP을 함께 업로드하면 네 상한을 다시 실행하고 결과표·차트·ZIP을 생성합니다.</p></div><div><a href="../notebooks/hysteresis_hard40_leverage_colab.ipynb" download>① Colab 노트북</a><a href="../bundles/hysteresis_hard40_leverage_colab_bundle.zip" download>② 실행 번들 ZIP</a></div></div></section></main>
<footer><div class="wrap">입력은 목표월보다 앞선 시점만 사용했습니다. 연구용 시뮬레이션이며 투자 조언이 아닙니다.</div></footer></body></html>'''
    replacements = {
        "__METRIC_CARDS__": _metric_cards(report),
        "__CAL_ROWS__": _calibration_rows(calibration),
        "__FULL_ROWS__": _period_rows(comparison, "full_2007_2026"),
        "__LOCKED_ROWS__": _period_rows(comparison, "locked_2018_2026"),
        "__PCAGR__": percent(boot["probability_cagr_improves"], 1),
        "__PSHARPE__": percent(boot["probability_sharpe_improves"], 1),
        "__PMDD__": percent(boot["probability_mdd_improves"], 1),
        "__PALL__": percent(boot["probability_all_three_improve"], 1),
        "__NAV__": nav,
        "__DRAWDOWN__": drawdown,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def make_notebook() -> dict:
    cells = [
        markdown_cell("""# 히스테리시스 + Hard 40% + 레버리지 상한 실험

노트북의 ±0.2 히스테리시스를 현재 Hard 40% 배분에만 적용하고, 최종 레버리지 상한 1.0·1.1·1.2·1.3을 비교합니다.

- `hysteresis_hard40_leverage_colab_bundle.zip`을 함께 업로드하세요.
- 상한 선택은 2017년까지의 자료만 사용하고, 2018–2026은 선택 후 검증합니다.
- 기존 무SJM 거시 확률, SLSQP 60%, 균형 L2 로지스틱, Robust VKOSPI는 유지됩니다.
- CPU 런타임에서 실행하며 GPU는 필요하지 않습니다.
"""),
        markdown_cell("## 1. 환경과 입력 번들 준비"),
        code_cell("""import sys, subprocess, zipfile, json
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "matplotlib"], check=True)

def safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)

if IN_COLAB:
    from google.colab import files
    uploaded = files.upload()
    bundle = next((Path(name) for name in uploaded if name.endswith(".zip")), None)
    if bundle is None:
        raise FileNotFoundError("hysteresis_hard40_leverage_colab_bundle.zip을 업로드하세요.")
    safe_extract(bundle, Path("/content"))
    PROJECT_ROOT = Path("/content/RegimeDecisionTest")
else:
    PROJECT_ROOT = Path.cwd().resolve()

required = [
    "strategies/stage09_hysteresis/hysteresis_hard40_leverage_experiment.py",
    "strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py",
    "strategies/core/regime_research.py",
    "cache/market_daily.csv",
    "raw_data/compass.db",
    "raw_data/VKOSPIData.csv",
    "results/openassetpricing_composites.csv",
    "results/vkospi_robust_dynamic_validation.json",
    "results/balanced_logistic_no_sjm_final_reconciled.csv",
]
missing = [name for name in required if not (PROJECT_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(missing)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("Python", sys.version.split()[0], "| 입력 확인 완료")
"""),
        markdown_cell("""## 2. 네 후보 실행

실험 코드는 성장·물가 점수를 ±0.2 히스테리시스로 분류하고, Hard 40% 이후의 월별 변동성 목표 레버리지에 각각 1.0·1.1·1.2·1.3 상한을 적용합니다. 기존 Robust VKOSPI 일간 상대수익을 마지막에 결합합니다.
"""),
        code_cell("""command = [sys.executable, "-m", "strategies.stage09_hysteresis.hysteresis_hard40_leverage_experiment"]
completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
print(completed.stdout)
if completed.returncode:
    print(completed.stderr)
    raise RuntimeError(f"실험 실패: exit={completed.returncode}")
"""),
        markdown_cell("## 3. 사전 선택표와 핵심 결론"),
        code_cell("""import pandas as pd
from IPython.display import display, Markdown

results = PROJECT_ROOT / "results"
report = json.loads((results / "hysteresis_hard40_leverage_validation.json").read_text(encoding="utf-8"))
calibration = pd.read_csv(results / "hysteresis_hard40_leverage_calibration.csv")
display(calibration[[
    "Candidate", "LeverageCap", "Cal_CAGR", "Cal_Sharpe", "Cal_MDD",
    "Validation_CAGR", "Validation_Sharpe", "Validation_MDD",
    "MultiObjectiveScore", "Selected", "StrictPrelockPass",
]].style.format({
    "Cal_CAGR": "{:.2%}", "Cal_Sharpe": "{:.3f}", "Cal_MDD": "{:.2%}",
    "Validation_CAGR": "{:.2%}", "Validation_Sharpe": "{:.3f}", "Validation_MDD": "{:.2%}",
    "MultiObjectiveScore": "{:.3f}",
}))
display(Markdown(
    f"**사전 선택:** `{report['selection']['selected_candidate']}`  \\n"
    f"**엄격 통과 후보:** {report['selection']['strict_eligible_count']}개  \\n"
    f"**운영 판단:** `{report['selection']['promotion_status']}`"
))
"""),
        markdown_cell("## 4. 2007–2026 및 2018–2026 비교"),
        code_cell("""comparison = pd.read_csv(results / "hysteresis_hard40_leverage_comparison.csv")
view = comparison.loc[comparison["Period"].isin(["full_2007_2026", "locked_2018_2026"]), [
    "Period", "Strategy", "Months", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar"
]]
display(view.style.format({
    "CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.3f}",
    "MDD": "{:.2%}", "Calmar": "{:.3f}",
}))
"""),
        markdown_cell("## 5. 경로 차트"),
        code_cell("""import matplotlib.pyplot as plt

current = pd.read_csv(results / "balanced_logistic_no_sjm_final_reconciled.csv", index_col=0)
selected = pd.read_csv(results / "hysteresis_hard40_leverage_selected_reconciled.csv", index_col=0)
current.index = pd.PeriodIndex(current.index, freq="M").to_timestamp()
selected.index = pd.PeriodIndex(selected.index, freq="M").to_timestamp()

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
axes[0].plot(current.index, current["nav"], label="Current / cap 1.5", lw=2)
axes[0].plot(selected.index, selected["nav"], label="Hysteresis / cap 1.0", lw=2)
axes[0].set_title("Cumulative NAV")
axes[0].legend()
axes[0].grid(alpha=.25)
axes[1].plot(current.index, current["drawdown"] * 100, label="Current / cap 1.5", lw=2)
axes[1].plot(selected.index, selected["drawdown"] * 100, label="Hysteresis / cap 1.0", lw=2)
axes[1].set_title("Drawdown (%)")
axes[1].legend()
axes[1].grid(alpha=.25)
plt.tight_layout()
plt.show()
"""),
        markdown_cell("## 6. 결과 ZIP 내려받기"),
        code_cell("""output = Path("/content/hysteresis_hard40_leverage_results.zip") if IN_COLAB else PROJECT_ROOT / "hysteresis_hard40_leverage_results.zip"
members = [
    "hysteresis_hard40_leverage_validation.json",
    "hysteresis_hard40_leverage_calibration.csv",
    "hysteresis_hard40_leverage_comparison.csv",
    "hysteresis_hard40_leverage_selected_medium.csv",
    "hysteresis_hard40_leverage_selected_reconciled.csv",
    "hysteresis_hard40_signals.csv",
    "hysteresis_hard40_features.csv",
    "hysteresis_hard40_factor.csv",
]
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in members:
        archive.write(results / name, arcname=name)
print(output)
if IN_COLAB:
    files.download(str(output))
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_bundle() -> None:
    source_files = [
        "strategies/stage09_hysteresis/hysteresis_hard40_leverage_experiment.py",
        "strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py",
        "strategies/stage06_vkospi/vkospi_model_robustness.py",
        "strategies/stage06_vkospi/vkospi_extended_diagnostics.py",
        "strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py",
        "strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py",
        "strategies/core/regime_research.py",
        "tests/test_hysteresis_hard40_leverage_experiment.py",
    ]
    data_files = [
        get_path(ROOT / "raw_data", "GDP 성장률.xlsx"),
        get_path(ROOT / "raw_data", "수출입 총괄_20260816.xlsx"),
        get_path(ROOT / "raw_data", "기업경기조사(전망).csv"),
        get_path(ROOT / "raw_data", "소비자물가 상승률.xlsx"),
        get_path(ROOT / "raw_data", "생산자물가 상승률.xlsx"),
        get_path(ROOT / "raw_data", "수출입물가 상승률.xlsx"),
        get_path(ROOT / "raw_data", "VKOSPIData.csv"),
        get_path(ROOT / "raw_data", "krx_bond_index.csv"),
        get_path(ROOT / "raw_data", "compass.db"),
        ROOT / "cache" / "market_daily.csv",
        RESULTS / "openassetpricing_composites.csv",
        RESULTS / "vkospi_robust_dynamic_validation.json",
        RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv",
    ]
    members = [ROOT / name for name in source_files] + data_files
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            if not path.exists():
                raise FileNotFoundError(path)
            archive.write(path, arcname=f"RegimeDecisionTest/{path.relative_to(ROOT).as_posix()}")


def main() -> None:
    report = json.loads(
        (RESULTS / "hysteresis_hard40_leverage_validation.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(RESULTS / "hysteresis_hard40_leverage_comparison.csv")
    calibration = pd.read_csv(RESULTS / "hysteresis_hard40_leverage_calibration.csv")
    HTML_PATH.write_text(make_html(report, comparison, calibration), encoding="utf-8")
    NOTEBOOK_PATH.write_text(
        json.dumps(make_notebook(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    make_bundle()
    for path in (HTML_PATH, NOTEBOOK_PATH, BUNDLE_PATH):
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
