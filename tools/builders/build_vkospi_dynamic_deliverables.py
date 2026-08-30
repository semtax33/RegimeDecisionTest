from __future__ import annotations

import html
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HTML_PATH = ROOT / "artifacts/reports/vkospi_dynamic_strategy_explainer.html"
NOTEBOOK_PATH = ROOT / "artifacts/notebooks/vkospi_dynamic_strategy_colab.ipynb"
BUNDLE_PATH = ROOT / "artifacts/bundles/vkospi_dynamic_colab_bundle.zip"


def percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def signed_percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:+.{digits}f}%p"


def svg_path(
    values: np.ndarray,
    width: int,
    height: int,
    low: float | None = None,
    high: float | None = None,
    padding: int = 28,
) -> str:
    finite = np.asarray(values, dtype=float)
    valid = np.isfinite(finite)
    if not valid.any():
        return ""
    low = float(np.nanmin(finite)) if low is None else low
    high = float(np.nanmax(finite)) if high is None else high
    if math.isclose(low, high):
        high = low + 1
    x = np.linspace(padding, width - padding, len(finite))
    y = height - padding - (finite - low) / (high - low) * (height - 2 * padding)
    points = [f"{xv:.2f},{yv:.2f}" for xv, yv, ok in zip(x, y, valid) if ok]
    return "M " + " L ".join(points)


def chart_svg(reference: pd.DataFrame, dynamic: pd.DataFrame, column: str, title: str) -> str:
    width, height = 960, 300
    common = reference.index.intersection(dynamic.index)
    ref = reference.loc[common, column].to_numpy(dtype=float)
    dyn = dynamic.loc[common, column].to_numpy(dtype=float)
    combined = np.r_[ref, dyn]
    lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
    ticks = np.linspace(lo, hi, 5)
    grid = []
    for tick in ticks:
        y = height - 28 - (tick - lo) / max(hi - lo, 1e-12) * (height - 56)
        label = f"{tick:.2f}" if column == "nav" else f"{tick * 100:.0f}%"
        grid.append(
            f'<line x1="28" y1="{y:.2f}" x2="932" y2="{y:.2f}" class="grid"/>'
            f'<text x="20" y="{y + 4:.2f}" text-anchor="end" class="axis">{label}</text>'
        )
    return f"""
    <figure class="chart-card">
      <figcaption><strong>{html.escape(title)}</strong><span>2007-04–2026-07</span></figcaption>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        {''.join(grid)}
        <path d="{svg_path(ref, width, height, lo, hi)}" class="line reference"/>
        <path d="{svg_path(dyn, width, height, lo, hi)}" class="line dynamic"/>
      </svg>
      <div class="legend"><span><i class="ref-dot"></i>기준 전략</span><span><i class="dyn-dot"></i>VKOSPI 동적 전략</span></div>
    </figure>"""


def metric_cards(locked: dict) -> str:
    reference = locked["reference"]
    candidate = locked["candidate"]
    deltas = locked["deltas"]
    specs = [
        ("CAGR", percent(reference["CAGR"]), percent(candidate["CAGR"]), signed_percent(deltas["CAGR"])),
        ("Sharpe", f'{reference["Sharpe"]:.3f}', f'{candidate["Sharpe"]:.3f}', f'{deltas["Sharpe"]:+.3f}'),
        ("MDD", percent(reference["MDD"]), percent(candidate["MDD"]), signed_percent(deltas["MDD"])),
        ("Calmar", f'{reference["Calmar"]:.3f}', f'{candidate["Calmar"]:.3f}', f'{deltas["Calmar"]:+.3f}'),
    ]
    cards = []
    for name, before, after, delta in specs:
        cards.append(
            f"""<article class="metric-card"><span>{name}</span><strong>{after}</strong>
            <div><s>{before}</s><em>{delta}</em></div></article>"""
        )
    return "".join(cards)


def cost_rows(costs: pd.DataFrame) -> str:
    rows = []
    for multiplier in (0.5, 1.0, 1.5, 2.0):
        period = f"cost_{multiplier:.1f}x_locked"
        view = costs.loc[costs["Period"] == period].set_index("Strategy")
        ref, dyn = view.loc["ReferenceDaily"], view.loc["VKOSPIDynamic"]
        rows.append(
            "<tr>"
            f"<td>{multiplier:.1f}×</td>"
            f"<td>{percent(ref['CAGR'])} → <b>{percent(dyn['CAGR'])}</b></td>"
            f"<td>{ref['Sharpe']:.3f} → <b>{dyn['Sharpe']:.3f}</b></td>"
            f"<td>{percent(ref['MDD'])} → <b>{percent(dyn['MDD'])}</b></td>"
            "</tr>"
        )
    return "".join(rows)


def subperiod_rows(subperiods: pd.DataFrame) -> str:
    rows = []
    for period in subperiods["Period"].drop_duplicates():
        view = subperiods.loc[subperiods["Period"] == period].set_index("Strategy")
        ref, dyn = view.loc["ReferenceMonthly"], view.loc["VKOSPIDynamicReconciled"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(period.replace('_', ' – '))}</td>"
            f"<td>{percent(ref['CAGR'])} → <b>{percent(dyn['CAGR'])}</b></td>"
            f"<td>{ref['Sharpe']:.3f} → <b>{dyn['Sharpe']:.3f}</b></td>"
            f"<td>{percent(ref['MDD'])} → <b>{percent(dyn['MDD'])}</b></td>"
            "</tr>"
        )
    return "".join(rows)


def make_html(report: dict, costs: pd.DataFrame, subperiods: pd.DataFrame) -> str:
    reference = pd.read_csv(RESULTS / "vkospi_selected_backtest.csv", index_col=0)
    dynamic = pd.read_csv(RESULTS / "vkospi_dynamic_reconciled_monthly.csv", index_col=0)
    reference.index = pd.PeriodIndex(reference.index, freq="M")
    dynamic.index = pd.PeriodIndex(dynamic.index, freq="M")
    locked = report["locked"]["monthly_reference_reconciled"]
    boot = report["locked"]["reconciled_multiobjective_bootstrap"]
    winner = report["winner"]
    nav_chart = chart_svg(reference, dynamic, "nav", "누적 자산가치")
    dd_chart = chart_svg(reference, dynamic, "drawdown", "드로다운")
    template = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VKOSPI 동적 위험 오버레이 전략</title>
<style>
:root{--ink:#13211c;--muted:#65716c;--paper:#f5f2e9;--card:#fffdf7;--line:#d8d4c8;--green:#007a5e;--green2:#00a676;--orange:#f28c45;--navy:#243b53}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Apple SD Gothic Neo","Noto Sans KR",sans-serif;line-height:1.65}
a{color:var(--green)}.wrap{width:min(1160px,calc(100% - 40px));margin:auto}.topbar{padding:18px 0;display:flex;justify-content:space-between;align-items:center;font-size:13px;letter-spacing:.04em}.topbar b{font-size:15px}.topbar nav a{margin-left:22px;text-decoration:none;color:var(--ink)}
header{position:relative;overflow:hidden;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:linear-gradient(135deg,#fbf8ef 0%,#e3f1e8 100%)}header:after{content:"";position:absolute;width:420px;height:420px;border:80px solid rgba(0,122,94,.08);border-radius:50%;right:-120px;top:-190px}
.hero{padding:80px 0 70px;position:relative;z-index:1}.eyebrow{text-transform:uppercase;color:var(--green);font-weight:700;letter-spacing:.13em;font-size:13px}.hero h1{font-family:Georgia,"Noto Serif KR",serif;font-size:clamp(43px,7vw,82px);line-height:1.02;letter-spacing:-.05em;max-width:890px;margin:14px 0 24px}.hero p{font-size:19px;max-width:760px;color:#35443e}.hero .stamp{display:inline-flex;gap:10px;align-items:center;border:1px solid rgba(0,122,94,.35);background:rgba(255,255,255,.55);padding:10px 15px;border-radius:999px;color:var(--green);font-size:13px;font-weight:700}
main{padding:64px 0 100px}section{margin:0 0 76px}.section-head{display:grid;grid-template-columns:160px 1fr;gap:22px;align-items:start;margin-bottom:28px}.section-head small{color:var(--green);font-weight:800;letter-spacing:.12em}.section-head h2{font-family:Georgia,"Noto Serif KR",serif;font-size:clamp(29px,4vw,43px);line-height:1.15;letter-spacing:-.03em;margin:0}.lede{font-size:18px;color:#43514b;max-width:760px;margin:12px 0 0}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.metric-card{padding:24px;border:1px solid var(--line);background:var(--card);border-radius:16px;box-shadow:0 8px 24px rgba(36,59,83,.05)}.metric-card>span{font-size:13px;color:var(--muted);letter-spacing:.08em}.metric-card strong{display:block;font-family:Georgia,serif;font-size:36px;line-height:1.2;margin:7px 0}.metric-card div{display:flex;justify-content:space-between;gap:10px}.metric-card s{color:#89928e}.metric-card em{font-style:normal;color:var(--green);font-weight:800}.note{border-left:3px solid var(--orange);padding:14px 18px;background:#fff8ee;margin-top:18px;color:#5d4937}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;counter-reset:step}.step{position:relative;background:var(--navy);color:white;padding:24px 19px;border-radius:14px;min-height:176px}.step:before{counter-increment:step;content:"0" counter(step);font:700 12px/1 Georgia;color:#85d6bd;letter-spacing:.1em}.step h3{font-size:18px;margin:22px 0 8px}.step p{font-size:14px;color:#d6e0dc;margin:0}.step:not(:last-child):after{content:"→";position:absolute;right:-18px;top:42%;z-index:2;color:var(--orange);font-size:24px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid2>*{min-width:0}.panel{border:1px solid var(--line);background:var(--card);padding:26px;border-radius:16px}.panel h3{margin:0 0 14px;font-size:19px}.parameter{display:grid;grid-template-columns:1fr auto;gap:12px;padding:9px 0;border-bottom:1px solid #ece8de}.parameter:last-child{border:0}.parameter span{color:var(--muted)}.parameter b{font-family:ui-monospace,Consolas,monospace;color:var(--green)}
.chart-grid{display:grid;gap:18px}.chart-card{margin:0;background:var(--card);border:1px solid var(--line);padding:20px;border-radius:16px;overflow:hidden}.chart-card figcaption{display:flex;justify-content:space-between}.chart-card figcaption span{color:var(--muted);font-size:13px}.chart-card svg{width:100%;height:auto;margin-top:12px}.grid{stroke:#e7e3d9;stroke-width:1}.axis{font-size:11px;fill:#818a86}.line{fill:none;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}.reference{stroke:#a8afac}.dynamic{stroke:var(--green)}.legend{display:flex;gap:20px;justify-content:flex-end;color:var(--muted);font-size:13px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.ref-dot{background:#a8afac}.dyn-dot{background:var(--green)}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:15px;background:var(--card)}table{width:100%;border-collapse:collapse;min-width:680px}th,td{text-align:left;padding:14px 16px;border-bottom:1px solid #e8e4d9;font-size:14px}th{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);background:#f8f5ed}tr:last-child td{border-bottom:0}td b{color:var(--green)}
.prob{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.prob div{background:#e7f3ed;padding:19px;border-radius:13px}.prob strong{display:block;color:var(--green);font:32px/1.2 Georgia,serif}.prob span{font-size:13px;color:#506059}.warning{margin-top:18px;padding:20px;border:1px dashed #c66b32;border-radius:13px;background:#fff8ee}.warning strong{color:#9f4d1b}
.source-list{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.source{padding:21px;border:1px solid var(--line);border-radius:14px;background:var(--card)}.source code{color:var(--green);font-weight:700}.source p{color:var(--muted);font-size:14px;margin-bottom:0}.download{background:var(--ink);color:white;border-radius:20px;padding:34px;display:grid;grid-template-columns:1fr auto;gap:25px;align-items:center}.download h2{margin:0 0 7px;font:34px Georgia,serif}.download p{margin:0;color:#cbd5d1}.download a{display:block;text-decoration:none;background:var(--green2);color:#06241a;font-weight:800;padding:13px 18px;border-radius:9px;margin:7px 0;text-align:center}
footer{border-top:1px solid var(--line);padding:30px 0 50px;color:var(--muted);font-size:13px}
@media(max-width:900px){.metrics,.prob{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:1fr 1fr}.step:not(:last-child):after{display:none}.source-list{grid-template-columns:1fr}.section-head{grid-template-columns:1fr}.grid2{grid-template-columns:1fr}.download{grid-template-columns:1fr}.topbar nav{display:none}}
@media(max-width:560px){.wrap{width:min(100% - 24px,1160px)}.hero{padding:54px 0}.metrics,.prob,.flow{grid-template-columns:1fr}.metric-card strong{font-size:30px}}
@media print{.topbar nav,.download{display:none}body{background:white}.hero{padding:38px 0}section{break-inside:avoid;margin-bottom:42px}.chart-card{break-inside:avoid}}
</style></head><body>
<div class="wrap topbar"><b>REGIME DECISION TEST · RESEARCH NOTE</b><nav><a href="#logic">전략</a><a href="#results">성과</a><a href="#robustness">강건성</a><a href="#run">실행</a></nav></div>
<header><div class="wrap hero"><span class="eyebrow">VKOSPI = Korea's VIX</span><h1>공포가 커질 때,<br>위험을 금으로 옮긴다.</h1><p>VKOSPI의 절대 수준과 단기 급등을 하루 늦춰 읽고, 기준 포트폴리오의 국내주식·원유 노출 일부를 GLD로 이동하는 동적 위험 오버레이입니다.</p><span class="stamp">✓ Locked 2018–2026 · CAGR / Sharpe / MDD 동시 개선</span></div></header>
<main class="wrap">
<section id="results"><div class="section-head"><small>01 · OUTCOME</small><div><h2>검증 구간에서 세 목표를 함께 개선</h2><p class="lede">월간 기준 전략과 일간 오버레이의 상대수익을 결합한 권위 비교 기준입니다. MDD의 양(+)의 변화는 낙폭이 줄었다는 뜻입니다.</p></div></div><div class="metrics">__METRICS__</div><div class="note">검증 구간: 2018-01–2026-07, 103개월. 동일 구간의 실제 일간 재구성에서도 CAGR 17.31% → 17.74%, Sharpe 1.178 → 1.294, MDD −12.83% → −11.59%로 세 지표가 모두 개선되었습니다.</div></section>
<section id="logic"><div class="section-head"><small>02 · SIGNAL</small><div><h2>미래를 보지 않는 다섯 단계</h2><p class="lede">t일 종가로 신호를 만든 뒤 t+1일 오픈-투-오픈 수익을 얻습니다. 당일 종가를 당일 거래에 쓰지 않습니다.</p></div></div><div class="flow"><article class="step"><h3>VKOSPI 원자료</h3><p>한국판 VIX의 일별 종가. 2003–2026 데이터를 정제·중복 제거.</p></article><article class="step"><h3>수준</h3><p>최근 252거래일 안에서 현재 값의 인과적 백분위. 126일 전에는 신호 없음.</p></article><article class="step"><h3>급등</h3><p>5거래일 변화율이 +15%를 넘는 정도를 연속형 강도로 환산.</p></article><article class="step"><h3>스트레스</h3><p>수준 강도와 급등 강도의 평균. 0–1 범위로 제한.</p></article><article class="step"><h3>오버레이</h3><p>KODEX200·USO에서 최대 25%를 GLD로 이동. 15% 밴드로 잦은 거래 억제.</p></article></div></section>
<section><div class="section-head"><small>03 · DESIGN</small><div><h2>선택된 파라미터와 자산 이동</h2></div></div><div class="grid2"><div class="panel"><h3>승자 설정</h3><div class="parameter"><span>결합 방식</span><b>__MODE__</b></div><div class="parameter"><span>VKOSPI 수준 문턱</span><b>__LEVEL__ percentile</b></div><div class="parameter"><span>모멘텀 창</span><b>__WINDOW__ days</b></div><div class="parameter"><span>급등 문턱</span><b>__SPIKE__</b></div><div class="parameter"><span>최대 위험 이전</span><b>__TRANSFER__</b></div><div class="parameter"><span>채권 배분 몫</span><b>__BOND__</b></div><div class="parameter"><span>리밸런싱 밴드</span><b>__BAND__</b></div></div><div class="panel"><h3>해석</h3><p>선택된 <b>bond_share=0</b>은 방어 이전분 전부가 GLD로 향한다는 뜻입니다. 원래 기준 비중을 통째로 대체하지 않고, 스트레스 강도에 비례해 KODEX200과 USO의 일부만 줄입니다.</p><p>탐색은 2007–2017 전체와 2013–2017 하위 검증에서 <b>CAGR·Sharpe·MDD가 모두 개선</b>되는 후보만 통과시켰습니다. 528개 거친 탐색 뒤 315개 정밀 탐색을 수행했고, 전체 836개 중 96개가 동시 기준을 충족했습니다.</p><p>2018년 이후 데이터는 파라미터 선택에 사용하지 않고 마지막에 한 번 잠갔습니다.</p></div></div></section>
<section><div class="section-head"><small>04 · PATH</small><div><h2>성과 경로</h2><p class="lede">전 구간에서는 CAGR과 Sharpe가 개선되고 MDD는 동일 수준입니다. 아래 선은 월간 권위 기준으로 재조정한 비교입니다.</p></div></div><div class="chart-grid">__NAV_CHART____DD_CHART__</div></section>
<section id="robustness"><div class="section-head"><small>05 · ROBUSTNESS</small><div><h2>비용과 시기별 점검</h2></div></div><div class="grid2"><div><h3>거래비용 민감도 · 실제 일간</h3><div class="table-wrap"><table><thead><tr><th>비용</th><th>CAGR</th><th>Sharpe</th><th>MDD</th></tr></thead><tbody>__COST_ROWS__</tbody></table></div></div><div><h3>하위기간 · 월간 재조정</h3><div class="table-wrap"><table><thead><tr><th>기간</th><th>CAGR</th><th>Sharpe</th><th>MDD</th></tr></thead><tbody>__SUBPERIOD_ROWS__</tbody></table></div></div></div></section>
<section><div class="section-head"><small>06 · UNCERTAINTY</small><div><h2>부트스트랩이 말하는 확률</h2><p class="lede">6개월 블록을 짝지어 5,000회 재표본화했습니다. 시계열 의존성과 기준/후보 간 공통 시장 충격을 보존합니다.</p></div></div><div class="prob"><div><strong>__P_CAGR__</strong><span>CAGR 개선</span></div><div><strong>__P_SHARPE__</strong><span>Sharpe 개선</span></div><div><strong>__P_MDD__</strong><span>MDD 개선</span></div><div><strong>__P_ALL__</strong><span>세 지표 동시 개선</span></div></div><div class="warning"><strong>해석 주의.</strong> Sharpe와 MDD 개선 확률은 높지만 CAGR 개선 확률은 약 63%입니다. 따라서 이 오버레이는 확정적 초과수익 엔진이라기보다, 위험 대비 수익과 낙폭 경로를 개선하려는 방어 규칙으로 보는 편이 타당합니다. 과거 시뮬레이션이며 미래 성과를 보장하지 않습니다.</div></section>
<section><div class="section-head"><small>07 · OAP MAP</small><div><h2>Open Asset Pricing 자료에서 가져온 설계 원칙</h2><p class="lede">종목 횡단면 신호를 그대로 복제한 것이 아니라, 공개 신호 문서의 변동성·추세 표현 방식을 시장 시계열 위험관리 문제에 맞게 변환했습니다.</p></div></div><div class="source-list"><article class="source"><code>betaVIX</code><p>Ang et al. (2006)의 체계적 변동성 신호. VIX 변화에 대한 민감도를 쓰는 아이디어를 VKOSPI 수준/변화 기반 시장 스트레스로 옮겼습니다.</p></article><article class="source"><code>RealizedVol</code><p>과거 일별 자료로 변동성 상태를 요약하는 가격 기반 신호. 여기서는 미래 정보 없는 252일 백분위로 상태를 표준화했습니다.</p></article><article class="source"><code>TrendFactor / Momentum</code><p>가격 변화 방향과 크기를 신호화하는 원칙을 VKOSPI 5일 급등 강도로 적용했습니다.</p></article></div><p class="note">출처: <a href="https://openassetpricing.com/SignalDoc-Browser.html">Chen–Zimmermann Signal Library · SignalDoc Browser</a>. OAP 신호는 개별 주식 수익 예측 문헌이고, 본 구현은 VKOSPI 기반 자산배분 오버레이이므로 경제적 적용 대상이 다릅니다.</p></section>
<section id="run"><div class="download"><div><h2>Colab에서 그대로 재현</h2><p>노트북을 열고 함께 만든 번들 ZIP을 업로드하면, 입력 데이터 확인 → 2단계 재탐색 → 잠금 검증 → 차트·표 → 결과 ZIP 저장까지 순서대로 실행됩니다.</p></div><div><a href="../notebooks/vkospi_dynamic_strategy_colab.ipynb" download>① Colab 노트북</a><a href="../bundles/vkospi_dynamic_colab_bundle.zip" download>② 실행 번들 ZIP</a></div></div></section>
</main><footer><div class="wrap">Generated from <code>results/vkospi_dynamic_validation.json</code> · 투자 조언이 아닌 연구용 시뮬레이션입니다.</div></footer>
</body></html>"""
    replacements = {
        "__METRICS__": metric_cards(locked),
        "__MODE__": html.escape(str(winner["mode"])),
        "__LEVEL__": f'{winner["level_threshold"]:.2f}',
        "__WINDOW__": str(winner["momentum_window"]),
        "__SPIKE__": percent(winner["spike_threshold"], 0),
        "__TRANSFER__": percent(winner["max_risk_transfer"], 0),
        "__BOND__": percent(winner["bond_share"], 0),
        "__BAND__": percent(winner["rebalance_band"], 0),
        "__NAV_CHART__": nav_chart,
        "__DD_CHART__": dd_chart,
        "__COST_ROWS__": cost_rows(costs),
        "__SUBPERIOD_ROWS__": subperiod_rows(subperiods),
        "__P_CAGR__": percent(boot["probability_cagr_improves"], 1),
        "__P_SHARPE__": percent(boot["probability_sharpe_improves"], 1),
        "__P_MDD__": percent(boot["probability_mdd_improves"], 1),
        "__P_ALL__": percent(boot["probability_all_three_improve"], 1),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def make_notebook() -> dict:
    cells = [
        markdown_cell(
            """# VKOSPI 동적 위험 오버레이 — Colab 재현 노트북

VKOSPI를 **한국판 VIX**로 해석하고, 전일 종가에서 만든 수준·5일 급등 신호로 기준 포트폴리오의 KODEX200/USO 일부를 GLD로 옮깁니다.

- 입력: `vkospi_dynamic_colab_bundle.zip` (이 노트북과 함께 생성됨)
- 기본값: Colab에서는 836개 후보를 다시 탐색하고 2018–2026 잠금 구간을 검증
- 예상 시간: 런타임에 따라 약 3–6분
- 주의: 연구용 과거 시뮬레이션이며 투자 조언이 아닙니다.
"""
        ),
        markdown_cell("## 1. 런타임 준비"),
        code_cell(
            """import sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "matplotlib"],
        check=True,
    )

import json, shutil, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

print("Colab runtime:", IN_COLAB)
print("Python:", sys.version.split()[0], "| pandas:", pd.__version__)
"""
        ),
        markdown_cell(
            """## 2. 실행 번들 불러오기

Colab이면 파일 선택창에서 **`vkospi_dynamic_colab_bundle.zip`**을 업로드하세요. 로컬 Jupyter에서는 노트북을 프로젝트 루트에 두면 현재 폴더를 그대로 사용합니다.
"""
        ),
        code_cell(
            """def safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)

if IN_COLAB:
    from google.colab import files
    uploaded = files.upload()
    bundle_name = next((name for name in uploaded if name.endswith(".zip")), None)
    if bundle_name is None:
        raise FileNotFoundError("vkospi_dynamic_colab_bundle.zip을 업로드해 주세요.")
    safe_extract(Path(bundle_name), Path("/content"))
    PROJECT_ROOT = Path("/content/RegimeDecisionTest")
else:
    PROJECT_ROOT = Path.cwd().resolve()

required = [
    "strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py",
    "raw_data/VKOSPIData.csv",
    "raw_data/compass.db",
    "raw_data/krx_bond_index.csv",
    "cache/market_daily.csv",
    "results/vkospi_selected_backtest.csv",
]
missing = [name for name in required if not (PROJECT_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(f"실행 번들에 필요한 파일이 없습니다: {missing}")

print("PROJECT_ROOT =", PROJECT_ROOT)
print("필수 입력 6개 확인 완료")
"""
        ),
        markdown_cell(
            """## 3. 전체 재탐색 및 잠금 검증

`RUN_FULL_RECALIBRATION=True`이면 2007–2017 자료만으로 528개 거친 후보와 315개 정밀 후보를 평가합니다. 2018년 이후 구간은 승자를 고르는 데 쓰지 않습니다. 빠르게 기존 산출물만 살펴보려면 `False`로 바꾸세요.
"""
        ),
        code_cell(
            """RUN_FULL_RECALIBRATION = True if IN_COLAB else False

if RUN_FULL_RECALIBRATION:
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "strategies.stage06_vkospi.vkospi_dynamic_risk_experiment",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    print(completed.stdout[-5000:])
else:
    print("재탐색을 생략하고 번들 안의 검증 결과를 읽습니다.")
"""
        ),
        markdown_cell("## 4. 선택된 규칙과 잠금 성과"),
        code_cell(
            """RESULTS = PROJECT_ROOT / "results"
report = json.loads((RESULTS / "vkospi_dynamic_validation.json").read_text(encoding="utf-8"))
locked = report["locked"]["monthly_reference_reconciled"]

display(Markdown("### 선택된 파라미터"))
display(pd.Series(report["winner"], name="value").to_frame())

metric_names = ["CAGR", "Sharpe", "MDD", "Calmar"]
metrics = pd.DataFrame({
    "Reference": pd.Series(locked["reference"])[metric_names],
    "VKOSPI Dynamic": pd.Series(locked["candidate"])[metric_names],
    "Delta": pd.Series(locked["deltas"])[metric_names],
})
display(Markdown("### Locked 2018–2026 · 월간 기준 재조정"))
display(metrics.style.format("{:.4f}"))
print("CAGR / Sharpe / MDD 동시 개선:", locked["passes_all_three"])
"""
        ),
        markdown_cell(
            """### 비교 기준 메모

`monthly_reference_reconciled`는 이미 검증된 월간 기준수익에 **동적 일간 전략 / 기준 일간 재구성**의 상대수익 팩터를 곱합니다. 월간→일간 변환 오차를 오버레이 알파로 오인하지 않기 위한 권위 비교이며, 별도로 실제 일간 재구성 결과도 보고합니다.
"""
        ),
        code_cell(
            """reference = pd.read_csv(RESULTS / "vkospi_selected_backtest.csv", index_col="month")
dynamic = pd.read_csv(RESULTS / "vkospi_dynamic_reconciled_monthly.csv", index_col="month")
common = reference.index.intersection(dynamic.index)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(common, reference.loc[common, "nav"], label="Reference", color="#9aa3a0", lw=2)
axes[0].plot(common, dynamic.loc[common, "nav"], label="VKOSPI Dynamic", color="#007a5e", lw=2.4)
axes[0].set_title("Cumulative NAV · monthly reconciled")
axes[0].set_ylabel("NAV")
axes[0].legend()
axes[0].grid(alpha=.2)
axes[1].fill_between(range(len(common)), 100 * reference.loc[common, "drawdown"], color="#aab2af", alpha=.35, label="Reference")
axes[1].plot(range(len(common)), 100 * dynamic.loc[common, "drawdown"], color="#007a5e", lw=2, label="VKOSPI Dynamic")
axes[1].set_ylabel("Drawdown (%)")
axes[1].set_xticks(range(0, len(common), 24), common[::24], rotation=45)
axes[1].grid(alpha=.2)
axes[1].legend()
plt.tight_layout()
plt.show()
"""
        ),
        markdown_cell("## 5. 실제 신호와 위험 이전"),
        code_cell(
            """daily = pd.read_csv(RESULTS / "vkospi_dynamic_daily.csv", parse_dates=["date", "signal_date"])
locked_daily = daily.loc[daily["date"] >= "2018-01-01"].copy()

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
axes[0].plot(locked_daily["date"], locked_daily["stress"], color="#f28c45", lw=1)
axes[0].set_ylabel("Stress 0–1")
axes[0].set_title("VKOSPI stress and defensive transfer")
axes[0].grid(alpha=.2)
axes[1].fill_between(locked_daily["date"], 100 * locked_daily["transfer_fraction"], color="#007a5e", alpha=.55)
axes[1].set_ylabel("Transfer (%)")
axes[1].grid(alpha=.2)
plt.tight_layout()
plt.show()

assert (locked_daily["signal_date"].dropna().to_numpy() < locked_daily.loc[locked_daily["signal_date"].notna(), "date"].to_numpy()).all()
print("Look-ahead 점검: 모든 신호일이 수익 발생일보다 앞섭니다.")
"""
        ),
        markdown_cell("## 6. 비용·하위기간·불확실성"),
        code_cell(
            """costs = pd.read_csv(RESULTS / "vkospi_dynamic_cost_sensitivity.csv")
subperiods = pd.read_csv(RESULTS / "vkospi_dynamic_subperiods.csv")

display(Markdown("### 거래비용 민감도"))
display(costs.loc[:, ["Period", "Strategy", "CAGR", "Sharpe", "MDD", "TotalCost"]].style.format({
    "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}", "TotalCost": "{:.4f}"
}))
display(Markdown("### 하위기간"))
display(subperiods.loc[:, ["Period", "Strategy", "CAGR", "Sharpe", "MDD"]].style.format({
    "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}"
}))

bootstrap = report["locked"]["reconciled_multiobjective_bootstrap"]
display(Markdown("### 6개월 블록 부트스트랩 · 5,000회"))
display(pd.Series({
    "CAGR 개선 확률": bootstrap["probability_cagr_improves"],
    "Sharpe 개선 확률": bootstrap["probability_sharpe_improves"],
    "MDD 개선 확률": bootstrap["probability_mdd_improves"],
    "세 지표 동시 개선 확률": bootstrap["probability_all_three_improve"],
}).to_frame("probability").style.format("{:.1%}"))
"""
        ),
        markdown_cell(
            """## 7. Open Asset Pricing 연결과 해석 한계

[Chen–Zimmermann Signal Library](https://openassetpricing.com/SignalDoc-Browser.html)의 `betaVIX`, `RealizedVol`, `TrendFactor/Momentum`에서 **변동성 상태와 변화율을 과거 관측만으로 표현**하는 원칙을 참고했습니다. 다만 원 자료는 개별 주식 횡단면 수익 예측 신호이고, 이 노트북은 VKOSPI 시장 시계열로 자산군 노출을 조절하므로 직접 복제나 동일한 경제적 검정은 아닙니다.

Sharpe/MDD 개선의 부트스트랩 확률보다 CAGR 개선 확률이 낮습니다. 미래 수익 보장이나 실거래 체결 보장이 아니며, 세금·슬리피지·상품 추적오차·운용 제약은 별도 검토가 필요합니다.
"""
        ),
        markdown_cell("## 8. 결과 묶음 저장"),
        code_cell(
            """output_zip = Path("/content/vkospi_dynamic_results.zip") if IN_COLAB else PROJECT_ROOT / "artifacts/bundles/vkospi_dynamic_results.zip"
with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RESULTS.glob("vkospi_dynamic_*")):
        archive.write(path, arcname=path.name)
print("저장 완료:", output_zip)

if IN_COLAB:
    from google.colab import files
    files.download(str(output_zip))
"""
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"vkospi-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_bundle() -> None:
    members = [
        ROOT / "strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py",
        ROOT / "tests/test_vkospi_dynamic_risk_experiment.py",
        ROOT / "raw_data" / "VKOSPIData.csv",
        ROOT / "raw_data" / "compass.db",
        ROOT / "raw_data" / "krx_bond_index.csv",
        ROOT / "cache" / "market_daily.csv",
        RESULTS / "vkospi_selected_backtest.csv",
        *sorted(RESULTS.glob("vkospi_dynamic_*")),
    ]
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            if not path.exists():
                raise FileNotFoundError(path)
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, arcname=f"RegimeDecisionTest/{relative}")


def main() -> None:
    report = json.loads(
        (RESULTS / "vkospi_dynamic_validation.json").read_text(encoding="utf-8")
    )
    costs = pd.read_csv(RESULTS / "vkospi_dynamic_cost_sensitivity.csv")
    subperiods = pd.read_csv(RESULTS / "vkospi_dynamic_subperiods.csv")
    HTML_PATH.write_text(make_html(report, costs, subperiods), encoding="utf-8")
    NOTEBOOK_PATH.write_text(
        json.dumps(make_notebook(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    make_bundle()
    print(f"HTML: {HTML_PATH} ({HTML_PATH.stat().st_size:,} bytes)")
    print(f"Notebook: {NOTEBOOK_PATH} ({NOTEBOOK_PATH.stat().st_size:,} bytes)")
    print(f"Bundle: {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
