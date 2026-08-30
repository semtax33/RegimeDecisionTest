from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path

import pandas as pd

from tools.builders.build_vkospi_dynamic_deliverables import chart_svg, code_cell, markdown_cell, percent


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HTML_PATH = ROOT / "artifacts/reports/vkospi_robust_dynamic_strategy_explainer.html"
NOTEBOOK_PATH = ROOT / "artifacts/notebooks/vkospi_robust_dynamic_strategy_colab.ipynb"
BUNDLE_PATH = ROOT / "artifacts/bundles/vkospi_robust_dynamic_colab_bundle.zip"


def metric_cards(report: dict) -> str:
    old = report["locked"]["existing"]
    new = report["locked"]["robust"]
    delta = report["locked"]["deltas"]
    specs = (
        ("CAGR", percent(old["CAGR"]), percent(new["CAGR"]), f"{100 * delta['CAGR']:+.2f}%p"),
        ("Sharpe", f"{old['Sharpe']:.3f}", f"{new['Sharpe']:.3f}", f"{delta['Sharpe']:+.3f}"),
        ("MDD", percent(old["MDD"]), percent(new["MDD"]), f"{100 * delta['MDD']:+.2f}%p"),
        ("Calmar", f"{old['Calmar']:.3f}", f"{new['Calmar']:.3f}", f"{delta['Calmar']:+.3f}"),
    )
    return "".join(
        f'<article class="metric"><span>{name}</span><strong>{after}</strong>'
        f'<div><s>{before}</s><b>{change}</b></div></article>'
        for name, before, after, change in specs
    )


def comparison_rows(comparison: pd.DataFrame) -> str:
    rows = []
    for period in (
        "calibration_2007_2017",
        "validation_2013_2017",
        "locked_2018_2026",
        "full_2007_2026",
    ):
        view = comparison.loc[
            (comparison["Period"] == period)
            & comparison["Strategy"].isin(
                ["ExistingDynamicReconciled", "RobustDynamicReconciled"]
            )
        ].set_index("Strategy")
        old = view.loc["ExistingDynamicReconciled"]
        new = view.loc["RobustDynamicReconciled"]
        label = {
            "calibration_2007_2017": "선정 · 2007–2017",
            "validation_2013_2017": "내부검증 · 2013–2017",
            "locked_2018_2026": "잠금 · 2018–2026",
            "full_2007_2026": "전체 · 2007–2026",
        }[period]
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{percent(old['CAGR'])} → <b>{percent(new['CAGR'])}</b></td>"
            f"<td>{old['Sharpe']:.3f} → <b>{new['Sharpe']:.3f}</b></td>"
            f"<td>{percent(old['MDD'])} → <b>{percent(new['MDD'])}</b></td></tr>"
        )
    return "".join(rows)


def cost_rows(costs: pd.DataFrame) -> str:
    rows = []
    for multiplier in (1.0, 2.0):
        period = f"cost_{multiplier:.1f}x_locked"
        view = costs.loc[costs["Period"] == period].set_index("Strategy")
        old, new = view.loc["ExistingDynamic"], view.loc["RobustDynamic"]
        rows.append(
            f"<tr><td>{multiplier:.0f}×</td>"
            f"<td>{percent(old['CAGR'])} → <b>{percent(new['CAGR'])}</b></td>"
            f"<td>{old['Sharpe']:.3f} → <b>{new['Sharpe']:.3f}</b></td>"
            f"<td>{percent(old['MDD'])} → <b>{percent(new['MDD'])}</b></td></tr>"
        )
    return "".join(rows)


def make_html(report: dict, comparison: pd.DataFrame, costs: pd.DataFrame) -> str:
    old_path = pd.read_csv(
        RESULTS / "vkospi_dynamic_reconciled_monthly.csv", index_col=0
    )
    new_path = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_reconciled_monthly.csv", index_col=0
    )
    old_path.index = pd.PeriodIndex(old_path.index, freq="M")
    new_path.index = pd.PeriodIndex(new_path.index, freq="M")
    nav = chart_svg(old_path, new_path, "nav", "누적 자산가치")
    drawdown = chart_svg(old_path, new_path, "drawdown", "드로다운")
    for before, after in (
        ("기준 전략", "기존 VKOSPI 동적"),
        ("VKOSPI 동적 전략", "새 robust VKOSPI"),
    ):
        nav = nav.replace(before, after)
        drawdown = drawdown.replace(before, after)
    winner = report["winner"]
    boot = report["locked"]["bootstrap"]
    template = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VKOSPI Robust Dynamic 개선 전략</title><style>
:root{--ink:#17211d;--muted:#66736d;--paper:#f3f1e9;--card:#fffdf7;--line:#d9d4c8;--green:#087f5b;--mint:#dff2e8;--gold:#d88934;--navy:#203a43}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Apple SD Gothic Neo","Noto Sans KR",sans-serif;line-height:1.65}a{color:var(--green)}
.wrap{width:min(1160px,calc(100% - 40px));margin:auto}.top{padding:18px 0;display:flex;justify-content:space-between;font-size:13px}.top nav a{margin-left:22px;text-decoration:none;color:var(--ink)}
header{border-block:1px solid var(--line);background:radial-gradient(circle at 87% 12%,rgba(8,127,91,.17),transparent 32%),linear-gradient(135deg,#fffaf0,#e7f2eb)}.hero{padding:78px 0}.eyebrow{color:var(--green);font-weight:800;letter-spacing:.13em;text-transform:uppercase}.hero h1{font:clamp(43px,7vw,80px)/1.02 Georgia,"Noto Serif KR",serif;letter-spacing:-.055em;margin:14px 0 22px;max-width:930px}.hero p{font-size:19px;max-width:800px;color:#405049}.stamp{display:inline-block;margin-top:12px;border:1px solid #98c9b7;border-radius:999px;padding:9px 14px;color:var(--green);font-weight:800;background:#ffffff99}
main{padding:64px 0 100px}section{margin-bottom:72px}.head{display:grid;grid-template-columns:150px 1fr;gap:22px;margin-bottom:26px}.head small{color:var(--green);font-weight:800;letter-spacing:.1em}.head h2{font:clamp(29px,4vw,43px)/1.15 Georgia,"Noto Serif KR",serif;letter-spacing:-.03em;margin:0}.lede{font-size:17px;color:#4f5e57;max-width:820px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.metric{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px}.metric span{font-size:13px;color:var(--muted);letter-spacing:.08em}.metric strong{display:block;font:36px/1.2 Georgia,serif;margin:7px 0}.metric div{display:flex;justify-content:space-between}.metric s{color:#8c9691}.metric b,td b{color:var(--green)}.note{margin-top:18px;border-left:3px solid var(--gold);background:#fff8eb;padding:14px 18px}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;counter-reset:step}.step{background:var(--navy);color:white;border-radius:14px;padding:22px 18px;min-height:190px}.step:before{counter-increment:step;content:"0" counter(step);color:#83d4b8;font:700 12px Georgia}.step h3{margin:20px 0 8px}.step p{margin:0;color:#d5e0dc;font-size:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid2>*{min-width:0}.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:25px}.panel h3{margin-top:0}.param{display:grid;grid-template-columns:1fr auto;padding:9px 0;border-bottom:1px solid #ece7db}.param span{color:var(--muted)}.param b{color:var(--green);font-family:Consolas,monospace}.feature-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.feature{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}.feature code{color:var(--green);font-weight:800}.feature p{font-size:14px;color:var(--muted);margin-bottom:0}
.chart-grid{display:grid;gap:18px}.chart-card{margin:0;background:var(--card);border:1px solid var(--line);padding:20px;border-radius:16px;overflow:hidden}.chart-card figcaption{display:flex;justify-content:space-between}.chart-card figcaption span{color:var(--muted);font-size:13px}.chart-card svg{width:100%;height:auto;margin-top:12px}.grid{stroke:#e7e3d9}.axis{font-size:11px;fill:#818a86}.line{fill:none;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}.reference{stroke:#9ca6a1}.dynamic{stroke:var(--green)}.legend{display:flex;justify-content:flex-end;gap:18px;color:var(--muted);font-size:13px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.ref-dot{background:#9ca6a1}.dyn-dot{background:var(--green)}
.table{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:15px}table{width:100%;border-collapse:collapse;min-width:680px}th,td{text-align:left;padding:14px 16px;border-bottom:1px solid #e8e4d9;font-size:14px}th{font-size:12px;color:var(--muted);background:#f8f5ed}.prob{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.prob div{background:var(--mint);border-radius:13px;padding:19px}.prob strong{display:block;color:var(--green);font:32px Georgia,serif}.prob span{font-size:13px;color:#53645c}.warning{margin-top:18px;border:1px dashed #c57732;background:#fff8eb;border-radius:13px;padding:18px}.download{background:var(--ink);color:white;border-radius:20px;padding:32px;display:grid;grid-template-columns:1fr auto;gap:22px;align-items:center}.download h2{margin:0;font:34px Georgia,serif}.download p{color:#c9d5d0}.download a{display:block;background:#24c38e;color:#07271d;text-decoration:none;font-weight:800;padding:12px 17px;border-radius:9px;margin:7px;text-align:center}footer{border-top:1px solid var(--line);padding:30px 0 50px;color:var(--muted);font-size:13px}
@media(max-width:900px){.head{grid-template-columns:1fr}.metrics,.prob{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:1fr 1fr}.feature-grid{grid-template-columns:1fr}.grid2,.download{grid-template-columns:1fr}.top nav{display:none}}@media(max-width:560px){.wrap{width:calc(100% - 24px)}.hero{padding:52px 0}.metrics,.prob,.flow{grid-template-columns:1fr}.metric strong{font-size:30px}}@media print{body{background:white}.top nav,.download{display:none}section{break-inside:avoid}}
</style></head><body>
<div class="wrap top"><b>REGIME DECISION TEST · RESEARCH NOTE</b><nav><a href="#result">성과</a><a href="#logic">가공</a><a href="#validation">검증</a><a href="#run">실행</a></nav></div>
<header><div class="wrap hero"><span class="eyebrow">VKOSPI = Korea's VIX · Robust reprocessing</span><h1>공포의 수준보다<br>급격한 가속을 본다.</h1><p>VKOSPI 장·단기 상대수준과 변동성 정규화 충격, 5일 가속을 결합해 방어 전환을 더 선택적으로 만든 일별 위험 오버레이입니다.</p><span class="stamp">✓ 2018–2026 잠금 구간 · CAGR / Sharpe / MDD 동시 개선</span></div></header>
<main class="wrap">
<section id="result"><div class="head"><small>01 · OUTCOME</small><div><h2>기존 VKOSPI 동적 전략을 다시 개선</h2><p class="lede">동일한 월별 기준수익과 비용 가정을 사용했습니다. MDD 변화가 양수이면 최대 낙폭이 줄었다는 뜻입니다.</p></div></div><div class="metrics">__METRICS__</div><div class="note">잠금 구간은 2018-01–2026-07의 103개월입니다. 실제 일별 경로에서도 CAGR 17.74% → 17.89%, Sharpe 1.294 → 1.328, MDD −11.59% → −11.44%로 세 지표가 함께 개선됐습니다.</div></section>
<section id="logic"><div class="head"><small>02 · PROCESS</small><div><h2>VKOSPI를 다섯 층으로 다시 표현</h2><p class="lede">모든 값은 t일 종가까지만 사용하며 t+1일 오픈-투-오픈 수익에 적용됩니다.</p></div></div><div class="flow"><article class="step"><h3>원자료</h3><p>2003–2026 VKOSPI OHLC. 날짜·숫자 정제와 중복 제거.</p></article><article class="step"><h3>상대 수준</h3><p>126/252일 인과적 백분위와 63/252일 median·MAD robust z-score.</p></article><article class="step"><h3>정규화 충격</h3><p>5/10/21일 로그변화를 후행 63일 변동성으로 나눠 시기 간 크기를 비교.</p></article><article class="step"><h3>경로 상태</h3><p>5일 가속, 21일 고점 거리, 종가 위치, 상승일 비율과 fast-minus-slow.</p></article><article class="step"><h3>위험 이전</h3><p>높은 수준·충격·가속이 겹칠 때 KODEX200·USO 일부를 GLD로 이전.</p></article></div></section>
<section><div class="head"><small>03 · FEATURES</small><div><h2>Open Asset Pricing에서 빌린 것은 표현 원칙</h2></div></div><div class="feature-grid"><article class="feature"><code>betaVIX</code><p>시장 변동성 변화에 대한 민감도를 다룬다는 발상을 VKOSPI 자체의 수준·충격 상태로 변환했습니다.</p></article><article class="feature"><code>RealizedVol</code><p>과거 가격경로로 변동성 상태를 표현하는 원칙을 63일 스케일 정규화에 반영했습니다.</p></article><article class="feature"><code>Momentum / Trend</code><p>방향과 지속성을 구분하는 원칙을 5/10/21일 충격, 가속, fast-minus-slow 특징으로 옮겼습니다.</p></article></div><div class="note">Chen–Zimmermann 자료는 212개 수준의 <b>개별 주식 횡단면 예측 신호 라이브러리</b>입니다. 본 전략은 이를 직접 복제하거나 동일 경제효과를 주장하지 않고, 시장 시계열 위험관리용 입력 표현에만 참고했습니다. <a href="https://openassetpricing.com/SignalDoc-Browser.html">SignalDoc Browser</a></div></section>
<section><div class="head"><small>04 · RULE</small><div><h2>2017년 이전에 고정한 승자 규칙</h2></div></div><div class="grid2"><div class="panel"><h3>선택된 설정</h3><div class="param"><span>스트레스 방식</span><b>__MODE__</b></div><div class="param"><span>252일 수준 문턱</span><b>__LEVEL__</b></div><div class="param"><span>정규화 충격 문턱</span><b>__SHOCK__</b></div><div class="param"><span>최대 위험 이전</span><b>__TRANSFER__</b></div><div class="param"><span>방어 수단</span><b>GLD 100%</b></div><div class="param"><span>리밸런싱 밴드</span><b>__BAND__</b></div></div><div class="panel"><h3>선택 절차</h3><p>5개 스트레스 결합 × 3개 수준 문턱 × 3개 충격 문턱 × 3개 이전 한도 × 2개 방어배분 × 3개 밴드, 총 <b>810개</b>를 평가했습니다.</p><p>2007–2017 전체와 2013–2017 내부검증에서 기존 동적 전략보다 CAGR·Sharpe·MDD가 모두 개선된 후보만 우선 통과시켰고, 그 조건을 만족한 후보는 <b>2개</b>였습니다.</p><p>2018년 이후 데이터는 파라미터 선택에 사용하지 않았습니다.</p></div></div></section>
<section><div class="head"><small>05 · PATH</small><div><h2>성과 경로</h2><p class="lede">월별 기준 전략에 일별 오버레이의 상대수익 팩터만 결합해, 월간↔일간 변환 오차를 알파로 세지 않았습니다.</p></div></div><div class="chart-grid">__NAV____DD__</div></section>
<section id="validation"><div class="head"><small>06 · VALIDATION</small><div><h2>선정·잠금·비용 점검</h2></div></div><div class="grid2"><div><h3>기간별 월간 재조정</h3><div class="table"><table><thead><tr><th>기간</th><th>CAGR</th><th>Sharpe</th><th>MDD</th></tr></thead><tbody>__COMPARISON__</tbody></table></div></div><div><h3>잠금 구간 실제 일별 · 비용 민감도</h3><div class="table"><table><thead><tr><th>비용</th><th>CAGR</th><th>Sharpe</th><th>MDD</th></tr></thead><tbody>__COSTS__</tbody></table></div></div></div></section>
<section><div class="head"><small>07 · UNCERTAINTY</small><div><h2>개선 폭은 작고, 불확실성은 남는다</h2><p class="lede">잠금 구간 월수익을 6개월 블록으로 5,000회 짝지어 재표본화했습니다.</p></div></div><div class="prob"><div><strong>__PCAGR__</strong><span>CAGR 개선 확률</span></div><div><strong>__PSHARPE__</strong><span>Sharpe 개선 확률</span></div><div><strong>__PMDD__</strong><span>MDD 개선 확률</span></div><div><strong>__PALL__</strong><span>세 지표 동시 개선 확률</span></div></div><div class="warning"><b>해석:</b> 관측 잠금 성과는 세 목표를 모두 개선했지만 CAGR 개선의 재표본 확률은 높지 않습니다. 이 결과는 큰 초과수익보다는 불필요한 방어 진입을 줄여 위험 대비 수익을 미세 조정한 결과로 보는 편이 타당합니다. 실거래·세금·상품 추적오차와 미래 구조변화는 보장하지 않습니다.</div></section>
<section id="run"><div class="download"><div><h2>Google Colab에서 재현</h2><p>노트북과 번들 ZIP을 함께 올리면 810개 후보 재탐색, 잠금 검증, 차트와 결과 ZIP 생성까지 실행됩니다.</p></div><div><a href="../notebooks/vkospi_robust_dynamic_strategy_colab.ipynb" download>① Colab 노트북</a><a href="../bundles/vkospi_robust_dynamic_colab_bundle.zip" download>② 실행 번들 ZIP</a></div></div></section>
</main><footer><div class="wrap">Generated from <code>results/vkospi_robust_dynamic_validation.json</code> · 연구용 시뮬레이션이며 투자 조언이 아닙니다.</div></footer></body></html>'''
    replacements = {
        "__METRICS__": metric_cards(report),
        "__MODE__": html.escape(str(winner["mode"])),
        "__LEVEL__": f"top {100 * (1 - winner['level_threshold']):.0f}%",
        "__SHOCK__": f"{winner['shock_threshold']:.1f}σ",
        "__TRANSFER__": percent(winner["max_risk_transfer"], 0),
        "__BAND__": percent(winner["rebalance_band"], 0),
        "__NAV__": nav,
        "__DD__": drawdown,
        "__COMPARISON__": comparison_rows(comparison),
        "__COSTS__": cost_rows(costs),
        "__PCAGR__": percent(boot["probability_cagr_improves"], 1),
        "__PSHARPE__": percent(boot["probability_sharpe_improves"], 1),
        "__PMDD__": percent(boot["probability_mdd_improves"], 1),
        "__PALL__": percent(boot["probability_all_three_improve"], 1),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def make_notebook() -> dict:
    cells = [
        markdown_cell("""# VKOSPI robust dynamic 개선 전략 — Colab 재현

VKOSPI를 한국판 VIX로 해석해 **장·단기 상대수준, robust z-score, 변동성 정규화 충격, 가속·고점거리**를 만들고, 2017년 이전 자료만으로 810개 일별 방어 오버레이를 다시 선택합니다.

- 함께 제공된 `vkospi_robust_dynamic_colab_bundle.zip`을 업로드하세요.
- 기본 실행은 약 4–8분이며 런타임에 따라 달라질 수 있습니다.
- 2018–2026은 선택에 쓰지 않는 잠금 검증입니다.
- 연구용 과거 시뮬레이션이며 투자 조언이 아닙니다.
"""),
        markdown_cell("## 1. 런타임과 번들 준비"),
        code_cell("""import sys, subprocess, json, zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "openpyxl",
        ],
        check=True,
    )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

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
        raise FileNotFoundError("vkospi_robust_dynamic_colab_bundle.zip을 업로드하세요.")
    safe_extract(bundle, Path("/content"))
    PROJECT_ROOT = Path("/content/RegimeDecisionTest")
else:
    PROJECT_ROOT = Path.cwd().resolve()

required = [
    "strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py",
    "strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py",
    "strategies/stage06_vkospi/vkospi_extended_diagnostics.py",
    "strategies/stage06_vkospi/vkospi_model_robustness.py",
    "strategies/core/regime_research.py",
    "raw_data/VKOSPIData.csv",
    "raw_data/compass.db",
    "raw_data/krx_bond_index.csv",
    "cache/market_daily.csv",
    "results/vkospi_selected_backtest.csv",
    "results/vkospi_dynamic_validation.json",
]
missing = [name for name in required if not (PROJECT_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(missing)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("Python", sys.version.split()[0], "| 입력 확인 완료")
"""),
        markdown_cell("""## 2. 810개 후보 재탐색

`RUN_SEARCH=True`가 기본입니다. 선정에는 2007–2017 전체와 2013–2017 내부검증만 쓰며 2018년 이후는 마지막에 평가합니다. 번들에 저장된 결과만 빠르게 열려면 `False`로 바꾸세요.
"""),
        code_cell("""RUN_SEARCH = True
if RUN_SEARCH:
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "strategies.stage06_vkospi.vkospi_robust_dynamic_experiment",
        ],
        cwd=PROJECT_ROOT, text=True, capture_output=True, check=True,
    )
    print(completed.stdout[-6000:])
else:
    print("재탐색을 생략하고 번들의 검증 결과를 읽습니다.")
"""),
        markdown_cell("""## 3. 상수·분류력·기간성과·오버피팅 진단

`RUN_DIAGNOSTICS=True`이면 배포 전략은 그대로 둔 채 다음 항목을 다시 계산합니다.

- 거시 상수 0.20·0.55·0.10·0.85의 1변수 민감도
- 16개 꼬리손실 설명변수의 AUC와 순차 로지스틱 계수
- ROC AUC·평균정밀도·Brier·LogLoss·ECE
- 요청한 2005~2026 구간의 데이터 가용성과 실제 전략 가능 구간
- 810개 격자의 승자 주변, 잠금 연도, 부트스트랩 기반 오버피팅 감사

이 진단은 잠금 결과를 보고 파라미터를 다시 고르는 데 쓰지 않습니다.
"""),
        code_cell("""RUN_DIAGNOSTICS = True
if RUN_DIAGNOSTICS:
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "strategies.stage06_vkospi.vkospi_extended_diagnostics",
        ],
        cwd=PROJECT_ROOT, text=True, capture_output=True, check=True,
    )
    print(completed.stdout[-5000:])
else:
    print("재계산을 생략하고 번들에 저장된 진단 결과를 읽습니다.")
"""),
        markdown_cell("""## 4. SJM·로지스틱 하이퍼파라미터 강건성

`RUN_MODEL_ROBUSTNESS=True`이면 다음 감사를 재실행합니다. 번들에는 1차 계산 캐시가 들어 있어 로지스틱 walk-forward는 재사용하고, SJM 후보 경로와 모든 요약·재현 단언을 다시 확인합니다.

- 로지스틱 28개: C, L1/L2/ElasticNet, liblinear/lbfgs/saga, 클래스 가중
- SJM 49개 시도: 점프 벌점, 희소 변수 수, 혼합비, SJM 미사용
- 2017년 이전 8블록 CSCV/PBO 70분할, 잠금 6개월 블록 부트스트랩 2,000회
- 2018년 이후는 재선정이 아니라 순위 안정성과 실패 확인에만 사용
"""),
        code_cell("""RUN_MODEL_ROBUSTNESS = True
if RUN_MODEL_ROBUSTNESS:
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "strategies.stage06_vkospi.vkospi_model_robustness",
        ],
        cwd=PROJECT_ROOT, text=True, capture_output=True, check=True,
    )
    print(completed.stdout[-8000:])
else:
    print("재계산을 생략하고 번들의 강건성 결과를 읽습니다.")

RESULTS = PROJECT_ROOT / "results"
model_audit = json.loads((RESULTS / "vkospi_model_robustness.json").read_text(encoding="utf-8"))
display(Markdown("### 핵심 선택위험 — 낮을수록 안정적인 PBO"))
display(pd.Series({
    "로지스틱 예측 Brier PBO": model_audit["logistic"]["prelock_prediction_brier_pbo"]["pbo"],
    "로지스틱 포트폴리오 Sharpe PBO": model_audit["logistic"]["prelock_portfolio_sharpe_pbo"]["pbo"],
    "수렴경고 제외 Sharpe PBO": model_audit["logistic"]["prelock_portfolio_sharpe_pbo_excluding_warning_candidates"]["pbo"],
    "SJM 거시 Brier PBO": model_audit["sjm"]["prelock_macro_brier_pbo"]["pbo"],
    "SJM soft Sharpe PBO": model_audit["sjm"]["prelock_soft_sharpe_pbo"]["pbo"],
}).to_frame("PBO").style.format("{:.1%}"))

display(Markdown(
    "포트폴리오 기준 PBO가 높아 **오버피팅이 없다고 결론낼 수 없습니다.** "
    "잠금 결과는 더 좋아 보이는 후보로 갈아타는 데 사용하지 않습니다."
))
"""),
        markdown_cell("## 5. 승자와 잠금 성과"),
        code_cell("""RESULTS = PROJECT_ROOT / "results"
report = json.loads((RESULTS / "vkospi_robust_dynamic_validation.json").read_text(encoding="utf-8"))
display(Markdown("### 2017년 이전에 선택된 설정"))
display(pd.Series(report["winner"], name="value").to_frame())

locked = report["locked"]
metrics = pd.DataFrame({
    "Existing dynamic": pd.Series(locked["existing"]),
    "Robust dynamic": pd.Series(locked["robust"]),
    "Delta": pd.Series(locked["deltas"]),
}).loc[["CAGR", "Sharpe", "MDD", "Calmar"]]
display(Markdown("### Locked 2018–2026 · 월간 기준 재조정"))
display(metrics.style.format("{:.4f}"))
assert locked["passes_all_three"]
assert locked["deltas"]["CAGR"] > 0 and locked["deltas"]["Sharpe"] > 0 and locked["deltas"]["MDD"] >= 0
print("CAGR / Sharpe / MDD 동시 개선 검증 완료")
"""),
        markdown_cell("## 6. 누적 성과와 드로다운"),
        code_cell("""old = pd.read_csv(RESULTS / "vkospi_dynamic_reconciled_monthly.csv", index_col=0)
new = pd.read_csv(RESULTS / "vkospi_robust_dynamic_reconciled_monthly.csv", index_col=0)
common = old.index.intersection(new.index)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(common, old.loc[common, "nav"], color="#9ca6a1", lw=2, label="Existing dynamic")
axes[0].plot(common, new.loc[common, "nav"], color="#087f5b", lw=2.4, label="Robust dynamic")
axes[0].set_ylabel("NAV"); axes[0].grid(alpha=.2); axes[0].legend()
axes[1].plot(range(len(common)), 100 * old.loc[common, "drawdown"], color="#9ca6a1", lw=2, label="Existing")
axes[1].plot(range(len(common)), 100 * new.loc[common, "drawdown"], color="#087f5b", lw=2, label="Robust")
axes[1].set_ylabel("Drawdown (%)"); axes[1].grid(alpha=.2); axes[1].legend()
axes[1].set_xticks(range(0, len(common), 24), common[::24], rotation=45)
plt.tight_layout(); plt.show()
"""),
        markdown_cell("## 7. VKOSPI 가공 신호와 look-ahead 점검"),
        code_cell("""daily = pd.read_csv(RESULTS / "vkospi_robust_dynamic_daily.csv", parse_dates=["date", "signal_date"])
locked_daily = daily.loc[daily["date"] >= "2018-01-01"].copy()

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
axes[0].plot(locked_daily["date"], locked_daily["stress"], color="#d88934", lw=1)
axes[0].set_ylabel("Robust stress"); axes[0].grid(alpha=.2)
axes[1].fill_between(locked_daily["date"], 100 * locked_daily["transfer_fraction"], color="#087f5b", alpha=.55)
axes[1].set_ylabel("Transfer (%)"); axes[1].grid(alpha=.2)
plt.tight_layout(); plt.show()

valid = daily["signal_date"].notna()
assert (daily.loc[valid, "signal_date"].to_numpy() < daily.loc[valid, "date"].to_numpy()).all()
assert daily["stress"].between(0, 1).all()
print("모든 신호일 < 수익일, stress ∈ [0, 1]")
"""),
        markdown_cell("## 8. 기간·비용·불확실성"),
        code_cell("""comparison = pd.read_csv(RESULTS / "vkospi_robust_dynamic_comparison.csv")
costs = pd.read_csv(RESULTS / "vkospi_robust_dynamic_cost_sensitivity.csv")
display(comparison[["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar"]].style.format({"CAGR":"{:.2%}", "Sharpe":"{:.3f}", "MDD":"{:.2%}", "Calmar":"{:.3f}"}))
display(costs[["Period", "Strategy", "CAGR", "Sharpe", "MDD"]].style.format({"CAGR":"{:.2%}", "Sharpe":"{:.3f}", "MDD":"{:.2%}"}))

boot = locked["bootstrap"]
display(pd.Series({
    "CAGR 개선": boot["probability_cagr_improves"],
    "Sharpe 개선": boot["probability_sharpe_improves"],
    "MDD 개선": boot["probability_mdd_improves"],
    "세 지표 동시 개선": boot["probability_all_three_improve"],
}).to_frame("6개월 블록 부트스트랩").style.format("{:.1%}"))
"""),
        markdown_cell("## 9. AUC·Brier·16개 설명변수와 2005 요청구간"),
        code_cell("""tail_prediction = pd.read_csv(RESULTS / "vkospi_tail_prediction_diagnostics.csv")
tail_features = pd.read_csv(RESULTS / "vkospi_tail_feature_diagnostics.csv")
period_performance = pd.read_csv(RESULTS / "vkospi_extended_period_performance.csv")
macro_sensitivity = pd.read_csv(RESULTS / "vkospi_macro_constant_sensitivity.csv")
overfit = json.loads((RESULTS / "vkospi_overfitting_diagnostics.json").read_text(encoding="utf-8"))

display(Markdown("### 꼬리손실 분류 성능 — AUC는 순위, Brier는 확률오차"))
display(tail_prediction.style.format({
    "event_rate": "{:.2%}", "roc_auc": "{:.4f}", "average_precision": "{:.4f}",
    "brier_score": "{:.4f}", "calibration_prevalence_brier": "{:.4f}",
    "recall_at_top_20pct": "{:.2%}", "precision_at_top_20pct": "{:.2%}",
}))
display(Markdown(
    "잠금 AUC가 0.7565여도 Brier 0.2193은 단순 사건률 고정예측 0.0903보다 나쁩니다. "
    "그래서 원시 확률을 그대로 비중으로 쓰지 않고 과거 예측 내 인과적 백분위로 바꿉니다."
))

display(Markdown("### 16개 입력변수 — 단변량 AUC와 표준화 로지스틱 계수"))
display(tail_features.style.format({
    "raw_univariate_auc": "{:.3f}",
    "direction_free_auc": "{:.3f}",
    "median_standardized_logit_coefficient": "{:+.3f}",
    "coefficient_sign_stability": "{:.1%}",
}))

display(Markdown("### 2005~2026 요청구간과 실제 측정 가능 구간"))
display(period_performance.style.format({
    "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}", "Calmar": "{:.3f}",
}))
display(Markdown(
    "네 자산 공통 월수익은 2006-04에 시작하고 24개월 워밍업 뒤 첫 거래월은 2007-04입니다. "
    "따라서 2005년부터의 동일 전략 성과는 만들 수 없으며, 2005~2026 KODEX200은 별도 시장 벤치마크입니다."
))

display(Markdown("### 거시 상수 민감도와 오버피팅 감사"))
display(macro_sensitivity.loc[
    macro_sensitivity["parameter"].eq("deployed")
    | macro_sensitivity["parameter"].eq("sjm_weight")
].style.format({"mean_brier": "{:.4f}", "quadrant_accuracy": "{:.2%}"}))
display(pd.Series({
    "후보 수": overfit["candidate_count"],
    "엄격 통과": overfit["strict_pass_count"],
    "승자-차점자 점수차": overfit["winner_runner_score_gap"],
    "잠금 연도 우위": f'{overfit["locked_years_robust_outperformed"]}/{overfit["locked_years_total"]}',
    "결론": overfit["conclusion"],
}).to_frame("audit"))
"""),
        markdown_cell("""## 10. Open Asset Pricing 연결과 한계

[Chen–Zimmermann SignalDoc Browser](https://openassetpricing.com/SignalDoc-Browser.html)는 개별 주식 횡단면 예측 신호 라이브러리입니다. 본 실험은 `betaVIX`, `RealizedVol`, momentum/trend 계열에서 **변동성 상태·방향·지속성을 과거 자료로 표현하는 원칙**만 참고해 VKOSPI 시장 시계열 입력으로 변환했습니다. 직접 복제나 같은 경제적 검정을 주장하지 않습니다.

관측 잠금 성과의 개선 폭은 작으며 부트스트랩의 세 지표 동시 개선 확률도 확정적이지 않습니다. 세금·슬리피지·추적오차·상품 교체와 미래 구조변화는 별도 검토해야 합니다.
"""),
        markdown_cell("## 11. 결과 다운로드"),
        code_cell("""output = Path("/content/vkospi_robust_dynamic_results.zip") if IN_COLAB else PROJECT_ROOT / "vkospi_robust_dynamic_results.zip"
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    output_patterns = (
        "vkospi_robust_dynamic_*",
        "vkospi_macro_constant_sensitivity.csv",
        "vkospi_tail_feature_diagnostics.csv",
        "vkospi_tail_prediction_diagnostics.csv",
        "vkospi_extended_period_performance.csv",
        "vkospi_robust_grid_neighborhood.csv",
        "vkospi_overfitting_diagnostics.json",
        "vkospi_locked_annual_relative_performance.csv",
        "vkospi_logistic_*",
        "vkospi_sjm_*",
        "vkospi_model_robustness.json",
    )
    output_files = []
    for pattern in output_patterns:
        output_files.extend(RESULTS.glob(pattern))
    for path in sorted(set(output_files)):
        archive.write(path, arcname=path.name)
print("저장 완료:", output)
if IN_COLAB:
    from google.colab import files
    files.download(str(output))
"""),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"robust-vkospi-{index:02d}"
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
        ROOT / "strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py",
        ROOT / "strategies/stage06_vkospi/vkospi_extended_diagnostics.py",
        ROOT / "strategies/stage06_vkospi/vkospi_model_robustness.py",
        ROOT / "strategies/core/regime_research.py",
        ROOT / "tests/test_vkospi_dynamic_risk_experiment.py",
        ROOT / "tests/test_vkospi_robust_dynamic_experiment.py",
        ROOT / "tests/test_vkospi_model_robustness.py",
        *sorted((ROOT / "raw_data").iterdir()),
        ROOT / "cache" / "market_daily.csv",
        RESULTS / "vkospi_selected_backtest.csv",
        RESULTS / "regime_signals.csv",
        RESULTS / "hard_crash_features.csv",
        RESULTS / "openassetpricing_composites.csv",
        RESULTS / "openassetpricing_medium_horizon_factor.csv",
        RESULTS / "openassetpricing_medium_horizon_backtest.csv",
        RESULTS / "proposed_backtest.csv",
        *sorted(RESULTS.glob("vkospi_dynamic_*")),
        *sorted(RESULTS.glob("vkospi_robust_dynamic_*")),
        RESULTS / "vkospi_macro_constant_sensitivity.csv",
        RESULTS / "vkospi_tail_feature_diagnostics.csv",
        RESULTS / "vkospi_tail_prediction_diagnostics.csv",
        RESULTS / "vkospi_extended_period_performance.csv",
        RESULTS / "vkospi_robust_grid_neighborhood.csv",
        RESULTS / "vkospi_overfitting_diagnostics.json",
        RESULTS / "vkospi_locked_annual_relative_performance.csv",
        *sorted(RESULTS.glob("vkospi_logistic_*")),
        *sorted(RESULTS.glob("vkospi_sjm_*")),
        RESULTS / "vkospi_model_robustness.json",
    ]
    unique = list(dict.fromkeys(members))
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in unique:
            if not path.exists():
                raise FileNotFoundError(path)
            archive.write(path, arcname=f"RegimeDecisionTest/{path.relative_to(ROOT).as_posix()}")


def main() -> None:
    report = json.loads(
        (RESULTS / "vkospi_robust_dynamic_validation.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(RESULTS / "vkospi_robust_dynamic_comparison.csv")
    costs = pd.read_csv(RESULTS / "vkospi_robust_dynamic_cost_sensitivity.csv")
    HTML_PATH.write_text(make_html(report, comparison, costs), encoding="utf-8")
    NOTEBOOK_PATH.write_text(
        json.dumps(make_notebook(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    make_bundle()
    for path in (HTML_PATH, NOTEBOOK_PATH, BUNDLE_PATH):
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
