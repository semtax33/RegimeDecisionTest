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
HTML_PATH = ROOT / "artifacts/reports/top3_regime_model_report.html"
NOTEBOOK_PATH = ROOT / "artifacts/notebooks/top3_regime_models_colab.ipynb"
BUNDLE_PATH = ROOT / "artifacts/bundles/top3_regime_models_colab_bundle.zip"


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def signed_pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:+.{digits}f}%p"


def choose_best_locked(comparison: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    locked = comparison.loc[comparison["Period"] == "locked_2018_2026"].copy()
    baseline = locked.loc[locked["Strategy"] == "Existing_VKOSPI_Dynamic"].iloc[0]
    challengers = locked.loc[locked["Strategy"] != "Existing_VKOSPI_Dynamic"].copy()
    for metric in ("CAGR", "Sharpe", "MDD", "Calmar"):
        challengers[f"rank_{metric}"] = challengers[metric].rank(pct=True)
    challengers["score"] = challengers[
        ["rank_CAGR", "rank_Sharpe", "rank_MDD", "rank_Calmar"]
    ].mean(axis=1)
    best = challengers.sort_values(["score", "Sharpe"], ascending=False).iloc[0]
    return baseline, best


def svg_paths(series: dict[str, pd.Series], width: int = 980, height: int = 320) -> str:
    combined = np.concatenate([value.to_numpy(dtype=float) for value in series.values()])
    low, high = float(np.nanmin(combined)), float(np.nanmax(combined))
    colors = {
        "Existing_VKOSPI_Dynamic": "#87928d",
        "CJM": "#377dff",
        "TVTP-HMM": "#e07a3f",
        "CJM+LightGBM": "#007a5e",
    }
    output = []
    for name, values in series.items():
        x = np.linspace(36, width - 26, len(values))
        y = height - 28 - (values.to_numpy(dtype=float) - low) / max(high - low, 1e-12) * (height - 56)
        path = "M " + " L ".join(f"{xv:.2f},{yv:.2f}" for xv, yv in zip(x, y))
        output.append(
            f'<path d="{path}" fill="none" stroke="{colors.get(name, "#333")}" stroke-width="2.5" stroke-linejoin="round"/>'
        )
    grid = []
    for tick in np.linspace(low, high, 5):
        y = height - 28 - (tick - low) / max(high - low, 1e-12) * (height - 56)
        grid.append(
            f'<line x1="36" x2="954" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>'
            f'<text x="30" y="{y + 4:.2f}" text-anchor="end" class="axis">{tick:.1f}</text>'
        )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(grid)}{"".join(output)}</svg>'


def performance_rows(comparison: pd.DataFrame, period: str) -> str:
    names = ["Existing_VKOSPI_Dynamic", "CJM", "TVTP-HMM", "CJM+LightGBM"]
    view = comparison.loc[comparison["Period"] == period].set_index("Strategy")
    rows = []
    for name in names:
        row = view.loc[name]
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(name.replace('_', ' '))}</b></td>"
            f"<td>{pct(row['CAGR'])}</td><td>{row['Sharpe']:.3f}</td>"
            f"<td>{pct(row['MDD'])}</td><td>{row['Calmar']:.3f}</td>"
            f"<td>{row['AvgTurnover']:.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def probability_rows(metrics: pd.DataFrame) -> str:
    view = metrics.loc[metrics["Period"] == "locked_2018_2026"]
    rows = []
    for model in ("SJM", "CJM", "TVTP-HMM", "CJM+LightGBM"):
        for _, row in view.loc[view["Model"] == model].sort_values("Horizon").iterrows():
            rows.append(
                "<tr>"
                f"<td>{html.escape(model)}</td><td>{int(row['Horizon'])}M</td>"
                f"<td>{row['Brier']:.3f}</td><td>{row['LogLoss']:.3f}</td>"
                f"<td>{row['ECE5']:.3f}</td><td>{row['TransitionRecall']:.1%}</td>"
                f"<td>{row['AUC']:.3f}</td>"
                "</tr>"
            )
    return "".join(rows)


def config_cards(report: dict) -> str:
    cards = []
    for model, detail in report["validation"].items():
        cfg = detail["selected_config"]
        deltas = detail["locked_deltas"]
        status = "세 지표 동시 개선" if detail["passes_all_three"] else "일부 지표만 개선"
        cards.append(
            f"""<article class="config"><span>{html.escape(model)}</span><h3>{status}</h3>
            <dl><dt>확률 문턱</dt><dd>{cfg['threshold']:.2f}</dd><dt>최대 방어 혼합</dt><dd>{pct(cfg['max_shift'],0)}</dd>
            <dt>채권 몫</dt><dd>{pct(cfg['bond_share'],0)}</dd><dt>1/3/6M 가중치</dt><dd>{' / '.join(f'{v:.2f}' for v in cfg['horizon_weights'])}</dd></dl>
            <p>CAGR {signed_pct(deltas['CAGR'])} · Sharpe {deltas['Sharpe']:+.3f} · MDD {signed_pct(deltas['MDD'])}</p></article>"""
        )
    return "".join(cards)


def make_html(report: dict, comparison: pd.DataFrame, metrics: pd.DataFrame) -> str:
    baseline, best = choose_best_locked(comparison)
    promoted = [name for name, detail in report["validation"].items() if detail["passes_all_three"]]
    if promoted:
        headline = f"{', '.join(promoted)}가 잠금 구간 세 목표를 함께 개선"
        decision = "개선안 승격 가능"
    else:
        headline = "상위 3개 모델을 적용했지만 잠금 구간의 기존 전략을 완전히 넘지는 못함"
        decision = "기존 전략 유지 권고"

    series = {}
    baseline_path = pd.read_csv(
        RESULTS / "vkospi_dynamic_reconciled_monthly.csv", index_col="month"
    )
    series["Existing_VKOSPI_Dynamic"] = baseline_path["nav"]
    for model, slug in (
        ("CJM", "cjm"),
        ("TVTP-HMM", "tvtp_hmm"),
        ("CJM+LightGBM", "cjm_plus_lightgbm"),
    ):
        frame = pd.read_csv(
            RESULTS / f"top3_regime_model_backtest_{slug}.csv", index_col="month"
        )
        series[model] = frame["nav"]
    nav_svg = svg_paths(series)

    template = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>추천 1~3위 국면모델 구현 보고서</title>
<style>
:root{--ink:#15211d;--muted:#66736d;--paper:#f4f1e8;--card:#fffdf8;--line:#d8d4c8;--green:#007a5e;--blue:#377dff;--orange:#e07a3f;--red:#b54d3b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Apple SD Gothic Neo",sans-serif;line-height:1.65}.wrap{width:min(1160px,calc(100% - 40px));margin:auto}a{color:var(--green)}
.bar{height:64px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);font-size:13px}.bar b{letter-spacing:.07em}.bar nav a{text-decoration:none;color:var(--ink);margin-left:24px}
header{background:linear-gradient(125deg,#172821 0%,#21483b 58%,#356759 100%);color:white}.hero{padding:86px 0 76px}.eyebrow{color:#7be0bc;text-transform:uppercase;font-size:13px;font-weight:800;letter-spacing:.14em}.hero h1{font:clamp(38px,6vw,70px)/1.08 Georgia,"Noto Serif KR",serif;letter-spacing:-.045em;max-width:980px;margin:15px 0 24px}.hero p{color:#d5e4de;font-size:18px;max-width:820px}.badge{display:inline-block;margin-top:18px;padding:10px 16px;border:1px solid #70caae;border-radius:999px;color:#8eedce;font-weight:700}
main{padding:68px 0 100px}section{margin-bottom:76px}.head{display:grid;grid-template-columns:150px 1fr;gap:25px;margin-bottom:28px}.head small{color:var(--green);font-weight:800;letter-spacing:.1em}.head h2{font:clamp(28px,4vw,42px)/1.18 Georgia,"Noto Serif KR",serif;letter-spacing:-.03em;margin:0}.head p{color:var(--muted);font-size:17px;max-width:780px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:13px}.metric{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:22px}.metric span{color:var(--muted);font-size:13px}.metric strong{display:block;font:34px Georgia,serif;margin:7px 0}.metric em{font-style:normal;color:var(--green);font-weight:800}.note{background:#fff7e9;border-left:3px solid var(--orange);padding:16px 18px;margin-top:18px}
.models{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.model{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px}.model i{font-style:normal;font:32px Georgia,serif;color:var(--green)}.model h3{margin:10px 0}.model p{font-size:14px;color:var(--muted)}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.step{background:#21372f;color:white;padding:21px;border-radius:13px;min-height:150px}.step b{color:#76dbb9;font-size:12px}.step h3{font-size:17px;margin:18px 0 6px}.step p{font-size:13px;color:#d0ddd8;margin:0}
.table{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:15px}table{width:100%;border-collapse:collapse;min-width:760px}th,td{padding:13px 15px;border-bottom:1px solid #e8e4d9;text-align:left;font-size:14px}th{background:#f8f5ed;color:var(--muted);text-transform:uppercase;font-size:11px;letter-spacing:.07em}tr:last-child td{border:0}
.configs{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.config{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:22px}.config>span{color:var(--green);font-weight:800}.config h3{margin:7px 0 15px}.config dl{display:grid;grid-template-columns:1fr auto;margin:0}.config dt,.config dd{padding:7px 0;border-bottom:1px solid #ebe7dc}.config dt{color:var(--muted)}.config dd{font-family:Consolas,monospace}.config p{font-size:13px;color:var(--muted)}
.chart{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px}.chart svg{width:100%;height:auto}.grid{stroke:#e7e3d9}.axis{fill:#808a85;font-size:10px}.legend{display:flex;gap:18px;flex-wrap:wrap;justify-content:flex-end;font-size:13px;color:var(--muted)}.legend i{width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:5px}
.warning{padding:22px;border:1px dashed var(--red);background:#fff4f0;border-radius:14px}.sources{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.source{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}.source code{color:var(--green);font-weight:800}.source p{font-size:14px;color:var(--muted)}
.download{display:grid;grid-template-columns:1fr auto;gap:22px;background:var(--ink);color:white;padding:32px;border-radius:18px}.download h2{font:32px Georgia,serif;margin:0}.download p{color:#cbd5d1}.download a{display:block;background:#31c692;color:#06261a;text-decoration:none;font-weight:800;padding:12px 17px;border-radius:8px;margin:7px;text-align:center}
footer{border-top:1px solid var(--line);padding:28px 0 50px;color:var(--muted);font-size:13px}
@media(max-width:900px){.metrics{grid-template-columns:repeat(2,1fr)}.models,.configs,.sources{grid-template-columns:1fr}.flow{grid-template-columns:1fr 1fr}.head{grid-template-columns:1fr}.download{grid-template-columns:1fr}.bar nav{display:none}}
@media(max-width:560px){.wrap{width:calc(100% - 24px)}.metrics,.flow{grid-template-columns:1fr}.hero{padding:55px 0}.metric strong{font-size:29px}}
@media print{body{background:white}.bar nav,.download{display:none}section{break-inside:avoid;margin-bottom:42px}}
</style></head><body>
<div class="wrap bar"><b>REGIME MODEL CHALLENGE · 2026</b><nav><a href="#models">모델</a><a href="#probability">확률</a><a href="#portfolio">성과</a><a href="#reproduce">재현</a></nav></div>
<header><div class="wrap hero"><span class="eyebrow">Feedback rank 1–3 · implemented</span><h1>__HEADLINE__</h1><p>CJM+LightGBM, TVTP-HMM, CJM을 실제 인과적 워크포워드로 구현하고 현재 VKOSPI 동적 전략 위에서 동일한 방어 규칙으로 비교했습니다.</p><span class="badge">__DECISION__ · Calibration ≤ 2017-12 · Locked ≥ 2018-01</span></div></header>
<main class="wrap">
<section><div class="head"><small>01 · LOCKED</small><div><h2>잠금 구간의 가장 균형 잡힌 도전자: __BEST_NAME__</h2><p>카드는 기존 VKOSPI 동적 전략과 도전자의 2018–2026 성과를 비교합니다. 이 선택은 보고용 종합 순위이며 파라미터 선택에는 잠금 데이터를 사용하지 않았습니다.</p></div></div><div class="metrics">
<article class="metric"><span>CAGR</span><strong>__BEST_CAGR__</strong><em>__DELTA_CAGR__</em></article><article class="metric"><span>Sharpe</span><strong>__BEST_SHARPE__</strong><em>__DELTA_SHARPE__</em></article><article class="metric"><span>MDD</span><strong>__BEST_MDD__</strong><em>__DELTA_MDD__</em></article><article class="metric"><span>Calmar</span><strong>__BEST_CALMAR__</strong><em>__DELTA_CALMAR__</em></article></div><div class="note">기존 잠금 성과: CAGR __BASE_CAGR__, Sharpe __BASE_SHARPE__, MDD __BASE_MDD__. “성과 개선안”은 구현과 검정의 대상이지 자동 승격을 뜻하지 않습니다.</div></section>
<section id="models"><div class="head"><small>02 · TOP 3</small><div><h2>피드백 상위 세 모델을 실제로 분리 구현</h2></div></div><div class="models"><article class="model"><i>🥇</i><h3>CJM → LightGBM</h3><p>공식 CJM 확률·변화량·지속기간을 1/3/6개월별 LightGBM에 투입하고 최근 검증구간으로 Platt calibration을 적용합니다.</p></article><article class="model"><i>🥈</i><h3>TVTP-HMM</h3><p>Gaussian MarkovRegression의 전이확률을 신용·금융여건·VIX/VKOSPI 함수로 만듭니다. 고정 전이 HMM이 아닙니다.</p></article><article class="model"><i>🥉</i><h3>CJM 단독</h3><p><code>JumpModel(cont=True)</code>의 온라인 확률과 경험적 전이행렬을 이용해 미래 Risk-Off 확률을 직접 전파합니다.</p></article></div></section>
<section><div class="head"><small>03 · TIMING</small><div><h2>월말 t에서 미래만 예측하는 경로</h2><p>Risk-Off 라벨은 KODEX200 월간 오픈-투-오픈 수익이 0보다 작은 달입니다. 모든 입력은 신호월 말까지로 제한됩니다.</p></div></div><div class="flow"><article class="step"><b>STEP 1</b><h3>Level</h3><p>매크로·신용·VIX/VKOSPI·시장 상태</p></article><article class="step"><b>STEP 2</b><h3>Momentum</h3><p>1M/3M 변화와 방향</p></article><article class="step"><b>STEP 3</b><h3>Z / Accel</h3><p>롤링 표준화와 변화속도</p></article><article class="step"><b>STEP 4</b><h3>1/3/6M</h3><p>각 horizon 별도 확률모델</p></article><article class="step"><b>STEP 5</b><h3>Overlay</h3><p>확률 문턱 이상만 채권/금 혼합</p></article></div></section>
<section id="probability"><div class="head"><small>04 · CALIBRATION</small><div><h2>잠금 구간 확률 품질</h2><p>Brier·LogLoss·ECE는 낮을수록, AUC·국면전환 Recall은 높을수록 좋습니다. 자산배분 전에 확률 자체를 따로 평가했습니다.</p></div></div><div class="table"><table><thead><tr><th>모델</th><th>시계</th><th>Brier ↓</th><th>LogLoss ↓</th><th>ECE ↓</th><th>Transition Recall ↑</th><th>AUC ↑</th></tr></thead><tbody>__PROBABILITY_ROWS__</tbody></table></div></section>
<section><div class="head"><small>05 · CONFIG</small><div><h2>2017년까지 선택된 방어 오버레이</h2></div></div><div class="configs">__CONFIG_CARDS__</div></section>
<section id="portfolio"><div class="head"><small>06 · PERFORMANCE</small><div><h2>전체기간과 잠금기간 성과</h2></div></div><h3>전체기간 · 2007–2026</h3><div class="table"><table><thead><tr><th>전략</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th><th>월평균 회전율</th></tr></thead><tbody>__FULL_ROWS__</tbody></table></div><h3>잠금기간 · 2018–2026</h3><div class="table"><table><thead><tr><th>전략</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th><th>월평균 회전율</th></tr></thead><tbody>__LOCKED_ROWS__</tbody></table></div></section>
<section><div class="head"><small>07 · PATH</small><div><h2>누적 NAV 경로</h2></div></div><div class="chart">__NAV_SVG__<div class="legend"><span><i style="background:#87928d"></i>기존 VKOSPI</span><span><i style="background:#377dff"></i>CJM</span><span><i style="background:#e07a3f"></i>TVTP-HMM</span><span><i style="background:#007a5e"></i>CJM+LightGBM</span></div></div></section>
<section><div class="head"><small>08 · LIMITS</small><div><h2>무엇을 믿고, 무엇을 경계할 것인가</h2></div></div><div class="warning"><b>작은 월간 표본과 상태 불확실성.</b> LightGBM은 2007년 이후 월간 자료를 사용하므로 복잡도를 강하게 제한했습니다. TVTP는 국면 라벨 전환과 수렴 불안정성이 있으며, CJM 상태는 경제적 진실이 아니라 데이터 기반 근사입니다. 부트스트랩도 구조 변화와 실거래 체결을 보장하지 않습니다.</div></section>
<section><div class="head"><small>09 · SOURCES</small><div><h2>구현 근거</h2></div></div><div class="sources"><article class="source"><code>jumpmodels</code><p>공식 라이브러리의 CJM, 온라인 확률 추론, scikit-learn형 API를 사용했습니다.</p><a href="https://github.com/Yizhan-Oliver-Shu/jump-models">GitHub</a></article><article class="source"><code>CJM paper</code><p>Aydınhan et al.의 Continuous Statistical Jump Model을 구현 근거로 사용했습니다.</p><a href="https://link.springer.com/article/10.1007/s10479-024-06035-z">Springer</a></article><article class="source"><code>statsmodels TVTP</code><p><code>MarkovRegression(exog_tvtp=...)</code>로 조건부 전이확률을 추정했습니다.</p><a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html">Documentation</a></article></div></section>
<section id="reproduce"><div class="download"><div><h2>Colab에서 전체 재현</h2><p>노트북과 번들 ZIP을 함께 사용하면 패키지 설치, 모델 재적합, 잠금 검증, 표·차트와 결과 ZIP 저장까지 실행됩니다.</p></div><div><a href="../notebooks/top3_regime_models_colab.ipynb" download>① Colab 노트북</a><a href="../bundles/top3_regime_models_colab_bundle.zip" download>② 실행 번들</a></div></div></section>
</main><footer><div class="wrap">Research simulation · Generated from top3_regime_model_validation.json · 투자 조언이 아닙니다.</div></footer></body></html>"""
    replacements = {
        "__HEADLINE__": html.escape(headline),
        "__DECISION__": html.escape(decision),
        "__BEST_NAME__": html.escape(str(best["Strategy"])),
        "__BEST_CAGR__": pct(best["CAGR"]),
        "__DELTA_CAGR__": signed_pct(best["CAGR"] - baseline["CAGR"]),
        "__BEST_SHARPE__": f'{best["Sharpe"]:.3f}',
        "__DELTA_SHARPE__": f'{best["Sharpe"] - baseline["Sharpe"]:+.3f}',
        "__BEST_MDD__": pct(best["MDD"]),
        "__DELTA_MDD__": signed_pct(best["MDD"] - baseline["MDD"]),
        "__BEST_CALMAR__": f'{best["Calmar"]:.3f}',
        "__DELTA_CALMAR__": f'{best["Calmar"] - baseline["Calmar"]:+.3f}',
        "__BASE_CAGR__": pct(baseline["CAGR"]),
        "__BASE_SHARPE__": f'{baseline["Sharpe"]:.3f}',
        "__BASE_MDD__": pct(baseline["MDD"]),
        "__PROBABILITY_ROWS__": probability_rows(metrics),
        "__CONFIG_CARDS__": config_cards(report),
        "__FULL_ROWS__": performance_rows(comparison, "full_2007_2026"),
        "__LOCKED_ROWS__": performance_rows(comparison, "locked_2018_2026"),
        "__NAV_SVG__": nav_svg,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def markdown_cell(source: str, identifier: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": identifier,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str, identifier: str) -> dict:
    return {
        "cell_type": "code",
        "id": identifier,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def make_notebook() -> dict:
    cells = [
        markdown_cell(
            """# 피드백 추천 1~3위 국면모델 구현 및 검증

이 노트북은 다음 세 개선안을 **실제 코드로 재현**하고 현재 `VKOSPI 동적 전략`과 비교합니다.

1. **CJM → LightGBM → Probability Calibration**
2. **TVTP-HMM**
3. **CJM 단독**

각 모델은 1·3·6개월 Risk-Off 확률을 별도로 만들며, Brier·LogLoss·ECE·국면전환 Recall과 CAGR·Sharpe·MDD를 함께 평가합니다. 연구용 시뮬레이션이며 투자 조언이 아닙니다.
""",
            "intro",
        ),
        markdown_cell("## 1. Colab 런타임과 패키지", "runtime-title"),
        code_cell(
            """import sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "lightgbm==4.6.0", "jumpmodels", "statsmodels==0.14.6",
        "openpyxl", "numpy", "pandas", "matplotlib", "scikit-learn",
    ], check=True)

import json, shutil, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

print("Colab:", IN_COLAB, "| Python:", sys.version.split()[0])
""",
            "runtime-code",
        ),
        markdown_cell(
            """## 2. 실행 번들 불러오기

Colab에서는 함께 제공된 `top3_regime_models_colab_bundle.zip`을 업로드합니다. ZIP은 경로 순회를 검사한 뒤 `/content/RegimeDecisionTest`에 해제합니다. 로컬에서는 노트북을 프로젝트 루트에서 실행하면 됩니다.
""",
            "bundle-title",
        ),
        code_cell(
            """def safe_extract(path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(path) as archive:
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
        raise FileNotFoundError("top3_regime_models_colab_bundle.zip을 업로드하세요.")
    safe_extract(bundle, Path("/content"))
    PROJECT_ROOT = Path("/content/RegimeDecisionTest")
else:
    PROJECT_ROOT = Path.cwd().resolve()

required = [
    "strategies/stage07_regime_models/top3_regime_model_experiment.py", "strategies/core/regime_research.py",
    "raw_data/compass.db", "cache/market_daily.csv",
    "cache/stress_monthly.csv", "results/vkospi_dynamic_reconciled_monthly.csv",
]
missing = [name for name in required if not (PROJECT_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(missing)
print("PROJECT_ROOT:", PROJECT_ROOT)
""",
            "bundle-code",
        ),
        markdown_cell(
            """## 3. 전체 워크포워드 실행

Colab 기본값은 `True`입니다. 약 수 분 동안 매월 다음 작업을 반복합니다.

- CJM: 과거 144개월 이내 자료만 robust scaling → `JumpModel(cont=True)` → 온라인 확률
- TVTP-HMM: 과거 수익률의 2상태 Gaussian emission + `exog_tvtp`
- LightGBM: horizon별로 **결과가 이미 알려진 라벨만** 학습 → 최근 15~24개월 Platt calibration
- 오버레이 선택: 2017-12까지만 사용, 2018-01 이후 잠금
""",
            "run-title",
        ),
        code_cell(
            """RUN_FULL_EXPERIMENT = True if IN_COLAB else False

if RUN_FULL_EXPERIMENT:
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "strategies.stage07_regime_models.top3_regime_model_experiment",
        ],
        cwd=PROJECT_ROOT, text=True, capture_output=True, check=True,
    )
    print(completed.stdout[-12000:])
else:
    print("번들의 사전 계산 결과를 사용합니다.")
""",
            "run-code",
        ),
        markdown_cell("## 4. 코드 구조와 입력변수", "features-title"),
        code_cell(
            """sys.path.insert(0, str(PROJECT_ROOT))
import strategies.stage07_regime_models.top3_regime_model_experiment as experiment
features, risk_off_label, asset_returns, baseline = experiment.build_master_features()
print("특성 패널:", features.shape, features.index.min(), "→", features.index.max())
display(pd.DataFrame({
    "CJM 입력": pd.Series(experiment.CJM_CANDIDATES),
    "TVTP 전이식": pd.Series(experiment.TVTP_CANDIDATES),
    "LightGBM 우선 입력": pd.Series(experiment.LGBM_PREFERRED),
}))
display(features.tail(3))
""",
            "features-code",
        ),
        markdown_cell(
            """### 입력변수 설계

- **Level:** GDP/수출/BSI, VIX, BAA spread, NFCI/STLFSI, VKOSPI, 시장 수익·변동성
- **Momentum:** 1개월·3개월 변화, 자산 3/6개월 누적수익
- **Z-score:** 60개월 스트레스 표준화, VKOSPI 63/252일 표준화
- **Acceleration:** 최근 변화량의 재변화
- **CJM 상태:** 현재 Risk-Off 확률, 1/3/6개월 전파확률, 확률 변화, 지속기간, 최근 전환 횟수

모든 행은 해당 월말까지 알려진 값입니다. Risk-Off 정답은 그 뒤 KODEX200 월간 오픈-투-오픈 수익이 음수인지로 정의합니다.
""",
            "features-explain",
        ),
        markdown_cell(
            """## 5. 세 모델의 실제 구현 방식

### CJM

공식 `jumpmodels.jump.JumpModel(cont=True, grid_size=0.10)`을 매월 다시 적합합니다. 수익률이 낮은 상태를 Risk-Off로 정렬하고 `predict_proba_online()`의 마지막 확률을 사용합니다. 전이행렬을 1·3·6번 곱해 미래 확률을 만듭니다.

### TVTP-HMM

`statsmodels.MarkovRegression(..., exog_tvtp=X)`의 전이확률은 고정 상수가 아니라 신용·금융여건·VIX/VKOSPI의 함수입니다. 전이식에는 한 달 전 관측치를 넣어 현재 상태 설명에 현재 정보를 역사용하지 않습니다.

### CJM + LightGBM

각 horizon을 별도 이진분류로 적합합니다. 작은 월간 표본을 고려해 깊이 3, 잎 7개로 제한하고, 마지막 검증구간 raw probability의 logit을 LogisticRegression에 넣어 Platt calibration합니다.
""",
            "models-explain",
        ),
        markdown_cell("## 6. 미래 정보 차단 감사", "audit-title"),
        code_cell(
            """RESULTS = PROJECT_ROOT / "results"
audit = pd.read_csv(RESULTS / "top3_regime_model_lgbm_audit.csv")
signal = pd.PeriodIndex(audit["signal_month"], freq="M")
max_label = pd.PeriodIndex(audit["max_label_month"], freq="M")
fit_end = pd.PeriodIndex(audit["fit_end_month"], freq="M")
assert (max_label <= signal).all()
assert (fit_end < signal).all()
display(audit.tail())
print("통과: 모든 LightGBM 학습 라벨의 실현월 ≤ 신호월, 학습 특성월 < 신호월")
""",
            "audit-code",
        ),
        markdown_cell("## 7. 확률 품질 비교", "prob-title"),
        code_cell(
            """prediction_metrics = pd.read_csv(RESULTS / "top3_regime_model_prediction_metrics.csv")
locked_probability = prediction_metrics.query("Period == 'locked_2018_2026'")
display(locked_probability.style.format({
    "Brier":"{:.3f}", "LogLoss":"{:.3f}", "ECE5":"{:.3f}",
    "AUC":"{:.3f}", "BalancedAccuracy":"{:.3f}", "TransitionRecall":"{:.1%}",
}))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, horizon in zip(axes, [1, 3, 6]):
    view = locked_probability.query("Horizon == @horizon").sort_values("Brier")
    ax.bar(view["Model"], view["Brier"], color=["#007a5e" if "LightGBM" in x else "#87928d" for x in view["Model"]])
    ax.set_title(f"{horizon}M Brier ↓")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()
""",
            "prob-code",
        ),
        markdown_cell("## 8. 2017년까지 선택된 포트폴리오 규칙", "config-title"),
        code_cell(
            """calibration = pd.read_csv(RESULTS / "top3_regime_model_calibration.csv")
winners = calibration.loc[calibration["Selected"].fillna(False)]
display(winners[[
    "Model","threshold","max_shift","bond_share","horizon_weights",
    "CAGR","Sharpe","MDD","Calmar","AvgDefensiveShift","ActiveMonths",
]].style.format({"CAGR":"{:.2%}","MDD":"{:.2%}","Sharpe":"{:.3f}","Calmar":"{:.3f}"}))
""",
            "config-code",
        ),
        markdown_cell(
            """### 방어 혼합 코드의 의미

확률 가중치는 `p = w1·p1M + w3·p3M + w6·p6M`입니다. `p`가 문턱보다 높을 때만 `shift = max_shift × (p-threshold)/(1-threshold)`를 활성화합니다. 기존 전략의 일부를 BOND/GLD 수익으로 대체하고 왕복 거래비용과 GLD 환전비용을 추가 차감합니다. 도전 모델끼리는 완전히 같은 규칙·그리드를 사용합니다.
""",
            "overlay-explain",
        ),
        markdown_cell("## 9. 전체기간·잠금기간 성과", "performance-title"),
        code_cell(
            """comparison = pd.read_csv(RESULTS / "top3_regime_model_comparison.csv")
display(comparison.style.format({
    "CAGR":"{:.2%}","Volatility":"{:.2%}","Sharpe":"{:.3f}",
    "MDD":"{:.2%}","Calmar":"{:.3f}","AvgTurnover":"{:.3f}",
}))

paths = {"Existing_VKOSPI_Dynamic": pd.read_csv(RESULTS/"vkospi_dynamic_reconciled_monthly.csv", index_col="month")}
for name, slug in [("CJM","cjm"),("TVTP-HMM","tvtp_hmm"),("CJM+LightGBM","cjm_plus_lightgbm")]:
    paths[name] = pd.read_csv(RESULTS/f"top3_regime_model_backtest_{slug}.csv", index_col="month")

fig, axes = plt.subplots(2,1,figsize=(14,9),sharex=True,gridspec_kw={"height_ratios":[2,1]})
colors={"Existing_VKOSPI_Dynamic":"#87928d","CJM":"#377dff","TVTP-HMM":"#e07a3f","CJM+LightGBM":"#007a5e"}
for name, path in paths.items():
    axes[0].plot(path.index, path["nav"], label=name, color=colors[name], lw=2)
    axes[1].plot(path.index, 100*path["drawdown"], label=name, color=colors[name], lw=1.6)
axes[0].set_title("Cumulative NAV"); axes[0].legend(); axes[0].grid(alpha=.2)
axes[1].set_title("Drawdown (%)"); axes[1].grid(alpha=.2)
axes[1].set_xticks(range(0,len(paths["CJM"]),24), paths["CJM"].index[::24], rotation=45)
plt.tight_layout(); plt.show()
""",
            "performance-code",
        ),
        markdown_cell("## 10. 잠금 부트스트랩과 최종 판단", "validation-title"),
        code_cell(
            """report = json.loads((RESULTS/"top3_regime_model_validation.json").read_text(encoding="utf-8"))
rows=[]
for model, detail in report["validation"].items():
    rows.append({"Model":model, **detail["locked_deltas"], **detail["bootstrap"], "passes_all_three":detail["passes_all_three"]})
validation_table=pd.DataFrame(rows)
display(validation_table.style.format({
    "CAGR":"{:+.2%}","Sharpe":"{:+.3f}","MDD":"{:+.2%}","Calmar":"{:+.3f}",
    "probability_cagr_improves":"{:.1%}","probability_sharpe_improves":"{:.1%}",
    "probability_mdd_improves":"{:.1%}","probability_all_three_improve":"{:.1%}",
}))

promoted=validation_table.loc[validation_table["passes_all_three"],"Model"].tolist()
print("세 지표 동시 개선 모델:", promoted if promoted else "없음 — 기존 전략 유지가 보수적 결론")
""",
            "validation-code",
        ),
        markdown_cell("## 11. LightGBM 변수 중요도", "importance-title"),
        code_cell(
            """importance=pd.read_csv(RESULTS/"top3_regime_model_feature_importance.csv").head(20)
display(importance)
plt.figure(figsize=(10,6)); plt.barh(importance["feature"][::-1],importance["mean_gain_share"][::-1],color="#007a5e")
plt.title("Mean LightGBM gain share"); plt.tight_layout(); plt.show()
""",
            "importance-code",
        ),
        markdown_cell(
            """## 12. 한계와 운영 전 체크리스트

- 월간 표본이 작아 복잡한 상호작용은 불안정할 수 있습니다.
- CJM/TVTP의 잠재상태는 관측 가능한 경제적 진실이 아니라 모델 근사입니다.
- Brier가 좋아도 포트폴리오 성과가 좋아진다는 보장은 없습니다.
- 세금, 상품 추적오차, 실제 체결 슬리피지, 레버리지 한도는 추가 반영해야 합니다.
- 운영 전에는 매월 데이터 가용일, 수정치(vintage), 모델 수렴, 확률 calibration drift를 모니터링해야 합니다.
""",
            "limits",
        ),
        markdown_cell("## 13. 결과 ZIP 저장", "export-title"),
        code_cell(
            """output = Path("/content/top3_regime_model_results.zip") if IN_COLAB else PROJECT_ROOT/"top3_regime_model_results.zip"
with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RESULTS.glob("top3_regime_model_*")):
        archive.write(path,arcname=path.name)
print("저장:",output)
if IN_COLAB:
    from google.colab import files
    files.download(str(output))
""",
            "export-code",
        ),
    ]
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
    raw_members = sorted(path for path in (ROOT / "raw_data").iterdir() if path.is_file())
    result_members = [
        RESULTS / "regime_signals.csv",
        RESULTS / "vkospi_features.csv",
        RESULTS / "vkospi_dynamic_reconciled_monthly.csv",
        *sorted(RESULTS.glob("top3_regime_model_*")),
    ]
    members = [
        ROOT / "strategies/stage07_regime_models/top3_regime_model_experiment.py",
        ROOT / "tests/test_top3_regime_model_experiment.py",
        ROOT / "strategies/core/regime_research.py",
        ROOT / "cache" / "market_daily.csv",
        ROOT / "cache" / "stress_monthly.csv",
        *raw_members,
        *result_members,
    ]
    seen = set()
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            if not path.exists():
                raise FileNotFoundError(path)
            relative = path.relative_to(ROOT).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            archive.write(path, arcname=f"RegimeDecisionTest/{relative}")


def main() -> None:
    report = json.loads(
        (RESULTS / "top3_regime_model_validation.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(RESULTS / "top3_regime_model_comparison.csv")
    metrics = pd.read_csv(RESULTS / "top3_regime_model_prediction_metrics.csv")
    HTML_PATH.write_text(make_html(report, comparison, metrics), encoding="utf-8")
    NOTEBOOK_PATH.write_text(
        json.dumps(make_notebook(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    make_bundle()
    print(f"HTML: {HTML_PATH} ({HTML_PATH.stat().st_size:,} bytes)")
    print(f"Notebook: {NOTEBOOK_PATH} ({NOTEBOOK_PATH.stat().st_size:,} bytes)")
    print(f"Bundle: {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
