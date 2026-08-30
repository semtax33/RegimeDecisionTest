from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/reports/economic_regime_implementation_guide.html"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_block(path: Path, prefix: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    rendered = []
    for index, line in enumerate(lines, start=1):
        rendered.append(
            f'<span class="code-line" id="src-{prefix}-L{index}">'
            f'<a class="line-no" href="#src-{prefix}-L{index}" aria-label="line {index}">{index}</a>'
            f'<span class="line-code">{esc(line) or " "}</span></span>'
        )
    return "\n".join(rendered)


def metric_rows(frame: pd.DataFrame) -> str:
    labels = {
        "proposed": "제안 전략",
        "soft": "Soft regime only",
        "hard": "Hard regime",
        "equal": "동일가중",
        "static_defensive": "정적 방어형",
        "kodex": "KODEX200 보유",
    }
    out = []
    for key, row in frame.iterrows():
        out.append(
            "<tr>"
            f"<th>{labels.get(key, esc(key))}</th>"
            f"<td>{int(row['Months'])}</td>"
            f"<td>{pct(float(row['CAGR']))}</td>"
            f"<td>{pct(float(row['Volatility']))}</td>"
            f"<td>{num(float(row['Sharpe']))}</td>"
            f"<td>{num(float(row['Sortino']))}</td>"
            f"<td>{pct(float(row['MDD']))}</td>"
            f"<td>{num(float(row['Calmar']))}</td>"
            f"<td>{float(row['FinalMultiple']):.2f}×</td>"
            f"<td>{pct(float(row['PositiveMonths']), 1)}</td>"
            "</tr>"
        )
    return "\n".join(out)


def main() -> None:
    summary = pd.read_csv(ROOT / "results" / "summary.csv", index_col=0)
    signals = pd.read_csv(ROOT / "results" / "regime_signals.csv")
    backtest = pd.read_csv(ROOT / "results" / "proposed_backtest.csv")
    calibration = pd.read_csv(ROOT / "results" / "calibration_grid.csv")
    config = json.loads((ROOT / "results" / "config.json").read_text(encoding="utf-8"))
    regime_metrics = json.loads((ROOT / "results" / "regime_metrics.json").read_text(encoding="utf-8"))

    latest_signal = signals.iloc[-1]
    latest_bt = backtest.iloc[-1]
    prob_keys = ["Goldilocks", "Overheating", "Slowdown", "Stagflation"]
    asset_keys = ["KODEX200", "BOND", "GLD", "USO"]
    anchors = {
        "Goldilocks": [0.58, 0.22, 0.15, 0.05],
        "Overheating": [0.30, 0.12, 0.23, 0.35],
        "Slowdown": [0.12, 0.66, 0.20, 0.02],
        "Stagflation": [0.08, 0.24, 0.50, 0.18],
    }
    latest_probabilities = [float(latest_signal[f"p_{key}"]) for key in prob_keys]
    latest_soft_anchor = [
        sum(latest_probabilities[r] * anchors[prob_keys[r]][a] for r in range(4))
        for a in range(4)
    ]
    strategic = [0.20, 0.45, 0.30, 0.05]
    latest_blended_anchor = [0.75 * latest_soft_anchor[i] + 0.25 * strategic[i] for i in range(4)]

    locked_rows = [
        ["proposed", 103, 0.090180, 0.075629, 1.182598, 2.512429, -0.080517, 1.120021, 2.098269, 0.640777],
        ["soft", 103, 0.136115, 0.125372, 1.084327, 2.034735, -0.144601, 0.941317, 2.990259, 0.660194],
        ["hard", 103, 0.225844, 0.239203, 0.978051, 1.630350, -0.266720, 0.846747, 5.742081, 0.621359],
        ["equal", 103, 0.134790, 0.143477, 0.957588, 1.571707, -0.273170, 0.493429, 2.960455, 0.660194],
        ["static_defensive", 103, 0.106960, 0.082503, 1.277294, 2.594808, -0.097093, 1.101630, 2.392216, 0.621359],
        ["kodex", 103, 0.164300, 0.289897, 0.666106, 1.246490, -0.334012, 0.491899, 3.690259, 0.553398],
    ]
    locked = pd.DataFrame(
        locked_rows,
        columns=["name", "Months", "CAGR", "Volatility", "Sharpe", "Sortino", "MDD", "Calmar", "FinalMultiple", "PositiveMonths"],
    ).set_index("name")

    cal_rows = []
    for _, row in calibration.head(10).iterrows():
        cal_rows.append(
            "<tr>"
            f"<td><code>{esc(row['name'])}</code></td>"
            f"<td>{pct(float(row['CAGR']))}</td>"
            f"<td>{num(float(row['Sharpe']))}</td>"
            f"<td>{pct(float(row['MDD']))}</td>"
            f"<td>{num(float(row['Calmar']))}</td>"
            f"<td>{pct(float(row['AvgTurnover']))}</td>"
            f"<td>{num(float(row['ValidationScore']))}</td>"
            "</tr>"
        )

    config_rows = "\n".join(
        f"<tr><th><code>{esc(key)}</code></th><td><code>{esc(value)}</code></td><td>{description}</td></tr>"
        for key, value, description in [
            ("name", config["name"], "설정 식별자"),
            ("target_vol", config["target_vol"], "연환산 기준 변동성 목표"),
            ("half_life", config["half_life"], "EWMA 공분산 반감기(월)"),
            ("invvol_tilt", config["invvol_tilt"], "기준비중에 적용하는 역변동성 기울기"),
            ("return_reward", config["return_reward"], "기대수익 보상 계수"),
            ("vol_penalty", config["vol_penalty"], "변동성 페널티"),
            ("cdar_penalty", config["cdar_penalty"], "경로 CDaR 페널티"),
            ("turnover_penalty", config["turnover_penalty"], "직전 거래 전 비중 대비 회전율 페널티"),
            ("tracking_penalty", config["tracking_penalty"], "사전 기준비중 이탈 페널티"),
            ("max_cdar", config["max_cdar"], "90% CDaR 절댓값 상한"),
            ("drawdown_guard", config["drawdown_guard"], "드로다운 발생 시 방어비중 혼합 강도"),
            ("regime_strength", config["regime_strength"], "국면 기준비중과 전략적 비중 혼합 강도"),
            ("use_regime", config["use_regime"], "국면 신호 활성화 여부"),
            ("use_risk_control", config["use_risk_control"], "위험 최적화 활성화 여부"),
        ]
    )

    regime_counts = signals["regime"].value_counts()
    proposed_metrics = summary.loc["proposed"]
    soft_metrics = summary.loc["soft"]
    hard_metrics = summary.loc["hard"]
    static_metrics = summary.loc["static_defensive"]
    soft_sharpe_gain = float(proposed_metrics["Sharpe"] / soft_metrics["Sharpe"] - 1)
    soft_vol_reduction = float(1 - proposed_metrics["Volatility"] / soft_metrics["Volatility"])
    soft_mdd_reduction = float(1 - abs(proposed_metrics["MDD"]) / abs(soft_metrics["MDD"]))
    hard_sharpe_gain = float(proposed_metrics["Sharpe"] / hard_metrics["Sharpe"] - 1)
    hard_vol_reduction = float(1 - proposed_metrics["Volatility"] / hard_metrics["Volatility"])
    hard_mdd_reduction = float(1 - abs(proposed_metrics["MDD"]) / abs(hard_metrics["MDD"]))
    static_sharpe_gain = float(proposed_metrics["Sharpe"] / static_metrics["Sharpe"] - 1)
    static_mdd_reduction = float(1 - abs(proposed_metrics["MDD"]) / abs(static_metrics["MDD"]))
    source_files = [
        ROOT / "strategies/core/regime_research.py",
        ROOT / "strategies/stage01_baseline/calibrate_configs.py",
        ROOT / "tools/builders/build_validated_notebook.py",
    ]
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    document = f"""<!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <meta name="color-scheme" content="light dark">
      <meta name="description" content="경제 국면 기반 멀티에셋 자산배분 코드의 상세 구현·설계 이유·근거 설명서">
      <title>경제 국면 기반 멀티에셋 자산배분 — 구현·설계 근거 설명서</title>
      <style>
        :root {{
          --bg:#f4f7fb; --paper:#ffffff; --ink:#172033; --muted:#5d687c; --line:#dce3ed;
          --blue:#185adb; --blue-2:#e9f0ff; --navy:#0d2045; --green:#087a5b; --green-bg:#e7f7f1;
          --amber:#a65d00; --amber-bg:#fff4dd; --red:#b83a3a; --red-bg:#fff0f0; --code:#0f172a;
          --shadow:0 16px 48px rgba(23,32,51,.09); --radius:18px; --mono:"Cascadia Code","Consolas",monospace;
        }}
        * {{ box-sizing:border-box; }}
        html {{ scroll-behavior:smooth; scroll-padding-top:24px; }}
        body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Pretendard","Noto Sans KR","Malgun Gothic",system-ui,sans-serif; line-height:1.72; }}
        a {{ color:var(--blue); text-decoration-thickness:1px; text-underline-offset:3px; }}
        code, pre {{ font-family:var(--mono); }}
        code {{ font-size:.91em; background:#eef2f7; border:1px solid #dfe6ef; border-radius:6px; padding:.08rem .34rem; overflow-wrap:anywhere; }}
        #progress {{ position:fixed; inset:0 auto auto 0; height:3px; width:0; background:linear-gradient(90deg,#185adb,#09a47b); z-index:100; }}
        .hero {{ background:radial-gradient(circle at 82% 16%,#2e71ed 0,transparent 28%),linear-gradient(135deg,#091a3b,#173c7e 62%,#095f5a); color:white; padding:70px 24px 54px; }}
        .hero-inner {{ max-width:1240px; margin:auto; }}
        .eyebrow {{ letter-spacing:.12em; text-transform:uppercase; font-size:.78rem; font-weight:800; opacity:.78; }}
        h1 {{ font-size:clamp(2.25rem,5vw,4.8rem); letter-spacing:-.055em; line-height:1.05; margin:.55rem 0 1.2rem; max-width:990px; }}
        .lead {{ font-size:clamp(1rem,1.8vw,1.25rem); max-width:900px; opacity:.9; }}
        .hero-meta {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:24px; }}
        .badge {{ display:inline-flex; align-items:center; gap:7px; padding:7px 11px; border:1px solid rgba(255,255,255,.25); border-radius:999px; background:rgba(255,255,255,.09); font-size:.86rem; }}
        .layout {{ display:grid; grid-template-columns:280px minmax(0,1fr); gap:28px; max-width:1420px; margin:28px auto 80px; padding:0 24px; align-items:start; }}
        .sidebar {{ position:sticky; top:18px; max-height:calc(100vh - 36px); overflow:auto; background:var(--paper); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); padding:18px; }}
        .sidebar strong {{ display:block; margin-bottom:10px; }}
        .filter {{ width:100%; border:1px solid var(--line); border-radius:10px; padding:9px 10px; background:var(--bg); color:var(--ink); margin-bottom:10px; }}
        .toc {{ list-style:none; padding:0; margin:0; font-size:.88rem; }}
        .toc li {{ margin:2px 0; }} .toc a {{ display:block; padding:5px 7px; border-radius:7px; text-decoration:none; color:var(--muted); }}
        .toc a:hover {{ background:var(--blue-2); color:var(--blue); }}
        .toc .sub {{ padding-left:18px; font-size:.82rem; }}
        .actions {{ display:flex; gap:8px; margin-top:14px; }}
        button {{ border:1px solid var(--line); background:var(--paper); color:var(--ink); padding:8px 10px; border-radius:9px; cursor:pointer; }}
        main {{ min-width:0; }}
        section {{ background:var(--paper); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); padding:clamp(22px,4vw,44px); margin-bottom:24px; }}
        h2 {{ font-size:clamp(1.55rem,3vw,2.35rem); letter-spacing:-.035em; line-height:1.2; margin:0 0 20px; }}
        h3 {{ font-size:1.25rem; letter-spacing:-.02em; margin:34px 0 12px; }}
        h4 {{ margin:26px 0 8px; font-size:1.04rem; }}
        p {{ margin:.65rem 0 1rem; }}
        .muted {{ color:var(--muted); }}
        .kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:20px 0; }}
        .kpi {{ border:1px solid var(--line); border-radius:14px; padding:16px; background:linear-gradient(160deg,#fff,#f7f9fc); }}
        .kpi .label {{ color:var(--muted); font-size:.82rem; }} .kpi .value {{ font-weight:850; font-size:1.55rem; letter-spacing:-.04em; }}
        .callout {{ border-left:4px solid var(--blue); background:var(--blue-2); padding:14px 16px; border-radius:0 11px 11px 0; margin:18px 0; }}
        .callout.warn {{ border-color:var(--amber); background:var(--amber-bg); }} .callout.good {{ border-color:var(--green); background:var(--green-bg); }}
        .callout.danger {{ border-color:var(--red); background:var(--red-bg); }}
        .evidence-legend {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 20px; }}
        .evidence-tag {{ display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:4px 9px; font-size:.76rem; font-weight:800; white-space:nowrap; }}
        .evidence-tag.lit {{ color:#1148a6; background:#e7efff; border:1px solid #bfd1f8; }}
        .evidence-tag.emp {{ color:#08654d; background:#e5f7f0; border:1px solid #b6e1d2; }}
        .evidence-tag.judge {{ color:#8a5108; background:#fff2d7; border:1px solid #efd39e; }}
        .evidence-tag.limit {{ color:#9d3030; background:#ffeded; border:1px solid #efc0c0; }}
        .decision {{ border:1px solid var(--line); border-radius:14px; padding:18px; background:var(--paper); margin:13px 0; }}
        .decision h4 {{ margin:0 0 8px; }} .decision p:last-child {{ margin-bottom:0; }}
        .why {{ font-weight:800; color:var(--blue); }}
        .grid-2 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
        .card {{ border:1px solid var(--line); border-radius:14px; padding:18px; background:#fbfcfe; }} .card h3,.card h4 {{ margin-top:0; }}
        .pipeline {{ display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:20px; margin:24px 0; }}
        .pipe {{ position:relative; border:1px solid #cbd8ef; border-radius:14px; padding:15px; background:#f5f8ff; min-height:132px; }}
        .pipe:not(:last-child)::after {{ content:"→"; position:absolute; right:-18px; top:42%; color:var(--blue); font-size:1.4rem; font-weight:900; }}
        .pipe b {{ display:block; color:var(--blue); margin-bottom:7px; }} .pipe small {{ color:var(--muted); }}
        .table-wrap {{ overflow:auto; margin:16px 0 24px; border:1px solid var(--line); border-radius:12px; }}
        table {{ width:100%; border-collapse:collapse; min-width:700px; font-size:.91rem; }}
        th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; text-align:left; }}
        thead th {{ position:sticky; top:0; background:#edf3ff; color:#19315f; z-index:1; }} tbody tr:last-child th,tbody tr:last-child td {{ border-bottom:0; }}
        tbody tr:hover {{ background:#f8faff; }}
        .formula {{ overflow:auto; padding:16px 18px; border-radius:12px; background:#101a33; color:#eaf1ff; font-family:"Cambria Math",serif; font-size:1.04rem; margin:14px 0; }}
        .formula code {{ border:0; background:transparent; color:inherit; padding:0; }}
        .steps {{ counter-reset:step; list-style:none; padding:0; }} .steps li {{ counter-increment:step; position:relative; padding:0 0 20px 52px; }}
        .steps li::before {{ content:counter(step); position:absolute; left:0; top:0; width:34px; height:34px; border-radius:50%; display:grid; place-items:center; background:var(--blue); color:#fff; font-weight:800; }}
        .steps li:not(:last-child)::after {{ content:""; position:absolute; left:16px; top:36px; bottom:2px; width:2px; background:var(--line); }}
        pre.block {{ background:var(--code); color:#d8e2f5; border-radius:13px; overflow:auto; padding:18px; line-height:1.55; font-size:.86rem; }}
        pre.block code {{ background:transparent; color:inherit; border:0; padding:0; }}
        .src {{ font-family:var(--mono); font-size:.76rem; white-space:nowrap; text-decoration:none; border:1px solid #cad8ee; background:#f1f5fb; padding:3px 6px; border-radius:6px; }}
        details {{ border:1px solid var(--line); border-radius:12px; padding:0 14px; margin:12px 0; background:#fbfcfe; }}
        summary {{ cursor:pointer; padding:13px 0; font-weight:750; }}
        .source-code {{ margin:0 -14px; border-radius:0 0 12px 12px; background:#0d1425; color:#d7e1f5; overflow:auto; max-height:72vh; padding:12px 0; line-height:1.48; font-size:.78rem; }}
        .code-line {{ display:grid; grid-template-columns:58px minmax(max-content,1fr); min-height:1.48em; }} .code-line:target {{ background:#303f65; }}
        .line-no {{ color:#71809d; text-align:right; padding-right:13px; user-select:none; text-decoration:none; }} .line-code {{ white-space:pre; padding-right:18px; }}
        .checklist {{ list-style:none; padding:0; }} .checklist li {{ padding:7px 0 7px 29px; position:relative; }} .checklist li::before {{ content:"✓"; position:absolute; left:0; color:var(--green); font-weight:900; }}
        .hash {{ word-break:break-all; font-family:var(--mono); font-size:.76rem; color:var(--muted); }}
        footer {{ color:var(--muted); text-align:center; padding:0 24px 50px; font-size:.86rem; }}
        @media (max-width:1100px) {{ .layout {{ grid-template-columns:1fr; }} .sidebar {{ position:relative; max-height:none; }} .pipeline {{ grid-template-columns:1fr 1fr; }} .pipe::after {{ display:none; }} }}
        @media (max-width:720px) {{ .hero {{ padding-top:46px; }} .layout {{ padding:0 12px; }} section {{ border-radius:14px; }} .kpis,.grid-2,.pipeline {{ grid-template-columns:1fr; }} h1 {{ font-size:2.35rem; }} }}
        @media print {{ #progress,.sidebar,.actions {{ display:none!important; }} body {{ background:#fff; font-size:10pt; }} .layout {{ display:block; margin:0; padding:0; }} section {{ box-shadow:none; border:0; page-break-before:auto; padding:16px 0; }} .hero {{ background:#0d2045!important; print-color-adjust:exact; -webkit-print-color-adjust:exact; }} details {{ break-inside:avoid; }} details:not([open]) > *:not(summary) {{ display:none; }} a {{ color:inherit; text-decoration:none; }} }}
        @media (prefers-color-scheme:dark) {{
          :root {{ --bg:#0b1020; --paper:#121a2d; --ink:#e9eef9; --muted:#a9b4c8; --line:#2b3850; --blue:#79a5ff; --blue-2:#172846; --green-bg:#102d27; --amber-bg:#342615; --red-bg:#351b22; --code:#080d19; }}
          code {{ background:#202b40; border-color:#35435e; }} .kpi,.card {{ background:#151f34; }} thead th {{ background:#1b2a45; color:#bed2ff; }} tbody tr:hover {{ background:#18243b; }} .pipe {{ background:#14233e; border-color:#29406a; }} .src {{ background:#1b2b48; border-color:#334c78; }}
        }}
      </style>
    </head>
    <body>
    <div id="progress"></div>
    <header class="hero">
      <div class="hero-inner">
        <div class="eyebrow">Implementation reference · validated snapshot</div>
        <h1>경제 국면 기반 멀티에셋 자산배분<br>상세 구현·설계 근거 설명서</h1>
        <p class="lead">현재 작업 트리의 Python 구현을 기준으로 무엇을 어떻게 구현했는지뿐 아니라, 왜 그 구조를 선택했는지, 어떤 문헌·내부 실험·보수적 판단이 근거인지, 대안과 한계는 무엇인지까지 추적할 수 있도록 작성한 독립 실행형 HTML 기술 문서입니다.</p>
        <div class="hero-meta">
          <span class="badge">문서 생성 {esc(generated_at)}</span>
          <span class="badge">핵심 모듈 {len((ROOT / 'strategies/core/regime_research.py').read_text(encoding='utf-8').splitlines())} lines</span>
          <span class="badge">신호 {len(signals)}개월</span>
          <span class="badge">설계 이유·대안·근거 포함</span>
          <span class="badge">외부 리소스 없는 단일 HTML</span>
        </div>
      </div>
    </header>

    <div class="layout">
      <aside class="sidebar" aria-label="문서 목차">
        <strong>문서 탐색</strong>
        <input id="tocFilter" class="filter" type="search" placeholder="목차 검색" aria-label="목차 검색">
        <ol class="toc" id="toc">
          <li><a href="#overview">1. 구현 개요</a></li>
          <li><a href="#architecture">2. 아키텍처와 실행 흐름</a></li>
          <li><a href="#rationale">설계 근거 총론: 왜 이렇게 구현했나</a></li>
          <li><a href="#files">3. 파일과 산출물</a></li>
          <li><a href="#macro">4. 거시 데이터 파이프라인</a></li>
          <li><a href="#market">5. 자산 가격·수익률 파이프라인</a></li>
          <li><a href="#sjm">6. Sparse Jump Model</a></li>
          <li><a href="#signals">7. 국면 확률 생성</a></li>
          <li><a href="#allocation">8. 기준비중과 위험제어</a></li>
          <li><a href="#backtest">9. 백테스트 엔진</a></li>
          <li><a href="#metrics">10. 성과·국면 평가</a></li>
          <li><a href="#calibration">11. 보정과 잠금 테스트</a></li>
          <li><a href="#results">12. 현재 실행 결과</a></li>
          <li><a href="#verification">13. 자동 검증과 불변조건</a></li>
          <li><a href="#runbook">14. 실행 방법</a></li>
          <li><a href="#maintenance">15. 수정·확장 가이드</a></li>
          <li><a href="#evidence">근거 감사: 문헌·내부 실증·판단</a></li>
          <li><a href="#limitations">16. 한계와 해석 주의사항</a></li>
          <li><a href="#reference">17. 함수 레퍼런스</a></li>
          <li><a href="#glossary">18. 용어집</a></li>
          <li><a href="#source">부록 A. 전체 소스 스냅샷</a></li>
          <li><a href="#integrity">부록 B. 무결성 정보</a></li>
        </ol>
        <div class="actions"><button type="button" onclick="window.print()">인쇄 / PDF</button><button type="button" onclick="window.scrollTo({{top:0}})">맨 위로</button></div>
      </aside>

      <main>
        <section id="overview">
          <h2>1. 구현 개요</h2>
          <p>이 프로젝트는 한국의 성장·물가 데이터를 두 개의 고/저 확률로 변환하고, 두 확률의 곱으로 네 경제 국면을 만든 뒤, 국면별 기준비중과 통계적 위험제어를 결합해 월별 포트폴리오를 산출합니다. 모든 투자 대상 월 <code>t</code>의 신호는 직전 월 <code>t−1</code>까지 이용 가능한 거시 데이터만 사용하며, 포트폴리오 최적화의 수익률 이력도 <code>t</code>보다 앞선 달만 포함합니다.</p>
          <div class="kpis">
            <div class="kpi"><div class="label">공통 자산 수익률</div><div class="value">244개월</div><div class="muted">2006-04 → 2026-07</div></div>
            <div class="kpi"><div class="label">실제 운용 신호</div><div class="value">232개월</div><div class="muted">2007-04 → 2026-07</div></div>
            <div class="kpi"><div class="label">최종 Sharpe</div><div class="value">{summary.loc['proposed','Sharpe']:.3f}</div><div class="muted">무위험수익률 0%</div></div>
            <div class="kpi"><div class="label">최종 MDD</div><div class="value">{pct(float(summary.loc['proposed','MDD']))}</div><div class="muted">비용 차감 후</div></div>
          </div>
          <div class="callout good"><strong>핵심 설계 의도.</strong> 예측모델이 포트폴리오를 직접 결정하지 않습니다. 먼저 경제적으로 해석 가능한 soft anchor를 만들고, 역변동성 사전분포·변동성/CDaR 제약·거래비용·드로다운 가드를 층층이 적용합니다. 따라서 “국면 추정”과 “위험 예산”이 분리되어 있습니다.</div>
          <h3>한 문장으로 보는 전체 알고리즘</h3>
          <div class="formula">rolling z-score 수준·3개월 변화 → 2상태 SJM + logistic 합성확률 → 4국면 결합확률 → 확률가중 앵커 → 비대칭 EWMA/SLSQP → 드로다운 방어 혼합 → 비용 차감 월별 수익률</div>
          <h3>구현의 중요 속성</h3>
          <ul class="checklist">
            <li>월별 walk-forward: 매월 과거 구간만 다시 적합합니다.</li>
            <li>hard switching 대신 네 국면 확률을 모두 반영하는 연속 비중을 사용합니다.</li>
            <li>주식 하락 충격에서 공분산을 더 빠르게 키우는 비대칭 EWMA를 사용합니다.</li>
            <li>롱온리, 완전투자, 자산별 상·하한, 목표 변동성, CDaR 제약을 동시에 적용합니다.</li>
            <li>거래 전 비중은 직전 목표비중이 자산 수익률에 따라 드리프트한 값입니다.</li>
            <li>매매 15bp와 해외자산 총비중 변화에 대한 환전 5bp를 별도로 차감합니다.</li>
          </ul>
        </section>

        <section id="architecture">
          <h2>2. 아키텍처와 실행 흐름</h2>
          <div class="pipeline" aria-label="전체 데이터 처리 파이프라인">
            <div class="pipe"><b>① 원천 데이터</b><small>한국 거시 Excel/CSV, SQLite 프록시, KRX 채권지수, Yahoo 일별 가격</small></div>
            <div class="pipe"><b>② 월별 특징</b><small>발표시차 반영, rolling z-score, 성장/물가 수준과 3개월 변화</small></div>
            <div class="pipe"><b>③ 국면 확률</b><small>SJM 확률 10% + 투명 합성확률 90%, 이전 확률 평활 15%</small></div>
            <div class="pipe"><b>④ 목표비중</b><small>soft anchor, inverse-vol prior, SLSQP, drawdown guard</small></div>
            <div class="pipe"><b>⑤ 검증 산출물</b><small>비용 차감 수익률, 성과표, 국면 정확도, CSV/JSON/노트북</small></div>
          </div>
          <h3><code>main()</code>의 호출 순서</h3>
          <ol class="steps">
            <li><strong>거시 특징 로드.</strong> <code>load_macro_data()</code>가 12개 MultiIndex 특징과 6개 핵심 z-score를 반환합니다. <a class="src" href="#src-regime-L42">L42</a></li>
            <li><strong>자산 수익률 로드.</strong> <code>load_monthly_asset_returns()</code>가 네 자산의 첫 거래일 원화 환산 수준과 월간 forward return을 만듭니다. <a class="src" href="#src-regime-L120">L120</a></li>
            <li><strong>walk-forward 국면 계산.</strong> 각 투자월마다 전월까지의 거시 이력으로 성장·물가 확률과 4분면 확률을 만듭니다. <a class="src" href="#src-regime-L265">L265</a></li>
            <li><strong>여섯 전략 비교.</strong> proposed, soft, hard, equal, static defensive, KODEX200 buy-and-hold를 같은 달과 비용 규칙으로 실행합니다. <a class="src" href="#src-regime-L444">L444</a></li>
            <li><strong>평가 및 저장.</strong> 성과 요약, 신호, 제안 백테스트, 국면 지표, 설정을 <code>results/</code>에 기록합니다. <a class="src" href="#src-regime-L570">L570</a></li>
          </ol>
          <div class="callout"><strong>책임 경계.</strong> <code>regime_research.py</code>는 재사용 가능한 연구 로직과 기본 실행점을 보유하고, <code>calibrate_configs.py</code>는 설정 선택만, <code>build_validated_notebook.py</code>는 설명·시각화 노트북 조립만 담당합니다.</div>
        </section>

        <section id="rationale">
          <h2>설계 근거 총론 — 왜 이렇게 구현했나</h2>
          <p>이 절은 코드의 설계 결정을 “목적 → 선택한 방법 → 선택 이유 → 대안 → 근거의 종류 → 남는 위험” 순서로 해설합니다. 아래 표식은 근거의 강도를 구분합니다. 문헌이 방법론의 일반적 타당성을 지지하더라도 이 프로젝트의 구체적인 숫자까지 보장하는 것은 아닙니다.</p>
          <div class="evidence-legend">
            <span class="evidence-tag lit">문헌 근거</span>
            <span class="evidence-tag emp">현재 프로젝트 내부 실증</span>
            <span class="evidence-tag judge">설계 판단·휴리스틱</span>
            <span class="evidence-tag limit">한계·반증 가능성</span>
          </div>
          <div class="callout warn"><strong>가장 중요한 구분.</strong> Jump model, 변동성 관리, CDaR, 거래비용을 포함한 최적화, drawdown에 따른 위험회피도 조절은 관련 문헌에서 아이디어를 얻었습니다. 그러나 <code>90%/10%</code> 앙상블, <code>0.55</code> sigmoid scale, <code>8%</code> 목표변동성, 국면 앵커, 자산 bounds, <code>−5%</code> 가드 발동선은 이 표본과 운용 목적에 맞춘 프로젝트 고유의 보수적 선택입니다. 문헌이 이 숫자들을 증명하지 않습니다.</div>

          <h3>R1. 월별 빈도와 첫 거래일→다음 첫 거래일 수익률</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>거시 신호와 리밸런싱을 월별로 통일하고, 월 <code>t</code> 첫 거래일 open에서 <code>t+1</code> 첫 거래일 open까지의 forward return을 투자월 <code>t</code>에 귀속했습니다.</p>
            <p><span class="why">왜:</span> GDP·물가·수출·BSI의 정보 갱신 속도는 일별 가격보다 훨씬 느립니다. 신호는 월말에 확정되고 다음 거래기회에 집행된다는 구조를 명시하면 “월말 정보를 같은 월 수익률에 소급 적용”하는 룩어헤드를 피할 수 있습니다. 일별 리밸런싱은 신호의 정보빈도와 맞지 않고 회전율·미시구조 잡음만 늘릴 가능성이 큽니다.</p>
            <p><span class="evidence-tag judge">설계 판단</span> <span class="evidence-tag emp">시점 assert</span> 현재 모든 행에서 <code>signal_month &lt; target_month</code>를 검사합니다. 다만 실제 주문시각·발표시각까지 intraday로 맞춘 것은 아닙니다.</p>
          </div>

          <h3>R2. 거시통계에 한 달 발표지연을 부여</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>GDP 분기치와 월별 수출·BSI·물가 지표를 코드상 다음 월말로 이동한 후 특징을 생성했습니다.</p>
            <p><span class="why">왜:</span> 데이터가 가리키는 경제활동 월과 그 수치가 실제로 알려지는 시점은 다릅니다. 원래 관측월 인덱스를 그대로 쓰면 아직 발표되지 않은 값을 사용한 것처럼 백테스트될 수 있습니다. 달력 한 달 지연은 완전한 빈티지 데이터가 없을 때 쓰는 단순하고 감사 가능한 안전장치입니다.</p>
            <p><span class="evidence-tag emp">코드 직접 근거</span> <code>MonthEnd(1)</code>/<code>MonthEnd(2)</code> 이동과 전월 신호 규칙이 중복 안전마진을 만듭니다. <span class="evidence-tag limit">한계</span> 지표별 실제 발표일·수정일을 조회하지 않으므로 지나치게 보수적이거나 일부 지표에는 여전히 부정확할 수 있습니다.</p>
          </div>

          <h3>R3. 전체표본 표준화 대신 rolling z-score</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>GDP 72개월, 수출·물가 36개월, BSI 24개월 창의 평균·표준편차만 사용하고 ±3으로 clip했습니다.</p>
            <p><span class="why">왜:</span> 서로 단위와 변동성이 다른 거시지표를 한 모델에 넣으려면 공통척도가 필요합니다. 전체표본 평균은 미래의 장기 평균을 과거에 누설하고 구조변화를 무시합니다. rolling 창은 당시까지의 상대적 고저를 표현하고, clip은 한 번의 극단치가 중심·분산·거리 계산을 지배하는 것을 줄입니다. 창 길이는 지표 빈도와 안정성의 절충으로, 느린 GDP에는 긴 창, 경기심리에 가까운 BSI에는 짧은 창을 사용했습니다.</p>
            <p><span class="evidence-tag judge">휴리스틱</span> 창 길이와 ±3은 보편적 최적값이 아닙니다. <span class="evidence-tag limit">대안</span> expanding robust scaler, 실시간 빈티지 percentile, median/MAD 표준화가 가능하며 민감도 검증이 추가로 필요합니다.</p>
          </div>

          <h3>R4. 수준(level)과 3개월 변화(d3)를 함께 사용</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>성장·물가 축마다 세 지표의 표준화 수준과 각 수준의 3개월 차분을 함께 넣었습니다.</p>
            <p><span class="why">왜:</span> 수준은 경제가 현재 높은지 낮은지를, 단기 변화는 방향 전환을 나타냅니다. 수준만 쓰면 정점 이후 둔화를 늦게 포착하고, 변화만 쓰면 낮은 수준에서의 작은 반등과 실제 고성장을 혼동할 수 있습니다. 두 종류를 함께 쓰되 logistic 합성점수에서는 변화의 비중을 20%로 제한해 상태(level)를 주축으로 유지했습니다.</p>
            <p><span class="evidence-tag judge">경제적 구조</span> 3개월은 월별 잡음을 어느 정도 완화하면서 분기 내 방향을 보는 휴리스틱입니다. 1·6·12개월 변화와의 OOS 민감도 비교는 현재 코드에 없습니다.</p>
          </div>

          <h3>R5. 성장과 물가를 각각 2상태로 분리</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>하나의 4상태 모델을 직접 적합하지 않고 성장 High/Low, 물가 High/Low 두 개의 2상태 모델을 만들고 결합했습니다.</p>
            <p><span class="why">왜:</span> 두 축은 경제적 의미와 입력변수가 분명하고, 각각의 오분류를 별도로 진단할 수 있습니다. 232개월이라는 작은 표본에서 4상태 중심을 직접 추정하면 상태별 관측수가 더 적어지고 상태 라벨 해석도 불안정해질 수 있습니다. 2×2 구조는 국면을 사람이 읽을 수 있고, 성장/물가 balanced accuracy를 분리해 검증할 수 있습니다.</p>
            <p><span class="evidence-tag emp">현재 표본</span> 최종 신호 표본 232개월에서 가장 적은 Stagflation도 30개월입니다. <span class="evidence-tag limit">모형 가정</span> 국면확률을 두 축 확률의 곱으로 만들기 때문에 성장·물가 오차의 조건부 상관은 모델링하지 않습니다.</p>
          </div>

          <h3>R6. Jump model과 전환 페널티</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>상태 중심 적합과 상태 경로 선택을 번갈아 수행하고, 경로 선택에는 상태가 바뀔 때마다 명시적 jump penalty를 부과했습니다.</p>
            <p><span class="why">왜:</span> 경제 국면은 매월 독립적으로 뒤집히는 분류라기보다 일정 기간 지속되는 순차 상태라는 가정을 반영하기 위해서입니다. 매월 가장 가까운 중심만 선택하면 경계 부근 잡음으로 상태가 자주 왕복할 수 있습니다. 전환비용을 둔 동적계획법은 전체 시계열 순서를 이용해 적합도와 지속성 사이의 최소비용 경로를 찾습니다.</p>
            <p><span class="evidence-tag lit">문헌 근거</span> <a href="https://web.stanford.edu/~boyd/papers/fitting_jump_models.html">Bemporad et al. (2018), Fitting Jump Models</a>는 모델 파라미터 적합과 이산 상태열 최소화를 교대하는 jump-model 틀을 제시합니다. <span class="evidence-tag limit">구현 차이</span> 현재 클래스는 논문의 일반모형 전체가 아니라 2상태 가중 중심거리 버전입니다.</p>
          </div>

          <h3>R7. 최대 4개 특징만 남기는 sparse weighting</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>상태간 중심 차이를 상태내 분산으로 나눈 분리도가 큰 특징 최대 4개만 사용합니다.</p>
            <p><span class="why">왜:</span> 각 축에 6개 특징밖에 없지만 유효 표본도 작고 level/d3 특징은 서로 상관될 수 있습니다. 모든 열을 동일하게 쓰면 상태와 무관한 잡음 특징이 거리 계산에 누적됩니다. sparse weighting은 그 시점의 과거 데이터에서 실제로 상태를 구분하는 소수 특징에 집중하고, 저장된 가중치로 선택 이유를 사후 감사할 수 있게 합니다.</p>
            <p><span class="evidence-tag lit">문헌 근거</span> <a href="https://orbit.dtu.dk/en/publications/feature-selection-in-jump-models/">Nystrup, Kolm &amp; Lindström (2021)</a>은 순차 데이터의 jump model에서 특징 선택·파라미터·상태열을 함께 추정하는 동기를 제시합니다. <span class="evidence-tag judge">구현 판단</span> 현재의 분리도 공식과 top-4 hard selection은 논문의 coordinate-descent penalty를 그대로 복제한 것이 아니라 단순화한 근사입니다.</p>
          </div>

          <h3>R8. SJM 10% + 해석 가능한 합성확률 90%</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>최종 성장·물가 확률은 SJM 확률 10%, 수준·모멘텀 logistic 합성확률 90%로 구성합니다.</p>
            <p><span class="why">왜:</span> 232개월 표본에서 복잡한 비지도 상태모델에 예측 권한을 전부 주기보다, 경제적 방향을 직접 설명할 수 있는 합성점수를 주모형으로 삼고 SJM은 희소선택·지속성 정보를 보조하도록 했습니다. 이 선택은 “모델이 복잡할수록 낫다”는 가정을 피하고, 신호가 왜 High인지 수준점수로 설명할 수 있게 합니다.</p>
            <p><span class="evidence-tag emp">내부 실증</span> 현재 SJM-composite 성장 balanced accuracy는 {pct(regime_metrics['growth_balanced_accuracy'],2)}로 단순 현재상태 지속성 {pct(0.8622713930909911,2)}보다 약 0.10%p 높고, 물가는 {pct(regime_metrics['inflation_balanced_accuracy'],2)} 대 {pct(0.885162837471958,2)}로 약 0.36%p 높습니다. 4분면 정확도도 {pct(regime_metrics['quadrant_accuracy'],2)} 대 {pct(0.7672413793103449,2)}로 약 0.43%p 차이에 불과합니다. 따라서 SJM의 증분 예측력은 <strong>양(+)이지만 매우 작으며</strong>, 10%만 반영한 보수적 구조와 일치합니다.</p>
            <p><span class="evidence-tag limit">반증 가능성</span> 이 작은 차이가 통계적으로 유의하다고 검정하지 않았고 같은 표본에서 설계되었습니다. 90/10이 최적이라는 증거는 아니며 nested walk-forward 또는 별도 국가 표본이 필요합니다.</p>
          </div>

          <h3>R9. 이전 확률 15% 평활과 확률 하한/상한</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>현재 raw 확률 85%와 이전 최종확률 15%를 섞고, SJM 단독 확률은 3%~97%로 자릅니다.</p>
            <p><span class="why">왜:</span> 경계에서 작은 데이터 수정이 큰 비중변경으로 이어지는 것을 줄이고, 불확실한 상태모델이 0%/100% 확신을 내는 것을 막기 위해서입니다. 평활은 신호 안정성과 회전율을 낮추는 장점이 있지만 전환 인식이 한 박자 늦어지는 비용이 있습니다.</p>
            <p><span class="evidence-tag judge">휴리스틱</span> 15%, 3%/97%, 온도 0.85는 문헌에서 도출한 상수가 아닙니다. 별도 OOS 민감도표가 없어 근거 강도는 낮습니다.</p>
          </div>

          <h3>R10. Hard regime 대신 확률가중 soft anchor</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>최빈 국면 하나의 극단비중으로 전환하지 않고 네 국면 앵커를 확률가중합니다.</p>
            <p><span class="why">왜:</span> 국면확률 49%와 51%는 거의 같은 불확실성을 뜻하지만 hard 분류는 전혀 다른 포트폴리오로 바뀔 수 있습니다. soft anchor는 모델 불확실성을 비중에 연속적으로 전달하여 경계에서의 점프와 거래비용을 줄입니다.</p>
            <p><span class="evidence-tag emp">내부 ablation</span> 전체 구간에서 제안 전략은 hard 전략보다 Sharpe가 {pct(hard_sharpe_gain,1)} 높고, 변동성은 {pct(hard_vol_reduction,1)}, MDD 절댓값은 {pct(hard_mdd_reduction,1)} 낮습니다. 대신 CAGR은 {pct(float(hard_metrics['CAGR'] - proposed_metrics['CAGR']))}p 낮습니다. 즉 soft+risk control은 절대수익 극대화가 아니라 경로위험 완화에 유리했다는 근거입니다.</p>
          </div>

          <h3>R11. 국면 앵커와 전략비중을 75%/25%로 혼합</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>확률가중 국면 앵커 75%에 장기 전략비중 <code>[20%,45%,30%,5%]</code> 25%를 섞습니다.</p>
            <p><span class="why">왜:</span> 거시국면 추정이 틀려도 포트폴리오가 한 테마에 완전히 종속되지 않도록 구조적 분산의 바닥을 남기기 위해서입니다. 국면 앵커는 경제적 prior이고 전략비중은 모델위험에 대한 shrinkage 대상입니다.</p>
            <p><span class="evidence-tag emp">보정 근거</span> 36개 제한 그리드에서 <code>regime_strength=0.75</code>가 2017-12까지의 선택점수 1위였습니다. <span class="evidence-tag limit">선택 편향</span> 같은 보정구간에서 고른 값이며 다른 표본에서 재현된 구조적 상수는 아닙니다.</p>
          </div>

          <h3>R12. 비대칭 EWMA와 역변동성 tilt</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>최근 84개월에 반감기 12개월 EWMA를 적용하고, KODEX200 하락월의 공분산 충격을 최대 2.5배까지 키웁니다. 국면 앵커에는 지수 0.35의 완만한 역변동성 tilt를 적용합니다.</p>
            <p><span class="why">왜:</span> 변동성·상관은 시간에 따라 달라지고 위험자산 급락기에 동반상승하는 경향이 있으므로 최근·하방 충격에 더 민감한 위험추정이 필요합니다. 완전한 inverse-vol portfolio가 경제적 신호를 지워버리지 않도록 0.35 지수와 prior 45%만 사용했습니다.</p>
            <p><span class="evidence-tag lit">문헌 근거</span> <a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513">Moreira &amp; Muir (2017)</a>은 변동성이 높을 때 위험을 낮추는 volatility-managed portfolio가 여러 요인에서 Sharpe와 효용을 높인 실증을 제시합니다. <span class="evidence-tag limit">구현 차이</span> 이 코드는 그 논문의 scaling rule을 복제한 것이 아니라 비대칭 EWMA·제약식에 아이디어를 적용했습니다.</p>
          </div>

          <h3>R13. 기대수익을 장기 80% + 최근 20%로 축소하고 clip</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>expanding 평균을 80%, 반감기 24개월 최근 평균을 20% 사용하고 월 −0.6%~+1.5% 범위로 제한합니다.</p>
            <p><span class="why">왜:</span> 평균수익률은 공분산보다 추정오차가 크고, 최적화는 작은 평균 차이에도 극단 비중을 만들기 쉽습니다. 장기 평균을 주축으로 두어 안정성을 확보하고 최근 정보를 조금만 반영하며, clip으로 비정상적인 표본평균이 목적함수를 지배하지 않게 합니다.</p>
            <p><span class="evidence-tag judge">정규화 판단</span> 이 shrinkage 비율과 clip 범위는 Bayesian posterior에서 나온 값이 아닙니다. 대안은 기대수익 항을 완전히 제거한 minimum-risk 최적화, Black–Litterman, robust mean uncertainty set입니다.</p>
          </div>

          <h3>R14. SLSQP로 수익·변동성·CDaR·회전율·tracking을 함께 최적화</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>하나의 scalar 목적함수에 다섯 항을 넣고 완전투자, 비선형 변동성/CDaR 제약, 자산별 bounds를 동시에 풉니다.</p>
            <p><span class="why">왜:</span> 이 문제는 단순 폐쇄형 mean-variance 해가 아니라 비선형 경로 CDaR와 거래 전 비중 의존 회전율을 포함합니다. SLSQP는 작은 4변수 문제에서 scalar objective, equality/inequality constraint, bounds를 한 인터페이스로 다룰 수 있습니다.</p>
            <p><span class="evidence-tag lit">공식 구현 근거</span> <a href="https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html">SciPy SLSQP 문서</a>는 bounds와 등식·부등식 제약을 가진 scalar 최소화를 지원합니다. <span class="evidence-tag limit">수치적 위험</span> 비볼록한 CDaR 경로와 수치미분 때문에 전역최적을 보장하지 않으며, 실패 시 prior fallback을 사용합니다.</p>
          </div>

          <h3>R15. 목표 변동성과 CDaR을 목적함수뿐 아니라 제약으로도 사용</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>연변동성 목표와 16% CDaR 한도를 hard constraint로 두고, 동시에 변동성·CDaR을 목적함수에서도 페널티로 둡니다.</p>
            <p><span class="why">왜:</span> penalty만 쓰면 기대수익이 높게 추정된 달에 위험한도를 넘을 수 있고, constraint만 쓰면 경계 안에서 위험이 낮은 해를 선호하지 않습니다. 두 층을 함께 쓰면 최대허용선과 내부 선호를 동시에 표현합니다. 변동성은 분산 크기를, CDaR은 손실 경로의 지속성을 보므로 역할이 다릅니다.</p>
            <p><span class="evidence-tag lit">문헌 근거</span> <a href="https://doi.org/10.1142/S0219024905002767">Chekhlov, Uryasev &amp; Zabarankin (2005)</a>은 drawdown의 꼬리평균을 포트폴리오 위험척도로 다루는 기반을 제공합니다. <span class="evidence-tag limit">구현 차이</span> 현재 <code>cdar()</code>는 과거 월별 drawdown의 최악 10% 단순평균이며 논문의 최적화 정식 전체를 구현한 것은 아닙니다.</p>
          </div>

          <h3>R16. 회전율·tracking penalty와 거래비용</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>목적함수에 직전 거래 전 비중 대비 회전율과 prior 대비 제곱거리 벌점을 넣고, 백테스트 수익률에서는 15bp 매매비용과 5bp 환전비용을 실제 차감합니다.</p>
            <p><span class="why">왜:</span> 추정오차가 있는 최적화에서 작은 신호 변화가 큰 거래로 증폭될 수 있습니다. 회전율과 tracking은 거래비용을 줄이는 동시에 불안정한 해를 prior 쪽으로 정규화합니다. 목적함수의 penalty와 사후 비용 차감은 역할이 다릅니다. 전자는 의사결정을 안정화하고 후자는 성과를 현실화합니다.</p>
            <p><span class="evidence-tag lit">문헌 근거</span> <a href="https://web.stanford.edu/~boyd/papers/cvx_portfolio.html">Boyd et al. (2017), Multi-Period Trading via Convex Optimization</a>은 기대수익·위험·거래/보유비용의 trade-off를 명시적으로 다루며, <a href="https://web.stanford.edu/~boyd/papers/multiperiod_portfolio_drawdown.html">Nystrup et al. (2019)</a>도 거래·보유비용이 추정오차에 대한 regularization 역할을 할 수 있다고 설명합니다.</p>
            <p><span class="evidence-tag emp">비용 스트레스</span> 비용 0배/1배/2배의 Sharpe는 1.156/1.144/1.132로 완만하게 저하합니다. 다만 실제 spread·market impact·세금은 빠져 있습니다.</p>
          </div>

          <h3>R17. −5% 이후 상태의존 drawdown guard</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>실현 drawdown이 −5%를 넘으면 심각도에 따라 최적비중의 18.75%~56.25%를 방어비중과 혼합합니다.</p>
            <p><span class="why">왜:</span> 같은 예상수익·공분산이라도 이미 손실이 누적된 상태에서는 추가 손실의 효용비용이 더 클 수 있습니다. 현재 상태를 다음 최적화에 반영해 위험회피도를 높이되, 위험자산을 완전히 제거하면 반등에 참여하지 못하는 문제를 피하려고 연속 혼합과 위험자산 바닥을 사용했습니다.</p>
            <p><span class="evidence-tag lit">아이디어 근거</span> <a href="https://web.stanford.edu/~boyd/papers/multiperiod_portfolio_drawdown.html">Nystrup et al. (2019)</a>은 realized drawdown에 따라 위험회피도를 조정하는 MPC 접근을 연구합니다. <span class="evidence-tag limit">중요한 차이</span> 현재 구현은 다기간 예측·정책을 푸는 완전한 MPC가 아니라 현재 drawdown에 반응하는 단일기간 휴리스틱입니다.</p>
          </div>

          <h3>R18. 롱온리·완전투자·자산별 상하한</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>레버리지·공매도·현금을 허용하지 않고 네 자산 비중 합을 1로 유지하며 자산별 집중 한도를 둡니다.</p>
            <p><span class="why">왜:</span> 원본 자산군을 유지하면서 신호의 효과를 비교하고, 작은 표본 평균·공분산 오차가 레버리지나 극단 공매도로 증폭되는 것을 막기 위해서입니다. bounds는 통계적 최적해보다 실제 운용 가능한 분산을 우선하는 guardrail입니다.</p>
            <p><span class="evidence-tag judge">운용 제약</span> 한도는 시장 유동성·법규에서 역산한 값이 아니라 연구 목적의 보수적 범위입니다. 현금자산이 없기 때문에 모든 자산이 위험한 국면에도 완전투자해야 하는 한계가 있습니다.</p>
          </div>

          <h3>R19. 36개 제한 그리드와 2018년 이후 잠금 구간</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>핵심 네 파라미터만 36개 조합으로 2017-12까지 보정하고, 2018-01 이후는 우승 설정 선택이 끝난 뒤 확인합니다.</p>
            <p><span class="why">왜:</span> 파라미터 수와 검색범위를 제한하면 같은 표본에 대한 과도한 최적화를 줄이고, 시간순 잠금 구간은 선택 후 성능 저하를 드러낼 최소한의 방어선이 됩니다. 점수도 Sharpe만 최대화하지 않고 Calmar, 15% MDD 초과, 회전율을 함께 봅니다.</p>
            <p><span class="evidence-tag emp">내부 실증</span> 우승 설정의 보정 Sharpe는 1.105, 잠금구간 Sharpe는 1.183이며 MDD는 각각 −8.93%, −8.05%입니다. 선택 뒤 붕괴하지 않았다는 긍정적 신호지만, 한 번의 temporal split만으로 일반화를 증명하지는 않습니다.</p>
          </div>

          <h3>R20. 프록시 접합과 캐시 우선 정책</h3>
          <div class="decision">
            <h4>결정</h4>
            <p>2009-03 이전 KODEX200 공백은 KOSPI200 프록시를 가격수준 앵커로 접합하고, Yahoo 데이터는 로컬 CSV 캐시를 기본 사용합니다.</p>
            <p><span class="why">왜:</span> 초기 공백을 그대로 두면 백테스트가 짧아지고 글로벌 금융위기 구간이 빠집니다. 수준비율 접합은 수익률 연속성을 단순하게 확보합니다. 캐시는 외부 공급자의 수정·네트워크 상태 때문에 같은 코드가 매번 다른 입력을 받는 문제를 줄입니다.</p>
            <p><span class="evidence-tag judge">데이터 공학 판단</span> 프록시는 ETF 총수익과 동일하지 않고 배당·추적오차가 빠집니다. 캐시도 원천의 진정한 빈티지 보관소는 아니며, 재현성을 위해서는 해시·수집시각·원천 응답을 manifest로 남겨야 합니다.</p>
          </div>
        </section>

        <section id="files">
          <h2>3. 파일과 산출물</h2>
          <h3>실행 코드</h3>
          <div class="table-wrap"><table><thead><tr><th>파일</th><th>역할</th><th>입력</th><th>출력/부작용</th></tr></thead><tbody>
            <tr><th><a href="../../strategies/core/regime_research.py"><code>regime_research.py</code></a></th><td>데이터 처리, SJM, 국면 확률, 위험 최적화, 백테스트, 성과·국면 평가의 단일 기준 구현</td><td><code>raw_data/</code>, <code>cache/market_daily.csv</code></td><td><code>results/</code>의 핵심 CSV/JSON 5종</td></tr>
            <tr><th><a href="../../strategies/stage01_baseline/calibrate_configs.py"><code>calibrate_configs.py</code></a></th><td>36개 제한 그리드에서 2017-12까지의 보정 점수로 설정 선택</td><td>핵심 모듈의 함수</td><td><code>results/calibration_grid.csv</code>, 콘솔 순위/잠금 성과</td></tr>
            <tr><th><a href="../../tools/builders/build_validated_notebook.py"><code>build_validated_notebook.py</code></a></th><td>설명, 구현 코드, 표, 그래프, 검증 assert가 포함된 재현 노트북 생성</td><td><code>regime_research.py</code></td><td><code>economic_regime_allocation_validated.ipynb</code></td></tr>
            <tr><th><code>build_implementation_guide.py</code></th><td>현재 소스·결과를 읽어 이 독립형 HTML 설명서를 생성</td><td>Python 소스와 <code>results/</code></td><td><code>economic_regime_implementation_guide.html</code></td></tr>
          </tbody></table></div>
          <h3>결과 파일 스키마</h3>
          <div class="table-wrap"><table><thead><tr><th>파일</th><th>핵심 내용</th><th>소비자</th></tr></thead><tbody>
            <tr><th><code>summary.csv</code></th><td>6개 전략 × 9개 성과지표</td><td>비교표, 보고서</td></tr>
            <tr><th><code>regime_signals.csv</code></th><td>투자월/신호월, 성장·물가 확률, SJM 확률, 네 국면 확률, 전환 수, 특징 가중치</td><td>백테스트, 국면 분석</td></tr>
            <tr><th><code>proposed_backtest.csv</code></th><td>월별 gross/net return, turnover, 거래·환전비용, NAV, drawdown, 네 목표비중</td><td>성과·시각화·감사</td></tr>
            <tr><th><code>regime_metrics.json</code></th><td>성장/물가 balanced accuracy, 4분면 정확도, 혼동행렬</td><td>신호 검증</td></tr>
            <tr><th><code>config.json</code></th><td>잠금된 <code>StrategyConfig</code> 전체 필드</td><td>재현성</td></tr>
            <tr><th><code>calibration_grid.csv</code></th><td>36개 후보의 설정, 보정구간 성과, 평균 회전율, 선택 점수</td><td>파라미터 선택 감사</td></tr>
          </tbody></table></div>
        </section>

        <section id="macro">
          <h2>4. 거시 데이터 파이프라인</h2>
          <p><code>load_macro_data()</code>는 파일별 형식을 정규화하고 공표 가능 시점을 보수적으로 뒤로 이동한 다음, 전체표본 통계 대신 rolling 통계로 표준화합니다. 반환값은 모델 입력용 <code>features</code>와 평가·진단용 <code>core</code>입니다.</p>
          <h3>4.1 파일명 정규화</h3>
          <p>macOS/Linux에서 만들어진 한글 파일명은 Unicode NFD, Windows 파일명은 NFC일 수 있습니다. <code>get_path()</code>는 요청 이름과 디렉터리 항목을 모두 NFC로 정규화한 후 비교하므로, 화면상 동일한 한글 이름이 내부 코드포인트 차이로 열리지 않는 문제를 방지합니다. 일치하지 않으면 조용히 다른 파일을 고르지 않고 <code>FileNotFoundError</code>를 발생시킵니다. <a class="src" href="#src-regime-L28">regime_research.py:L28–34</a></p>
          <h3>4.2 입력별 처리와 시점 규칙</h3>
          <div class="table-wrap"><table><thead><tr><th>모듈</th><th>원천/선택 열</th><th>날짜 처리</th><th>변환</th><th>rolling 창</th></tr></thead><tbody>
            <tr><th>GDP</th><td><code>GDP 성장률.xlsx</code>, QoQ/YoY</td><td>분기 Period → 분기말 월 → <strong>다음 월말</strong>, 이후 월별 forward-fill</td><td>YoY z-score</td><td>72개월</td></tr>
            <tr><th>수출</th><td><code>수출입 총괄_20260816.xlsx</code>, 수출 금액/수입금액</td><td><code>YYYY.MM</code> → <strong>다음 월말</strong></td><td>쉼표 제거, 12개월 수출 증가율</td><td>36개월</td></tr>
            <tr><th>BSI</th><td><code>기업경기조사(전망).csv</code>, 제조업·업황전망BSI</td><td><code>월</code>/공백 제거 → 해당 표기의 <strong>다음 월말</strong></td><td>cp949 읽기, 필터 후 z-score</td><td>24개월</td></tr>
            <tr><th>CPI</th><td><code>소비자물가 상승률.xlsx</code>, YoY</td><td><code>YYYY-MM</code> → <code>MonthEnd(2)</code>, 즉 파싱된 월초 기준 다음 월말</td><td>YoY z-score</td><td>36개월</td></tr>
            <tr><th>PPI</th><td><code>생산자물가 상승률.xlsx</code>, YoY</td><td>CPI와 동일</td><td>YoY z-score</td><td>36개월</td></tr>
            <tr><th>수입물가</th><td><code>수출입물가 상승률.xlsx</code>, ImportPrice_YoY</td><td>CPI와 동일</td><td>수입물가 YoY z-score</td><td>36개월</td></tr>
          </tbody></table></div>
          <div class="callout warn"><strong>정확한 코드 의미.</strong> 위 “다음 월말”은 경제통계의 실제 빈티지 공표시각을 데이터베이스에서 조회한 것이 아니라 코드에 구현된 보수적 달력 이동입니다. 따라서 revision bias까지 제거한 point-in-time 데이터셋은 아닙니다.</div>
          <h3>4.3 Rolling z-score</h3>
          <div class="formula"><code>z<sub>t</sub> = clip((x<sub>t</sub> − mean(x<sub>t−W+1:t</sub>)) / std<sub>ddof=1</sub>(x<sub>t−W+1:t</sub>), −3, +3)</code></div>
          <p><code>min_periods=W</code>이므로 창이 완전히 채워지기 전에는 값이 나오지 않습니다. 표준편차 0은 NaN으로 바꾸며, 극단치가 상태 중심과 거리 계산을 지배하지 않도록 ±3에서 자릅니다. 이는 미래 평균·표준편차를 사용하는 전체표본 표준화와 다릅니다. <a class="src" href="#src-regime-L36">L36–39</a></p>
          <h3>4.4 특징 행렬</h3>
          <div class="grid-2">
            <div class="card"><h4>성장 모듈: 6열</h4><p><code>GDP_level</code>, <code>Export_level</code>, <code>BSI_level</code>과 각각의 <code>_d3</code>(3개월 차분).</p></div>
            <div class="card"><h4>물가 모듈: 6열</h4><p><code>CPI_level</code>, <code>PPI_level</code>, <code>ImportPrice_level</code>과 각각의 <code>_d3</code>.</p></div>
          </div>
          <p>두 모듈은 열 상위 레벨이 <code>growth</code>/<code>inflation</code>인 MultiIndex DataFrame으로 결합됩니다. 마지막 <code>dropna()</code>는 12개 열이 모두 관측되는 공통 구간만 남깁니다. 현재 결과는 <strong>256행 × 12열, 2005-04-30~2026-07-31</strong>이며 결측치는 0개입니다.</p>
        </section>

        <section id="market">
          <h2>5. 자산 가격·수익률 파이프라인</h2>
          <h3>5.1 시장 캐시</h3>
          <p><code>download_market_cache(refresh=False)</code>는 <code>cache/market_daily.csv</code>가 있으면 네트워크를 사용하지 않고 즉시 읽습니다. 새로 받을 때는 yfinance의 <code>069500.KS</code>, <code>GLD</code>, <code>USO</code>, <code>KRW=X</code>를 2000-01-01부터 순차 다운로드합니다. KODEX200·GLD·USO는 <code>auto_adjust=True</code>, 환율은 원시 값 유지를 위해 <code>False</code>입니다. MultiIndex 열을 평탄화하고 UTC timezone을 제거한 정규화 날짜, open, close, 내부 symbol만 캐시에 저장합니다. <a class="src" href="#src-regime-L98">L98–117</a></p>
          <h3>5.2 KODEX200 프록시 접합</h3>
          <ol>
            <li><code>compass.db</code>의 <code>etf_prices</code>에서 <code>symbol='1028'</code>인 KOSPI200 프록시를 읽습니다.</li>
            <li>Yahoo KODEX200은 2009-03-31 이후의 연속 구간만 실제 시계열로 사용합니다.</li>
            <li>첫 실제 KODEX200 open과 같은 날의 프록시 open을 앵커로 잡습니다. 정확한 날짜가 없으면 절대 날짜 차이가 가장 작은 한 행을 사용합니다.</li>
            <li>프록시 open/close 전체에 <code>actual_anchor / proxy_anchor</code>를 곱해 수준을 맞춥니다.</li>
            <li>첫 실제 날짜 이전 프록시와 그 이후 실제 KODEX200을 이어 붙입니다.</li>
          </ol>
          <div class="formula"><code>ProxyAdjusted<sub>t</sub> = Proxy<sub>t</sub> × ActualOpen<sub>splice</sub> / ProxyOpen<sub>splice</sub></code></div>
          <h3>5.3 채권·환율·해외자산 원화 환산</h3>
          <p>국내 채권은 <code>krx_bond_index.csv</code>의 첫 번째 열을 날짜, 두 번째 열을 지수 수준으로 읽고 open과 close를 같은 값으로 둡니다. USD/KRW는 일력으로 재색인한 뒤 forward-fill합니다. GLD와 USO의 매월 첫 거래일 open에 해당 날짜까지의 USD/KRW 종가를 곱해 원화 수준을 만듭니다.</p>
          <div class="callout warn"><strong>시간대 불일치.</strong> 미국 ETF의 첫 거래가격과 USD/KRW 종가는 동일 순간 가격이 아닙니다. 구현은 월간 연구 단순화를 위해 날짜 기준으로 결합하며, 문서의 한계에 명시합니다.</div>
          <h3>5.4 월간 forward return</h3>
          <p>각 자산은 날짜를 월 Period로 바꾼 뒤 <code>groupby(month).first()</code>로 첫 거래일 행을 선택합니다. 네 자산 수준을 공통 월에 맞추고 아래 식으로 투자월 수익률을 계산합니다.</p>
          <div class="formula"><code>R<sub>t</sub> = Level<sub>t+1, first open</sub> / Level<sub>t, first open</sub> − 1</code></div>
          <p>코드의 <code>levels.shift(-1).div(levels).sub(1)</code> 때문에 인덱스 <code>t</code>는 “월 <code>t</code> 첫 거래일에 진입하여 다음 달 첫 거래일까지 보유한 수익률”을 뜻합니다. 네 자산 중 하나라도 없는 달은 제거됩니다. 현재 공통 결과는 <strong>244행 × 4열, 2006-04~2026-07</strong>입니다. <a class="src" href="#src-regime-L174">L174–179</a></p>
        </section>

        <section id="sjm">
          <h2>6. Sparse Jump Model 상세</h2>
          <p><code>SparseJump2</code>는 외부 전용 패키지 없이 구현한 투명한 2상태 jump model입니다. 상태별 중심 및 희소 특징 가중치 추정과, 상태 전환 페널티가 포함된 동적계획법 경로 선택을 번갈아 수행합니다. 기본값은 <code>jump_penalty=3.0</code>, <code>keep_features=4</code>, <code>max_iter=30</code>입니다.</p>
          <h3>6.1 Robust scaling</h3>
          <ol>
            <li>각 열 중앙값을 위치로, 75백분위−25백분위(IQR)를 척도로 사용합니다.</li>
            <li>IQR이 0.15보다 작으면 해당 열의 전체 과거 표준편차로 대체합니다.</li>
            <li>그래도 1e−6보다 작으면 1로 대체합니다.</li>
            <li>표준화된 값은 −5~+5로 clip합니다.</li>
          </ol>
          <div class="formula"><code>x̃<sub>tj</sub> = clip((x<sub>tj</sub> − median<sub>j</sub>) / scale<sub>j</sub>, −5, 5)</code></div>
          <p>여기서 중앙값·IQR은 매 투자월의 <code>hist</code>에 대해서만 다시 계산되므로 해당 투자월 이후의 데이터는 포함되지 않습니다.</p>
          <h3>6.2 초기 상태</h3>
          <p>입력 열의 앞쪽 세 개가 수준 특징이라는 열 순서 계약을 이용합니다. 첫 세 열 평균 score가 그 과거 창의 score 중앙값보다 높으면 초기 상태 1, 아니면 0입니다. 최초 특징 가중치는 모든 열에 동일하게 <code>1 / p</code>를 둡니다.</p>
          <h3>6.3 희소 특징 선택</h3>
          <p>각 상태의 열별 중심과 상태내 분산을 계산한 뒤, 상태 분리도를 아래처럼 정의합니다. 분모의 0.20은 분산이 지나치게 작은 열의 폭주를 완화하는 안정화 상수입니다.</p>
          <div class="formula"><code>separation<sub>j</sub> = (center<sub>1j</sub> − center<sub>0j</sub>)² / (withinVar<sub>j</sub> + 0.20)</code></div>
          <p>분리도가 큰 최대 4개 열만 남기고 나머지 가중치는 0으로 만듭니다. 선택된 열의 가중치는 최소 1e−4를 적용한 분리도에 비례하도록 합이 1이 되게 정규화합니다. 각 관측치와 상태 중심 사이의 거리는 선택된 특징만 반영한 가중 제곱거리입니다.</p>
          <div class="formula"><code>distance(t,k) = Σ<sub>j</sub> weight<sub>j</sub> × (x̃<sub>tj</sub> − center<sub>kj</sub>)²</code></div>
          <h3>6.4 동적계획법 상태 경로</h3>
          <p>시점 <code>t</code>에서 상태 <code>k</code>로 끝나는 최소 누적비용은 직전 두 상태 중 작은 값을 선택합니다. 직전 상태가 현재 상태와 다르면 <code>jump_penalty</code>를 더합니다.</p>
          <div class="formula"><code>C(t,k) = distance(t,k) + min<sub>j∈{{0,1}}</sub>[C(t−1,j) + jump × 1(j≠k)]</code></div>
          <p><code>back[t,k]</code>에 최적 직전 상태를 저장하고 마지막 최소비용 상태에서 역추적합니다. 전체 복잡도는 상태 수가 2로 고정되어 대략 <code>O(T×4)</code>이며, 각 반복의 중심·거리 계산은 <code>O(T×p)</code>입니다. 새 경로가 이전 경로와 같으면 최대 30회 전에 조기 종료합니다. <a class="src" href="#src-regime-L201">L201–215</a></p>
          <h3>6.5 “High” 상태 확률</h3>
          <p>수렴 후 두 상태 중심에서 앞 세 수준 특징의 평균이 큰 상태를 <code>high_state</code>로 해석합니다. 마지막 관측치의 국소비용에는 직전 상태와 다를 때 원래 jump penalty의 55%를 넣고, 온도 0.85의 softmax로 확률화합니다.</p>
          <div class="formula"><code>localCost(k)=distance(last,k)+0.55×jump×1(k≠previousState)</code><br><code>P(k)=softmax(−localCost(k)/0.85), &nbsp; pHigh=clip(P(highState),0.03,0.97)</code></div>
          <p>3%/97% 절단은 단일 모델이 완전 확신을 내지 못하게 합니다. 함수는 <code>p_high</code>와 함께 마지막 상태, high state 식별자, 전체 전환 횟수, 특징별 가중치를 반환하여 신호를 감사할 수 있게 합니다. <a class="src" href="#src-regime-L217">L217–262</a></p>
        </section>

        <section id="signals">
          <h2>7. 국면 확률 생성</h2>
          <h3>7.1 Walk-forward 시점 정렬</h3>
          <p><code>compute_regime_signals()</code>는 자산 수익률의 각 <code>target_month</code>를 순회합니다. 신호 기준월은 항상 <code>target_month − 1</code>이며, 거시 특징은 그 신호월의 달력 월말까지만 자릅니다. 과거 특징이 24행보다 적으면 해당 투자월은 건너뜁니다.</p>
          <pre class="block"><code>for target_month in returns.index:
        signal_month = target_month - 1
        hist = features.loc[: signal_month.to_timestamp("M")]
        if len(hist) &lt; 24:
            continue</code></pre>
          <p>이 구조 때문에 자산 수익률 244개월 중 초기 12개월이 학습기간으로 빠지고, 실제 신호/백테스트는 232개월이 됩니다.</p>
          <h3>7.2 해석 가능한 logistic 합성확률</h3>
          <p>성장과 물가 각각에 대해 세 수준 특징 평균과 세 3개월 변화 평균을 계산합니다. 수준에 3개월 모멘텀의 20%를 더하고 scale 0.55로 나눈 뒤 sigmoid(<code>expit</code>)를 적용합니다.</p>
          <div class="formula"><code>pComposite = sigmoid((mean(levels) + 0.20 × mean(Δ3 levels)) / 0.55)</code></div>
          <h3>7.3 SJM 앙상블과 시간 평활</h3>
          <div class="formula"><code>pRaw = 0.10 × pSJM + 0.90 × pComposite</code><br><code>p<sub>t</sub> = 0.85 × pRaw<sub>t</sub> + 0.15 × p<sub>t−1</sub></code></div>
          <p>작은 월별 표본에서 복잡한 상태모델이 과도하게 지배하지 않도록 투명한 합성확률이 90%를 차지합니다. SJM은 희소 특징 선택과 전환 지속성을 보조합니다. 이전 확률은 성장·물가 각각 0.5로 시작하며 15% 평활이 월간 확률 점프를 줄입니다.</p>
          <h3>7.4 네 국면 결합확률</h3>
          <div class="table-wrap"><table><thead><tr><th>국면</th><th>조건 해석</th><th>확률식</th><th>현재 빈도</th></tr></thead><tbody>
            <tr><th>Goldilocks</th><td>성장 High, 물가 Low</td><td><code>pg × (1−pi)</code></td><td>{int(regime_counts.get('Goldilocks',0))}개월</td></tr>
            <tr><th>Overheating</th><td>성장 High, 물가 High</td><td><code>pg × pi</code></td><td>{int(regime_counts.get('Overheating',0))}개월</td></tr>
            <tr><th>Slowdown</th><td>성장 Low, 물가 Low</td><td><code>(1−pg) × (1−pi)</code></td><td>{int(regime_counts.get('Slowdown',0))}개월</td></tr>
            <tr><th>Stagflation</th><td>성장 Low, 물가 High</td><td><code>(1−pg) × pi</code></td><td>{int(regime_counts.get('Stagflation',0))}개월</td></tr>
          </tbody></table></div>
          <p>네 확률은 독립적인 두 Bernoulli 축의 곱으로 만들어 합이 항상 1입니다. 문자열 <code>regime</code>은 최대확률 국면이지만, 제안 전략의 soft anchor는 최대값 하나가 아니라 네 확률 전체를 사용합니다.</p>
          <h3>7.5 신호 출력 16열</h3>
          <p><code>signal_month</code>, 성장/물가 최종 확률 2개, SJM 단독 확률 2개, 현재 수준 합성점수 2개, 최빈 국면, 국면 확률 4개, 성장/물가 상태 전환 수 2개, 성장/물가 특징 가중치 JSON 2개가 저장됩니다. 특징 가중치는 <code>ensure_ascii=False</code>로 직렬화되어 한글도 보존할 수 있습니다.</p>
        </section>

        <section id="allocation">
          <h2>8. 기준비중과 위험제어 최적화</h2>
          <h3>8.1 국면별 경제적 기준비중</h3>
          <div class="table-wrap"><table><thead><tr><th>국면</th><th>KODEX200</th><th>BOND</th><th>GLD</th><th>USO</th><th>의도</th></tr></thead><tbody>
            <tr><th>Goldilocks</th><td>58%</td><td>22%</td><td>15%</td><td>5%</td><td>성장 수혜 주식 중심</td></tr>
            <tr><th>Overheating</th><td>30%</td><td>12%</td><td>23%</td><td>35%</td><td>인플레이션 민감 원유·금 확대</td></tr>
            <tr><th>Slowdown</th><td>12%</td><td>66%</td><td>20%</td><td>2%</td><td>채권 중심 방어</td></tr>
            <tr><th>Stagflation</th><td>8%</td><td>24%</td><td>50%</td><td>18%</td><td>금 중심 인플레이션 방어</td></tr>
          </tbody></table></div>
          <p><code>soft_anchor()</code>는 국면확률 행벡터와 위 4×4 행렬을 곱합니다. 각 국면 행의 합이 1이고 확률 합도 1이므로 결과 비중 합도 1입니다. 이어 <code>regime_strength=0.75</code>를 적용해 전략적 비중 <code>[20%,45%,30%,5%]</code>와 혼합합니다.</p>
          <div class="formula"><code>anchorSoft = pRegimeᵀ × AnchorMatrix</code><br><code>anchor = 0.75 × anchorSoft + 0.25 × strategic</code></div>
          <h3>8.2 비대칭 EWMA 공분산</h3>
          <p>최적화에는 최대 최근 84개월이 들어갑니다. 반감기 12개월의 감쇠계수는 <code>α=1−exp(log(0.5)/12)≈0.0561</code>입니다. 초기 공분산은 이력 앞쪽 최대 24개월의 표본 공분산이고, 이후 매 행을 순회하며 갱신합니다.</p>
          <div class="formula"><code>Σ<sub>t</sub> = (1−α)Σ<sub>t−1</sub> + α × m<sub>t</sub> × shock<sub>t</sub>shock<sub>t</sub>ᵀ</code><br><code>m<sub>t</sub> = 1 + min(max(−R<sub>KODEX,t</sub>,0)/0.08, 1.5)</code></div>
          <p>KODEX200 월수익이 음수일 때만 multiplier가 1보다 커지며 최대 2.5입니다. 즉 주식 급락기에 전체 자산 충격의 outer product를 더 강하게 반영합니다. 마지막에 대각선 1e−7을 더해 수치적 양의 정부호성을 보강합니다. 이 구현은 Bayesian stochastic volatility가 아니라 계산이 투명한 비대칭 EWMA 대용치입니다.</p>
          <h3>8.3 역변동성 prior</h3>
          <div class="formula"><code>tilted<sub>i</sub> ∝ anchor<sub>i</sub> × (median(vol)/vol<sub>i</sub>)<sup>0.35</sup></code><br><code>prior = 0.55 × anchor + 0.45 × tilted</code></div>
          <p>자산별 월 변동성은 최소 0.5%로 자르고, 역변동성 tilt 후 합이 1이 되게 정규화합니다. prior는 경제 국면과 통계 위험 사이의 시작점이자 tracking penalty의 기준입니다.</p>
          <h3>8.4 기대수익과 동적 목표 변동성</h3>
          <p>장기 평균은 해당 투자월 이전 전체 이력의 expanding mean, 최근 평균은 최대 84개월에 대한 반감기 24개월 EWM입니다. 두 값을 80%/20%로 섞고 월 −0.6%~+1.5%로 clip하여 기대수익 추정치의 폭주를 제한합니다.</p>
          <div class="formula"><code>μ = clip(0.80×μ<sub>expanding</sub> + 0.20×μ<sub>EWM24</sub>, −0.006, 0.015)</code><br><code>targetVol = 8% × (0.86 + 0.20 × pGrowthHigh)</code></div>
          <p>성장확률 0~1에 따라 목표 연변동성은 6.88%~8.48%입니다. 성장 국면이 강할수록 아주 조금 더 위험 예산을 허용합니다.</p>
          <h3>8.5 SLSQP 목적함수</h3>
          <div class="formula"><code>min<sub>w</sub> −1.15×Return<sub>ann</sub> + 0.18×Vol<sub>ann</sub> + 0.25×|CDaR<sub>90</sub>| + 0.05×TurnoverSmooth + 0.32×TrackingError²</code></div>
          <ul>
            <li><code>Return_ann = 12 × wᵀμ</code></li>
            <li><code>Vol_ann = √(12 × wᵀΣw)</code></li>
            <li><code>CDaR90</code>은 동일 비중을 과거 최대 84개월에 적용한 경로에서 가장 나쁜 하위 10% drawdown의 평균입니다.</li>
            <li><code>TurnoverSmooth = 0.5 × Σ√((w−pretrade)²+1e−6)</code>로 절댓값의 미분 불가능 지점을 완화합니다.</li>
            <li><code>TrackingError² = Σ(w−prior)²</code>입니다.</li>
          </ul>
          <h3>8.6 제약과 bounds</h3>
          <div class="table-wrap"><table><thead><tr><th>종류</th><th>코드 의미</th></tr></thead><tbody>
            <tr><th>완전투자</th><td><code>Σw = 1</code></td></tr>
            <tr><th>목표 변동성</th><td><code>target − √(12wᵀΣw) ≥ 0</code></td></tr>
            <tr><th>CDaR 상한</th><td><code>0.16 + CDaR90 ≥ 0</code>; CDaR이 음수이므로 절댓값 16% 이하 의미</td></tr>
            <tr><th>KODEX200</th><td>2% ≤ w ≤ 68%</td></tr>
            <tr><th>BOND</th><td>5% ≤ w ≤ 88%</td></tr>
            <tr><th>GLD</th><td>2% ≤ w ≤ 62%</td></tr>
            <tr><th>USO</th><td>0% ≤ w ≤ 38%</td></tr>
          </tbody></table></div>
          <p>SLSQP는 prior에서 시작해 최대 80회, <code>ftol=1e−8</code>로 풉니다. 성공하지 않았거나 결과가 유한하지 않으면 prior를 안전한 fallback으로 사용합니다. 그 후 음수값을 0으로 clip하고 다시 합 1로 정규화합니다.</p>
          <h3>8.7 드로다운 가드</h3>
          <p>현재 NAV drawdown이 −5%보다 작아지면 최적화 비중을 방어비중 <code>[5%,72%,23%,0%]</code>와 추가 혼합합니다.</p>
          <div class="formula"><code>severity = clip((−currentDD − 0.05)/0.12, 0, 1)</code><br><code>blend = 0.75 × (0.25 + 0.50×severity)</code><br><code>wFinal = (1−blend)×wOptimized + blend×Defensive</code></div>
          <p>가드가 막 발동하면 약 18.75%, 심각도가 최대면 56.25%를 방어비중으로 혼합합니다. 위험자산을 완전히 제거하지 않아 회복 참여를 위한 바닥 비중을 남깁니다.</p>
        </section>

        <section id="backtest">
          <h2>9. 백테스트 엔진</h2>
          <h3>9.1 비교 모드</h3>
          <div class="table-wrap"><table><thead><tr><th><code>mode</code></th><th>비중 규칙</th><th>용도</th></tr></thead><tbody>
            <tr><th><code>proposed</code></th><td><code>controlled_weights()</code> 전체</td><td>최종 제안 전략</td></tr>
            <tr><th><code>soft</code></th><td>국면 확률가중 anchor만 사용</td><td>위험제어의 추가효과 분리</td></tr>
            <tr><th><code>hard</code></th><td>Goldilocks=[1,0,0,0], Overheating=[0,0,0,1], Slowdown=[.6,.4,0,0], Stagflation=[0,0,1,0]</td><td>단일국면 hard switching 비교</td></tr>
            <tr><th><code>equal</code></th><td>[25%,25%,25%,25%]</td><td>단순 분산 기준</td></tr>
            <tr><th><code>static_defensive</code></th><td>[20%,45%,30%,5%]</td><td>동적 신호 없는 방어 기준</td></tr>
            <tr><th><code>kodex</code></th><td>[100%,0,0,0]</td><td>국내 주식 buy-and-hold 기준</td></tr>
          </tbody></table></div>
          <h3>9.2 한 달의 상태 전이</h3>
          <ol class="steps">
            <li><strong>허용 월 결정.</strong> 신호와 수익률 인덱스의 교집합을 만든 뒤 선택적 start/end Period로 자릅니다.</li>
            <li><strong>과거만 추출.</strong> <code>history = returns.loc[returns.index &lt; month]</code>로 현재 투자월 수익률을 최적화에서 제외합니다.</li>
            <li><strong>현재 drawdown 계산.</strong> 거래 직전 <code>nav/peak−1</code>을 드로다운 가드 상태로 넘깁니다.</li>
            <li><strong>목표비중 계산.</strong> mode에 따라 네 자산 비중 <code>w</code>를 만듭니다.</li>
            <li><strong>거래량·비용 계산.</strong> 직전 월말 드리프트 비중 <code>pretrade</code>와 목표비중 차이 <code>delta</code>를 사용합니다.</li>
            <li><strong>월 수익률 반영.</strong> <code>gross = wᵀR</code>, 비용 차감 후 NAV와 peak를 갱신합니다.</li>
            <li><strong>다음 달 거래 전 비중.</strong> 각 자산의 월 수익률을 반영한 end weight를 저장합니다.</li>
          </ol>
          <h3>9.3 회전율과 비용</h3>
          <div class="formula"><code>delta = wTarget − wPretrade</code><br><code>turnover = firstTrade ? Σ|delta| : 0.5×Σ|delta|</code><br><code>tradeCost = Σ|delta| × 0.0015 × costMultiplier</code><br><code>fxCost = |(wGLD+wUSO) − (preGLD+preUSO)| × 0.0005 × costMultiplier</code></div>
          <p>보고 회전율은 보통 매수·매도를 중복 계산하지 않도록 절반을 취하지만, 첫 진입은 현금 100%에서 시작하므로 합계 1을 그대로 기록합니다. 매매비용은 매수/매도 명목 각각 15bp를 뜻하도록 <code>Σ|delta|</code> 전체에 적용됩니다. 예를 들어 A에서 B로 10% 옮기면 절댓값 합이 20%이므로 자산 교체 비용은 포트폴리오의 3bp입니다. 환전비용은 GLD+USO <em>합계</em> 비중의 순변화에만 5bp를 적용하므로 두 해외자산 사이 교체에는 별도 환전비용이 없습니다.</p>
          <h3>9.4 순수익률·NAV·드리프트 비중</h3>
          <div class="formula"><code>netReturn = grossReturn − tradeCost − fxCost</code><br><code>NAV<sub>t</sub> = NAV<sub>t−1</sub> × (1 + netReturn<sub>t</sub>)</code><br><code>endWeight<sub>i</sub> = w<sub>i</sub> × (1+R<sub>i</sub>) / (1+grossReturn)</code></div>
          <p>비용은 수익률에서 단순 차감합니다. 다음 달 <code>pretrade</code>는 목표비중을 그대로 복사하지 않고 자산별 gross return으로 드리프트시킨 값입니다. 분모가 portfolio gross factor이므로 end weights의 합은 수치 오차를 제외하면 1입니다. <a class="src" href="#src-regime-L444">L444–510</a></p>
        </section>

        <section id="metrics">
          <h2>10. 성과·국면 평가</h2>
          <h3>10.1 성과지표</h3>
          <div class="table-wrap"><table><thead><tr><th>지표</th><th>구현식/정의</th><th>주의</th></tr></thead><tbody>
            <tr><th>Months</th><td>결측 제거 후 월수익률 개수</td><td>연수는 Months/12</td></tr>
            <tr><th>CAGR</th><td><code>FinalMultiple^(1/years)−1</code></td><td>월별 복리</td></tr>
            <tr><th>Volatility</th><td><code>std(ddof=1)×√12</code></td><td>월수익률 연환산</td></tr>
            <tr><th>Sharpe</th><td><code>mean/std×√12</code></td><td>무위험수익률 0%</td></tr>
            <tr><th>Sortino</th><td><code>mean×12 / [sqrt(mean(min(r,0)²))×√12]</code></td><td>0보다 낮은 월수익률만 downside</td></tr>
            <tr><th>MDD</th><td><code>min(wealth/runningMax−1)</code></td><td>음수로 표시</td></tr>
            <tr><th>Calmar</th><td><code>CAGR/|MDD|</code></td><td>MDD가 0이면 NaN</td></tr>
            <tr><th>FinalMultiple</th><td><code>Π(1+r)</code></td><td>초기자본 1 기준</td></tr>
            <tr><th>PositiveMonths</th><td><code>mean(r&gt;0)</code></td><td>0% 월은 성공에 포함하지 않음</td></tr>
          </tbody></table></div>
          <h3>10.2 CDaR 구현</h3>
          <p><code>cdar()</code>는 수익률 경로로 wealth와 drawdown을 만들고, drawdown을 오름차순 정렬하여 가장 작은(가장 손실이 큰) <code>ceil((1−α)T)</code>개 평균을 반환합니다. α=0.90이면 최악 10% drawdown 평균이며 값은 음수입니다. 최적화 목적함수는 절댓값을 벌점으로, 제약은 <code>max_cdar + cdar ≥ 0</code> 형태로 사용합니다.</p>
          <h3>10.3 국면 예측 평가</h3>
          <p>사후 정답은 <code>core</code>의 성장/물가 3개 z-score 평균에 대해 <strong>다음 1·2·3개월 공표치의 평균</strong>이 0 이상인지로 정의합니다. 신호 인덱스를 <code>target_month</code>가 아니라 <code>signal_month</code> 월말 timestamp로 바꿔 정답과 inner join합니다. 예측은 확률 0.5 이상을 High로 분류합니다.</p>
          <div class="grid-2">
            <div class="card"><h4>성장</h4><p>Balanced accuracy <strong>{pct(regime_metrics['growth_balanced_accuracy'],1)}</strong></p><p>혼동행렬 [[{regime_metrics['growth_confusion'][0][0]}, {regime_metrics['growth_confusion'][0][1]}], [{regime_metrics['growth_confusion'][1][0]}, {regime_metrics['growth_confusion'][1][1]}]]</p></div>
            <div class="card"><h4>물가</h4><p>Balanced accuracy <strong>{pct(regime_metrics['inflation_balanced_accuracy'],1)}</strong></p><p>혼동행렬 [[{regime_metrics['inflation_confusion'][0][0]}, {regime_metrics['inflation_confusion'][0][1]}], [{regime_metrics['inflation_confusion'][1][0]}, {regime_metrics['inflation_confusion'][1][1]}]]</p></div>
          </div>
          <p>두 축을 모두 맞힌 4분면 정확도는 <strong>{pct(regime_metrics['quadrant_accuracy'],1)}</strong>, 평가 표본은 {regime_metrics['n_months']}개월입니다. Balanced accuracy는 각 클래스 recall의 평균이라 High/Low 표본 수 불균형에 일반 accuracy보다 덜 민감합니다.</p>
        </section>

        <section id="calibration">
          <h2>11. 파라미터 보정과 잠금 테스트</h2>
          <h3>11.1 제한 그리드</h3>
          <p><code>calibrate_configs.py</code>는 아래 네 파라미터만 Cartesian product로 탐색합니다. 나머지 목적함수 계수와 CDaR 한도는 고정입니다.</p>
          <div class="formula"><code>regime_strength ∈ {{0.25,0.50,0.75}}</code><br><code>target_vol ∈ {{0.08,0.10,0.12}}</code><br><code>invvol_tilt ∈ {{0.15,0.35}}</code><br><code>drawdown_guard ∈ {{0.40,0.75}}</code><br><strong>총 3×3×2×2 = 36개</strong></div>
          <h3>11.2 선택 점수와 기간 분리</h3>
          <div class="formula"><code>breach = max(|MDD| − 0.15, 0)</code><br><code>ValidationScore = Sharpe + 0.35×Calmar − 4×breach − 0.20×AvgTurnover</code></div>
          <p>설정 선택에는 백테스트 시작~2017-12까지만 사용합니다. 2018-01 이후는 우승 설정을 결정한 후 평가하는 잠금 구간입니다. 따라서 보정 스크립트가 2018년 이후 성과를 점수에 섞지는 않습니다. 다만 원천 거시파일 자체는 최신 개정치이므로 빈티지 데이터 수준의 완전한 시간분리는 아닙니다.</p>
          <h3>11.3 상위 10개 후보</h3>
          <div class="table-wrap"><table><thead><tr><th>설정</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th><th>평균 회전율</th><th>점수</th></tr></thead><tbody>{''.join(cal_rows)}</tbody></table></div>
          <div class="callout good"><strong>잠금 우승 설정.</strong> <code>rs0.75_tv0.08_iv0.35_dg0.75</code>가 점수 {calibration.iloc[0]['ValidationScore']:.3f}으로 1위입니다. 이는 <code>StrategyConfig</code> 기본값과 일치합니다.</div>
          <h3>11.4 잠금 설정 전체</h3>
          <div class="table-wrap"><table><thead><tr><th>필드</th><th>값</th><th>의미</th></tr></thead><tbody>{config_rows}</tbody></table></div>
        </section>

        <section id="results">
          <h2>12. 현재 실행 결과</h2>
          <p class="muted">수치는 현재 <code>results/</code> 산출물과 동일한 소스 스냅샷의 재실행 결과입니다. 수익률은 거래·환전비용 차감 후이며, 과거 성과는 미래 성과를 보장하지 않습니다.</p>
          <h3>12.1 전체 구간: 2007-04~2026-07</h3>
          <div class="table-wrap"><table><thead><tr><th>전략</th><th>개월</th><th>CAGR</th><th>변동성</th><th>Sharpe</th><th>Sortino</th><th>MDD</th><th>Calmar</th><th>최종배수</th><th>양(+)의 월</th></tr></thead><tbody>{metric_rows(summary)}</tbody></table></div>
          <h3>12.2 잠금 구간: 2018-01~2026-07</h3>
          <div class="table-wrap"><table><thead><tr><th>전략</th><th>개월</th><th>CAGR</th><th>변동성</th><th>Sharpe</th><th>Sortino</th><th>MDD</th><th>Calmar</th><th>최종배수</th><th>양(+)의 월</th></tr></thead><tbody>{metric_rows(locked)}</tbody></table></div>
          <div class="callout"><strong>결과 해석.</strong> 제안 전략은 전체 구간에서 CAGR {pct(float(summary.loc['proposed','CAGR']))}, 연변동성 {pct(float(summary.loc['proposed','Volatility']))}, Sharpe {summary.loc['proposed','Sharpe']:.3f}, MDD {pct(float(summary.loc['proposed','MDD']))}입니다. hard 전략보다 절대수익은 낮지만 MDD와 변동성을 크게 줄이는 것이 설계 목표입니다. 잠금 구간 Sharpe는 {locked.loc['proposed','Sharpe']:.3f}입니다.</div>
          <h3>12.3 최신 신호와 비중</h3>
          <div class="grid-2">
            <div class="card"><h4>{esc(latest_signal['target_month'])} 투자 신호</h4><p>신호월 <strong>{esc(latest_signal['signal_month'])}</strong><br>최빈 국면 <strong>{esc(latest_signal['regime'])}</strong><br>P(Growth High) <strong>{pct(float(latest_signal['p_growth_high']),1)}</strong><br>P(Inflation High) <strong>{pct(float(latest_signal['p_inflation_high']),1)}</strong></p></div>
            <div class="card"><h4>네 국면 확률</h4><p>Goldilocks {pct(latest_probabilities[0],1)}<br>Overheating {pct(latest_probabilities[1],1)}<br>Slowdown {pct(latest_probabilities[2],1)}<br>Stagflation {pct(latest_probabilities[3],1)}</p></div>
          </div>
          <div class="table-wrap"><table><thead><tr><th>단계</th><th>KODEX200</th><th>BOND</th><th>GLD</th><th>USO</th></tr></thead><tbody>
            <tr><th>확률가중 soft anchor</th>{''.join(f'<td>{pct(v,1)}</td>' for v in latest_soft_anchor)}</tr>
            <tr><th>75% 국면 + 25% 전략 anchor</th>{''.join(f'<td>{pct(v,1)}</td>' for v in latest_blended_anchor)}</tr>
            <tr><th>최종 위험제어 목표비중</th>{''.join(f'<td>{pct(float(latest_bt[f"w_{a}"]),1)}</td>' for a in asset_keys)}</tr>
          </tbody></table></div>
          <p>최신 월은 gross return {pct(float(latest_bt['gross_return']))}, net return {pct(float(latest_bt['return']))}, 회전율 {pct(float(latest_bt['turnover']))}, 월말 NAV {float(latest_bt['nav']):.3f}, drawdown {pct(float(latest_bt['drawdown']))}입니다. 최종비중이 국면 anchor와 크게 다른 이유는 역변동성 prior, 변동성/CDaR 제약, 기대수익, 직전비중 회전율 페널티, 당시 drawdown guard가 모두 순차 반영되기 때문입니다.</p>
          <h3>12.4 제안 전략 운용 진단</h3>
          <div class="table-wrap"><table><thead><tr><th>항목</th><th>현재 값</th><th>계산 근거</th></tr></thead><tbody>
            <tr><th>평균 월 회전율</th><td>2.506%</td><td><code>proposed_backtest.turnover.mean()</code></td></tr>
            <tr><th>누적 매매비용률 합</th><td>1.595%</td><td>월별 return 차감분의 산술 합</td></tr>
            <tr><th>누적 환전비용률 합</th><td>0.148%</td><td>월별 return 차감분의 산술 합</td></tr>
            <tr><th>평균 목표비중</th><td>KODEX 20.4% / BOND 51.4% / GLD 26.6% / USO 1.6%</td><td>232개월 목표비중 평균</td></tr>
            <tr><th>비용 0배 Sharpe</th><td>1.156</td><td>동일 전략, 비용만 0</td></tr>
            <tr><th>기본 비용 Sharpe</th><td>1.144</td><td>15bp 매매 + 5bp 환전</td></tr>
            <tr><th>비용 2배 Sharpe</th><td>1.132</td><td>30bp 매매 + 10bp 환전</td></tr>
          </tbody></table></div>
        </section>

        <section id="verification">
          <h2>13. 자동 검증과 불변조건</h2>
          <p>검증 노트북 마지막 셀은 단순 성과 출력이 아니라 시점·비중·비용·목표에 대한 실행 중단 조건을 둡니다.</p>
          <ul class="checklist">
            <li><code>signals</code>와 <code>asset_returns</code> 인덱스가 오름차순이며 중복이 없습니다.</li>
            <li>모든 행에서 <code>signal_month &lt; target_month</code>입니다.</li>
            <li>제안 전략 net return은 모두 유한값입니다.</li>
            <li>네 목표비중의 합은 행마다 오차 1e−8 이내로 1입니다.</li>
            <li>목표비중은 음수가 아닙니다.</li>
            <li>거래비용과 환전비용은 음수가 아닙니다.</li>
            <li>전체 구간 제안 전략 MDD가 −20%보다 큽니다.</li>
            <li>2018-01 이후 잠금 구간 제안 전략 Sharpe가 1.0보다 큽니다.</li>
          </ul>
          <div class="callout warn"><strong>검증 범위.</strong> 이 assert들은 중요한 회귀 오류를 잡지만, 원천 데이터의 경제적 정확성, 실제 발표시간, 빈티지 개정, 실거래 체결 가능성까지 증명하지는 않습니다. 또한 SLSQP가 매월 제약을 엄밀히 만족했는지를 별도 로그로 보존하지 않으며, 실패 시 prior fallback을 허용합니다.</div>
          <h3>추가로 확인할 수 있는 불변조건</h3>
          <ul>
            <li>네 국면 확률의 행별 합이 1인지 허용오차로 검사.</li>
            <li>optimizer 성공/실패, 최종 제약 slack, fallback 발생 여부를 월별 저장.</li>
            <li>원천 파일 해시와 시장 캐시 기준일을 결과 manifest에 기록.</li>
            <li><code>gross − trade_cost − fx_cost = return</code>을 행별 검사.</li>
            <li><code>nav</code>가 net return 누적곱과 일치하는지 검사.</li>
          </ul>
        </section>

        <section id="runbook">
          <h2>14. 실행 방법</h2>
          <h3>14.1 Windows PowerShell 환경 활성화</h3>
          <pre class="block"><code>Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    &amp; d:\Programming\python_example\Arcana\.venv-llama\Scripts\Activate.ps1
    Set-Location d:\Programming\python_example\RegimeDecisionTest</code></pre>
          <p><code>-Scope Process</code>이므로 실행 정책 변경은 현재 PowerShell 프로세스에만 적용됩니다.</p>
          <h3>14.2 기본 연구 실행</h3>
          <pre class="block"><code>python -m strategies.core.regime_research</code></pre>
          <p>기본 main은 <code>refresh=False</code>라 기존 <code>cache/market_daily.csv</code>를 재사용합니다. 실행 후 <code>results/summary.csv</code>, <code>regime_signals.csv</code>, <code>proposed_backtest.csv</code>, <code>regime_metrics.json</code>, <code>config.json</code>을 갱신합니다.</p>
          <h3>14.3 시장 캐시 강제 갱신</h3>
          <pre class="block"><code>python -c "from regime_research import main; main(refresh=True)"</code></pre>
          <div class="callout warn">강제 갱신은 네트워크와 yfinance 응답에 의존하고 기존 캐시 파일을 새 데이터로 덮어씁니다. 재현 가능한 과거 결과를 보존해야 한다면 먼저 캐시의 별도 사본과 해시를 관리해야 합니다.</div>
          <h3>14.4 보정·노트북·설명서 재생성</h3>
          <pre class="block"><code>python -m strategies.stage01_baseline.calibrate_configs
    python -m tools.builders.build_validated_notebook
    python -m tools.builders.build_implementation_guide</code></pre>
          <p>보정 스크립트는 import만 해도 top-level 루프가 실행됩니다. 라이브러리처럼 import할 계획이라면 향후 <code>main()</code>과 <code>if __name__ == "__main__"</code> 가드로 감싸는 것이 안전합니다.</p>
          <h3>14.5 노트북 실행/HTML 내보내기 예시</h3>
          <pre class="block"><code>jupyter nbconvert --to notebook --execute economic_regime_allocation_validated.ipynb ^
      --output economic_regime_allocation_validated_executed.ipynb
    jupyter nbconvert --to html economic_regime_allocation_validated_executed.ipynb ^
      --output economic_regime_allocation_validated.html</code></pre>
        </section>

        <section id="maintenance">
          <h2>15. 수정·확장 가이드</h2>
          <h3>15.1 거시변수 추가</h3>
          <ol>
            <li><code>load_macro_data()</code>에서 파일을 읽고 실제 시점가용성 규칙을 날짜 인덱스에 반영합니다.</li>
            <li>rolling 창과 단위를 명시하고 <code>core</code>에 z-score 열을 추가합니다.</li>
            <li>growth 또는 inflation 모듈의 level 열과 <code>diff(3)</code> 열에 포함합니다.</li>
            <li>SJM의 “앞 세 열이 수준 특징” 가정과 high_state 판별 코드를 함께 수정합니다.</li>
            <li><code>evaluate_regimes()</code>의 realized composite 정의도 같은 경제 축에 맞게 갱신합니다.</li>
          </ol>
          <h3>15.2 자산 추가</h3>
          <p><code>ASSETS</code>만 늘리면 끝나지 않습니다. 시장 로딩, FX 처리, <code>REGIME_ANCHORS</code>의 모든 행, <code>DEFENSIVE</code>/<code>STRATEGIC</code>, bounds, <code>pretrade=np.zeros(4)</code>, equal weight의 길이, hard mapping, FX 비용 대상 인덱스, 노트북 표·그래프를 모두 동기화해야 합니다. 현재 여러 위치에 자산 수 4와 해외자산 인덱스 2·3이 하드코딩되어 있습니다.</p>
          <h3>15.3 리밸런싱 주기 변경</h3>
          <p>현재 데이터·수익률·신호월·연환산 상수 12·반감기 단위·비용 산정이 모두 월간을 전제로 합니다. 주간/분기로 바꾸려면 단지 resample만 변경하지 말고 <code>sqrt(12)</code>, <code>12×mean</code>, 3개월 target, 24/84개월 창, 공표시차 의미까지 함께 재설계해야 합니다.</p>
          <h3>15.4 실거래 시스템으로 분리</h3>
          <div class="grid-2">
            <div class="card"><h4>연구 계층</h4><p>빈티지 데이터 저장, 특징 버전, 모델 보정, OOS 리포트, 파라미터 manifest.</p></div>
            <div class="card"><h4>운용 계층</h4><p>실시간 공표 캘린더, 최신 신호 snapshot, 주문가능 수량/호가, 세금·슬리피지, 체결 및 사후 reconciliation.</p></div>
          </div>
          <h3>15.5 권장 리팩터링 우선순위</h3>
          <ol>
            <li>자산 메타데이터(통화, bounds, ticker)를 dataclass/dict 하나로 통합.</li>
            <li>신호 앙상블 계수 0.10/0.90, 평활 0.85/0.15, SJM 온도 0.85 등을 설정 객체로 이동.</li>
            <li>optimizer 진단과 fallback 플래그를 백테스트 행에 기록.</li>
            <li>보정 스크립트 top-level 실행을 함수화하고 calibration/test 기간을 명시적 인자로 분리.</li>
            <li>pytest로 날짜 지연, 수익률 방향, 비용, 비중 드리프트, DP 최적경로의 작은 결정적 사례를 단위 테스트.</li>
          </ol>
        </section>

        <section id="evidence">
          <h2>근거 감사 — 문헌, 내부 실증, 설계 판단을 분리해서 읽기</h2>
          <p>설계 근거는 모두 같은 강도가 아닙니다. 아래 표는 각 주장에 대해 무엇이 실제로 확인되었고 무엇이 아직 추론인지 명시합니다. “논문에 비슷한 아이디어가 있다”는 사실은 현재 구현의 특정 파라미터나 미래 성과를 검증하지 않습니다.</p>
          <h3>근거 강도 레지스터</h3>
          <div class="table-wrap"><table><thead><tr><th>설계 주장</th><th>외부 방법론 근거</th><th>현재 내부 근거</th><th>판정</th><th>추가로 필요한 검증</th></tr></thead><tbody>
            <tr><th>순차 상태에는 jump penalty가 적합하다</th><td>Jump-model 교대최적화·이산 상태열 문헌</td><td>상태 전환 수와 특징가중치가 월별 저장됨</td><td><span class="evidence-tag lit">방법론 지지</span></td><td>HMM·change-point·단순 threshold와 동일 OOS 비교</td></tr>
            <tr><th>희소 특징선택이 잡음을 줄인다</th><td>Sparse jump model 문헌</td><td>top-4 가중치가 생성되지만 비희소 모델 ablation 없음</td><td><span class="evidence-tag limit">내부 근거 부족</span></td><td><code>keep_features=1..6</code> nested OOS 민감도</td></tr>
            <tr><th>SJM을 합성점수에 추가하면 예측이 개선된다</th><td>직접 해당 없음</td><td>성장 +0.10%p, 물가 +0.36%p, 4분면 +0.43%p 대 naive</td><td><span class="evidence-tag emp">작은 양의 증분</span></td><td>통계적 유의성, 국가별/빈티지 외부검증</td></tr>
            <tr><th>Soft allocation이 hard switching보다 경로위험이 낮다</th><td>확률·불확실성의 연속 반영이라는 설계 논리</td><td>제안 전략은 hard 대비 변동성 {pct(hard_vol_reduction,1)}, MDD {pct(hard_mdd_reduction,1)} 감소</td><td><span class="evidence-tag emp">현재 표본 강한 지지</span></td><td>동일 risk target의 hard 전략 비교로 공정성 강화</td></tr>
            <tr><th>위험제어가 soft anchor 단독보다 유리하다</th><td>Volatility management·drawdown control 문헌</td><td>soft 대비 Sharpe {pct(soft_sharpe_gain,1)} 증가, 변동성 {pct(soft_vol_reduction,1)}, MDD {pct(soft_mdd_reduction,1)} 감소; CAGR은 1.45%p 감소</td><td><span class="evidence-tag emp">위험조정 성과 지지</span></td><td>EWMA/CDaR/turnover/guard 각각의 단독 ablation</td></tr>
            <tr><th>동적 국면이 정적 방어비중보다 낫다</th><td>직접 해당 없음</td><td>전체 Sharpe +{pct(static_sharpe_gain,1)}, MDD {pct(static_mdd_reduction,1)} 감소지만 잠금구간 Sharpe는 1.183 대 정적 1.277로 열위</td><td><span class="evidence-tag limit">혼합·불충분</span></td><td>더 긴 OOS, rolling 재보정, 경제국면별 attribution</td></tr>
            <tr><th>거래비용을 넣어도 결과가 견고하다</th><td>비용·turnover regularization 문헌</td><td>비용 0/1/2배 Sharpe 1.156/1.144/1.132</td><td><span class="evidence-tag emp">제한적 지지</span></td><td>bid-ask, market impact, 세금, 환헤지 비용 포함</td></tr>
            <tr><th>잠금구간에서 선택 후 성능이 유지된다</th><td>시간순 validation 원칙</td><td>보정/잠금 Sharpe 1.105/1.183, MDD −8.93%/−8.05%</td><td><span class="evidence-tag emp">한 번의 split 통과</span></td><td>multiple rolling-origin splits와 deflated Sharpe</td></tr>
            <tr><th>8% 목표변동성·16% CDaR·−5% guard가 최적이다</th><td>개념만 문헌 지지</td><td>일부 값만 제한 그리드에서 선택; CDaR/guard threshold 전체 탐색 아님</td><td><span class="evidence-tag judge">휴리스틱</span></td><td>사전 선언된 넓은 민감도 및 효용함수 기반 설정</td></tr>
            <tr><th>미래 실거래에서도 Sharpe 1 이상이다</th><td>없음</td><td>역사적 시뮬레이션뿐</td><td><span class="evidence-tag limit">근거 없음</span></td><td>빈티지 데이터, paper trading, 실체결 기록</td></tr>
          </tbody></table></div>

          <h3>내부 비교를 어떻게 해석해야 하는가</h3>
          <div class="grid-2">
            <div class="card"><h4>긍정적으로 볼 수 있는 부분</h4><ul><li>제안 전략은 hard/soft 단독보다 위험조정 성과와 MDD가 개선되었습니다.</li><li>2018년 이후 잠금구간에서도 목표 Sharpe와 MDD 기준을 통과했습니다.</li><li>비용을 2배로 높여도 Sharpe 감소가 완만했습니다.</li><li>국면평가·비중·비용·시점의 결과가 파일로 남아 감사 가능합니다.</li></ul></div>
            <div class="card"><h4>과대해석하면 안 되는 부분</h4><ul><li>SJM의 naive 대비 분류개선은 1%p 미만으로 매우 작습니다.</li><li>잠금구간에서 정적 방어비중의 Sharpe가 제안 전략보다 높습니다.</li><li>여러 구성요소가 함께 바뀌므로 어떤 하나가 성과를 만들었는지 완전히 분리되지 않았습니다.</li><li>최신 개정 거시데이터와 프록시 사용 때문에 진정한 실시간 OOS는 아닙니다.</li></ul></div>
          </div>

          <h3>참고문헌과 현재 구현에 실제로 연결되는 지점</h3>
          <div class="table-wrap"><table><thead><tr><th>출처</th><th>문헌이 제공하는 근거</th><th>현재 코드와의 연결</th><th>그대로 복제하지 않은 부분</th></tr></thead><tbody>
            <tr><th><a href="https://web.stanford.edu/~boyd/papers/fitting_jump_models.html">Bemporad et al. (2018), Fitting Jump Models</a></th><td>모델 적합과 이산 상태열 적합을 교대하는 일반 jump-model 프레임</td><td><code>SparseJump2</code>의 중심/가중치와 DP 상태열 교대</td><td>일반 loss·regularizer·다상태 모델 전체</td></tr>
            <tr><th><a href="https://orbit.dtu.dk/en/publications/feature-selection-in-jump-models/">Nystrup, Kolm &amp; Lindström (2021), Feature Selection in Jump Models</a></th><td>순차데이터에서 상태와 관련된 특징을 함께 선택할 동기와 알고리즘</td><td>상태분리도 기반 top-4 가중치</td><td>논문의 coordinate descent와 sparsity formulation</td></tr>
            <tr><th><a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513">Moreira &amp; Muir (2017), Volatility-Managed Portfolios</a></th><td>변동성이 높을 때 위험을 줄이는 timing의 다수 자산요인 실증</td><td>동적 target vol과 EWMA 위험제약의 방향성</td><td>논문의 직접 volatility scaling portfolio</td></tr>
            <tr><th><a href="https://web.stanford.edu/~boyd/papers/multiperiod_portfolio_drawdown.html">Nystrup et al. (2019), Multi-Period Portfolio Selection with Drawdown Control</a></th><td>실현 drawdown 기반 위험회피도 조절, MPC, 거래비용 regularization</td><td>drawdown guard와 turnover/tracking penalty의 아이디어</td><td>다기간 예측과 MPC 최적정책</td></tr>
            <tr><th><a href="https://doi.org/10.1142/S0219024905002767">Chekhlov, Uryasev &amp; Zabarankin (2005), Drawdown Measure in Portfolio Optimization</a></th><td>drawdown 꼬리 위험을 포트폴리오 목적·제약에 쓰는 이론적 기반</td><td>과거 경로의 90% CDaR penalty와 16% constraint</td><td>원 논문의 정교한 CDaR 최적화 정식</td></tr>
            <tr><th><a href="https://web.stanford.edu/~boyd/papers/cvx_portfolio.html">Boyd et al. (2017), Multi-Period Trading via Convex Optimization</a></th><td>기대수익·위험·거래비용·보유비용의 명시적 trade-off</td><td>수익/위험/turnover/tracking 혼합 목적함수</td><td>다기간 planning과 convex formulation 전체</td></tr>
            <tr><th><a href="https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html">SciPy: minimize(method='SLSQP')</a></th><td>등식·부등식·bounds가 있는 scalar 최적화 API의 공식 계약</td><td>완전투자, vol/CDaR, 자산 bounds를 한 번에 전달</td><td>전역최적성·모델 타당성 보장</td></tr>
          </tbody></table></div>

          <h3>현재 문서에서 의도적으로 하지 않는 주장</h3>
          <ul class="checklist">
            <li><code>SparseJump2</code>가 논문 구현과 수학적으로 동일하다고 주장하지 않습니다.</li>
            <li>비대칭 EWMA를 stochastic volatility 또는 Bayesian SV라고 부르지 않습니다.</li>
            <li>drawdown guard를 완전한 MPC 구현이라고 부르지 않습니다.</li>
            <li>90/10, 75/25, 8%, 16%, −5%가 보편적 최적 파라미터라고 주장하지 않습니다.</li>
            <li>historical Sharpe와 MDD가 미래에 반복된다고 주장하지 않습니다.</li>
            <li>현재 개정치 기반 백테스트를 point-in-time vintage backtest라고 주장하지 않습니다.</li>
          </ul>
        </section>

        <section id="limitations">
          <h2>16. 한계와 해석 주의사항</h2>
          <div class="callout danger"><strong>투자 의사결정 주의.</strong> 이 결과는 연구용 역사 시뮬레이션이며 수익을 보장하지 않습니다. 세금, 시장충격, 실제 스프레드, 주문 지연, ETF 추적오차, 운용규모 제약은 반영하지 않았습니다.</div>
          <ul>
            <li><strong>Revision bias:</strong> GDP·물가 파일은 현재 최신 개정치이며 당시 발표 빈티지 자체가 아닙니다.</li>
            <li><strong>프록시 위험:</strong> 2009-03 이전 KODEX200은 KOSPI200 가격지수 프록시이고 배당 반영이 완전하지 않습니다.</li>
            <li><strong>가격 동기화:</strong> 미국 ETF 첫 거래가격과 USD/KRW 종가의 시각이 일치하지 않습니다.</li>
            <li><strong>평균수익 추정:</strong> expanding/최근 평균 혼합은 과거 표본 의존적이며 안정적 기대수익 모델을 보장하지 않습니다.</li>
            <li><strong>최적화 fallback:</strong> SLSQP 실패 시 prior를 사용하므로 그 달에는 변동성/CDaR 제약이 엄밀히 충족되지 않을 수 있습니다.</li>
            <li><strong>CDaR 근사:</strong> 과거 자산수익률에 현재 고정비중을 적용한 경로 CDaR이며, 매월 동적 리밸런싱의 미래 경로 최적화는 아닙니다.</li>
            <li><strong>국면 독립성:</strong> 네 국면 확률은 성장과 물가 확률의 곱이므로 두 축의 조건부 의존구조를 별도 모델링하지 않습니다.</li>
            <li><strong>보정 편향:</strong> 36개로 제한했지만 보정구간에 대한 모델 선택 편향은 여전히 존재합니다.</li>
            <li><strong>기준금리:</strong> Sharpe/Sortino에서 무위험수익률은 0%입니다.</li>
            <li><strong>단일 시장 표본:</strong> 한국 중심 한 기간의 역사이며 다른 국가·제도·유동성 환경으로 일반화되지 않습니다.</li>
          </ul>
        </section>

        <section id="reference">
          <h2>17. 함수·클래스 레퍼런스</h2>
          <div class="table-wrap"><table><thead><tr><th>심볼</th><th>위치</th><th>입력 → 출력</th><th>책임/계약</th></tr></thead><tbody>
            <tr><th><code>get_path</code></th><td><a class="src" href="#src-regime-L28">L28</a></td><td>directory, filename → Path</td><td>Unicode NFC 파일명 해결; 없으면 예외</td></tr>
            <tr><th><code>rolling_zscore</code></th><td><a class="src" href="#src-regime-L36">L36</a></td><td>Series, window, clip → Series</td><td>과거 rolling 표준화, ±clip</td></tr>
            <tr><th><code>load_macro_data</code></th><td><a class="src" href="#src-regime-L42">L42</a></td><td>없음 → (features, core)</td><td>거시 파일 읽기, 시차, 12개 특징 생성</td></tr>
            <tr><th><code>download_market_cache</code></th><td><a class="src" href="#src-regime-L98">L98</a></td><td>refresh → daily DataFrame</td><td>캐시 우선 또는 Yahoo 다운로드</td></tr>
            <tr><th><code>load_monthly_asset_returns</code></th><td><a class="src" href="#src-regime-L120">L120</a></td><td>refresh → (returns, levels)</td><td>프록시 접합, 원화환산, first-open forward return</td></tr>
            <tr><th><code>_softmax</code></th><td><a class="src" href="#src-regime-L182">L182</a></td><td>ndarray → ndarray</td><td>최댓값을 빼는 수치안정 softmax</td></tr>
            <tr><th><code>SparseJump2._dp</code></th><td><a class="src" href="#src-regime-L201">L201</a></td><td>distance[T,2], jump → states, costs</td><td>전환 페널티 최소비용 상태경로</td></tr>
            <tr><th><code>SparseJump2.fit_predict_high</code></th><td><a class="src" href="#src-regime-L217">L217</a></td><td>feature frame → p_high, detail</td><td>robust scaling, 희소선택, 반복 적합, 확률</td></tr>
            <tr><th><code>compute_regime_signals</code></th><td><a class="src" href="#src-regime-L265">L265</a></td><td>features, returns → signals</td><td>월별 walk-forward 성장·물가·4국면 확률</td></tr>
            <tr><th><code>soft_anchor</code></th><td><a class="src" href="#src-regime-L331">L331</a></td><td>signal → ndarray[4]</td><td>국면확률 × 기준비중 행렬</td></tr>
            <tr><th><code>ewma_cov</code></th><td><a class="src" href="#src-regime-L336">L336</a></td><td>history, half_life, leverage → 4×4 covariance</td><td>주식 하락 비대칭 EWMA</td></tr>
            <tr><th><code>cdar</code></th><td><a class="src" href="#src-regime-L350">L350</a></td><td>return path, alpha → float</td><td>최악 drawdown tail 평균</td></tr>
            <tr><th><code>StrategyConfig</code></th><td><a class="src" href="#src-regime-L357">L357</a></td><td>14개 설정 필드</td><td>포트폴리오 위험·목적함수 설정</td></tr>
            <tr><th><code>controlled_weights</code></th><td><a class="src" href="#src-regime-L375">L375</a></td><td>signal, history, pretrade, dd, cfg → weights</td><td>anchor부터 SLSQP와 drawdown guard까지</td></tr>
            <tr><th><code>hard_regime_weights</code></th><td><a class="src" href="#src-regime-L434">L434</a></td><td>signal → weights</td><td>최빈국면 단일 hard mapping</td></tr>
            <tr><th><code>run_backtest</code></th><td><a class="src" href="#src-regime-L444">L444</a></td><td>returns, signals, cfg, mode, 기간, 비용배수 → DataFrame</td><td>월별 상태전이·비용·비중 드리프트</td></tr>
            <tr><th><code>performance_summary</code></th><td><a class="src" href="#src-regime-L513">L513</a></td><td>returns → 9개 지표 Series</td><td>월수익률 성과 연환산</td></tr>
            <tr><th><code>evaluate_regimes</code></th><td><a class="src" href="#src-regime-L540">L540</a></td><td>signals, core → joined, metrics</td><td>다음 3개월 상태 정답과 분류 평가</td></tr>
            <tr><th><code>main</code></th><td><a class="src" href="#src-regime-L570">L570</a></td><td>refresh → None</td><td>전체 파이프라인 실행·결과 저장</td></tr>
            <tr><th><code>score_metrics</code></th><td><a class="src" href="#src-calibrate-L18">calibrate:L18</a></td><td>metrics, turnover → score</td><td>보정구간 선택 목적함수</td></tr>
          </tbody></table></div>
        </section>

        <section id="glossary">
          <h2>18. 용어집</h2>
          <div class="table-wrap"><table><thead><tr><th>용어</th><th>이 프로젝트에서의 의미</th></tr></thead><tbody>
            <tr><th>signal month</th><td>거시정보를 관측하는 월. 항상 투자 target month보다 한 Period 앞섭니다.</td></tr>
            <tr><th>target month</th><td>해당 월 첫 거래일에서 다음 달 첫 거래일까지 수익률이 귀속되는 투자월.</td></tr>
            <tr><th>soft regime</th><td>최빈 국면 하나가 아니라 네 국면 확률을 모두 사용한 연속적 자산배분.</td></tr>
            <tr><th>hard regime</th><td>최대확률 국면 하나만 선택해 미리 정한 극단 비중을 적용하는 비교전략.</td></tr>
            <tr><th>anchor</th><td>경제 국면이 제시하는 기본 목표비중. 최종비중이 아니라 최적화 prior의 출발점.</td></tr>
            <tr><th>pretrade weight</th><td>직전 목표비중이 한 달 자산수익률로 드리프트한 뒤, 새 거래 직전에 형성된 비중.</td></tr>
            <tr><th>EWMA</th><td>최근 충격에 지수적으로 더 큰 가중치를 주는 공분산 추정.</td></tr>
            <tr><th>CDaR</th><td>drawdown 분포의 최악 꼬리 평균. 본 구현에서는 과거 경로 하위 10% 평균.</td></tr>
            <tr><th>Balanced accuracy</th><td>Low recall과 High recall의 산술평균.</td></tr>
            <tr><th>locked test</th><td>2017-12까지 보정한 뒤 설정 선택에 사용하지 않은 2018-01 이후 평가구간.</td></tr>
          </tbody></table></div>
        </section>

        <section id="source">
          <h2>부록 A. 전체 소스 스냅샷</h2>
          <p>설명과 구현을 한 파일에서 대조할 수 있도록 문서 생성 시점의 전체 Python 소스를 줄 번호와 함께 포함했습니다. 각 줄 번호는 직접 링크가 가능하며, 위 본문의 소스 링크도 이 부록으로 이동합니다.</p>
          <details><summary><code>regime_research.py</code> — 핵심 연구 구현</summary><pre class="source-code"><code>{source_block(ROOT / 'strategies/core/regime_research.py', 'regime')}</code></pre></details>
          <details><summary><code>calibrate_configs.py</code> — 제한 그리드 보정</summary><pre class="source-code"><code>{source_block(ROOT / 'strategies/stage01_baseline/calibrate_configs.py', 'calibrate')}</code></pre></details>
          <details><summary><code>build_validated_notebook.py</code> — 재현 노트북 생성</summary><pre class="source-code"><code>{source_block(ROOT / 'tools/builders/build_validated_notebook.py', 'notebook')}</code></pre></details>
        </section>

        <section id="integrity">
          <h2>부록 B. 무결성·재현 정보</h2>
          <p>아래 SHA-256은 이 HTML이 설명한 세 소스 파일의 생성 시점 내용에 대한 지문입니다. 코드가 바뀌면 설명서를 다시 생성하여 해시와 문서 내용을 함께 갱신해야 합니다.</p>
          <div class="table-wrap"><table><thead><tr><th>파일</th><th>SHA-256</th></tr></thead><tbody>
            {''.join(f'<tr><th><code>{esc(path.name)}</code></th><td class="hash">{sha256(path)}</td></tr>' for path in source_files)}
          </tbody></table></div>
          <p><strong>문서 생성기:</strong> <code>build_implementation_guide.py</code><br><strong>생성 시각:</strong> {esc(generated_at)}<br><strong>기준 작업 폴더:</strong> <code>{esc(ROOT)}</code></p>
        </section>
      </main>
    </div>

    <footer>Economic Regime Allocation · implementation guide · generated from the current local worktree</footer>
    <script>
      const progress = document.getElementById('progress');
      const updateProgress = () => {{
        const max = document.documentElement.scrollHeight - innerHeight;
        progress.style.width = (max > 0 ? scrollY / max * 100 : 0) + '%';
      }};
      addEventListener('scroll', updateProgress, {{passive:true}}); updateProgress();
      const filter = document.getElementById('tocFilter');
      filter.addEventListener('input', () => {{
        const q = filter.value.trim().toLowerCase();
        document.querySelectorAll('#toc li').forEach(li => {{ li.hidden = !li.textContent.toLowerCase().includes(q); }});
      }});
      const revealSourceTarget = () => {{
        if (!location.hash.startsWith('#src-')) return;
        const target = document.getElementById(location.hash.slice(1));
        const container = target && target.closest('details');
        if (container) container.open = true;
      }};
      document.querySelectorAll('a[href^="#src-"]').forEach(link => link.addEventListener('click', () => {{
        const target = document.getElementById(link.hash.slice(1));
        const container = target && target.closest('details');
        if (container) container.open = true;
      }}));
      addEventListener('hashchange', revealSourceTarget);
      revealSourceTarget();
    </script>
    </body>
    </html>
    """

    OUTPUT.write_text(document, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
