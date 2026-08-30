from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BODY_PATH = (
    ROOT
    / "artifacts"
    / "archive"
    / "humanization_sessions"
    / "2026-08-28-001"
    / "final.md"
)
OUTPUT_PATH = ROOT / "artifacts/reports/vix6_case1_strategy_report.html"


def build() -> Path:
    body = BODY_PATH.read_text(encoding="utf-8")
    # The humanized body predates the repository refactor.  Keep that archived
    # prose immutable and translate its local links when the report is built.
    link_rewrites = {
        'href="vix6_case1_strategy.py"': 'href="../../strategies/stage08_options/vix6_case1_strategy.py"',
        'href="vix6_case1_model_comparison.py"': 'href="../../strategies/stage08_options/vix6_case1_model_comparison.py"',
        'href="test_vix6_case1_strategy.py"': 'href="../../tests/test_vix6_case1_strategy.py"',
        'href="results/': 'href="../../results/',
    }
    for old, new in link_rewrites.items():
        body = body.replace(old, new)
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="description" content="KOSPI200 옵션 VIX6 분해, VKOSPI 오버레이, 로지스틱 입력변수 및 최종 전략 선택 보고서">
  <title>VIX6 Case 1 연구 보고서 | 최종 전략 선택과 알고리즘</title>
  <style>
    :root {{
      --bg: #f3f5f2;
      --paper: #fffefa;
      --ink: #17231f;
      --muted: #617069;
      --line: #dce3de;
      --green: #0a6a50;
      --green-2: #128063;
      --green-soft: #e0f3ea;
      --blue: #214f69;
      --blue-soft: #e7f0f5;
      --gold: #9b6714;
      --gold-soft: #fff0cf;
      --red: #9d3b35;
      --red-soft: #fae9e6;
      --shadow: 0 16px 44px rgba(18, 44, 35, .08);
      --radius: 18px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 95% 3%, rgba(18,128,99,.11), transparent 28rem), var(--bg);
      color: var(--ink);
      font-family: Pretendard, "Noto Sans KR", "Malgun Gothic", system-ui, sans-serif;
      line-height: 1.72;
      word-break: keep-all;
    }}
    a {{ color: var(--green); text-underline-offset: 3px; }}
    code {{
      font-family: var(--mono); font-size: .91em; word-break: break-word;
      padding: .1rem .34rem; border: 1px solid var(--line); border-radius: 6px;
      background: rgba(33,79,105,.055);
    }}
    .hero {{
      padding: 70px 28px 58px;
      color: #f6fff9;
      background: linear-gradient(118deg, #075740, #183f56);
      position: relative; overflow: hidden;
    }}
    .hero::after {{
      content: ""; position: absolute; width: 440px; height: 440px;
      right: -120px; top: -240px; border-radius: 50%;
      border: 58px solid rgba(255,255,255,.055);
    }}
    .hero-inner {{ max-width: 1120px; margin: 0 auto; position: relative; z-index: 1; }}
    .kicker {{ letter-spacing: .16em; font-size: .78rem; opacity: .78; font-weight: 800; }}
    h1 {{ margin: .65rem 0 1rem; max-width: 850px; font-size: clamp(2.15rem, 5vw, 4.5rem); line-height: 1.08; letter-spacing: -.045em; }}
    .hero p {{ max-width: 820px; margin: 0; color: rgba(255,255,255,.82); font-size: 1.08rem; }}
    .hero-badges {{ display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.6rem; }}
    .hero-badges span {{ padding: .38rem .7rem; border: 1px solid rgba(255,255,255,.22); border-radius: 999px; background: rgba(255,255,255,.08); font-size: .85rem; }}
    .layout {{ max-width: 1220px; margin: 0 auto; padding: 34px 24px 72px; display: grid; grid-template-columns: 235px minmax(0, 1fr); gap: 28px; align-items: start; }}
    nav {{ position: sticky; top: 20px; padding: 18px; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,254,250,.92); box-shadow: var(--shadow); }}
    nav strong {{ display: block; margin-bottom: .8rem; color: var(--muted); font-size: .76rem; letter-spacing: .12em; }}
    nav a {{ display: block; padding: .46rem .55rem; color: var(--ink); text-decoration: none; border-radius: 8px; font-size: .9rem; }}
    nav a:hover {{ color: var(--green); background: var(--green-soft); }}
    main {{ min-width: 0; }}
    .section {{ margin-bottom: 24px; padding: clamp(24px, 4.5vw, 50px); background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); scroll-margin-top: 24px; }}
    .lead-section {{ border-top: 5px solid var(--green); }}
    .eyebrow {{ margin: 0 0 .5rem; color: var(--green); font-weight: 800; font-size: .76rem; letter-spacing: .14em; text-transform: uppercase; }}
    h2 {{ margin: 0 0 1.15rem; font-size: clamp(1.55rem, 3vw, 2.25rem); line-height: 1.26; letter-spacing: -.035em; }}
    h3 {{ margin: 1.6rem 0 .65rem; font-size: 1.08rem; }}
    p {{ margin: .8rem 0; }}
    .lead {{ font-size: 1.12rem; color: #263a33; }}
    .callout {{ display: grid; gap: .1rem; margin: 1.45rem 0; padding: 1.15rem 1.25rem; border-radius: 14px; }}
    .callout.keep {{ background: var(--green-soft); border-left: 5px solid var(--green); }}
    .callout strong {{ font-size: .78rem; color: var(--green); letter-spacing: .1em; }}
    .callout span {{ font-size: 1.27rem; font-weight: 850; }}
    .callout small {{ color: var(--muted); }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 13px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 660px; }}
    th, td {{ padding: .82rem .9rem; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f1; color: var(--muted); font-size: .8rem; }}
    tr:last-child td {{ border-bottom: 0; }}
    tr.winner td {{ background: rgba(224,243,234,.62); font-weight: 750; }}
    .note {{ padding: 1rem 1.1rem; border-radius: 12px; background: var(--gold-soft); color: #5b471e; font-size: .94rem; }}
    .flow {{ display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr; gap: 9px; align-items: stretch; margin: 1.4rem 0; }}
    .flow > div {{ display: grid; gap: .22rem; padding: .95rem; border: 1px solid var(--line); border-radius: 12px; background: #f8faf8; }}
    .flow b {{ width: 1.65rem; height: 1.65rem; display: grid; place-items: center; border-radius: 50%; color: white; background: var(--green); }}
    .flow span {{ font-weight: 800; }} .flow small {{ color: var(--muted); }} .flow i {{ align-self: center; color: var(--green); font-style: normal; font-size: 1.2rem; }}
    .factor-grid, .input-columns, .search-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 1.3rem 0; }}
    .factor-grid article, .input-columns > div, .search-grid article {{ padding: 1rem 1.05rem; border: 1px solid var(--line); border-radius: 13px; background: #fafbf9; }}
    .factor-grid h3 {{ margin-top: 0; color: var(--blue); }}
    .factor-grid p, .search-grid p {{ margin-bottom: 0; font-size: .93rem; color: #384942; }}
    .formula-list {{ display: grid; gap: 8px; margin: 1.2rem 0; }}
    .formula-list p {{ display: grid; grid-template-columns: minmax(300px, .9fr) 1fr; gap: 14px; align-items: center; margin: 0; padding: .85rem 1rem; border: 1px solid var(--line); border-radius: 11px; }}
    .formula-list code {{ background: var(--blue-soft); color: var(--blue); font-weight: 750; }}
    .formula-list span {{ color: var(--muted); font-size: .93rem; }}
    ol, ul {{ padding-left: 1.25rem; }} li {{ margin: .35rem 0; }}
    .input-columns h3 {{ margin-top: 0; }} .input-columns ul {{ margin-bottom: 0; }}
    .search-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .search-grid strong {{ color: var(--green); font-size: 1.02rem; }}
    .audit-list {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; padding: 0; list-style: none; }}
    .audit-list li {{ margin: 0; padding: .85rem 1rem; border-radius: 11px; background: var(--blue-soft); border: 1px solid #cfdee6; }}
    .file-list {{ display: grid; gap: 9px; }}
    .file-list a {{ display: grid; gap: .12rem; padding: .9rem 1rem; border: 1px solid var(--line); border-radius: 11px; text-decoration: none; background: #fafbf9; }}
    .file-list a:hover {{ border-color: var(--green-2); background: var(--green-soft); }}
    .file-list b {{ color: var(--ink); font-family: var(--mono); font-size: .9rem; }} .file-list span {{ color: var(--muted); font-size: .9rem; }}
    .footnote {{ border-left: 5px solid var(--gold); }}
    footer {{ padding: 0 24px 54px; text-align: center; color: var(--muted); font-size: .84rem; }}
    footer a {{ margin-left: .4rem; }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }} nav {{ position: static; display: none; }}
      .flow {{ grid-template-columns: 1fr; }} .flow i {{ transform: rotate(90deg); justify-self: center; }}
      .search-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 660px) {{
      .hero {{ padding: 50px 20px 42px; }} .layout {{ padding: 20px 12px 50px; }}
      .section {{ padding: 24px 19px; border-radius: 14px; }}
      .factor-grid, .input-columns, .audit-list {{ grid-template-columns: 1fr; }}
      .formula-list p {{ grid-template-columns: 1fr; gap: 5px; }}
    }}
    @media print {{
      body {{ background: white; }} nav {{ display: none; }} .layout {{ display: block; max-width: none; padding: 0; }}
      .hero {{ padding: 35px; }} .section {{ box-shadow: none; break-inside: avoid; }} a {{ color: inherit; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <div class="kicker">REGIME DECISION TEST · VIX6 CASE 1</div>
      <h1>VIX6를 넣어도<br>최종 전략은 바꾸지 않았다</h1>
      <p>KOSPI200 옵션 860,380행을 30일 변동성 곡면으로 재구성하고, 여섯 성분·가공 입력·오버레이를 같은 비용 조건에서 비교한 기록이다.</p>
      <div class="hero-badges"><span>4개 투자자산</span><span>옵션은 신호 전용</span><span>15 tests passed</span><span>Post-lock exploratory</span></div>
    </div>
  </header>
  <div class="layout">
    <nav aria-label="보고서 목차">
      <strong>CONTENTS</strong>
      <a href="#decision">최종 결정</a><a href="#scoreboard">성과표</a>
      <a href="#what-overlay">VKOSPI 오버레이</a><a href="#vix6">VIX6 여섯 성분</a>
      <a href="#meta">메타 신호·상태</a><a href="#inputs">입력변수 전체</a>
      <a href="#classifier">분류 알고리즘</a><a href="#search">후보·파라미터 탐색</a>
      <a href="#why">효과 해석</a><a href="#audit">시차·재현성</a><a href="#files">관련 파일</a>
    </nav>
    <main>{body}</main>
  </div>
  <footer>
    2026-08-28 · RegimeDecisionTest · humanize-korean strict pass
    <a href="https://cdn.cboe.com/resources/vix/VIX-Decomposition-2025-08-01.pdf">Cboe VIX Decomposition 공식 자료</a>
  </footer>
</body>
</html>
"""
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build())
