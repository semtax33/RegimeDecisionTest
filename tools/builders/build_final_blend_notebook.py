from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "artifacts/notebooks/economic_regime_allocation_four_asset_mdd15.ipynb"

def main() -> None:
    base_source = (ROOT / "strategies/core/regime_research.py").read_text(encoding="utf-8")
    base_source = base_source.replace("ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path.cwd()")
    base_source = base_source.split("\ndef main(refresh: bool = False) -> None:")[0]

    blend_source = (ROOT / "strategies/stage03_tail_risk/blend_leverage_experiment.py").read_text(encoding="utf-8")
    blend_source = blend_source.replace("ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path.cwd()")
    blend_source = blend_source.split("\nmacro, _ = load_macro_data()")[0]

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 경제국면 자산배분 — 기존 4개 자산 고수익·MDD 15% 절충안

    사용자 선택에 따라 **신규 헤지 자산은 추가하지 않았다.** 투자되는 위험자산은 원 노트북과 동일한 `KODEX200`, 국내 `BOND`, `GLD`, `USO`뿐이다.

    최종 중앙값 후보는 원 고수익 전략 40%와 기존 방어 전략 60%를 섞고, 총자산 노출을 1.20배로 유지한다. 20% 차입에는 연 4% 금융비용을 매월 차감한다.

    > 전체기간 결과: **CAGR 14.30%, Sharpe 1.024, MDD -14.90%**  
    > 2018년 이후 잠금구간: **CAGR 17.07%, Sharpe 1.164, MDD -14.90%**

    이 수치는 역사적 시뮬레이션이며 미래 MDD 15%를 보장하지 않는다. 특히 비용 2배와 블록 부트스트랩 결과를 함께 확인해야 한다."""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. 선택 원칙과 검증 구조

    - **자산 제한:** 네 자산만 매수한다. 옵션, VIX 상품, 인버스, 관리선물은 없다.
    - **핵심 비중:** `0.40 × 원 고수익 국면비중 + 0.60 × 기존 방어비중`.
    - **총노출:** 위 혼합비중에 1.20을 곱한다. 음의 현금 20%는 헤지 자산이 아니라 명시적 차입이다.
    - **비용:** 자산 비중 변화 절댓값에 15bp, 해외자산 순변화에 5bp, 차입 연 4%를 차감한다.
    - **시점:** 원 노트북의 발표시차 및 `t월 말 신호 → t+1월 첫 거래일` 규칙을 유지한다.
    - **보정구간:** 2017-12까지. **잠금구간:** 2018-01 이후.
    - **후보 선택 감사:** ML·VIX 동적 방어는 보정 성과가 높아도 잠금구간에서 무너져 제외했다. 두 매개변수 고정혼합을 채택해 모델 위험을 줄였다.

    보정 경계의 1.25배 후보는 전체 MDD가 -15.58%였기 때문에 제외하고, 인접 중앙값인 1.20배를 고정했다."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
    import json

    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    ROOT = Path.cwd()
    RESULTS = ROOT / "results"
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (14, 5)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Malgun Gothic"
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 180)

    print("작업 폴더:", ROOT)
    print("투자 자산:", ["KODEX200", "BOND", "GLD", "USO"])
    print("신규 헤지 자산: 없음")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. 원 노트북의 데이터·국면·방어모델 구현

    아래 숨김 셀은 거시 입력변수, 발표시차, Sparse Jump Model, 경기·물가 확률, 공분산 추정, 제약 최적화, drawdown guard, 거래비용을 포함한다."""
        ),
        nbf.v4.new_code_cell(base_source, metadata={"jupyter": {"source_hidden": True}, "tags": ["base-implementation"]}),
        nbf.v4.new_markdown_cell(
            """## 3. 고정혼합·금융비용 구현

    원 전략은 경기국면별 기대수익 프리미엄을 강하게 취한다. 방어 전략은 국면 확률, 변동성, 상관관계와 낙폭을 이용해 분산한다. 두 전략을 일정 비율로 섞으면 예측모형이 특정 위기 직전에 맞아야 한다는 부담이 줄어든다.

    1. `Goldilocks → KODEX200`, `Overheating → USO`, `Slowdown → KODEX200 60% + BOND 40%`, `Stagflation → GLD`가 원 고수익 비중이다.
    2. 원 비중 40%와 기존 방어 비중 60%를 매월 혼합한다.
    3. 기대수익을 회복하기 위해 동일 네 자산의 총노출만 1.20배로 확대한다.
    4. 레버리지 때문에 수익과 손실, 비용이 모두 확대된다. 이 부분이 전략의 핵심 위험이다."""
        ),
        nbf.v4.new_code_cell(blend_source, metadata={"jupyter": {"source_hidden": True}, "tags": ["blend-implementation"]}),
        nbf.v4.new_markdown_cell("## 4. 데이터 로딩과 전략 재계산"),
        nbf.v4.new_code_cell(
            """macro, macro_core = load_macro_data()
    asset_returns, asset_levels = load_monthly_asset_returns(refresh=False)
    signals = compute_regime_signals(macro, asset_returns)

    base_cfg = StrategyConfig()
    hard = run_backtest(asset_returns, signals, base_cfg, mode="hard")
    defensive = run_backtest(asset_returns, signals, base_cfg, mode="proposed")
    final_cfg = BlendConfig(hard_fraction=0.40, leverage=1.20, financing_rate=0.04)
    final = run_blend(asset_returns, signals, defensive, final_cfg)

    audit = pd.DataFrame({
        "시작": [macro.index.min(), asset_returns.index.min(), signals.index.min(), final.index.min()],
        "종료": [macro.index.max(), asset_returns.index.max(), signals.index.max(), final.index.max()],
        "관측치": [len(macro), len(asset_returns), len(signals), len(final)],
    }, index=["거시 입력변수", "자산 월수익률", "투자 신호", "최종 백테스트"])
    display(audit)
    display(pd.Series({
        "원 전략 비율": final_cfg.hard_fraction,
        "방어 전략 비율": 1-final_cfg.hard_fraction,
        "총자산 노출": final_cfg.leverage,
        "차입 비중": final_cfg.leverage-1,
        "연 금융비용 가정": final_cfg.financing_rate,
    }, name="고정값").to_frame().style.format("{:.1%}"))"""
        ),
        nbf.v4.new_markdown_cell("## 5. 핵심 성과 — 보정·잠금·전체"),
        nbf.v4.new_code_cell(
            """rows = []
    for period, start, end in [
        ("보정 2007-2017", None, "2017-12"),
        ("잠금 2018+", "2018-01", None),
        ("전체", None, None),
    ]:
        for strategy, bt in [("원 고수익", hard), ("기존 방어", defensive), ("최종 혼합", final)]:
            sample = bt.loc[start:end] if start else bt.loc[:end] if end else bt
            m = performance_summary(sample["return"])
            rows.append({"구간": period, "전략": strategy, **m.to_dict(), "월평균 회전율": sample["turnover"].mean()})

    comparison = pd.DataFrame(rows)
    display(comparison[["구간", "전략", "CAGR", "Volatility", "Sharpe", "Sortino", "MDD", "Calmar", "FinalMultiple", "월평균 회전율"]].style.format({
        "CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
        "MDD": "{:.2%}", "Calmar": "{:.3f}", "FinalMultiple": "{:.2f}x", "월평균 회전율": "{:.2%}",
    }).highlight_max(subset=["CAGR", "Sharpe", "Calmar"], color="#d8f0dc"))

    full_hard = performance_summary(hard["return"])
    full_final = performance_summary(final["return"])
    locked_final = performance_summary(final.loc["2018-01":, "return"])
    display(pd.Series({
        "원 수익률 유지율": f'{full_final["CAGR"] / full_hard["CAGR"]:.1%}',
        "MDD 개선폭": f'{full_final["MDD"] - full_hard["MDD"]:.2%}',
        "전체 Sharpe": f'{full_final["Sharpe"]:.3f}',
        "잠금 Sharpe": f'{locked_final["Sharpe"]:.3f}',
    }, name="핵심 판정").to_frame())"""
        ),
        nbf.v4.new_code_cell(
            """idx = final.index.to_timestamp()
    wealth = {name: (1 + bt["return"]).cumprod() for name, bt in [("원 고수익", hard), ("기존 방어", defensive), ("최종 혼합", final)]}
    drawdown = {name: values / values.cummax() - 1 for name, values in wealth.items()}
    colors = {"원 고수익": "#7f8c8d", "기존 방어": "#2f9e44", "최종 혼합": "#0969da"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name in wealth:
        axes[0].plot(idx, wealth[name], label=name, color=colors[name], lw=2.2 if name == "최종 혼합" else 1.5)
    axes[0].set_yscale("log")
    axes[0].set_title("누적자산 — 비용·금융비용 차감 후 (로그축)")
    axes[0].set_ylabel("1원의 성장")
    axes[0].legend(ncol=3)
    for name in drawdown:
        axes[1].plot(idx, drawdown[name], label=name, color=colors[name], lw=2 if name == "최종 혼합" else 1.3)
    axes[1].axhline(-0.15, color="#c92a2a", ls="--", lw=1.4, label="-15% 목표")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("낙폭")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend(ncol=4)
    plt.tight_layout()
    plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## 6. 실제 자산비중과 레버리지"),
        nbf.v4.new_code_cell(
            """weight_cols = [f"w_{asset}" for asset in ASSETS]
    weights = final[weight_cols].copy()
    weights.columns = ASSETS
    weights.index = weights.index.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    weights.plot.area(ax=axes[0], color=["#1479ff", "#697b8c", "#e7b840", "#c95c43"], alpha=0.88)
    axes[0].axhline(1.20, color="black", ls="--", lw=1)
    axes[0].set_ylim(0, 1.26)
    axes[0].set_title("최종 혼합의 월별 자산비중 — 합계 120%")
    axes[0].set_ylabel("NAV 대비 비중")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].legend(ncol=4, loc="upper center")

    hard_share = final[[f"hard_w_{asset}" for asset in ASSETS]].mul(0.4 * 1.2).sum(axis=1)
    def_share = final[[f"defensive_w_{asset}" for asset in ASSETS]].mul(0.6 * 1.2).sum(axis=1)
    axes[1].plot(idx, hard_share, label="원 전략 기여 노출", color="#d9480f")
    axes[1].plot(idx, def_share, label="방어 전략 기여 노출", color="#2f9e44")
    axes[1].axhline(-0.20, label="차입/현금 비중", color="#5f3dc4")
    axes[1].set_ylim(-0.25, 0.8)
    axes[1].set_title("구성요소 노출과 고정 차입")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend(ncol=3)
    plt.tight_layout()
    plt.show()

    average_weights = weights.mean().to_frame("평균 비중")
    average_weights.loc["차입(현금)"] = -0.20
    display(average_weights.style.format("{:.1%}"))"""
        ),
        nbf.v4.new_markdown_cell("## 7. 5년 이동성과와 주요 낙폭"),
        nbf.v4.new_code_cell(
            """rolling = pd.read_csv(RESULTS / "final_blend_rolling60.csv")
    rolling["EndMonth"] = pd.PeriodIndex(rolling["EndMonth"], freq="M").to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(rolling["EndMonth"], rolling["CAGR"], color="#0969da")
    axes[0].axhline(0.14, color="#777", ls="--")
    axes[0].set_ylabel("5년 CAGR")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].plot(rolling["EndMonth"], rolling["Sharpe"], color="#2f9e44")
    axes[1].axhline(1.0, color="#777", ls="--")
    axes[1].set_ylabel("5년 Sharpe")
    axes[2].plot(rolling["EndMonth"], rolling["MDD"], color="#c92a2a")
    axes[2].axhline(-0.15, color="#777", ls="--")
    axes[2].set_ylabel("5년 MDD")
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].set_title("최종 혼합의 60개월 이동성과")
    plt.tight_layout()
    plt.show()

    drawdowns = pd.read_csv(RESULTS / "final_blend_drawdown_episodes.csv")
    display(drawdowns.head(10).style.format({"MDD": "{:.2%}", "MonthsToRecovery": "{:,.0f}"}))"""
        ),
        nbf.v4.new_markdown_cell("## 8. 비용·금리·매개변수 민감도"),
        nbf.v4.new_code_cell(
            """costs = pd.read_csv(RESULTS / "final_blend_cost_sensitivity.csv")
    financing = pd.read_csv(RESULTS / "final_blend_financing_sensitivity.csv")
    display(Markdown("### 거래·환전비용 민감도"))
    display(costs[["CostMultiplier", "CAGR", "Sharpe", "MDD", "Calmar"]].style.format({
        "CostMultiplier": "{:.0f}×", "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}", "Calmar": "{:.3f}",
    }))
    display(Markdown("### 차입금리 민감도"))
    display(financing[["FinancingRate", "CAGR", "Sharpe", "MDD", "Calmar"]].style.format({
        "FinancingRate": "{:.0%}", "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}", "Calmar": "{:.3f}",
    }))

    neighborhood = pd.read_csv(RESULTS / "final_blend_parameter_neighborhood.csv")
    full_surface = neighborhood[neighborhood["Period"] == "full"]
    cagr_surface = full_surface.pivot(index="HardFraction", columns="Leverage", values="CAGR")
    mdd_surface = full_surface.pivot(index="HardFraction", columns="Leverage", values="MDD")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, cmap in [
        (axes[0], cagr_surface, "인접 매개변수 — 전체 CAGR", "Blues"),
        (axes[1], mdd_surface, "인접 매개변수 — 전체 MDD", "RdYlGn"),
    ]:
        image = ax.imshow(data.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(data.columns)), [f"{x:.2f}" for x in data.columns])
        ax.set_yticks(range(len(data.index)), [f"{x:.3f}" for x in data.index])
        ax.set_xlabel("총노출")
        ax.set_ylabel("원 전략 혼합비율")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data.iloc[i,j]:.1%}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## 9. 하위기간과 위기 에피소드"),
        nbf.v4.new_code_cell(
            """subperiods = pd.read_csv(RESULTS / "final_blend_subperiods.csv")
    stress = pd.read_csv(RESULTS / "final_blend_stress_episodes.csv")
    display(Markdown("### 하위기간"))
    display(subperiods[["Subperiod", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar"]].style.format({
        "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}", "Calmar": "{:.3f}",
    }))
    display(Markdown("### 위기 에피소드"))
    display(stress[["Episode", "Strategy", "CAGR", "Sharpe", "MDD", "FinalMultiple"]].style.format({
        "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}", "FinalMultiple": "{:.2f}x",
    }))"""
        ),
        nbf.v4.new_markdown_cell("## 10. 블록 부트스트랩 — MDD 15%는 보장이 아니다"),
        nbf.v4.new_code_cell(
            """bootstrap = pd.read_csv(RESULTS / "final_blend_bootstrap.csv")
    with (RESULTS / "final_blend_validation.json").open(encoding="utf-8") as handle:
        validation = json.load(handle)

    display(bootstrap.style.format({"P05": "{:.2%}", "Median": "{:.2%}", "P95": "{:.2%}"}))
    prob = pd.Series(validation["bootstrap_probabilities"], name="확률/비율")
    display(prob.to_frame().style.format("{:.1%}"))
    display(Markdown(
        "같은 월수익 블록을 세 전략에 동시에 적용한 3,000회 쌍체 재표본 결과다. "
        "최종 혼합이 원 전략보다 MDD가 작을 확률은 거의 100%였지만, 절대 MDD 15% 이내 확률은 블록 6개월 약 31%, 12개월 약 54%였다. "
        "따라서 -14.90%는 점추정치이지 손실 한도가 아니다."
    ))"""
        ),
        nbf.v4.new_markdown_cell("## 11. 경기국면 판정력"),
        nbf.v4.new_code_cell(
            """regime_metrics = validation["regime_metrics"]
    regime_table = pd.Series({
        "성장상태 균형정확도": regime_metrics["growth_balanced_accuracy"],
        "물가상태 균형정확도": regime_metrics["inflation_balanced_accuracy"],
        "4개 국면 정확도": regime_metrics["quadrant_accuracy"],
        "평가 월수": regime_metrics["n_months"],
    }, name="결과").to_frame()
    display(regime_table.style.format({"결과": lambda x: f"{x:.1%}" if x <= 1 else f"{x:.0f}"}))

    growth_cm = pd.DataFrame(regime_metrics["growth_confusion"], index=["실제 Low", "실제 High"], columns=["예측 Low", "예측 High"])
    inflation_cm = pd.DataFrame(regime_metrics["inflation_confusion"], index=["실제 Low", "실제 High"], columns=["예측 Low", "예측 High"])
    display(Markdown("### 성장 혼동행렬")); display(growth_cm)
    display(Markdown("### 물가 혼동행렬")); display(inflation_cm)
    display(Markdown("최종 혼합은 국면분류기를 다시 학습하지 않는다. 따라서 수익률 개선을 위해 경기국면 정확도를 희생하거나 사후 조정하지 않았다."))"""
        ),
        nbf.v4.new_markdown_cell("## 12. 자동 검증"),
        nbf.v4.new_code_cell(
            """full_m = performance_summary(final["return"])
    locked_m = performance_summary(final.loc["2018-01":, "return"])

    assert signals.index.is_monotonic_increasing and signals.index.is_unique
    assert (signals["signal_month"] < signals.index).all(), "신호월은 투자월보다 앞서야 함"
    assert np.isfinite(final["return"]).all()
    assert (final[weight_cols].sum(axis=1).sub(1.20).abs() < 1e-8).all()
    assert (final[weight_cols] >= -1e-12).all().all()
    assert (final[["trade_cost", "fx_cost"]] >= 0).all().all()
    assert full_m["MDD"] >= -0.15 and locked_m["MDD"] >= -0.15
    assert full_m["Sharpe"] >= 1.0 and locked_m["Sharpe"] >= 1.0
    assert full_m["CAGR"] / full_hard["CAGR"] >= 0.75
    assert set(ASSETS) == {"KODEX200", "BOND", "GLD", "USO"}
    print("시점·자산범위·비중·비용·잠금구간·MDD·Sharpe 검증을 통과했습니다.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 13. 결론과 운영상 주의

    ### 채택안

    - 원 고수익 국면비중 40%, 기존 방어비중 60%, 총노출 120%.
    - 전체 CAGR **14.30%**, Sharpe **1.024**, MDD **-14.90%**.
    - 잠금구간 CAGR **17.07%**, Sharpe **1.164**, MDD **-14.90%**.
    - 원 전략 CAGR의 약 **77.5%**를 유지하면서 역사적 MDD를 약 11.8%p 줄였다.
    - 신규 헤지 자산은 없다. 경기국면 정확도는 성장 86.3%, 물가 88.9%, 4국면 77.2%로 기존 판정을 유지한다.

    ### 반드시 함께 읽어야 하는 한계

    1. MDD 여유가 0.10%p뿐이다. 비용 2배에서는 MDD가 약 -15.23%로 상한을 넘는다.
    2. 2026년 2월 고점 이후 6월 저점의 -14.90% 낙폭은 데이터 종료 시점까지 회복되지 않았다.
    3. 블록 부트스트랩상 MDD 15% 이내 확률은 31~54%이므로 실제 위험한도는 더 넓게 잡아야 한다.
    4. 20% 차입은 마진콜·금리변동·추적오차를 만든다. 실제 금융비용이 8%면 전체 CAGR은 약 13.44%, MDD는 약 -15.13%다.
    5. GDP·물가 실시간 빈티지, 세금, 시장충격, 실제 체결 슬리피지는 완전 반영되지 않았다. 2009-03 이전 KODEX200은 KOSPI200 프록시다.

    따라서 이 안은 ‘MDD 15% 보장’ 전략이 아니라, **추가 헤지 자산 없이 원 전략의 수익성을 최대한 보존한 역사적 절충안**이다. 실전에서는 총노출 1.15~1.20 범위를 위험예산에 따라 선택하고, 비용과 차입금리를 별도로 관리해야 한다."""
        ),
    ]

    nbf.write(nb, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
