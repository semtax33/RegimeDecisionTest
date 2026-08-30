from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "strategies/core/regime_research.py"
NOTEBOOK_PATH = ROOT / "artifacts/notebooks/economic_regime_allocation_validated.ipynb"

def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    source = source.replace("ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path.cwd()")
    source = source.split("\ndef main(refresh: bool = False) -> None:")[0]

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 경제 국면 기반 멀티에셋 자산배분 — 검증 완료 버전

    **목표:** 원본 노트북의 입력변수와 자산군만 사용해 경제적으로 설명 가능한 국면 확률을 만들고, MDD를 10~20% 미만으로 억제하면서 Sharpe 1 이상을 지향한다.

    이 노트북은 원본의 실행 오류와 룩어헤드 가능성을 고친 뒤, 다음 구조를 월별 walk-forward로 검증한다.

    `경제 합성점수 + Sparse Jump Model → soft regime probabilities → 비대칭 EWMA 위험예측 → 8% vol target + CDaR/turnover 제약 → drawdown guard`

    > 결과는 연구용 백테스트이며 투자수익을 보장하지 않는다. Sharpe는 원본과 동일하게 무위험수익률 0%를 사용한다."""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. 원본 진단과 수정 사항

    - 원본은 `regime_df.index >= 2026-07-30`으로 잘려 신호가 1개뿐이었다.
    - 다음 달 가격이 없어 전략 결과가 빈 표가 되었고 `KeyError: net_factor`로 중단됐다.
    - Yahoo의 KODEX200 초창기 데이터에는 2007~2009 공백이 있어, 2009-03까지 KOSPI200 프록시를 사용하고 이후 실제 KODEX200 조정가격으로 접합했다.
    - 모든 거시변수에 발표시차를 반영하고, `t월 말 신호 → t+1월 첫 거래일 리밸런싱`을 강제한다.
    - 자산은 원본과 동일한 **KODEX200, 국내 채권 총수익지수, GLD, USO**만 사용한다. 레버리지·공매도·신규 현금자산은 없다.
    - 거래비용은 매수/매도 명목 각각 15bp, USD 비중 변화에는 환전비용 5bp를 추가한다."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    ROOT = Path.cwd()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (13, 5)
    plt.rcParams["axes.unicode_minus"] = False
    for font in ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = font
            break
        except Exception:
            pass

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 160)
    print("작업 폴더:", ROOT)
    print("자산:", ["KODEX200", "BOND", "GLD", "USO"])"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. 재현 가능한 전체 구현

    아래 셀에는 데이터 로딩, Sparse Jump Model, soft 국면 확률, 비대칭 EWMA 공분산, CDaR 목적함수, 변동성 목표, 드로다운 제어, 비용 차감 백테스트가 모두 들어 있다. 외부 사용자 정의 패키지에 의존하지 않는다."""
        ),
        nbf.v4.new_code_cell(source, metadata={"jupyter": {"source_hidden": True}, "tags": ["implementation"]}),
        nbf.v4.new_markdown_cell(
            """## 3. 데이터와 시점가용성 감사

    성장 모듈은 GDP YoY(분기 발표 후 한 달), 수출 YoY(다음 월말), 제조업 BSI 전망(해당 월말), 물가 모듈은 CPI/PPI/수입물가 YoY(다음 월말)를 사용한다. 각 수준의 rolling z-score와 3개월 변화만 만든다. 전 구간 평균·표준편차를 쓰지 않는다."""
        ),
        nbf.v4.new_code_cell(
            """features, core = load_macro_data()
    asset_returns, asset_levels = load_monthly_asset_returns(refresh=False)
    signals = compute_regime_signals(features, asset_returns, jump_penalty=3.0, min_history=24)

    data_audit = pd.DataFrame({
        "Start": [features.index.min(), asset_returns.index.min().to_timestamp(), signals.index.min().to_timestamp()],
        "End": [features.index.max(), asset_returns.index.max().to_timestamp(), signals.index.max().to_timestamp()],
        "Observations": [len(features), len(asset_returns), len(signals)],
    }, index=["Macro features", "Common asset returns", "Tradable signals"])
    display(data_audit)
    display(asset_returns.describe().T[["mean", "std", "min", "max"]].style.format("{:.3%}"))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. 경제적 구조와 국면 확률

    성장·물가를 각각 고/저 확률로 만든 뒤 4개 국면의 결합확률을 계산한다.

    | 국면 | 경제 해석 | 자산 방향 |
    |---|---|---|
    | Goldilocks | 성장 높음, 물가 낮음 | 주식 중심, 채권·금 분산 |
    | Overheating | 성장·물가 높음 | 원유·금 확대, 주식 축소 |
    | Slowdown | 성장·물가 낮음 | 채권 중심, 금 보조 |
    | Stagflation | 성장 낮음, 물가 높음 | 금 중심, 원유 보조 |

    표본이 232개월로 작기 때문에 국면확률의 90%는 직접 해석 가능한 현재 합성점수, 10%만 SJM의 희소 변수선택·전환 억제를 반영한다. 이 비율은 복잡한 모델이 단순 지속성 기준보다 나빠지는 것을 막기 위한 보수적 결정이다."""
        ),
        nbf.v4.new_code_cell(
            """regime_eval, regime_metrics = evaluate_regimes(signals, core)

    # 단순 지속성 기준: 현재 합성점수의 부호가 다음 3개월에도 유지된다고 예측
    current_state = pd.DataFrame({
        "naive_growth": core[["GDP", "Export", "BSI"]].mean(axis=1) >= 0,
        "naive_inflation": core[["CPI", "PPI", "ImportPrice"]].mean(axis=1) >= 0,
    })
    regime_compare = regime_eval.join(current_state, how="left")
    naive_growth = balanced_accuracy_score(regime_compare["growth_high_realized"], regime_compare["naive_growth"])
    naive_inflation = balanced_accuracy_score(regime_compare["inflation_high_realized"], regime_compare["naive_inflation"])
    naive_quadrant = ((regime_compare["naive_growth"] == regime_compare["growth_high_realized"]) &
                      (regime_compare["naive_inflation"] == regime_compare["inflation_high_realized"])).mean()

    accuracy_table = pd.DataFrame({
        "SJM-composite ensemble": [regime_metrics["growth_balanced_accuracy"], regime_metrics["inflation_balanced_accuracy"], regime_metrics["quadrant_accuracy"]],
        "Naive persistence": [naive_growth, naive_inflation, naive_quadrant],
    }, index=["Growth balanced accuracy", "Inflation balanced accuracy", "4-quadrant accuracy"])
    display(accuracy_table.style.format("{:.1%}"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [
        (axes[0], "growth_confusion", "성장 상태 혼동행렬"),
        (axes[1], "inflation_confusion", "물가 상태 혼동행렬"),
    ]:
        cm = np.array(regime_metrics[key])
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["Low", "High"])
        ax.set_yticks([0, 1], ["Low", "High"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Realized next 3M")
        ax.set_title(title)
    plt.tight_layout()
    plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """prob_plot = signals[["p_growth_high", "p_inflation_high"]].copy()
    prob_plot.index = prob_plot.index.to_timestamp()
    ax = prob_plot.plot(figsize=(14, 4), color=["#1479FF", "#F59E0B"], lw=1.8)
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_title("Walk-forward 성장·물가 고국면 확률")
    ax.set_ylabel("Probability")
    ax.legend(["P(Growth High)", "P(Inflation High)"])
    plt.show()

    latest_signal = signals.iloc[-1]
    latest_probs = pd.Series({r: latest_signal[f"p_{r}"] for r in REGIME_ANCHORS.index}, name="Probability")
    display(Markdown(f"**최신 투자 대상 월:** {signals.index[-1]} / **최빈 국면:** {latest_signal['regime']}"))
    display(latest_probs.sort_values(ascending=False).to_frame().style.format("{:.1%}"))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. 자산배분·위험제어와 보정 원칙

    1. 국면별 경제적 기준비중을 확률가중한다.
    2. 보정에서 선택된 국면 강도 75%와 장기 전략적 비중 25%를 혼합한다.
    3. 최근 84개월 수익률로 downside shock에 더 큰 가중치를 주는 비대칭 EWMA 공분산을 계산한다. **이는 Bayesian SV라고 주장하지 않는 안정적 대용치**다.
    4. 목표 변동성 8%, 90% CDaR 상한 16%, 거래회전율 및 기준비중 이탈 페널티를 둔다.
    5. 현재 drawdown이 -5%를 넘으면 MPC 논문의 아이디어처럼 위험회피도를 상태의존적으로 높인다.

    파라미터 후보는 36개로 제한하고, 2007-04~2017-12만 사용해 `Sharpe + 0.35×Calmar − MDD 초과 − turnover` 점수로 결정했다. 2018-01 이후는 선택이 끝날 때까지 보지 않은 잠금 구간이다."""
        ),
        nbf.v4.new_code_cell(
            """cfg = StrategyConfig()
    config_table = pd.Series(asdict(cfg), name="Locked value").to_frame()
    display(config_table)

    calibration_grid = pd.read_csv(RESULTS_DIR / "calibration_grid.csv")
    calibration_cols = ["name", "Sharpe", "MDD", "Calmar", "CAGR", "AvgTurnover", "ValidationScore"]
    display(calibration_grid[calibration_cols].head(10).style.format({
        "Sharpe": "{:.3f}", "MDD": "{:.2%}", "Calmar": "{:.3f}", "CAGR": "{:.2%}", "AvgTurnover": "{:.2%}", "ValidationScore": "{:.3f}"
    }))"""
        ),
        nbf.v4.new_markdown_cell("## 6. 전체 walk-forward 백테스트와 구성요소 비교"),
        nbf.v4.new_code_cell(
            """mode_labels = {
        "proposed": "Proposed: Regime + Risk control",
        "soft": "Soft regime only",
        "hard": "Original-style hard regime",
        "equal": "Equal weight",
        "static_defensive": "Static defensive",
        "kodex": "KODEX200 B&H",
    }
    backtests = {mode: run_backtest(asset_returns, signals, cfg, mode=mode) for mode in mode_labels}
    summary = pd.DataFrame({mode_labels[k]: performance_summary(v["return"]) for k, v in backtests.items()}).T
    display(summary.style.format({
        "Months": "{:.0f}", "CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
        "MDD": "{:.2%}", "Calmar": "{:.3f}", "FinalMultiple": "{:.2f}x", "PositiveMonths": "{:.1%}"
    }).highlight_max(subset=["Sharpe", "Calmar"], color="#d7f5dd").highlight_min(subset=["MDD"], color="#ffe0e0"))"""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"proposed": "#0B63CE", "static_defensive": "#16A085", "kodex": "#7F8C8D", "hard": "#D35400"}
    for key in ["proposed", "static_defensive", "kodex", "hard"]:
        bt = backtests[key]
        idx = bt.index.to_timestamp()
        wealth = (1 + bt["return"]).cumprod()
        axes[0].plot(idx, wealth, label=mode_labels[key], color=colors[key], lw=2 if key == "proposed" else 1.2)
        if key in ["proposed", "kodex"]:
            dd = wealth / wealth.cummax() - 1
            axes[1].plot(idx, dd, label=mode_labels[key], color=colors[key], lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_title("누적자산 (로그축, 거래·환전비용 차감)")
    axes[0].set_ylabel("Growth of 1 KRW")
    axes[0].legend(ncol=2)
    axes[1].set_title("Drawdown")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend()
    plt.tight_layout()
    plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """proposed = backtests["proposed"]
    weights = proposed[[f"w_{a}" for a in ASSETS]].copy()
    weights.columns = ASSETS
    weights.index = weights.index.to_timestamp()
    ax = weights.plot.area(figsize=(14, 5), color=["#1479FF", "#6C7A89", "#E5B94E", "#C85A3D"], alpha=0.88)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("제안 전략의 월별 목표비중")
    ax.set_ylabel("Weight")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.show()

    avg_weights = weights.mean().rename("Average weight")
    latest_weights = weights.iloc[-1].rename("Latest weight")
    display(pd.concat([avg_weights, latest_weights], axis=1).style.format("{:.1%}"))"""
        ),
        nbf.v4.new_markdown_cell("## 7. 잠금 테스트, 비용 스트레스, 위기 구간"),
        nbf.v4.new_code_cell(
            """locked_backtests = {mode: run_backtest(asset_returns, signals, cfg, mode=mode, start="2018-01") for mode in mode_labels}
    locked_summary = pd.DataFrame({mode_labels[k]: performance_summary(v["return"]) for k, v in locked_backtests.items()}).T
    display(Markdown("### 2018-01~2026-07 잠금 구간"))
    display(locked_summary.style.format({
        "Months": "{:.0f}", "CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
        "MDD": "{:.2%}", "Calmar": "{:.3f}", "FinalMultiple": "{:.2f}x", "PositiveMonths": "{:.1%}"
    }))"""
        ),
        nbf.v4.new_code_cell(
            """cost_rows = []
    for multiplier in [0.0, 1.0, 2.0]:
        bt = run_backtest(asset_returns, signals, cfg, mode="proposed", cost_multiplier=multiplier)
        m = performance_summary(bt["return"])
        cost_rows.append({"Cost multiplier": multiplier, **m.to_dict(), "Avg turnover": bt["turnover"].mean()})
    cost_sensitivity = pd.DataFrame(cost_rows).set_index("Cost multiplier")
    display(Markdown("### 거래·환전비용 민감도"))
    display(cost_sensitivity[["CAGR", "Sharpe", "MDD", "Calmar", "Avg turnover"]].style.format({
        "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}", "Calmar": "{:.3f}", "Avg turnover": "{:.2%}"
    }))

    crises = {
        "Global Financial Crisis": ("2007-10", "2009-03"),
        "COVID shock": ("2020-01", "2020-12"),
        "Inflation shock": ("2022-01", "2022-12"),
    }
    crisis_rows = []
    for label, (start, end) in crises.items():
        lo, hi = pd.Period(start, "M"), pd.Period(end, "M")
        for key in ["proposed", "static_defensive", "kodex"]:
            r = backtests[key].loc[lo:hi, "return"]
            if len(r):
                wealth = (1 + r).cumprod()
                crisis_rows.append({
                    "Episode": label, "Strategy": mode_labels[key], "Cumulative return": wealth.iloc[-1] - 1,
                    "Episode MDD": (wealth / wealth.cummax() - 1).min(), "Volatility": r.std() * np.sqrt(12)
                })
    crisis_table = pd.DataFrame(crisis_rows)
    display(Markdown("### 주요 위기 에피소드"))
    display(crisis_table.pivot(index="Episode", columns="Strategy", values=["Cumulative return", "Episode MDD"]).style.format("{:.1%}"))"""
        ),
        nbf.v4.new_markdown_cell("## 8. 블록 부트스트랩 불확실성"),
        nbf.v4.new_code_cell(
            """def block_bootstrap_metrics(series, n_boot=1000, block=12, seed=42):
        r = np.asarray(series, dtype=float)
        rng = np.random.default_rng(seed)
        rows = []
        starts = np.arange(0, len(r) - block + 1)
        for _ in range(n_boot):
            sample = []
            while len(sample) < len(r):
                s = int(rng.choice(starts))
                sample.extend(r[s:s+block])
            sample = pd.Series(sample[:len(r)])
            m = performance_summary(sample)
            rows.append([m["Sharpe"], m["MDD"], m["CAGR"]])
        return pd.DataFrame(rows, columns=["Sharpe", "MDD", "CAGR"])

    boot = block_bootstrap_metrics(proposed["return"])
    bootstrap_ci = boot.quantile([0.025, 0.50, 0.975]).rename(index={0.025: "2.5%", 0.5: "Median", 0.975: "97.5%"})
    display(bootstrap_ci.style.format({"Sharpe": "{:.3f}", "MDD": "{:.2%}", "CAGR": "{:.2%}"}))
    display(Markdown("블록 부트스트랩은 월별 의존성을 일부 보존한 표본 불확실성 점검이며 미래 성과 예측구간이 아니다."))"""
        ),
        nbf.v4.new_markdown_cell("## 9. 자동 검증"),
        nbf.v4.new_code_cell(
            """assert signals.index.is_monotonic_increasing and signals.index.is_unique
    assert asset_returns.index.is_monotonic_increasing and asset_returns.index.is_unique
    assert (signals["signal_month"] < signals.index).all(), "신호월은 투자월보다 반드시 앞서야 함"
    assert np.isfinite(proposed["return"]).all()
    assert (weights.sum(axis=1).sub(1).abs() < 1e-8).all()
    assert (weights >= -1e-12).all().all()
    assert (proposed[["trade_cost", "fx_cost"]] >= 0).all().all()
    assert summary.loc["Proposed: Regime + Risk control", "MDD"] > -0.20
    assert locked_summary.loc["Proposed: Regime + Risk control", "Sharpe"] > 1.0
    print("모든 시점·비중·비용·목표 검증을 통과했습니다.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 10. 결론과 한계

    ### 결론

    - **경제적 설명:** 성장/물가 4분면에 따라 주식·채권·금·원유를 연속적으로 기울인다.
    - **국면:** 다음 3개월 기준 성장·물가 balanced accuracy와 4분면 정확도를 단순 지속성 기준과 함께 공개한다.
    - **MDD:** 변동성 목표, CDaR, drawdown guard가 hard switching의 큰 손실을 줄이는 핵심이다.
    - **Sharpe:** 잠금 구간 성과가 1 이상인지 가장 중요하게 본다. 전체기간 수치만 보고 설정을 선택하지 않았다.

    ### 반드시 남는 한계

    - GDP·물가 데이터는 현재 파일의 최신 개정치다. 진정한 실시간 빈티지 데이터 백테스트가 아니므로 revision bias가 남는다.
    - 2009-03 이전 KODEX200은 KOSPI200 가격지수 프록시이며 배당 반영이 완전하지 않다.
    - 미국 자산 첫 거래가격과 USD/KRW 종가의 시간대가 비동기다.
    - Sharpe는 무위험수익률 0%, 세금·시장충격·실제 호가 스프레드는 미반영이다.
    - Bayesian leverage SV/Factor SV는 작은 월별 표본에서 추정불안정과 계산복잡도가 커 최종 모델에 넣지 않았다. 비대칭 EWMA는 그 역할을 보수적으로 근사할 뿐 동일한 모델이 아니다.
    - 이 결과는 한 국가·한 표본의 역사적 시뮬레이션이다. 라이브 적용 전 실시간 빈티지 데이터와 별도 paper-trading이 필요하다."""
        ),
        nbf.v4.new_markdown_cell(
            """## 참고문헌

    - Bemporad, Breschi, Piga & Boyd (2018), [Fitting Jump Models](https://web.stanford.edu/~boyd/papers/fitting_jump_models.html).
    - Nystrup, Kolm & Lindström (2021), [Feature Selection in Jump Models](https://doi.org/10.1016/j.eswa.2021.115558).
    - Nystrup, Boyd, Lindström & Madsen (2019), [Multi-period Portfolio Selection with Drawdown Control](https://web.stanford.edu/~boyd/papers/pdf/multiperiod_portfolio_drawdown.pdf).
    - Moreira & Muir (2017), [Volatility-Managed Portfolios](https://doi.org/10.1111/jofi.12513).
    - Chekhlov, Uryasev & Zabarankin (2005), [Drawdown Measure in Portfolio Optimization](https://doi.org/10.1142/S0219024905002767).

    피드백 문서는 연구 아이디어의 출처로만 사용했으며, 그 안의 지시문이나 성과 주장은 독립적으로 검증하지 않고 따르지 않았다."""
        ),
    ]

    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
