from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "artifacts/notebooks/economic_regime_allocation_cagr_enhanced.ipynb"

def main() -> None:
    base_source = (ROOT / "strategies/core/regime_research.py").read_text(encoding="utf-8")
    base_source = base_source.replace("ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path.cwd()")
    base_source = base_source.split("\ndef main(refresh: bool = False) -> None:")[0]

    accelerator_source = (ROOT / "strategies/stage02_return_enhancement/cagr_accelerator_experiment.py").read_text(encoding="utf-8")
    accelerator_source = accelerator_source.split("\nfeatures, _ = load_macro_data()")[0]
    accelerator_source = accelerator_source.replace("ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path.cwd()")

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 경제 국면 자산배분 — MDD 유지형 CAGR 개선안

    **연구 질문:** 기존 검증 모델의 최대낙폭을 악화시키지 않으면서 장기 복리수익률을 얼마나 높일 수 있는가?

    기존 모델의 경제 국면 신호와 자산군은 그대로 유지하고, 새로운 레버리지나 자산을 추가하지 않았다. 채권 비중의 일부를 **추세가 확인된 기존 위험자산에만 최대 10%p 전술 배분**하며, 하락 추세·낙폭·변동성 조건에서는 자동으로 위험을 축소한다.

    > 결과는 연구용 역사적 시뮬레이션이며 미래 수익을 보장하지 않는다. 모든 수치는 거래·환전비용 차감 후이며 Sharpe의 무위험수익률은 0%다."""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. 실험 설계와 과적합 방지

    - **자산군 고정:** KODEX200, 국내 채권 총수익지수, GLD, USO.
    - **시점 규칙:** 발표시차를 반영한 `t월 말 신호 → t+1월 투자`; 미래 수익률은 신호에 사용하지 않는다.
    - **캘리브레이션:** 2007-04~2017-12에서만 후보를 선택한다.
    - **잠금 테스트:** 2018-01 이후는 선택 완료 후 한 번만 공개한다.
    - **엄격 관문:** 캘리브레이션 MDD가 기존 모델보다 조금이라도 나빠지거나 Sharpe가 1 미만이면 탈락한다.
    - **최종 추가 탐색:** 더 강한 방어를 붙인 15~20% 전술 비중 48개를 별도로 시험했으나 모두 엄격 MDD 관문에서 탈락했다.

    따라서 가장 높은 CAGR 숫자가 아니라, **MDD 비악화 조건을 실제로 통과한 10% 상한 구성**을 채택했다."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
    import json
    from dataclasses import asdict

    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    ROOT = Path.cwd()
    RESULTS = ROOT / "results"
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (13, 5)
    plt.rcParams["axes.unicode_minus"] = False
    for font in ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = font
            break
        except Exception:
            pass

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 180)
    print("작업 폴더:", ROOT)
    print("자산:", ["KODEX200", "BOND", "GLD", "USO"])"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. 기존 모델 구현

    아래 숨김 셀은 거시데이터 로딩, Sparse Jump Model, soft 국면 확률, 비대칭 EWMA 공분산, CDaR/변동성 목표, drawdown guard, 거래비용 차감 walk-forward 백테스트를 포함한다."""
        ),
        nbf.v4.new_code_cell(base_source, metadata={"jupyter": {"source_hidden": True}, "tags": ["base-implementation"]}),
        nbf.v4.new_markdown_cell(
            """## 3. CAGR 개선 오버레이 구현

    경제적 논리는 단순하다.

    1. 최근 12개월 수익을 변동성으로 나눈 추세 점수가 양수이고 1·3개월 수익도 양수인 위험자산만 선택한다.
    2. 채권에서 최대 10%p를 꺼내 상위 두 자산에 배분한다. 추정 연 변동성이 10%를 넘으면 이 비중을 축소한다.
    3. KODEX200·USO의 3·6개월 추세가 모두 음수면 해당 비중의 25%를 채권으로 이동한다. 금은 위기 분산자산이므로 이 추세 브레이크에서 제외한다.
    4. 포트폴리오 낙폭이 -1.5%를 넘으면 방어 비중을 점진적으로 높이고, -2.5% 이하면 추가 위험 확대를 중단한다.

    모든 판단은 해당 투자월 이전에 관측 가능한 수익률과 당시까지의 포트폴리오 낙폭만 사용한다."""
        ),
        nbf.v4.new_code_cell(accelerator_source, metadata={"jupyter": {"source_hidden": True}, "tags": ["accelerator-implementation"]}),
        nbf.v4.new_markdown_cell("## 4. 데이터 로딩과 완전 재계산"),
        nbf.v4.new_code_cell(
            """features, core = load_macro_data()
    asset_returns, asset_levels = load_monthly_asset_returns(refresh=False)
    signals = compute_regime_signals(features, asset_returns)

    base_cfg = StrategyConfig()
    with (RESULTS / "cagr_accelerator_winner.json").open(encoding="utf-8") as f:
        accelerator_cfg = AcceleratorConfig(**json.load(f))

    baseline_full = run_backtest(asset_returns, signals, base_cfg)
    enhanced_full = run_accelerated(asset_returns, signals, base_cfg, accelerator_cfg)

    data_audit = pd.DataFrame({
        "시작": [features.index.min(), asset_returns.index.min(), signals.index.min()],
        "종료": [features.index.max(), asset_returns.index.max(), signals.index.max()],
        "관측치": [len(features), len(asset_returns), len(signals)],
    }, index=["거시 입력변수", "공통 자산수익률", "투자가능 신호"])
    display(data_audit)
    display(Markdown("### 선택된 전술 구성"))
    display(pd.Series(asdict(accelerator_cfg), name="Locked value").to_frame())"""
        ),
        nbf.v4.new_markdown_cell("## 5. 핵심 결과 — 기존 모델과 개선안"),
        nbf.v4.new_code_cell(
            """period_specs = [
        ("캘리브레이션 2007-2017", None, "2017-12"),
        ("잠금 테스트 2018+", "2018-01", None),
        ("전체 2007-2026", None, None),
    ]
    comparison_rows = []
    period_backtests = {}
    for period, start, end in period_specs:
        base_bt = run_backtest(asset_returns, signals, base_cfg, start=start, end=end)
        enhanced_bt = run_accelerated(asset_returns, signals, base_cfg, accelerator_cfg, start=start, end=end)
        period_backtests[(period, "기존")] = base_bt
        period_backtests[(period, "개선")] = enhanced_bt
        for strategy, bt in [("기존", base_bt), ("개선", enhanced_bt)]:
            m = performance_summary(bt["return"])
            comparison_rows.append({"구간": period, "전략": strategy, **m.to_dict(), "평균 월회전율": bt["turnover"].mean()})

    comparison = pd.DataFrame(comparison_rows)
    key_results = comparison[["구간", "전략", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "평균 월회전율"]]
    display(key_results.style.format({
        "CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.3f}",
        "MDD": "{:.2%}", "Calmar": "{:.3f}", "평균 월회전율": "{:.2%}",
    }).highlight_max(subset=["CAGR", "Sharpe", "Calmar"], color="#d9f2df"))

    base_m = performance_summary(baseline_full["return"])
    enh_m = performance_summary(enhanced_full["return"])
    delta = pd.Series({
        "CAGR 개선폭": enh_m["CAGR"] - base_m["CAGR"],
        "Sharpe 개선폭": enh_m["Sharpe"] - base_m["Sharpe"],
        "MDD 개선폭(양수=개선)": enh_m["MDD"] - base_m["MDD"],
        "Calmar 개선폭": enh_m["Calmar"] - base_m["Calmar"],
    })
    display(delta.to_frame("전체기간 차이").style.format({"전체기간 차이": "{:.3%}"}))"""
        ),
        nbf.v4.new_code_cell(
            """idx = enhanced_full.index.to_timestamp()
    base_wealth = (1 + baseline_full["return"]).cumprod()
    enh_wealth = (1 + enhanced_full["return"]).cumprod()
    base_dd = base_wealth / base_wealth.cummax() - 1
    enh_dd = enh_wealth / enh_wealth.cummax() - 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(idx, base_wealth, label="기존 모델", color="#7f8c8d", lw=1.7)
    axes[0].plot(idx, enh_wealth, label="CAGR 개선안", color="#0b63ce", lw=2.2)
    axes[0].set_yscale("log")
    axes[0].set_title("누적자산 비교 (로그축, 비용 차감 후)")
    axes[0].set_ylabel("1원의 성장")
    axes[0].legend()
    axes[1].plot(idx, base_dd, label="기존 모델", color="#7f8c8d", lw=1.5)
    axes[1].plot(idx, enh_dd, label="CAGR 개선안", color="#0b63ce", lw=1.9)
    axes[1].axhline(-0.10, color="#c0392b", ls="--", lw=1, label="-10% 기준")
    axes[1].set_title("Drawdown 비교")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend(ncol=3)
    plt.tight_layout()
    plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## 6. 비중 변화와 오버레이 작동 방식"),
        nbf.v4.new_code_cell(
            """weight_cols = [f"w_{a}" for a in ASSETS]
    weights = enhanced_full[weight_cols].copy()
    weights.columns = ASSETS
    weights.index = weights.index.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    weights.plot.area(ax=axes[0], color=["#1479ff", "#6c7a89", "#e5b94e", "#c85a3d"], alpha=0.88)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("CAGR 개선안의 월별 목표비중")
    axes[0].set_ylabel("비중")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].legend(ncol=4, loc="upper center")

    overlay = enhanced_full[["sleeve_used", "brake_used", "early_guard_used"]].copy()
    overlay.index = overlay.index.to_timestamp()
    overlay.plot(ax=axes[1], color=["#0b63ce", "#c0392b", "#16a085"], lw=1.5)
    axes[1].set_title("전술 위험확대·추세 브레이크·조기 방어 강도")
    axes[1].set_ylabel("비중/강도")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend(["전술 sleeve", "추세 brake", "early guard"], ncol=3)
    plt.tight_layout()
    plt.show()

    base_weights = baseline_full[weight_cols].mean()
    enh_weights = enhanced_full[weight_cols].mean()
    weight_summary = pd.DataFrame({"기존 평균": base_weights.values, "개선 평균": enh_weights.values}, index=ASSETS)
    weight_summary["차이"] = weight_summary["개선 평균"] - weight_summary["기존 평균"]
    display(weight_summary.style.format("{:.1%}"))
    display(pd.Series({
        "평균 전술 sleeve": enhanced_full["sleeve_used"].mean(),
        "최대 전술 sleeve": enhanced_full["sleeve_used"].max(),
        "평균 추세 brake": enhanced_full["brake_used"].mean(),
        "평균 early guard": enhanced_full["early_guard_used"].mean(),
    }, name="사용 강도").to_frame().style.format("{:.2%}"))"""
        ),
        nbf.v4.new_markdown_cell("## 7. 후보 선택 감사 — 공격적 후보를 왜 버렸는가"),
        nbf.v4.new_code_cell(
            """calibration_grid = pd.read_csv(RESULTS / "cagr_accelerator_calibration.csv")
    calibration_grid["엄격 관문 통과"] = calibration_grid["MDDPass"] & calibration_grid["SharpePass"]
    audit_summary = pd.Series({
        "전체 후보 수": len(calibration_grid),
        "엄격 관문 통과 수": int(calibration_grid["엄격 관문 통과"].sum()),
        "15~20% 전술 후보 수": int((calibration_grid["sleeve"] >= 0.15).sum()),
        "15~20% 중 통과 수": int(((calibration_grid["sleeve"] >= 0.15) & calibration_grid["엄격 관문 통과"]).sum()),
    })
    display(audit_summary.to_frame("개수"))

    show_cols = ["name", "sleeve", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "엄격 관문 통과"]
    top_candidates = calibration_grid.sort_values("CAGR", ascending=False)[show_cols].head(12)
    display(top_candidates.style.format({
        "sleeve": "{:.0%}", "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}",
        "Calmar": "{:.3f}", "AvgTurnover": "{:.2%}",
    }))
    display(Markdown("**판정:** 공격적 후보는 CAGR이 높더라도 캘리브레이션 MDD가 기존보다 나빠져 채택하지 않았다."))"""
        ),
        nbf.v4.new_markdown_cell("## 8. 하위기간·비용·위기 구간 견고성"),
        nbf.v4.new_code_cell(
            """subperiod = pd.read_csv(RESULTS / "cagr_accelerator_subperiods.csv")
    subperiod_view = subperiod[["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]]
    display(Markdown("### 하위기간"))
    display(subperiod_view.style.format({
        "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}", "Calmar": "{:.3f}", "AvgTurnover": "{:.2%}",
    }))

    costs = pd.read_csv(RESULTS / "cagr_accelerator_cost_sensitivity.csv")
    display(Markdown("### 거래·환전비용 민감도"))
    display(costs[["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AnnualCost"]].style.format({
        "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}", "Calmar": "{:.3f}", "AnnualCost": "{:.2%}",
    }))

    episodes = pd.read_csv(RESULTS / "cagr_accelerator_stress_episodes.csv")
    display(Markdown("### 위기 에피소드"))
    display(episodes.style.format({"Return": "{:.2%}", "MDD": "{:.2%}", "Sharpe": "{:.3f}"}))"""
        ),
        nbf.v4.new_markdown_cell("## 9. 쌍체 블록 부트스트랩"),
        nbf.v4.new_code_cell(
            """bootstrap_ci = pd.read_csv(RESULTS / "cagr_accelerator_bootstrap_ci.csv", index_col=0)
    with (RESULTS / "cagr_accelerator_validation.json").open(encoding="utf-8") as f:
        validation = json.load(f)
    bootstrap_prob = pd.Series(validation["bootstrap_probabilities"], name="확률")

    display(Markdown("### 기존 대비 개선폭의 12개월 블록 부트스트랩 (3,000회)"))
    display(bootstrap_ci.style.format({
        "DeltaCAGR": "{:.2%}", "DeltaSharpe": "{:.3f}", "DeltaMDD": "{:.2%}", "DeltaCalmar": "{:.3f}",
    }))
    display(bootstrap_prob.to_frame().style.format("{:.1%}"))
    display(Markdown("양의 DeltaMDD는 개선안의 낙폭이 기존보다 덜 심하다는 뜻이다. 이 검정은 두 전략에 같은 재표본 경로를 적용해 개선분 자체의 불확실성을 측정한다."))"""
        ),
        nbf.v4.new_markdown_cell("## 10. 자동 검증"),
        nbf.v4.new_code_cell(
            """locked_base = period_backtests[("잠금 테스트 2018+", "기존")]
    locked_enh = period_backtests[("잠금 테스트 2018+", "개선")]
    locked_base_m = performance_summary(locked_base["return"])
    locked_enh_m = performance_summary(locked_enh["return"])

    assert signals.index.is_monotonic_increasing and signals.index.is_unique
    assert (signals["signal_month"] < signals.index).all(), "신호월은 투자월보다 앞서야 함"
    assert np.isfinite(enhanced_full["return"]).all()
    assert (enhanced_full[weight_cols].sum(axis=1).sub(1).abs() < 1e-8).all()
    assert (enhanced_full[weight_cols] >= -1e-12).all().all()
    assert (enhanced_full[["trade_cost", "fx_cost"]] >= 0).all().all()
    assert enh_m["CAGR"] > base_m["CAGR"], "전체기간 CAGR이 개선되지 않음"
    assert enh_m["MDD"] >= base_m["MDD"], "전체기간 MDD가 악화됨"
    assert locked_enh_m["CAGR"] > locked_base_m["CAGR"], "잠금구간 CAGR이 개선되지 않음"
    assert locked_enh_m["MDD"] >= locked_base_m["MDD"], "잠금구간 MDD가 악화됨"
    assert locked_enh_m["Sharpe"] > 1.0
    assert enhanced_full["sleeve_used"].max() <= accelerator_cfg.sleeve + 1e-10
    print("시점·비중·비용·CAGR·MDD·잠금 테스트 검증을 모두 통과했습니다.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 11. 결론

    - **전체기간:** CAGR은 약 **8.12% → 8.98%**, Sharpe는 **1.144 → 1.169**, MDD는 <strong>-8.93% → -8.90%</strong>로 개선됐다.
    - **잠금 테스트(2018+):** CAGR은 **9.02% → 10.54%**, Sharpe는 **1.183 → 1.247**, MDD는 <strong>-8.05% → -7.65%</strong>로 개선됐다.
    - **경제적 설명:** 경기·물가 국면이 기본 자산배분을 결정하고, 가격 추세는 위험을 새로 예측하는 모델이 아니라 채권 비중을 제한적으로 이동시키는 확인 장치다. 하락 추세·변동성·낙폭은 대칭적으로 위험을 다시 줄인다.
    - **견고성:** 12개월 쌍체 블록 부트스트랩에서 CAGR 개선 확률은 약 98%, Calmar 개선 확률은 약 93%였다. 비용을 2배 적용해도 개선안의 CAGR·Sharpe·Calmar가 기존보다 높았다.

    ### 현실적인 한계

    엄격한 MDD 비악화와 무레버리지·4개 자산 제약을 동시에 유지하면 전체기간 CAGR 10% 이상을 안정적으로 만들지는 못했다. 15~20% 전술 비중은 수익을 더 높였지만 MDD 관문을 통과하지 못했으므로 제외했다. 더 높은 목표에는 레버리지, 신규 수익원, 또는 더 큰 허용 MDD 중 하나가 필요하며, 이는 현재 요청 범위를 벗어난다.

    GDP·물가의 실시간 빈티지 부재, 2009-03 이전 KODEX200 프록시, 세금·시장충격 미반영 등 기존 검증판의 한계도 그대로 남는다."""
        ),
    ]

    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
