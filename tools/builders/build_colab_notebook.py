from __future__ import annotations

import zipfile
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "strategies/core/regime_research.py"
NOTEBOOK_PATH = ROOT / "artifacts/notebooks/economic_regime_allocation_colab.ipynb"
BUNDLE_PATH = ROOT / "artifacts/bundles/regime_colab_inputs.zip"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str, tags: list[str] | None = None):
    metadata = {"tags": tags} if tags else {}
    return nbf.v4.new_code_cell(source.strip(), metadata=metadata)


def build_input_bundle() -> None:
    # The baseline notebook does not need the 113 MB integrated option table,
    # the historical option directory, or VKOSPI. Keeping those out prevents
    # an unrelated later experiment from inflating/breaking this small bundle.
    excluded = {"KOSPI200OptionPrice.csv", "VKOSPIData.csv"}
    inputs = [
        path
        for path in sorted((ROOT / "raw_data").iterdir())
        if path.is_file() and path.name not in excluded
    ] + [ROOT / "cache" / "market_daily.csv"]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Colab inputs: {missing}")
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in inputs:
            archive.write(path, path.relative_to(ROOT).as_posix())


def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    source = source.replace("ROOT = Path(__file__).resolve().parents[2]", "ROOT = PROJECT_ROOT")
    source = source.split("\ndef main(refresh: bool = False) -> None:")[0]

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "colab": {
            "name": NOTEBOOK_PATH.name,
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    nb["cells"] = [
        markdown(
            """
    # 경제 국면 기반 멀티에셋 자산배분 — Colab 통합 실행본

    이 노트북은 다음 세 파일을 하나의 위→아래 실행 흐름으로 통합한 버전입니다.

    - `economic_regime_allocation_validated.ipynb`: 분석·시각화·검증 구조
    - `regime_research.py`: 데이터 로딩, 국면 신호, 비중 최적화, 백테스트 구현
    - `calibrate_configs.py`: 36개 후보 설정 보정과 우승 설정 선택

    ## Colab 실행 방법

    1. 이 노트북과 함께 생성된 `regime_colab_inputs.zip`을 내려받습니다.
    2. 노트북을 Google Colab에서 엽니다.
    3. **런타임 → 모두 실행**을 누릅니다.
    4. 입력자료 셀에서 요청하면 `regime_colab_inputs.zip`을 업로드합니다.

    > 입력 ZIP에는 `raw_data/`와 재현용 `cache/market_daily.csv`가 들어 있습니다. 결과는 `/content/RegimeDecisionTest/results/`에 저장됩니다. 역사적 시뮬레이션이며 미래 수익을 보장하지 않습니다. Sharpe 무위험수익률은 0%입니다.
    """
        ),
        markdown(
            """
    ## 1. Colab 환경 준비

    Colab에서만 필요한 패키지를 설치합니다. 로컬 Jupyter에서 실행할 때는 현재 Python 환경을 그대로 사용합니다. 이 전략에는 GPU가 필요하지 않습니다.
    """
        ),
        code(
            """
    import subprocess
    import sys

    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "yfinance>=0.2.54", "openpyxl>=3.1", "scipy>=1.11",
            "scikit-learn>=1.4", "koreanize-matplotlib>=0.1.1",
        ])
        print("Colab 의존성 설치 완료")
    else:
        print("로컬 Python 환경 사용")
    """,
            ["colab-setup"],
        ),
        markdown(
            """
    ## 2. 입력 데이터 준비

    Colab의 `/content/RegimeDecisionTest`를 프로젝트 폴더로 사용합니다. 원천자료가 없으면 파일 업로드 창이 열립니다. 함께 제공된 `regime_colab_inputs.zip`을 선택하세요.
    """
        ),
        code(
            """
    from pathlib import Path
    import zipfile

    PROJECT_ROOT = Path("/content/RegimeDecisionTest") if IN_COLAB else Path.cwd()
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "raw_data").mkdir(exist_ok=True)
    (PROJECT_ROOT / "cache").mkdir(exist_ok=True)
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)

    def inputs_ready(root: Path) -> bool:
        raw = root / "raw_data"
        return (
            (raw / "compass.db").exists()
            and (raw / "krx_bond_index.csv").exists()
            and len(list(raw.glob("*.xlsx"))) >= 5
            and len(list(raw.glob("*.csv"))) >= 2
            and (root / "cache" / "market_daily.csv").exists()
        )

    def safe_extract(archive_path: Path, destination: Path) -> None:
        destination = destination.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if target != destination and destination not in target.parents:
                    raise ValueError(f"안전하지 않은 ZIP 경로: {member.filename}")
            archive.extractall(destination)

    if not inputs_ready(PROJECT_ROOT):
        if not IN_COLAB:
            raise FileNotFoundError("raw_data/와 cache/market_daily.csv가 필요합니다.")
        from google.colab import files
        print("regime_colab_inputs.zip을 업로드하세요.")
        uploaded = files.upload()
        zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
        if len(zip_names) != 1:
            raise ValueError("입력 ZIP 파일 하나를 업로드해야 합니다.")
        safe_extract(Path(zip_names[0]), PROJECT_ROOT)

    if not inputs_ready(PROJECT_ROOT):
        raise FileNotFoundError("ZIP에서 필수 원천자료 또는 market_daily.csv를 찾지 못했습니다.")

    print("프로젝트 폴더:", PROJECT_ROOT)
    print("원천파일 수:", len(list((PROJECT_ROOT / "raw_data").iterdir())))
    print("시장 캐시:", PROJECT_ROOT / "cache" / "market_daily.csv")
    """,
            ["data-upload"],
        ),
        code(
            """
    import json
    import itertools
    from dataclasses import asdict

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from IPython.display import Markdown, display

    if IN_COLAB:
        import koreanize_matplotlib  # noqa: F401

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (13, 5)
    plt.rcParams["axes.unicode_minus"] = False
    if not IN_COLAB:
        plt.rcParams["font.family"] = "Malgun Gothic"

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 180)
    print("분석 환경 준비 완료")
    """
        ),
        markdown(
            """
    ## 3. 전략 전체 구현

    아래 셀은 `regime_research.py`를 그대로 포함합니다. 사용자 정의 모듈을 별도로 업로드할 필요가 없습니다.

    `거시 데이터 → Sparse Jump + 합성점수 → 네 국면 확률 → soft anchor → 비대칭 EWMA → SLSQP 위험제어 → 드로다운 가드 → 비용 차감 백테스트`
    """
        ),
        code(source, ["implementation"]),
        markdown(
            """
    ## 4. 데이터 로딩과 시점 감사

    - 성장: GDP, 수출, 제조업 BSI
    - 물가: CPI, PPI, 수입물가
    - 투자자산: KODEX200, 국내채권, GLD, USO
    - 신호월은 항상 투자 대상 월보다 앞서야 합니다.
    """
        ),
        code(
            """
    features, core = load_macro_data()
    asset_returns, asset_levels = load_monthly_asset_returns(refresh=False)
    signals = compute_regime_signals(features, asset_returns, jump_penalty=3.0, min_history=24)

    data_audit = pd.DataFrame({
        "Start": [features.index.min(), asset_returns.index.min().to_timestamp(), signals.index.min().to_timestamp()],
        "End": [features.index.max(), asset_returns.index.max().to_timestamp(), signals.index.max().to_timestamp()],
        "Observations": [len(features), len(asset_returns), len(signals)],
    }, index=["Macro features", "Common asset returns", "Tradable signals"])

    display(data_audit)
    display(asset_returns.describe().T[["mean", "std", "min", "max"]].style.format("{:.3%}"))
    """
        ),
        markdown(
            """
    ## 5. 경제 국면 확률

    성장·물가 각각의 고국면 확률을 만들고 네 결합확률을 계산합니다.

    | 국면 | 조건 | 자산 방향 |
    |---|---|---|
    | Goldilocks | 성장↑, 물가↓ | 주식 중심 |
    | Overheating | 성장↑, 물가↑ | 원유·금 확대 |
    | Slowdown | 성장↓, 물가↓ | 채권 중심 |
    | Stagflation | 성장↓, 물가↑ | 금 중심 |

    합성점수가 90%, Sparse Jump Model이 10%이며 직전 확률 15%를 섞어 신호를 평활합니다.
    """
        ),
        code(
            """
    regime_eval, regime_metrics = evaluate_regimes(signals, core)

    current_state = pd.DataFrame({
        "naive_growth": core[["GDP", "Export", "BSI"]].mean(axis=1) >= 0,
        "naive_inflation": core[["CPI", "PPI", "ImportPrice"]].mean(axis=1) >= 0,
    })
    regime_compare = regime_eval.join(current_state, how="left")
    naive_growth = balanced_accuracy_score(regime_compare["growth_high_realized"], regime_compare["naive_growth"])
    naive_inflation = balanced_accuracy_score(regime_compare["inflation_high_realized"], regime_compare["naive_inflation"])
    naive_quadrant = (
        (regime_compare["naive_growth"] == regime_compare["growth_high_realized"])
        & (regime_compare["naive_inflation"] == regime_compare["inflation_high_realized"])
    ).mean()

    accuracy_table = pd.DataFrame({
        "SJM-composite ensemble": [
            regime_metrics["growth_balanced_accuracy"],
            regime_metrics["inflation_balanced_accuracy"],
            regime_metrics["quadrant_accuracy"],
        ],
        "Naive persistence": [naive_growth, naive_inflation, naive_quadrant],
    }, index=["Growth balanced accuracy", "Inflation balanced accuracy", "4-quadrant accuracy"])
    display(accuracy_table.style.format("{:.1%}"))

    prob_plot = signals[["p_growth_high", "p_inflation_high"]].copy()
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
    display(latest_probs.sort_values(ascending=False).to_frame().style.format("{:.1%}"))
    """
        ),
        markdown(
            """
    ## 6. 36개 설정 보정 — `calibrate_configs.py` 통합

    보정 기간은 시작 시점부터 **2017-12**까지입니다. 다음 네 항목만 탐색합니다.

    - `regime_strength ∈ {0.25, 0.50, 0.75}`
    - `target_vol ∈ {0.08, 0.10, 0.12}`
    - `invvol_tilt ∈ {0.15, 0.35}`
    - `drawdown_guard ∈ {0.40, 0.75}`

    목적함수 계수 `1.15, 0.18, 0.25, 0.05, 0.32`와 CDaR 상한 16%는 탐색하지 않고 모든 후보에 고정합니다.

    `ValidationScore = Sharpe + 0.35×Calmar − 4×max(|MDD|−15%, 0) − 0.20×AvgTurnover`
    """
        ),
        code(
            """
    def score_metrics(metrics: pd.Series, turnover: float) -> float:
        breach = max(abs(float(metrics["MDD"])) - 0.15, 0.0)
        return float(metrics["Sharpe"] + 0.35 * metrics["Calmar"] - 4.0 * breach - 0.20 * turnover)

    calibration_rows = []
    calibration_configs = []
    candidate_values = list(itertools.product(
        [0.25, 0.50, 0.75],
        [0.08, 0.10, 0.12],
        [0.15, 0.35],
        [0.40, 0.75],
    ))

    for candidate_number, (regime_strength, target_vol, invvol_tilt, drawdown_guard) in enumerate(candidate_values, 1):
        candidate = StrategyConfig(
            name=f"rs{regime_strength}_tv{target_vol}_iv{invvol_tilt}_dg{drawdown_guard}",
            regime_strength=regime_strength,
            target_vol=target_vol,
            invvol_tilt=invvol_tilt,
            drawdown_guard=drawdown_guard,
            return_reward=1.15,
            vol_penalty=0.18,
            cdar_penalty=0.25,
            turnover_penalty=0.05,
            tracking_penalty=0.32,
            max_cdar=0.16,
        )
        calibration_bt = run_backtest(asset_returns, signals, candidate, mode="proposed", end="2017-12")
        metrics = performance_summary(calibration_bt["return"])
        avg_turnover = float(calibration_bt["turnover"].mean())
        calibration_rows.append({
            "name": candidate.name,
            **asdict(candidate),
            **metrics.to_dict(),
            "AvgTurnover": avg_turnover,
            "ValidationScore": score_metrics(metrics, avg_turnover),
        })
        calibration_configs.append(candidate)
        if candidate_number % 6 == 0:
            print(f"보정 진행: {candidate_number:02d}/{len(candidate_values)}")

    calibration_grid = (
        pd.DataFrame(calibration_rows)
        .sort_values("ValidationScore", ascending=False)
        .reset_index(drop=True)
    )
    calibration_grid.to_csv(RESULTS_DIR / "calibration_grid.csv", index=False)

    winner_name = calibration_grid.iloc[0]["name"]
    winner = next(candidate for candidate in calibration_configs if candidate.name == winner_name)
    cfg = StrategyConfig(**{**asdict(winner), "name": "Proposed"})

    display(Markdown(f"### 선택된 설정: `{winner_name}`"))
    display(calibration_grid[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "ValidationScore"]].head(10).style.format({
        "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}",
        "Calmar": "{:.3f}", "AvgTurnover": "{:.2%}", "ValidationScore": "{:.3f}",
    }))
    display(pd.Series(asdict(cfg), name="Locked value").to_frame())

    with (RESULTS_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(cfg), file, ensure_ascii=False, indent=2)
    """,
            ["calibration"],
        ),
        markdown(
            """
    ## 7. 전체기간 walk-forward 백테스트

    우승 설정을 잠근 뒤 proposed, soft, hard, equal weight, static defensive, KODEX200 buy-and-hold를 같은 기간과 비용 규칙으로 비교합니다.
    """
        ),
        code(
            """
    mode_labels = {
        "proposed": "Proposed: Regime + Risk control",
        "soft": "Soft regime only",
        "hard": "Original-style hard regime",
        "equal": "Equal weight",
        "static_defensive": "Static defensive",
        "kodex": "KODEX200 B&H",
    }

    backtests = {mode: run_backtest(asset_returns, signals, cfg, mode=mode) for mode in mode_labels}
    summary_by_mode = pd.DataFrame({mode: performance_summary(bt["return"]) for mode, bt in backtests.items()}).T
    summary = summary_by_mode.rename(index=mode_labels)

    summary_by_mode.to_csv(RESULTS_DIR / "summary.csv")
    signals.to_csv(RESULTS_DIR / "regime_signals.csv")
    backtests["proposed"].to_csv(RESULTS_DIR / "proposed_backtest.csv")

    display(summary.style.format({
        "Months": "{:.0f}", "CAGR": "{:.2%}", "Volatility": "{:.2%}",
        "Sharpe": "{:.3f}", "Sortino": "{:.3f}", "MDD": "{:.2%}",
        "Calmar": "{:.3f}", "FinalMultiple": "{:.2f}x", "PositiveMonths": "{:.1%}",
    }).highlight_max(subset=["Sharpe", "Calmar"], color="#d7f5dd"))
    """
        ),
        code(
            """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"proposed": "#0B63CE", "static_defensive": "#16A085", "kodex": "#7F8C8D", "hard": "#D35400"}
    for key in ["proposed", "static_defensive", "kodex", "hard"]:
        bt = backtests[key]
        idx = bt.index.to_timestamp()
        wealth = (1 + bt["return"]).cumprod()
        axes[0].plot(idx, wealth, label=mode_labels[key], color=colors[key], lw=2 if key == "proposed" else 1.2)
        if key in ["proposed", "kodex"]:
            drawdown = wealth / wealth.cummax() - 1
            axes[1].plot(idx, drawdown, label=mode_labels[key], color=colors[key], lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_title("누적자산 — 로그축, 거래·환전비용 차감")
    axes[0].set_ylabel("Growth of 1 KRW")
    axes[0].legend(ncol=2)
    axes[1].set_title("Drawdown")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    proposed = backtests["proposed"]
    weights = proposed[[f"w_{asset}" for asset in ASSETS]].copy()
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

    display(pd.concat([
        weights.mean().rename("Average weight"),
        weights.iloc[-1].rename("Latest weight"),
    ], axis=1).style.format("{:.1%}"))
    """
        ),
        markdown(
            """
    ## 8. 잠금 테스트와 강건성 점검

    2018-01 이후는 36개 후보의 우승 설정을 고를 때 점수에 포함하지 않은 잠금 구간입니다. 비용 0배·1배·2배, 주요 위기구간, 12개월 블록 부트스트랩도 함께 확인합니다.
    """
        ),
        code(
            """
    locked_backtests = {
        mode: run_backtest(asset_returns, signals, cfg, mode=mode, start="2018-01")
        for mode in mode_labels
    }
    locked_summary_by_mode = pd.DataFrame({
        mode: performance_summary(bt["return"]) for mode, bt in locked_backtests.items()
    }).T
    locked_summary = locked_summary_by_mode.rename(index=mode_labels)
    locked_summary_by_mode.to_csv(RESULTS_DIR / "locked_summary.csv")

    display(Markdown(f"### 2018-01~{signals.index[-1]} 잠금 구간"))
    display(locked_summary.style.format({
        "Months": "{:.0f}", "CAGR": "{:.2%}", "Volatility": "{:.2%}",
        "Sharpe": "{:.3f}", "Sortino": "{:.3f}", "MDD": "{:.2%}",
        "Calmar": "{:.3f}", "FinalMultiple": "{:.2f}x", "PositiveMonths": "{:.1%}",
    }))

    cost_rows = []
    for multiplier in [0.0, 1.0, 2.0]:
        scenario = run_backtest(asset_returns, signals, cfg, mode="proposed", cost_multiplier=multiplier)
        metrics = performance_summary(scenario["return"])
        cost_rows.append({"CostMultiplier": multiplier, **metrics.to_dict(), "AvgTurnover": scenario["turnover"].mean()})
    cost_sensitivity = pd.DataFrame(cost_rows).set_index("CostMultiplier")
    cost_sensitivity.to_csv(RESULTS_DIR / "cost_sensitivity.csv")

    display(Markdown("### 거래·환전비용 민감도"))
    display(cost_sensitivity[["CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].style.format({
        "CAGR": "{:.2%}", "Sharpe": "{:.3f}", "MDD": "{:.2%}",
        "Calmar": "{:.3f}", "AvgTurnover": "{:.2%}",
    }))
    """
        ),
        code(
            """
    crises = {
        "Global Financial Crisis": ("2007-10", "2009-03"),
        "COVID shock": ("2020-01", "2020-12"),
        "Inflation shock": ("2022-01", "2022-12"),
    }
    crisis_rows = []
    for episode, (start, end) in crises.items():
        lo, hi = pd.Period(start, "M"), pd.Period(end, "M")
        for key in ["proposed", "static_defensive", "kodex"]:
            episode_returns = backtests[key].loc[lo:hi, "return"]
            if len(episode_returns):
                wealth = (1 + episode_returns).cumprod()
                crisis_rows.append({
                    "Episode": episode,
                    "Strategy": mode_labels[key],
                    "CumulativeReturn": wealth.iloc[-1] - 1,
                    "EpisodeMDD": (wealth / wealth.cummax() - 1).min(),
                    "Volatility": episode_returns.std() * np.sqrt(12),
                })
    crisis_table = pd.DataFrame(crisis_rows)
    crisis_table.to_csv(RESULTS_DIR / "crisis_episodes.csv", index=False)

    display(Markdown("### 주요 위기 에피소드"))
    display(crisis_table.pivot(
        index="Episode", columns="Strategy", values=["CumulativeReturn", "EpisodeMDD"]
    ).style.format("{:.1%}"))
    """
        ),
        code(
            """
    def block_bootstrap_metrics(series, n_boot=1000, block=12, seed=42):
        values = np.asarray(series, dtype=float)
        rng = np.random.default_rng(seed)
        starts = np.arange(0, len(values) - block + 1)
        rows = []
        for _ in range(n_boot):
            sample = []
            while len(sample) < len(values):
                start = int(rng.choice(starts))
                sample.extend(values[start:start + block])
            metrics = performance_summary(pd.Series(sample[:len(values)]))
            rows.append([metrics["Sharpe"], metrics["MDD"], metrics["CAGR"]])
        return pd.DataFrame(rows, columns=["Sharpe", "MDD", "CAGR"])

    bootstrap = block_bootstrap_metrics(proposed["return"])
    bootstrap_ci = bootstrap.quantile([0.025, 0.50, 0.975]).rename(
        index={0.025: "2.5%", 0.50: "Median", 0.975: "97.5%"}
    )
    bootstrap_ci.to_csv(RESULTS_DIR / "bootstrap_ci.csv")

    display(Markdown("### 12개월 블록 부트스트랩"))
    display(bootstrap_ci.style.format({"Sharpe": "{:.3f}", "MDD": "{:.2%}", "CAGR": "{:.2%}"}))
    display(Markdown("블록 부트스트랩은 월별 의존성을 일부 보존한 표본 불확실성 점검이며 미래 성과 예측구간이 아닙니다."))
    """
        ),
        markdown(
            """
    ## 9. 자동 검증과 결과 파일

    시점, 설정 수, 비중, 비용, 잠금구간 기준을 자동으로 확인합니다. 검증을 통과하면 주요 결과 CSV가 `results/`에 남습니다.
    """
        ),
        code(
            """
    assert len(calibration_grid) == 36
    assert winner_name == calibration_grid.iloc[0]["name"]
    assert signals.index.is_monotonic_increasing and signals.index.is_unique
    assert asset_returns.index.is_monotonic_increasing and asset_returns.index.is_unique
    assert (signals["signal_month"] < signals.index).all(), "신호월은 투자월보다 반드시 앞서야 함"
    assert np.isfinite(proposed["return"]).all()
    assert (weights.sum(axis=1).sub(1).abs() < 1e-8).all()
    assert (weights >= -1e-12).all().all()
    assert (proposed[["trade_cost", "fx_cost"]] >= 0).all().all()
    assert summary_by_mode.loc["proposed", "MDD"] > -0.20
    assert locked_summary_by_mode.loc["proposed", "Sharpe"] > 1.0

    expected_outputs = [
        "calibration_grid.csv", "config.json", "summary.csv", "locked_summary.csv",
        "regime_signals.csv", "proposed_backtest.csv", "cost_sensitivity.csv",
        "crisis_episodes.csv", "bootstrap_ci.csv",
    ]
    for filename in expected_outputs:
        assert (RESULTS_DIR / filename).exists(), filename

    print("모든 시점·설정·비중·비용·잠금구간 검증을 통과했습니다.")
    print("선택 설정:", winner_name)
    print("전체기간 Sharpe:", round(float(summary_by_mode.loc["proposed", "Sharpe"]), 6))
    print("잠금구간 Sharpe:", round(float(locked_summary_by_mode.loc["proposed", "Sharpe"]), 6))
    print("결과 폴더:", RESULTS_DIR)
    """,
            ["validation"],
        ),
        markdown(
            """
    ## 10. 해석과 한계

    ### 현재 설계가 의미하는 것

    - 경제 국면은 최종 비중의 방향을 제시하고, 비대칭 EWMA·변동성/CDaR 제약·회전율·tracking·드로다운 가드가 경로위험을 줄입니다.
    - `calibration_grid.csv`는 외부 입력이 아니라 이 노트북의 보정 셀이 직접 생성합니다.
    - 목적함수의 다섯 계수는 36개 그리드에서 선택된 값이 아니라 고정 휴리스틱입니다.

    ### 반드시 남는 한계

    - 거시자료는 현재 개정치이므로 진정한 point-in-time 빈티지 백테스트가 아닙니다.
    - 2009-03 이전 KODEX200은 KOSPI200 프록시입니다.
    - 실제 호가 스프레드, 시장충격, 세금, 추적오차는 완전히 반영되지 않습니다.
    - 한 번의 보정/잠금 분할은 선택편향을 완전히 제거하지 못합니다.
    - 역사적 Sharpe와 MDD는 미래 실거래 성과를 보장하지 않습니다.

    ### 참고문헌

    - Bemporad et al. (2018), [Fitting Jump Models](https://web.stanford.edu/~boyd/papers/fitting_jump_models.html)
    - Nystrup, Kolm & Lindström (2021), [Feature Selection in Jump Models](https://doi.org/10.1016/j.eswa.2021.115558)
    - Nystrup et al. (2019), [Multi-period Portfolio Selection with Drawdown Control](https://web.stanford.edu/~boyd/papers/pdf/multiperiod_portfolio_drawdown.pdf)
    - Moreira & Muir (2017), [Volatility-Managed Portfolios](https://doi.org/10.1111/jofi.12513)
    - Chekhlov, Uryasev & Zabarankin (2005), [Drawdown Measure in Portfolio Optimization](https://doi.org/10.1142/S0219024905002767)
    """
        ),
    ]

    nbf.validate(nb)
    nbf.write(nb, NOTEBOOK_PATH)
    build_input_bundle()

    print(NOTEBOOK_PATH)
    print(BUNDLE_PATH)


if __name__ == "__main__":
    main()
