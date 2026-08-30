from __future__ import annotations

import ast
import hashlib
import json
import textwrap
import unicodedata
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_OUTPUT = ROOT / "artifacts/notebooks/robust_vkospi_reference_single_colab.ipynb"
DATA_OUTPUT = ROOT / "artifacts/bundles/robust_vkospi_colab_data.zip"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip() + "\n",
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip() + "\n",
    }


def normalized_child(directory: Path, filename: str) -> Path:
    target = unicodedata.normalize("NFC", filename)
    for path in directory.iterdir():
        if unicodedata.normalize("NFC", path.name) == target:
            return path
    raise FileNotFoundError(directory / filename)


def selected_source(relative_path: str, names: list[str]) -> str:
    """Copy complete top-level definitions into visible notebook cells."""
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    found: dict[str, str] = {}
    for node in tree.body:
        name = getattr(node, "name", None)
        if name not in names:
            continue
        start = node.lineno
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, *(decorator.lineno for decorator in decorators))
        found[name] = "\n".join(lines[start - 1 : node.end_lineno])
    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError(f"{relative_path}: missing definitions {missing}")
    return "\n\n\n".join(found[name] for name in names)


def write_data_bundle() -> dict[str, object]:
    inputs = {
        ROOT / "raw_data/compass.db": "RegimeDecisionData/raw_data/compass.db",
        ROOT / "raw_data/krx_bond_index.csv": "RegimeDecisionData/raw_data/krx_bond_index.csv",
        ROOT / "raw_data/VKOSPIData.csv": "RegimeDecisionData/raw_data/VKOSPIData.csv",
        ROOT / "cache/market_daily.csv": "RegimeDecisionData/cache/market_daily.csv",
        ROOT / "results/openassetpricing_composites.csv": (
            "RegimeDecisionData/input_data/openassetpricing_composites.csv"
        ),
    }
    for filename in [
        "GDP 성장률.xlsx",
        "기업경기조사(전망).csv",
        "수출입 총괄_20260816.xlsx",
        "생산자물가 상승률.xlsx",
        "소비자물가 상승률.xlsx",
        "수출입물가 상승률.xlsx",
    ]:
        source = normalized_child(ROOT / "raw_data", filename)
        inputs[source] = f"RegimeDecisionData/raw_data/{source.name}"

    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, archive_name in inputs.items():
            archive.write(source, archive_name)
    payload = DATA_OUTPUT.read_bytes()
    return {
        "filename": DATA_OUTPUT.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "compressed_bytes": len(payload),
        "file_count": len(inputs),
        "members": sorted(inputs.values()),
    }


def extracted_modules() -> dict[str, str]:
    modules: dict[str, str] = {}
    modules["loaders"] = selected_source(
        "strategies/core/regime_research.py",
        [
            "get_path",
            "rolling_zscore",
            "load_macro_data",
            "download_market_cache",
            "load_monthly_asset_returns",
        ],
    )
    modules["allocation"] = selected_source(
        "strategies/core/regime_research.py",
        [
            "soft_anchor",
            "ewma_cov",
            "cdar",
            "StrategyConfig",
            "controlled_weights",
            "hard_regime_weights",
            "run_backtest",
            "performance_summary",
        ],
    )
    modules["macro"] = "\n\n\n".join(
        [
            selected_source(
                "strategies/stage06_vkospi/vkospi_extended_diagnostics.py",
                ["_macro_probabilities"],
            ),
            selected_source(
                "strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py",
                ["build_no_sjm_components"],
            ),
            selected_source(
                "strategies/stage06_vkospi/vkospi_model_robustness.py",
                ["build_macro_signals"],
            ),
        ]
    )
    modules["domestic"] = selected_source(
        "strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py",
        [
            "_daily_asset_returns",
            "build_domestic_features",
            "run_neutral_factor_blend",
            "forward_path_loss",
            "balanced_logistic_spec",
        ],
    )
    modules["tail"] = selected_source(
        "strategies/stage06_vkospi/vkospi_model_robustness.py",
        [
            "causal_percentile",
            "_transfer",
            "apply_factor_tilt",
            "run_factor_vol_target",
            "make_logistic_model",
            "fit_logistic_candidate",
            "make_tail_factor",
        ],
    )
    modules["tail"] = modules["tail"].replace(
        "def causal_percentile(", "def tail_causal_percentile("
    ).replace(
        'causal_percentile(output["p_tail_raw"])',
        'tail_causal_percentile(output["p_tail_raw"])',
    )
    modules["daily"] = selected_source(
        "strategies/stage06_vkospi/vkospi_dynamic_risk_experiment.py",
        [
            "DynamicRiskConfig",
            "load_vkospi_daily",
            "load_daily_open_levels",
            "causal_percentile",
            "build_daily_vkospi_signals",
            "prepare_arrays",
            "simulate",
            "reconcile_to_monthly_reference",
        ],
    )
    modules["daily"] = modules["daily"].replace(
        "def causal_percentile(", "def daily_causal_percentile("
    ).replace(
        'output["level_percentile"] = causal_percentile(close)',
        'output["level_percentile"] = daily_causal_percentile(close)',
    )
    modules["robust"] = selected_source(
        "strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py",
        [
            "RobustStressConfig",
            "robust_zscore",
            "build_robust_daily_features",
            "align_features_to_arrays",
            "stress_from_features",
        ],
    ).replace("causal_percentile(", "daily_causal_percentile(")
    return modules


def build_notebook(bundle: dict[str, object]) -> dict[str, object]:
    module = extracted_modules()
    cells: list[dict[str, object]] = [
        markdown(
            f"""
            # Robust VKOSPI 기준 전략 — 실행코드 공개형 Google Colab

            이 노트북은 '무SJM 거시 국면 + 균형 L2 로지스틱 + Robust VKOSPI 일간
            오버레이' 기준 전략을 **압축된 코드나 프로젝트 모듈 없이** 재현합니다.
            각 함수와 계산식은 아래 코드 셀에 그대로 보이며 단계별로 나눴습니다.

            실행은 노트북을 Colab에서 연 뒤 '런타임 → 모두 실행'을 누르고,
            업로드 창에서 {bundle["filename"]} 하나를 선택하면 됩니다.
            데이터 ZIP에는 원자료와 OAP 가공 입력만 있으며 Python 코드나 백테스트
            결과는 들어 있지 않습니다. 결과는 연구용이며 투자 조언이 아닙니다.
            """
        ),
        markdown(
            """
            ## 0. 계산 순서

            데이터 적재 → 무SJM 거시확률 → SLSQP 기본배분 → 16개 설명변수
            → 균형 L2 로지스틱 → 꼬리위험 틸트·변동성 타깃
            → Robust VKOSPI 일간 오버레이 → 성과·인과성 검증 순서입니다.
            """
        ),
        markdown("## 1. 데이터 ZIP 업로드와 실행환경 준비"),
        code(
            f'''
            import json
            import math
            import shutil
            import sqlite3
            import subprocess
            import sys
            import tempfile
            import unicodedata
            import warnings
            import zipfile
            from dataclasses import asdict, dataclass
            from pathlib import Path

            IN_COLAB = "google.colab" in sys.modules
            if IN_COLAB:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "openpyxl>=3.1"],
                    check=True,
                )

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from scipy.optimize import minimize
            from scipy.special import expit
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                average_precision_score, brier_score_loss, roc_auc_score
            )
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            EXPECTED_DATA_ZIP = "{bundle["filename"]}"


            def safe_extract(archive_path: Path, destination: Path) -> None:
                destination = destination.resolve()
                with zipfile.ZipFile(archive_path) as archive:
                    for member in archive.infolist():
                        target = (destination / member.filename).resolve()
                        if target != destination and destination not in target.parents:
                            raise ValueError(f"안전하지 않은 ZIP 경로: {{member.filename}}")
                    archive.extractall(destination)


            if IN_COLAB:
                from google.colab import files

                uploaded = files.upload()
                zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
                if len(zip_names) != 1:
                    raise ValueError("데이터 ZIP 파일 하나만 업로드해 주세요.")
                DATA_ARCHIVE = Path(zip_names[0])
                RUNTIME_PARENT = Path("/content/robust_vkospi_runtime")
                if RUNTIME_PARENT.exists():
                    shutil.rmtree(RUNTIME_PARENT)
                RUNTIME_PARENT.mkdir(parents=True)
            else:
                DATA_ARCHIVE = Path(
                    globals().get(
                        "DATA_ARCHIVE_OVERRIDE",
                        Path.cwd() / "artifacts/bundles" / EXPECTED_DATA_ZIP,
                    )
                )
                RUNTIME_PARENT = Path(tempfile.mkdtemp(prefix="robust_vkospi_visible_"))

            safe_extract(DATA_ARCHIVE, RUNTIME_PARENT)
            ROOT = RUNTIME_PARENT / "RegimeDecisionData"
            RAW_DIR = ROOT / "raw_data"
            CACHE_DIR = ROOT / "cache"
            INPUT_DIR = ROOT / "input_data"
            OUTPUT_DIR = ROOT / "outputs"
            VKOSPI_PATH = RAW_DIR / "VKOSPIData.csv"
            OUTPUT_DIR.mkdir(exist_ok=True)
            print("데이터:", DATA_ARCHIVE)
            print("실행 루트:", ROOT)
            print("Python:", sys.version.split()[0])
            '''
        ),
        markdown(
            """
            ## 2. 공통 설정과 입력변수 16개

            국내 포트폴리오 상태 12개와 Open Asset Pricing 기반 스트레스 합성치
            4개를 월간 로지스틱 입력으로 사용합니다. VKOSPI 변수는 월간 로지스틱에
            섞지 않고 마지막 일간 오버레이에서 독립적으로 사용합니다.
            """
        ),
        code(
            '''
            ASSETS = ["KODEX200", "BOND", "GLD", "USO"]
            DOMESTIC_FEATURES = [
                "base_USO", "base_GLD", "base_KODEX200", "p_inflation_high",
                "proxy_mom1", "proxy_mom6", "proxy_vol6",
                "daily_mom21", "daily_mom252", "daily_vol21",
                "daily_downvol21", "daily_mean_corr63",
            ]
            OAP_COMPOSITES = [
                "oap_momentum_trend_stress",
                "oap_reversal_crowding_stress",
                "oap_low_risk_tail_stress",
                "oap_liquidity_activity_stress",
            ]
            TAIL_FEATURES = DOMESTIC_FEATURES + OAP_COMPOSITES
            RNG_SEED = 20260828

            REGIME_ANCHORS = pd.DataFrame(
                {
                    "Goldilocks": [0.58, 0.22, 0.15, 0.05],
                    "Overheating": [0.30, 0.12, 0.23, 0.35],
                    "Slowdown": [0.12, 0.66, 0.20, 0.02],
                    "Stagflation": [0.08, 0.24, 0.50, 0.18],
                },
                index=ASSETS,
            ).T
            DEFENSIVE = np.array([0.05, 0.72, 0.23, 0.00])
            STRATEGIC = np.array([0.20, 0.45, 0.30, 0.05])


            @dataclass(frozen=True)
            class NotebookConfig:
                hard_weight: float = 0.40
                slsqp_weight: float = 0.60
                initial_leverage: float = 1.20
                tail_horizon_months: int = 2
                tail_loss_threshold: float = -0.05
                logistic_c: float = 0.10
                maximum_tail_shift: float = 0.20
                target_volatility: float = 0.15
                calibration_end: str = "2017-12"
                locked_start: str = "2018-01"


            CONFIG = NotebookConfig()
            display(pd.Series(asdict(CONFIG), name="value").to_frame())
            display(pd.DataFrame({"input_variable": TAIL_FEATURES}))
            '''
        ),
        markdown(
            """
            ## 3. 모듈 A — 원자료 적재

            한국어 파일명은 Unicode 정규화로 찾습니다. KODEX200은 초기 KOSPI200
            프록시와 실제 ETF를 이어 붙이고, GLD·USO는 USDKRW를 곱해 원화
            수익률로 바꿉니다.
            """
        ),
        code(module["loaders"]),
        code(
            '''
            monthly_returns, monthly_levels = load_monthly_asset_returns(False)
            macro_features, macro_core = load_macro_data()
            print("월별 수익률:", monthly_returns.index.min(), "→", monthly_returns.index.max())
            print("거시 특징:", macro_features.index.min().date(), "→", macro_features.index.max().date())
            display(monthly_returns.head())
            '''
        ),
        markdown(
            """
            ## 4. 모듈 B — SLSQP 위험제어 배분

            합계 100%, 자산별 상·하한, 목표 변동성, CDaR 제약을 두고 기대수익,
            변동성, 꼬리손실, 회전율, 국면 기준점 이탈을 목적함수에서 절충합니다.
            아래 셀에 목적함수와 제약식 전체가 있습니다.
            """
        ),
        code(module["allocation"]),
        markdown(
            """
            ## 5. 모듈 C — 무SJM 거시국면

            성장축은 GDP·수출·BSI, 물가축은 CPI·PPI·수입물가의 표준화 수준과
            3개월 변화를 사용합니다. SJM 가중치는 0입니다. 현재 확률 85%와
            직전 확률 15%를 섞어 월간 급변을 완화합니다.
            """
        ),
        code(module["macro"]),
        code(
            '''
            def build_no_sjm_signals(returns: pd.DataFrame):
                components = build_no_sjm_components(returns)
                probabilities = _macro_probabilities(
                    components, d3_weight=0.20, sigmoid_scale=0.55,
                    sjm_weight=0.0, current_weight=0.85,
                )
                signals = build_macro_signals(probabilities, components)
                assert (signals["signal_month"] < signals.index).all()
                assert np.isfinite(probabilities.to_numpy(dtype=float)).all()
                return signals, probabilities


            signals, macro_probabilities = build_no_sjm_signals(monthly_returns)
            defensive = run_backtest(
                monthly_returns, signals, StrategyConfig(), mode="proposed"
            )
            print("첫 거시 예측 대상월:", signals.index.min())
            display(signals.head())
            '''
        ),
    ]
    cells.extend(
        [
            markdown(
                """
                ## 6. 모듈 D — 16개 입력과 꼬리사건 라벨

                40% hard 국면과 60% SLSQP를 섞고 1.2배 노출을 적용한 중립경로를
                만듭니다. 그 경로의 앞으로 2개월 진행 중 누적손실이 -5%보다 작으면
                꼬리사건입니다. 마지막 두 달은 미래가 완성되지 않아 결측입니다.
                """
            ),
            code(module["domestic"]),
            code(
                '''
                domestic_features = build_domestic_features(signals, monthly_returns)
                oap = pd.read_csv(
                    INPUT_DIR / "openassetpricing_composites.csv", index_col=0
                )
                oap.index = pd.PeriodIndex(oap.index, freq="M")

                neutral_baseline = run_neutral_factor_blend(
                    monthly_returns, signals, defensive
                )
                model_data = domestic_features[DOMESTIC_FEATURES].join(
                    oap[OAP_COMPOSITES], how="left"
                )
                model_data = model_data.loc[
                    model_data.index.intersection(neutral_baseline.index)
                ].copy()
                path_loss = forward_path_loss(
                    neutral_baseline.loc[model_data.index, "return"],
                    horizon=CONFIG.tail_horizon_months,
                )
                model_data["tail_event"] = (
                    path_loss < CONFIG.tail_loss_threshold
                ).where(path_loss.notna()).astype(float)

                print("학습 패널:", model_data.shape)
                print("꼬리사건 수:", int(model_data["tail_event"].sum()))
                display(model_data.tail())
                '''
            ),
            markdown(
                """
                ## 7. 모듈 E — 균형 L2 로지스틱 워크포워드

                매월 모델을 새로 학습합니다. 해당 월에서 두 달을 비우는 embargo를
                두고 그보다 과거만 사용합니다. 최소 36개 행, 양성 4개, 음성 12개가
                모이기 전에는 확률을 만들지 않습니다. 결측치 중앙값 대체와 표준화도
                매 시점의 학습표본에서 다시 적합합니다. class_weight는 balanced,
                L2 규제 C는 0.1, solver는 liblinear입니다.
                """
            ),
            code(module["tail"]),
            code(
                '''
                probability, fit_stats = fit_logistic_candidate(
                    model_data, balanced_logistic_spec()
                )
                factor = make_tail_factor(probability, model_data["tail_event"])

                prediction_view = factor.dropna(
                    subset=["p_tail_raw", "tail_event"]
                )
                y = prediction_view["tail_event"].astype(int)
                p = prediction_view["p_tail_raw"].clip(1e-6, 1 - 1e-6)
                prediction_scores = pd.Series(
                    {
                        "observations": len(prediction_view),
                        "events": int(y.sum()),
                        "ROC_AUC": roc_auc_score(y, p),
                        "AveragePrecision": average_precision_score(y, p),
                        "BrierScore": brier_score_loss(y, p),
                        **fit_stats,
                    }
                )
                print("첫 유효 로지스틱 예측월:", factor["p_tail_raw"].first_valid_index())
                display(prediction_scores.to_frame("value"))
                display(factor.tail(12))
                '''
            ),
            markdown(
                """
                ## 8. 모듈 F — 꼬리위험 틸트와 15% 변동성 타깃

                최근 60개 유효 확률 안에서 현재 확률의 인과적 백분위를 계산합니다.
                80백분위 이하에서는 이동하지 않고 80~100백분위에서 위험강도가
                0에서 1로 커집니다. 최대 20%를 주식·원유에서 채권·금으로 옮긴 뒤,
                최근 24개월 가중 변동성으로 총 노출을 0.5~1.5배에서 조정합니다.
                """
            ),
            code(
                '''
                medium = run_factor_vol_target(
                    monthly_returns,
                    signals,
                    defensive,
                    factor,
                    max_shift=CONFIG.maximum_tail_shift,
                    target_vol=CONFIG.target_volatility,
                )
                display(medium[[
                    "return", "leverage", "risk_score",
                    "w_KODEX200", "w_BOND", "w_GLD", "w_USO",
                ]].tail())
                '''
            ),
            markdown(
                """
                ## 9. 모듈 G — 일간 자산·VKOSPI 정렬과 실행

                월간 기준비중을 각 영업일에 펼치고 다음 영업일 시가 수익률에
                적용합니다. VKOSPI 신호는 행동일 전일까지 잘라 사용합니다. 월초에는
                리밸런싱하고 월중 목표와 현재 비중 차이의 절반합이 20% 이상일 때만
                거래해 회전율을 억제합니다.
                """
            ),
            code(module["daily"]),
            markdown(
                """
                ## 10. 모듈 H — Robust VKOSPI 특징과 스트레스

                VKOSPI 수준은 126·252일 인과적 백분위와 median/MAD z-score,
                충격은 5·10·21일 로그변화를 직전 63일 변동성으로 나눠 만듭니다.
                배포 설정 acceleration은 수준 40%, 5일 충격 35%, 5일 가속도
                25%입니다. 최대 35%의 주식·원유 비중을 줄여 전부 금으로 옮깁니다.
                """
            ),
            code(module["robust"]),
            code(
                '''
                ROBUST_CONFIG = RobustStressConfig(
                    mode="acceleration",
                    level_threshold=0.90,
                    shock_threshold=1.00,
                    max_risk_transfer=0.35,
                    bond_share=0.00,
                    rebalance_band=0.20,
                    financing_rate=0.04,
                )


                def run_robust_vkospi_overlay(reference: pd.DataFrame):
                    levels = load_daily_open_levels()
                    arrays = prepare_arrays(
                        levels, reference, build_daily_vkospi_signals()
                    )
                    features = align_features_to_arrays(
                        build_robust_daily_features(), arrays
                    )
                    stress = stress_from_features(
                        features,
                        ROBUST_CONFIG.mode,
                        ROBUST_CONFIG.level_threshold,
                        ROBUST_CONFIG.shock_threshold,
                    )
                    _, neutral_monthly = simulate(
                        arrays, None, keep_daily=False
                    )
                    daily, overlay_monthly = simulate(
                        arrays,
                        ROBUST_CONFIG.dynamic_config(),
                        keep_daily=True,
                        stress_override=stress,
                    )
                    reconciled = reconcile_to_monthly_reference(
                        reference, neutral_monthly, overlay_monthly
                    )
                    valid = daily["signal_date"].notna()
                    assert (
                        daily.index[valid].to_numpy()
                        > pd.DatetimeIndex(
                            daily.loc[valid, "signal_date"]
                        ).to_numpy()
                    ).all()
                    return daily, overlay_monthly, reconciled


                final_daily, overlay_monthly, final = (
                    run_robust_vkospi_overlay(medium)
                )
                print(
                    "최종 월별 경로:",
                    final.index.min(), "→", final.index.max(), len(final)
                )
                display(pd.Series(asdict(ROBUST_CONFIG), name="value").to_frame())
                '''
            ),
        ]
    )
    cells.extend(
        [
            markdown(
                """
                ## 11. 성과와 누수 방지 검증

                전체기간과 2018년 이후 잠금구간을 따로 봅니다. 기대값은 결과파일에서
                읽는 값이 아니라 공개형 계산이 기준전략과 같은지 검사하는 회귀 테스트
                상수입니다. 데이터나 로직이 바뀌면 검증은 의도적으로 실패합니다.
                """
            ),
            code(
                '''
                def period_metrics(path: pd.DataFrame, start: str | None = None):
                    view = (
                        path
                        if start is None
                        else path.loc[pd.Period(start, "M"):]
                    )
                    return performance_summary(view["return"])


                metrics = pd.concat(
                    {
                        "Full_2007_2026": period_metrics(final),
                        "Locked_2018_2026": period_metrics(
                            final, CONFIG.locked_start
                        ),
                    },
                    axis=1,
                ).T
                display(metrics[[
                    "Months", "CAGR", "Sharpe", "MDD",
                    "Calmar", "FinalMultiple",
                ]])

                EXPECTED = {
                    "Full_2007_2026": {
                        "CAGR": 0.15636553973863565,
                        "Sharpe": 1.1325286822772858,
                        "MDD": -0.12958799769553853,
                    },
                    "Locked_2018_2026": {
                        "CAGR": 0.20907823332988906,
                        "Sharpe": 1.4969186464577038,
                        "MDD": -0.09760421330851343,
                    },
                }
                for period, expected in EXPECTED.items():
                    for metric, value in expected.items():
                        actual = float(metrics.loc[period, metric])
                        assert np.isclose(
                            actual, value, atol=5e-11, rtol=0
                        ), (period, metric, actual, value)

                assert (signals["signal_month"] < signals.index).all()
                assert final.index.equals(medium.index)
                assert final["return"].notna().all()
                print(
                    "PASS: 기준 성과, 월간 정렬, 거시 1개월 시차, "
                    "VKOSPI 전일 시차 확인"
                )
                '''
            ),
            code(
                '''
                wealth = pd.DataFrame(
                    {
                        "SLSQP base": (1 + defensive["return"]).cumprod(),
                        "Tail + vol target": (1 + medium["return"]).cumprod(),
                        "Robust VKOSPI final": (1 + final["return"]).cumprod(),
                    }
                )
                ax = wealth.plot(figsize=(12, 5), logy=True, grid=True)
                ax.set_title("Robust VKOSPI reference strategy")
                ax.set_ylabel("Wealth multiple (log scale)")
                plt.show()
                '''
            ),
            markdown(
                """
                ## 12. 결과 내보내기

                월별 수익률·비중, 거시확률, 16개 입력패널, 로지스틱 확률,
                일간 VKOSPI 거래경로와 성과표를 CSV/JSON으로 저장하고 ZIP으로
                묶습니다.
                """
            ),
            code(
                '''
                signals.to_csv(OUTPUT_DIR / "macro_signals.csv")
                macro_probabilities.to_csv(
                    OUTPUT_DIR / "macro_probabilities.csv"
                )
                model_data.to_csv(OUTPUT_DIR / "logistic_input_panel.csv")
                factor.to_csv(OUTPUT_DIR / "tail_factor.csv")
                medium.to_csv(OUTPUT_DIR / "medium_monthly.csv")
                final.to_csv(OUTPUT_DIR / "final_reconciled_monthly.csv")
                final_daily.to_csv(OUTPUT_DIR / "robust_vkospi_daily.csv")
                metrics.to_csv(OUTPUT_DIR / "performance_summary.csv")
                (OUTPUT_DIR / "run_config.json").write_text(
                    json.dumps(
                        {
                            "notebook": asdict(CONFIG),
                            "robust_vkospi": asdict(ROBUST_CONFIG),
                            "tail_features": TAIL_FEATURES,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                archive = shutil.make_archive(
                    str(ROOT / "robust_vkospi_results"), "zip", OUTPUT_DIR
                )
                print("저장 완료:", archive)
                if IN_COLAB:
                    files.download(archive)
                '''
            ),
            markdown(
                """
                ## 해석할 때 주의할 점

                이 노트북은 재현용 기준전략입니다. 2018년 이후 구간도 전략 연구
                과정에서 이미 관찰됐으므로 완전히 새로운 미사용 표본으로 해석하면
                안 됩니다. 거래비용은 구현값을 반영하지만 세금, 추적오차, 실제
                호가충격은 별도입니다. OAP 4개 합성치는 ZIP에 이미 월별 입력으로
                들어 있으며 원 논문의 개별 신호를 다시 내려받는 구조는 아닙니다.
                """
            ),
        ]
    )
    for number, cell in enumerate(cells, start=1):
        cell["id"] = f"robust-vkospi-{number:02d}"
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": NOTEBOOK_OUTPUT.name, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    bundle = write_data_bundle()
    notebook = build_notebook(bundle)
    NOTEBOOK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "notebook": str(NOTEBOOK_OUTPUT),
                "notebook_bytes": NOTEBOOK_OUTPUT.stat().st_size,
                "data_bundle": str(DATA_OUTPUT),
                **bundle,
                "code_cells": sum(
                    cell["cell_type"] == "code" for cell in notebook["cells"]
                ),
                "markdown_cells": sum(
                    cell["cell_type"] == "markdown" for cell in notebook["cells"]
                ),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
