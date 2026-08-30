from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyStage:
    id: str
    title: str
    objective: str
    module: str
    status: str
    outcome: str
    report: str | None = None
    builder: str | None = None


STAGES = (
    StrategyStage(
        id="01",
        title="거시 국면 베이스라인",
        objective="성장·물가 국면, SJM, SLSQP 방어배분의 기준선 구축",
        module="strategies.core.regime_research",
        status="baseline",
        outcome="전체 232개월 Proposed: CAGR 8.12%, Sharpe 1.144, MDD -8.93%",
        report="artifacts/reports/economic_regime_implementation_guide.html",
        builder="tools.builders.build_implementation_guide",
    ),
    StrategyStage(
        id="02",
        title="CAGR·Hard 배분 강화",
        objective="목표변동성, 모멘텀, Hard 국면, 레버리지로 CAGR을 높이는 탐색",
        module="strategies.stage02_return_enhancement.cagr_accelerator_experiment",
        status="superseded",
        outcome="수익 확대 아이디어를 검증한 중간 단계; 이후 MDD 15% 제약 전략으로 통합",
        report="artifacts/reports/economic_regime_allocation_cagr_enhanced.html",
        builder="tools.builders.build_cagr_enhanced_notebook",
    ),
    StrategyStage(
        id="03",
        title="꼬리위험·MDD 15% 제약",
        objective="Crash feature, 일별 guard, stop-loss, synthetic put, blend를 비교",
        module="strategies.stage03_tail_risk.validate_final_blend",
        status="milestone",
        outcome="Final blend 전체: CAGR 14.30%, Sharpe 1.024, MDD -14.90%; 네 검증 gate 통과",
        report="artifacts/reports/economic_regime_allocation_four_asset_mdd15.html",
        builder="tools.builders.build_final_blend_notebook",
    ),
    StrategyStage(
        id="04",
        title="LightGBM·피드백·시장구조",
        objective="거시 확률을 ML·tail meta·시장구조 특징으로 보완",
        module="strategies.stage04_ml_feedback.regime_lightgbm_factor_experiment",
        status="not_promoted",
        outcome="LightGBM 최종 max_shift=0; 잠금 성과 개선 gate를 통과하지 못해 기준 경로 유지",
        report="artifacts/reports/regime_strategy_sharpe_1_1_explainer.html",
    ),
    StrategyStage(
        id="05",
        title="Open Asset Pricing 입력",
        objective="OAP 아이디어를 한국 시장 시계열 합성변수로 변환",
        module="strategies.stage05_openassetpricing.openassetpricing_signal_experiment",
        status="feature_adopted",
        outcome="16변수 균형 L2 로지스틱의 OAP 합성 4개 입력으로 후속 전략에 편입",
        builder="tools.builders.build_openassetpricing_colab_notebook",
    ),
    StrategyStage(
        id="06",
        title="VKOSPI·Robust VKOSPI",
        objective="한국판 VIX의 수준·충격·가속으로 일별 위험예산을 조절",
        module="strategies.stage06_vkospi.balanced_logistic_no_sjm_strategy",
        status="current_reference",
        outcome="현재 비교 기준 전체: CAGR 15.64%, Sharpe 1.133, MDD -12.96%; 2018+ Sharpe 1.497",
        report="artifacts/reports/vkospi_robust_dynamic_technical_report.html",
        builder="tools.builders.build_vkospi_robust_dynamic_technical_report",
    ),
    StrategyStage(
        id="07",
        title="CJM·TVTP-HMM·CJM+LightGBM",
        objective="Hard 전환을 연속확률 국면모델로 대체할 수 있는지 비교",
        module="strategies.stage07_regime_models.top3_regime_model_experiment",
        status="not_promoted",
        outcome="세 후보 모두 기준 대비 CAGR·Sharpe·MDD 동시 개선에 실패",
        report="artifacts/reports/top3_regime_model_report.html",
        builder="tools.builders.build_top3_regime_model_deliverables",
    ),
    StrategyStage(
        id="08",
        title="VIX6·KOSPI200 옵션",
        objective="VIX6 decomposition으로 옵션 매수·청산·비중을 결정",
        module="strategies.stage08_options.option_asset_slippage_experiment",
        status="not_promoted",
        outcome="Base·보수적 슬리피지 모두에서 세 지표를 개선한 옵션 비중 0개; 옵션 비중 0% 유지",
        report="artifacts/reports/vix6_case1_strategy_report.html",
        builder="tools.builders.build_vix6_case1_html",
    ),
    StrategyStage(
        id="09",
        title="노트북 히스테리시스·레버리지 상한",
        objective="±0.2 히스테리시스와 상한 1.0~1.3을 Hard 40% 경로에 결합",
        module="strategies.stage09_hysteresis.hysteresis_hard40_leverage_experiment",
        status="research_only",
        outcome="상한 1.0은 전체 Sharpe·MDD 개선, CAGR 하락; 엄격 사전 gate 실패로 현재 전략 유지",
        report="artifacts/reports/hysteresis_hard40_leverage_report.html",
        builder="tools.builders.build_hysteresis_hard40_leverage_deliverables",
    ),
    StrategyStage(
        id="10",
        title="VIX6 조건부 위기 라우터",
        objective="Macro→위험예산→위기유형→4자산·옵션 실행의 계층형 구조를 검증",
        module="strategies.stage10_vix6_router.vix6_conditional_router_strategy",
        status="research_only",
        outcome=(
            "24개 사전 후보와 옵션 구조 모두 채택 gate 실패; 안전 폴백으로 현재 "
            "Robust VKOSPI 성과를 정확히 보존"
        ),
    ),
)


STAGE_BY_ID = {stage.id: stage for stage in STAGES}
