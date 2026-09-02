# Stage 47 — Guarded Ratio/Downside Objectives

Stage 36을 동결 기준선으로 두고, 피드백에서 제안한 기대수익 대비 위험
목적함수 네 가지를 같은 조건에서 비교한 연구다. 결론은 **Stage 36 유지**다.
후보 B~E 중 전체·공통·잠금 구간의 성과 게이트를 모두 통과한 전략은
없었다.

## 고정 실험

| 실험 | 기대수익·위험 | 최적화 목적 |
| --- | --- | --- |
| A | 기존 Stage 36 | 동결 기준선 |
| B | Stage 36 기대수익 + Ledoit–Wolf(LW) 공분산 | 기존 Stage 36 utility |
| C | Stage 36 기대수익 + LW 공분산 | ex-ante Sharpe 직접 최대화 |
| D | Stage 36 기대수익 + LW/downside | 비용 차감 기대 초과수익 - downside semivariance |
| E | Stage 36 기대수익 + LW/downside | ex-ante Sortino 직접 최대화 |

새 수익률 예측 모델이나 parameter grid는 넣지 않았다. 무위험수익률은
해당 월에 이미 알려진 국고채 3년물 연수익률을 12로 나눈 값이다.
downside risk는 목표 월 이전 월별 포트폴리오 초과수익의 음수 부분
semivariance로 측정한다.

모든 shadow 최적화는 long-only, 완전투자, 연 13% ex-ante 변동성,
과거 CDaR -16%, Stage 36 turnover budget을 적용한다. 배치 경로는 공통으로
최대 5% tilt, drawdown 구간의 tilt 감쇠, GVZ/OVX가 반영된 LW 위험 veto,
Stage 36 fallback을 적용한다.

## 성과와 결론

거래비용 차감 후 2007-04~2026-07 결과다.

| 구간 | 전략 | CAGR | Sharpe | Sortino | MDD | 월평균 turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 전체 | A Stage 36 | 10.50% | 1.1049 | 2.2019 | -12.41% | 3.70% |
| 전체 | B 기존 목적+LW | 10.54% | 1.1043 | 2.2018 | -12.41% | 3.72% |
| 전체 | C 직접 Sharpe | 10.32% | **1.1068** | 2.2009 | **-12.14%** | 3.84% |
| 전체 | D downside penalty | **10.55%** | 1.1042 | **2.2033** | -12.41% | 3.74% |
| 전체 | E 직접 Sortino | 10.32% | 1.1045 | 2.1972 | -12.14% | 3.83% |
| 2010+ | A Stage 36 | 9.23% | **1.0592** | 2.0495 | -12.41% | 2.79% |
| 2010+ | B 기존 목적+LW | 9.29% | 1.0585 | 2.0509 | -12.41% | 2.82% |
| 2010+ | C 직접 Sharpe | 9.01% | 1.0586 | 2.0321 | **-12.14%** | 2.93% |
| 2010+ | D downside penalty | **9.30%** | 1.0583 | **2.0527** | -12.41% | 2.84% |
| 2010+ | E 직접 Sortino | 9.02% | 1.0560 | 2.0296 | -12.14% | 2.92% |
| 2018+ | A Stage 36 | 12.63% | **1.2235** | **2.5378** | -11.93% | 2.62% |
| 2018+ | B 기존 목적+LW | 12.69% | 1.2203 | 2.5317 | -12.00% | 2.67% |
| 2018+ | C 직접 Sharpe | 12.24% | 1.2150 | 2.4904 | -11.77% | 2.78% |
| 2018+ | D downside penalty | **12.70%** | 1.2201 | 2.5344 | -12.00% | 2.72% |
| 2018+ | E 직접 Sortino | 12.27% | 1.2128 | 2.4889 | **-11.73%** | 2.75% |

D는 세 구간 CAGR을 각각 약 +5.3bp, +6.4bp, +7.9bp 높였고 전체·공통
Sortino도 소폭 개선했다. 그러나 세 구간 모두 Sharpe와 MDD가 악화되어
승격할 수 없다. C는 전체 구간 Sharpe와 MDD를 개선했지만 2010+와 2018+
Sharpe가 하락했다. E는 MDD는 개선했지만 실현 Sortino까지 세 구간 모두
하락했다. B 역시 CAGR 이외의 핵심 게이트를 충족하지 못했다.

따라서 사전 우선순위 `D → C → B → E`를 적용해도 승격 후보가 없으며,
`Stage36_GVZ_OVXAssetRisk`를 유지한다.

## Paired block bootstrap

12개월 paired block, 2,000회 재표집의 `delta > 0` 확률이다.

| 후보 | 구간 | CAGR | Sharpe | MDD |
| --- | --- | ---: | ---: | ---: |
| B | 전체 / 2010+ / 2018+ | 94.85% / 96.60% / 85.60% | 39.15% / 38.00% / 20.05% | 11.25% / 12.55% / 5.90% |
| C | 전체 / 2010+ / 2018+ | 1.85% / 1.10% / 0.10% | 67.10% / 48.00% / 15.10% | 84.50% / 88.40% / 83.45% |
| D | 전체 / 2010+ / 2018+ | 95.55% / 97.55% / 87.30% | 36.85% / 34.80% / 16.25% | 5.40% / 6.80% / 1.75% |
| E | 전체 / 2010+ / 2018+ | 1.50% / 0.75% / 0.25% | 51.45% / 34.85% / 12.30% | 83.70% / 86.70% / 81.95% |

C/E의 MDD 개선과 B/D의 CAGR 개선은 방향성이 뚜렷하지만, 위험조정성과와
성장률을 동시에 개선하는 후보는 아니다. 특히 직접 비율 최적화 C/E의
구간 민감성이 피드백의 불안정성 우려와 일치한다.

## 검증

- 네 shadow solver의 위험·CDaR·turnover·합계 제약은 수치 허용오차 안에서
  모두 유효하다.
- 네 배치 경로 모두 long-only이고 매월 비중 합계가 1이다.
- 최대 실제 objective tilt는 모두 5%다.
- 각 LW 공분산은 목표 월 이전의 완전한 252 거래일만 사용한다.
- Stage 36·45·46 동결 파일은 실행 전후 SHA-256이 동일하다.

## 실행

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
(& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\Activate.ps1')
python -m strategies.stage47_guarded_ratio_downside_objectives.guarded_ratio_downside_slsqp
python -m pytest tests\test_stage47_guarded_ratio_downside_objectives.py -q
```

## 산출물

- `guarded_ratio_downside_slsqp.py`: A~E 최적화·guard·백테스트 코드
- `outputs/stage47_*_guarded_*_monthly.csv`: B~E의 guarded 월별 경로
- `outputs/stage47_*_shadow.csv`: 배치 guard 전 최적화 경로
- `outputs/performance_comparison.csv`: 5개 경로, 3개 구간 성과표
- `outputs/paired_block_bootstrap_vs_stage36.csv`: paired bootstrap 결과
- `outputs/validation_report.json`: 게이트·인과성·solver·동결 해시 감사

연 13%는 사전 추정 위험한도이며 실현 변동성을 보장하지 않는다. 과거
백테스트 결과는 미래 성과를 보장하지 않는다.
