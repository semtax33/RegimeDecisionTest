# Strict Hard 100/0 + SLSQP

이 폴더는 Zero-Tune 거시확률을 사용하되 Hard 국면배분을 한 자산 100%,
나머지 0%로 강제하고 현재 기본배분 구조의 SLSQP까지만 적용한 실험이다.

## Hard 매핑

| 국면 | KODEX200 | BOND | GLD | USO |
|---|---:|---:|---:|---:|
| Goldilocks | 100% | 0% | 0% | 0% |
| Overheating | 0% | 0% | 0% | 100% |
| Slowdown | 0% | 100% | 0% | 0% |
| Stagflation | 0% | 0% | 100% | 0% |

모든 월에 정확히 하나의 자산만 1이고 나머지는 0인지 자동 검사한다.
기존 Hard 규칙과 달리 Slowdown의 주식 60%·채권 40%를 없애고 채권
100%로 바꿨다.

## SLSQP 적용 범위

현재 기준전략의 기본배분 구조와 맞추기 위해 다음처럼 계산했다.

    최종 월간 기본비중 = 40% × Strict Hard + 60% × SLSQP

SLSQP는 목표 변동성·CDaR·자산별 범위·합계 100% 제약 아래에서 기대수익,
변동성, 꼬리손실, 회전율과 국면 anchor 이탈을 절충한다.

그 이후의 다음 계층은 적용하지 않았다.

- 균형 L2 꼬리위험 로지스틱
- tail tilt
- 15% 변동성 타깃
- 레버리지
- Robust VKOSPI 가속도 신호
- 현재 35% 금 이전 오버레이
- 월간·일간 frequency reconciliation

“Zero-Tune VKOSPI 기반”이라는 조건의 해석을 확인할 수 있도록 두 결과를
모두 저장했다.

1. SLSQP까지만 계산한 월간 경로
2. 그 월간 비중에 기존 Zero-Tune expanding VKOSPI overlay만 적용한 경로

## 전체기간 결과

공통기간은 2007-04~2026-07, 232개월이다.

| 전략 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| Strict Hard 100% 단독 월간 | 11.66% | 0.709 | -25.66% |
| Strict Hard 100% + Zero-Tune VKOSPI | 7.93% | 0.628 | -25.78% |
| **Strict Hard 40% + SLSQP 60% 월간** | **9.60%** | **1.049** | **-11.45%** |
| Strict Hard 40% + SLSQP 60% + Zero-Tune VKOSPI | 7.35% | 0.933 | -12.72% |
| 기존 Zero-Tune VKOSPI | 7.14% | 0.845 | -14.01% |
| 현재 Robust VKOSPI | 15.64% | 1.133 | -12.96% |

SLSQP의 순수한 역할은 Hard 단독과 월간끼리 비교하는 것이 가장 정확하다.

- CAGR: 11.66% → 9.60%, -2.06%p
- Sharpe: 0.709 → 1.049, +0.340
- MDD: -25.66% → -11.45%, +14.21%p 개선
- 연 변동성: 17.78% → 9.17%

즉 SLSQP는 최고수익을 높인 계층이 아니라 집중된 100% Hard 배분의 위험을
분산해 Sharpe와 MDD를 크게 개선한 계층이었다.

Zero-Tune VKOSPI 오버레이를 추가하면 월간 SLSQP 경로보다 CAGR, Sharpe,
MDD가 모두 나빠졌다. 이 조합에서는 expanding percentile을 위험이전 비율로
직접 사용하는 Zero-Tune 오버레이가 지나치게 방어적으로 작동했다.

## 2018-01~2026-07

| 전략 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| Strict Hard 100% 단독 월간 | 11.29% | 0.694 | -21.82% |
| Strict Hard 40% + SLSQP 60% 월간 | 10.19% | 1.202 | -10.74% |
| Strict Hard 40% + SLSQP 60% + Zero-Tune VKOSPI | 6.24% | 1.186 | -11.50% |

## 파일

- strict_hard_slsqp.py: 전체 실행 코드
- outputs/strict_hard_weights.csv: 매월 100/0 Hard 비중과 국면
- outputs/slsqp_path.csv: SLSQP 계산 경로
- outputs/strict_hard40_slsqp60_weights.csv: 최종 월간 기본비중
- outputs/strict_hard40_slsqp60_monthly.csv: SLSQP까지만 적용한 월간 성과
- outputs/strict_hard40_slsqp60_zero_tune_vkospi_monthly.csv:
  Zero-Tune VKOSPI까지 유지한 월간 성과
- outputs/performance_comparison.csv: 모든 비교전략의 전체·2018년 이후 성과
- outputs/strict_hard_slsqp_report.json: 설정과 인과성 검사

## 실행

프로젝트 루트에서 실행한다.

    & 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' -m strategies.stage09_strict_hard_slsqp.strict_hard_slsqp
