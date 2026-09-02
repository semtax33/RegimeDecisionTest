# Stage40 — Stage36 기반 Calmar·Sortino 최대화 SLSQP

## 결론

Stage36의 데이터, 기대수익·공분산, GVZ→GLD·OVX→USO 위험조정, 거래비용,
long-only·무레버리지 조건, 연변동성 13%와 역사적 CDaR -16% 제약을 그대로
두고 SLSQP 목적함수만 다음 두 가지로 교체했다.

- `Stage40_CausalCalmar`: 인과적으로 계산한 예상 Calmar 최대화
- `Stage40_CausalSortino`: 인과적으로 계산한 예상 Sortino 최대화

SLSQP는 최소화 알고리즘이므로 코드에서는 각각 `-Calmar`, `-Sortino`를
최소화한다. 수학적으로 해당 비율을 최대화하는 것과 같다.

두 후보 모두 2007-04~2026-07 전체구간의 실현 Sharpe가 1 이상이어서 사용자가
요청한 전체구간 조건은 통과했다. 그러나 기대비율의 위험 분모를 줄이는 과정에서
채권에 약 90~95% 집중했고, CAGR과 실제 Calmar·Sortino가 Stage36보다 낮아졌다.
2018년 이후 Sharpe도 1 아래였다. 따라서 연구후보와 결과는 보존하지만 Stage36을
교체하지 않는다.

## 성과

### 전체구간 2007-04~2026-07

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Stage36 동결 | **10.499%** | 9.472% | 1.105 | **2.202** | -12.407% | **0.846** |
| Calmar 최대화 | 4.404% | 3.587% | **1.222** | 2.157 | **-6.844%** | 0.644 |
| Sortino 최대화 | 3.730% | **3.562%** | 1.047 | 1.748 | -9.137% | 0.408 |

Calmar 목적은 MDD를 5.56%p 줄이고 전체 Sharpe를 높였지만 CAGR이 6.09%p
낮아져 실현 Calmar 자체는 오히려 `0.846 → 0.644`로 하락했다. Sortino 목적도
하방변동성을 줄였지만 수익 감소가 더 커 실현 Sortino가 `2.202 → 1.748`로
낮아졌다.

### 공통구간 2010-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Stage36 동결 | 9.232% | 8.719% | 1.059 | 2.049 | -12.407% | 0.744 |
| Calmar 최대화 | 3.941% | 3.561% | **1.105** | 1.817 | **-6.844%** | 0.576 |
| Sortino 최대화 | 3.224% | 3.562% | 0.910 | 1.422 | -9.137% | 0.353 |

### 잠금구간 2018-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Stage36 동결 | **12.626%** | 10.178% | **1.224** | **2.538** | -11.931% | **1.058** |
| Calmar 최대화 | 4.245% | 4.514% | 0.945 | 1.496 | **-6.844%** | 0.620 |
| Sortino 최대화 | 2.520% | 4.541% | 0.571 | 0.857 | -9.137% | 0.276 |

전체구간 Sharpe 1 조건은 두 후보 모두 통과하지만 최근 잠금구간에서는 통과하지
못했다. 이 차이를 숨기지 않고 `validation_report.json`의 구간별 gate에 기록했다.

## 목적함수 정의

사후에 확인되는 전체 백테스트 CAGR·MDD·Sortino를 매월 목적함수에 넣으면
미래정보를 사용하는 것이 된다. Stage40은 목표월보다 앞선 정보만으로 다음의
대용 목적함수를 계산한다.

### Calmar 최대화

후보비중 `w`의 Stage36 월 기대수익과 공분산으로 거래비용 차감 후 예상 CAGR을
계산한다.

```text
net_mu_m(w) = w'μ_t - estimated_transaction_cost(w, pretrade_w)

expected_CAGR(w)
  = exp{12 × [net_mu_m(w) - 0.5 × w'Σ_t w]} - 1
```

목표월 이전의 모든 자산 월수익에 후보비중을 고정 적용해 역사적 MDD를 계산한다.

```text
historical_path_i(w) = R_i w,  i < target_month
historical_MDD(w)    = MDD(historical_path(w))

causal_Calmar(w)
  = expected_CAGR(w) / abs(historical_MDD(w))

SLSQP objective = -causal_Calmar(w)
```

Calmar의 분자는 현재 추정 기대수익이고 분모는 과거 스트레스 경로다. 최종 성과표의
실현 Calmar와 동일한 정보집합은 아니며, 이것이 목적함수 최대화와 사후 실현비율
개선이 일치하지 않을 수 있는 핵심 이유다.

### Sortino 최대화

목표월 이전 역사에서 후보비중의 음수수익만 사용해 하방편차를 계산한다.

```text
downside_deviation(w)
  = sqrt(mean[min(R_i w, 0)^2]) × sqrt(12)

causal_Sortino(w)
  = 12 × net_mu_m(w) / downside_deviation(w)

SLSQP objective = -causal_Sortino(w)
```

양의 변동성은 벌점에 들어가지 않는다. 그러나 기대수익 추정오차가 있고 과거
하방편차가 매우 작은 채권이 존재하면 분모 축소가 분자 확대보다 쉬워진다.

## Stage36에서 보존한 구조

- 성장·물가 soft-regime 조건부 기대수익·공분산
- VKOSPI·VIX6 stress/recovery
- K-ratio·RSI 기술신뢰도와 ATR `DΣD`
- EPS revision·밸류에이션·신용 조정
- GVZ→GLD와 OVX→USO 자산별 위험배수
- 비중합 1, 각 자산 0~1
- 무공매도·무레버리지
- 단일자산 과반금지 없음
- 예상 연변동성 13% 이하
- 역사적 CDaR(90%) -16% 이상
- 국내 거래비용 15bp와 해외 순비중변화 비용 5bp
- 직전 drift 비중과 월별 walk-forward 순서

따라서 Stage36 대비 결과차이는 목적함수 교체에서 발생한다.

## 왜 채권에 집중됐는가

| 후보 | KODEX200 평균 | BOND 평균 | GLD 평균 | USO 평균 |
|---|---:|---:|---:|---:|
| Calmar 최대화 | 1.96% | **89.91%** | 4.71% | 3.43% |
| Sortino 최대화 | 1.39% | **95.28%** | 2.05% | 1.28% |

비율 목적함수에는 잘 알려진 분모 최소화 문제가 있다.

1. 기대수익 `μ`는 잡음이 크다.
2. 채권의 과거 MDD·음의 하방편차는 매우 작다.
3. SLSQP는 불확실한 분자를 높이기보다 확실히 작은 분모를 선택한다.
4. 그 결과 예상비율은 높지만 절대 기대수익과 실현 CAGR은 낮아진다.

단일자산 과반금지 조건을 사용자가 해제했기 때문에 이러한 집중을 인위적으로
막지 않았다. 결과를 본 뒤 채권상한이나 최소 주식비중을 추가하면 목적함수의
문제를 숨기는 사후 하이퍼파라미터가 되므로 이번 Stage에서는 하지 않았다.

## Sharpe 1 조건과 최종판정

| 후보 | 전체 Sharpe ≥ 1 | 공통구간 Sharpe ≥ 1 | 2018+ Sharpe ≥ 1 | 목표 실현비율이 Stage36보다 높음 |
|---|---:|---:|---:|---:|
| Calmar 최대화 | 통과 | 통과 | 실패 | 실패 |
| Sortino 최대화 | 통과 | 실패 | 실패 | 실패 |

사용자의 `Sharpe 1 이상 유지` 조건은 전체 백테스트 성과를 기준으로 두 후보 모두
통과한다. 다만 하위구간 안정성과 실제 목표비율 개선까지 보면 둘 다 Stage36
교체조건을 통과하지 못한다. 최종 유지전략은 `Stage36_Frozen`이다.

## SLSQP와 fallback

Calmar의 MDD는 최대·최소 연산을 포함해 미분이 매끄럽지 않다. Sortino도 수익이
0을 통과할 때 기울기가 바뀐다. 기본 초기점에서 SLSQP가 실패하면 동일 목적함수를
동일가중·역분산 초기점으로 다시 시도한다. 그래도 실패하는 초기 짧은 이력에서는
Stage36과 같은 최소분산 fallback을 사용한다.

- Calmar: 목적함수 재시작 3개월, 최소분산 fallback 4개월
- Sortino: 목적함수 재시작 3개월, 최소분산 fallback 5개월
- 모든 최종 해 성공
- 비중합 최대오차 `3.33e-16` 이하
- 모든 월 변동성·CDaR 제약 충족
- fallback은 2007~2008년의 짧은 초기 역사에만 발생

## Bootstrap

12개월 블록 2,000회 paired bootstrap에서 Stage36 대비 전체구간 개선확률은 다음과
같다.

| 후보 | ΔCAGR > 0 | ΔSharpe > 0 | ΔMDD > 0 |
|---|---:|---:|---:|
| Calmar 최대화 | 0.25% | 66.75% | 96.35% |
| Sortino 최대화 | 0.15% | 45.80% | 93.60% |

MDD 감소는 일관되지만 CAGR 손실도 매우 일관된다. Calmar 후보의 Sharpe 개선은
가능성이 있으나 목표인 실현 Calmar 개선으로 연결되지 않았다.

## 과최적화 방지

- Calmar·Sortino 두 목적을 사전 고정
- 목적함수 혼합비율 없음
- ratio floor는 0 나눗셈 방지용 `1e-10`만 사용
- lookback·분모 가중치·채권상한 탐색 없음
- 미래 실현수익은 비중결정에 미사용
- 실현 Sharpe는 백테스트가 끝난 뒤 gate에만 사용
- Stage36 소스와 저장결과 SHA-256 불변 확인

## 실행

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m strategies.stage40_calmar_sortino_objectives.ratio_objective_slsqp
```

## 파일

- `ratio_objective_slsqp.py`: 목적함수, SLSQP, 워크포워드, 성과·gate 생성
- `outputs/stage40_calmar_monthly.csv`: Calmar 최대화 월별 비중·성과·solver 기록
- `outputs/stage40_sortino_monthly.csv`: Sortino 최대화 월별 비중·성과·solver 기록
- `outputs/performance_comparison.csv`: 전체·공통·잠금구간 비교
- `outputs/paired_block_bootstrap_vs_stage36.csv`: Stage36 대비 재표집 결과
- `outputs/validation_report.json`: 인과성·제약·Sharpe gate·동결파일 감사
- `tests/test_stage40_calmar_sortino_objectives.py`: 공식·산출물·제약 회귀검증

과거 백테스트는 미래성과를 보장하지 않는다.
