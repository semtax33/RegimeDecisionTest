# Stage43 — 동적 MDD 기준을 -12%로 완화한 재검증

## 결론

Stage42와 같은 알고리즘에서 MDD 기준만 `-10%`에서 `-12%`로 완화해 다시
테스트했다. 새 폴더를 만들었고 Stage36·Stage42 파일은 수정하지 않았다.

결과는 기대와 반대였다. 허용 손실폭을 넓히자 expected geometric growth
optimizer가 더 위험한 비중을 선택했고, 2008-10 drawdown이 `-13.891%`까지
커졌다. Stage43도 2008-11 시작 시점에 이미 바닥을 넘었으므로 문턱을 추가로
완화하거나 Stage36 비중으로 대체하지 않고 중단했다.

따라서 Stage43의 2007-2026 CAGR·Sharpe는 존재하지 않는다. 실행되지 않은 이후
213개월을 다른 규칙으로 채워 전체성과처럼 표시하지 않았다.

## 전체 gate 판정

| 전략 | 평가 상태 | CAGR | Sharpe | MDD | MDD ≥ -12% | Sharpe ≥ 1.1 |
|---|---|---:|---:|---:|---:|---:|
| Stage36 동결 | 2007-04~2026-07 완주 | 10.499% | 1.105 | -12.407% | 실패 | 통과 |
| Stage43 | 2007-04~2008-10 후 중단 | 산출 안 함 | 산출 안 함 | 산출 안 함 | 실패 | 실패 |

Stage36은 MDD 기준을 약 `0.407%p` 초과한다. 잠금구간 2018-01~2026-07만 보면
Sharpe `1.224`, MDD `-11.931%`로 두 조건을 통과하지만, 요청한 전체 2007-2026
gate는 통과하지 못한다.

## 실행 가능했던 19개월

아래는 전체성과가 아니라 실패 전 동일구간 진단치다.

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Stage36 동결 | 18.650% | 13.082% | 1.378 | 3.021 | -5.517% | 3.381 |
| Stage42, MDD -10% | 9.375% | 14.856% | 0.675 | 1.124 | -11.519% | 0.814 |
| **Stage43, MDD -12%** | **8.683%** | **16.403%** | **0.587** | **0.932** | **-13.891%** | **0.625** |

19개월 CAGR은 짧은 구간 연율화이므로 장기 기대수익으로 해석하면 안 된다. 다만
MDD를 2%p 완화했을 때 위험과 성과가 어느 방향으로 움직였는지는 보여준다.

## Stage42에서 바꾼 것은 하나

Stage42:

```text
NAV floor = 0.90 * running peak
MDD floor = -10%
L_t = 0.90 / (1 + d_t) - 1
```

Stage43:

```text
NAV floor = 0.88 * running peak
MDD floor = -12%
L_t = 0.88 / (1 + d_t) - 1
```

그 외에는 동일하다.

- Stage36의 월간 조건부 `mu_t`, `Sigma_t`
- expected geometric growth 목적함수
- ex-ante Sharpe 하한 1.10
- Stage36의 역사적 CDaR 정의와 90% 신뢰수준
- long-only, 무레버리지, 비중합 1, 각 비중 0~1
- 단일자산 과반금지 없음
- 국내 비중변화 15bp, GLD·USO 순비중변화 추가 5bp
- 월말 신호로 다음 달 비중을 정하는 인과적 워크포워드

새 lambda, risk multiplier, lookback, threshold grid는 추가하지 않았다.

## 목적함수

Stage36의 거시·기술·펀더멘털·VIX6·GVZ·OVX 정보로 현재 월의 조건부 기대수익과
공분산을 만든다. 그다음 다음 식을 최대화한다.

```text
G_t(w) = w' mu_t - 0.5 * w' Sigma_t w - C(w,w_pre)
```

`mu_t`와 `Sigma_t`는 월 단위다. 작은 월수익에서

```text
E[log(1+r)] ≈ E[r] - 0.5 Var(r)
```

이므로 평균수익만 최대화하는 것보다 CAGR과 가까운 방향이다. `0.5`는 임의로
맞춘 위험회피값이 아니라 로그함수의 2차 근사 계수다. SLSQP는 최소화 API이므로
코드에서는 `-G_t(w)`를 반환한다.

`C(w,w_pre)`는 임의 turnover penalty가 아니다. Stage36과 같은 실제 거래비용을
당월 기대성장률에서 차감한다.

## ex-ante Sharpe 제약

```text
Sharpe_exante(w)
  = sqrt(12) * [w' mu_t - C(w,w_pre)]
    / sqrt(w' Sigma_t w)

Sharpe_exante(w) >= 1.10
```

실행된 19개월의 최소 slack은 약 `-7.62e-10`이다. 허용 수치오차 `1e-7` 안에서
모든 달의 제약을 통과했다.

ex-ante Sharpe는 현재 forecast에 대한 조건이고, 그 뒤 실제 수익으로 계산하는
실현 Sharpe와 같지 않다. 매월 ex-ante 1.1을 만족해도 19개월 실현 Sharpe는
`0.587`이었다.

## 실제 NAV 기반 동적 손실예산

직전 달까지 실제 전략 NAV에서 현재 drawdown을 계산한다.

```text
d_t = NAV_t / Peak_t - 1
```

현재 NAV가 `(1+d_t) * Peak_t`이고 허용 바닥이 `0.88 * Peak_t`이므로, 바닥까지
남은 추가 손실률은 항등적으로 다음과 같다.

```text
(1+d_t)(1+L_t) = 0.88

L_t = 0.88 / (1+d_t) - 1
```

| 현재 drawdown | 남은 추가 손실예산 |
|---:|---:|
| 0% | -12.000% |
| -5% | -7.368% |
| -8% | -4.348% |
| -10% | -2.222% |
| -11% | -1.124% |

여기에는 drawdown 단계별 임의 배수나 위험감축 속도가 없다.

## tail-risk constraint

target month보다 앞선 수익률만 사용해 후보 고정비중의 역사적 CDaR90을 계산한다.

```text
historical_CDaR90(R_hist w) >= L_t
```

실행된 19개월의 최소 slack은 약 `-8.31e-12`로, 모든 의사결정에서 제약을
지켰다. `history_end_month < target_month`도 전부 참이다.

하지만 이는 다음 달 실현수익의 hard floor가 아니다. `L_t`는 현재 NAV에서 정확히
계산되지만 `historical_CDaR90`은 과거로 추정한 위험값이다. 아직 보지 못한 다음 달
충격이 그 추정치를 초과할 수 있다.

## 2008-10 실패 분석

2008-10 비중을 정하기 직전 Stage43의 실제 drawdown은 `-7.221%`였다.

```text
L_t = 0.88 / (1 - 0.072213) - 1
    = -5.1506%
```

Optimizer 결과:

| 항목 | 값 |
|---|---:|
| KODEX200 | 0.000% |
| 채권 | 36.745% |
| GLD | 60.838% |
| USO | 2.417% |
| ex-ante Sharpe | 1.100000 |
| 역사적 CDaR90 | -5.0215% |
| 동적 허용예산 | -5.1506% |
| 기대 월수익 | 1.820% |
| ex-ante 연변동성 | 18.303% |
| 실제 2008-10 순수익 | -7.189% |
| 충격 후 drawdown | -13.891% |

역사적 CDaR는 허용예산보다 약 `0.129%p` 좋았고 ex-ante Sharpe도 수치오차 범위에서
1.1을 지켰다. 그러나 실제 손실은 허용예산보다 약 `2.038%p` 컸다.

2008-11 시작 시점의 계산은 다음과 같다.

```text
0.88 / (1 - 0.138908) - 1 = +2.1958%
```

양수인 이유는 이미 NAV가 peak의 88% 아래에 있기 때문이다. 이때 새 포트폴리오를
찾는다고 이미 발생한 MDD 위반이 없어지지 않는다. 코드가
`DrawdownFloorAlreadyBreached`를 기록하고 중단하는 이유다.

## 왜 -12% 완화가 더 나빠졌나

완화는 손실을 줄이는 장치가 아니다. feasible set을 넓히는 장치다.

```text
Feasible(-10%) subset of Feasible(-12%)
```

목적함수는 그 넓어진 집합 안에서 expected geometric growth가 높은 비중을 찾는다.
2008-10을 비교하면 다음과 같다.

| 항목 | Stage42 -10% | Stage43 -12% |
|---|---:|---:|
| 채권 | 51.961% | 36.745% |
| GLD | 46.421% | 60.838% |
| ex-ante 연변동성 | 14.083% | 18.303% |
| 실제 월수익 | -5.102% | -7.189% |
| 충격 후 drawdown | -11.519% | -13.891% |

Stage43은 더 넓은 tail budget 안에서 채권을 줄이고 당시 기대수익이 높게 추정된
GLD를 늘렸다. ex-ante Sharpe 1.1 하한은 “위험을 작게 유지하라”가 아니라 예측
수익 대비 위험의 비율을 요구한다. 기대수익이 충분히 높으면 절대변동성 18.3%도
허용할 수 있다.

따라서 결과는 역설이 아니다. **MDD constraint를 완화하면서 CAGR 목적함수를 그대로
두면 optimizer가 여유 위험예산을 수익추구에 사용하는 것**이 정상적인 동작이다.

## 불가능 해와 전체성과 처리

2008-10까지는 solver가 성공했고 모든 제약을 사후 재계산해 통과했다. 하지만 실제
drawdown이 -12%를 넘은 뒤에는 다음 조치를 하지 않았다.

- MDD를 -14%로 다시 완화
- Sharpe 하한을 1.0으로 완화
- 2008-11 이후 Stage36 비중으로 연결
- CDaR confidence 또는 lookback 변경
- 실패 월을 제외하고 NAV 재시작

그래서 전체·2010 공통·2018 잠금구간의 Stage43 성과는 `NaN`이며
`CompletePeriod=False`다. 이는 데이터 누락이 아니라 엄격한 전략이 해당 기간에
존재하지 않는다는 명시적 결과다.

## Stage36 13% 변동성 guard를 넣지 않은 이유

이번 요청은 Stage42의 MDD 문턱만 -10%에서 -12%로 바꾸는 재시험이다. Stage42에
없던 13% volatility cap을 Stage43에서 갑자기 복원하면 두 조건이 동시에 바뀌어
순수한 비교가 아니게 된다. 따라서 Stage43도 ex-ante Sharpe와 동적 CDaR budget만
사용했다.

2008-10 ex-ante 변동성이 18.303%까지 올라간 결과는 이 선택의 중요한 진단이다.
향후 13% cap을 복원하는 실험을 한다면 Stage43 수정본이 아니라 별도 stage의
사전선언된 risk-ablation으로 진행해야 한다.

## 과최적화 방지

- Stage42 대비 변경값은 NAV floor `0.90 → 0.88` 하나
- 후보 전략 한 개
- ex-ante Sharpe 1.10 유지
- CDaR 90% 정의 유지
- threshold·lookback·objective weight 탐색 없음
- 미래 실현수익을 optimizer에 사용하지 않음
- 실패 후 기준을 다시 변경하지 않음
- Stage36 소스·결과·검증 JSON의 실행 전후 hash 일치 확인

SLSQP는 전월 drift 비중, Stage36 기준 비중, 동일비중과 네 단일자산 꼭짓점에서
동일 문제를 반복해 푼다. 이는 서로 다른 전략을 고르는 parameter search가 아니라
비매끄러운 CDaR 제약에서 국소해 오판을 줄이는 deterministic multi-start다.

## 실행

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m strategies.stage43_dynamic_drawdown_budget_12.dynamic_drawdown_budget_12_slsqp
```

## 파일

- `dynamic_drawdown_budget_12_slsqp.py`: Stage43 전체 구현
- `outputs/stage43_dynamic_dd12_budget_monthly.csv`: 실행된 19개월 비중과 제약
- `outputs/performance_comparison.csv`: 완전한 기간만 인정한 성과표
- `outputs/partial_failure_window_performance.csv`: 실패 전 동기간 진단
- `outputs/infeasible_events.csv`: 2008-11 중단 사유
- `outputs/validation_report.json`: 인과성·제약·gate·hash 감사
- `tests/test_stage43_dynamic_drawdown_budget_12.py`: 공식과 성과 회귀 테스트

코드는 `remaining_loss_budget` → `build_stage36_forecast` →
`portfolio_forecast_statistics` → `solve_weights` → `run_backtest` →
`run_research` 순서로 읽으면 된다.

## 최종 해석

MDD 허용폭을 -12%로 완화해도 현재 4자산·무레버리지·월간 리밸런싱 구조는 전체
기간을 완주하지 못했다. 더구나 완화된 위험예산을 optimizer가 수익추구에 사용하면서
2008년 손실이 Stage42보다 커졌다.

이번 결과는 “-12%가 너무 엄격해서 해가 없었다”보다 더 구체적이다. 최적화 해는
매월 존재했고 ex-ante 제약도 지켰지만, 예측 tail을 넘어선 실제 월간 충격 때문에
hard MDD gate가 깨졌다. 위험추정 제약과 손실보장은 같은 것이 아니다.
