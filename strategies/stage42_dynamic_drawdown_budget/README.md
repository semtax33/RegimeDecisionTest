# Stage42 — Stage36 예측 + 실제 NAV 기반 동적 drawdown 예산

## 결론

Stage42는 Stage41의 가장 큰 결함이었던 **상태 무시**를 고쳤다. 매월 새로운 비중을
정할 때 과거 고정비중 포트폴리오만 보는 대신, 직전 달까지 실제로 쌓인 전략 NAV와
고점에서 현재 drawdown을 계산한다. 이미 손실이 누적됐다면 다음 달에 허용하는 tail
loss도 자동으로 줄어든다.

하지만 2007-04부터 실행한 엄격한 Stage42는 2008-10의 실제 충격으로 drawdown이
`-11.519%`가 되면서 `MDD >= -10%` 기준을 넘었다. 2008-11에는 이미 NAV가 바닥선
아래였으므로 제약을 완화하거나 다른 전략 비중으로 대체하지 않고 실행을 중단했다.
따라서 Stage42의 2007-2026 CAGR·Sharpe는 계산하지 않는다. 존재하지 않는 213개월을
Stage36이나 사후 규칙으로 채워 만든 숫자는 Stage42의 성과가 아니기 때문이다.

이 결과가 뜻하는 바는 분명하다.

> 월 1회 리밸런싱하는 4개 위험자산 포트폴리오에서 ex-ante tail-risk 제약은 손실
> 가능성을 줄일 수는 있어도, 다음 한 달의 실제 손실을 확정적으로 -10% 바닥 안에
> 가둘 수 없다.

요청한 기준을 통과한 전략은 없었다. Stage36은 전체 Sharpe `1.105`로 1.1을
통과하지만 MDD가 `-12.407%`이고, Stage42는 전체기간을 완주하지 못했다. 문턱은
사후에 낮추지 않았다.

## 전체 판정

| 전략 | 전체기간 | CAGR | Sharpe | MDD | MDD ≥ -10% | Sharpe ≥ 1.1 |
|---|---:|---:|---:|---:|---:|---:|
| Stage36 동결 | 2007-04~2026-07 | 10.499% | 1.105 | -12.407% | 실패 | 통과 |
| Stage42 | 2007-04~2008-10에서 중단 | 산출 안 함 | 산출 안 함 | 산출 안 함 | 실패 | 실패 |

Stage42 행의 수치가 비어 있는 것은 계산 오류가 아니다. 완전한 2007-2026 경로가
없기 때문에 비교 가능한 전체성과가 없다는 뜻이다.

## 실행 가능했던 19개월의 진단 성과

이 표는 전략의 전체성과가 아니라 **실패 전 구간의 진단치**다. 같은 2007-04~
2008-10 구간으로 Stage36을 맞춰 비교했다.

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Stage36 동결 | 18.650% | 13.082% | 1.378 | 3.021 | -5.517% | 3.381 |
| Stage42 | 9.375% | 14.856% | 0.675 | 1.124 | -11.519% | 0.814 |

19개월 CAGR은 짧은 구간 연율화라 장기 기대수익으로 해석하면 안 된다. 이 표의
용도는 Stage42가 중단되기 전에도 Stage36보다 나았다고 주장할 근거가 없다는 것을
보이는 데 있다.

## 목표함수

Stage41처럼 과거 고정비중 CAGR을 직접 최대화하지 않는다. Stage36이 target month
직전에 만든 월간 조건부 기대수익 벡터 `mu_t`와 공분산 행렬 `Sigma_t`를 그대로
사용한다.

```text
maximize
    w' mu_t - 0.5 * w' Sigma_t w - C(w, w_pre)
```

SLSQP는 최소화 문제를 풀기 때문에 실제 코드의 반환값은 위 식의 음수다.

작은 월수익에서는 다음 근사가 성립한다.

```text
E[log(1 + r_p)] ≈ E[r_p] - 0.5 Var(r_p)
```

따라서 `w' mu - 0.5 w' Sigma w`는 평균수익만 높이는 산술 목적함수보다 복리성장에
가깝다. `0.5`는 로그함수의 2차 Taylor 전개에서 나오므로 새로 맞춘 위험회피
계수가 아니다. `C(w,w_pre)`도 임의 turnover penalty가 아니라 Stage36 백테스트가
실제로 차감하는 비용률을 쓴다.

- 국내 자산 비중 변화: 15bp
- GLD·USO 합산 순비중 변화: 추가 5bp
- 기대수익·분산·비용: 모두 월 단위

Stage36의 downside semivariance 항은 새 목적함수에서 제거했다. 이번 실험의 목적은
별도의 lambda 없이 expected geometric growth와 절대 제약만으로 목표가 가능한지
보는 것이기 때문이다.

## Stage36의 `mu_t`, `Sigma_t`에서 보존한 정보

`build_stage36_forecast`가 다음 단계를 Stage36과 같은 순서로 실행한다.

1. 네 가지 거시국면 확률로 과거 수익률의 조건부 평균과 공분산을 계산한다.
2. VIX6 stress·recovery 회귀조정을 기대수익과 stress covariance에 반영한다.
3. 일간 기술 신호로 기대수익을 신뢰도 조정하고 공분산을 조정한다.
4. KOSPI200 forward EPS와 valuation 정보를 KODEX200 기대수익에 반영한다.
5. 신용 spread 확인값으로 KODEX200 stress 효과와 분산을 조정한다.
6. GVZ는 GLD 분산축, OVX는 USO 분산축만 1~2배 범위에서 조정한다.

GVZ·OVX로 GLD·USO의 기대수익 방향을 바꾸지는 않는다. Stage42가 바꾼 부분은 이
정보를 받은 뒤 비중을 고르는 목적함수와 제약뿐이다.

## ex-ante Sharpe 1.1 제약

월간 `mu_t`, `Sigma_t`와 당월 예상 거래비용으로 다음 값을 계산한다.

```text
Sharpe_exante(w)
    = sqrt(12) * (w' mu_t - C(w,w_pre))
      / sqrt(w' Sigma_t w)
```

Stage36 체계에 별도의 risk-free series가 없으므로 초과수익 기준은 0이다. 이를
중간에 새 금리 series로 바꾸지 않았다.

```text
Sharpe_exante(w) - 1.10 >= 0
```

2007-04~2008-10의 19개 의사결정에서 최소 slack은 약
`-3.7e-10`이었다. 이는 SLSQP 수치오차 범위이며 허용오차 `1e-7`보다 작다. 즉 실행한
모든 달의 ex-ante Sharpe 제약은 통과했다.

여기서 ex-ante Sharpe와 실현 Sharpe는 다른 값이다. 전자는 그 시점까지 관측한
Stage36 forecast로 계산하고, 후자는 그 뒤 실제로 발생한 월수익 경로로 계산한다.
ex-ante Sharpe 1.1을 매월 지켰다고 실현 Sharpe 1.1이 보장되지는 않는다.

## 실제 NAV에서 남은 손실예산 계산

직전 달까지 실제 NAV와 running peak로 현재 drawdown을 계산한다.

```text
d_t = NAV_t / Peak_t - 1
```

NAV 바닥은 peak의 90%다. 다음 기간 수익률이 정확히 바닥에 닿는 값 `L_t`는
다음 항등식에서 바로 나온다.

```text
(1 + d_t) * (1 + L_t) = 0.90

L_t = 0.90 / (1 + d_t) - 1
```

| 현재 실제 DD | 남은 추가 손실예산 |
|---:|---:|
| 0% | -10.000% |
| -5% | -5.263% |
| -8% | -2.174% |
| -9% | -1.099% |

`0.90`, 현재 NAV와 현재 peak만 사용한다. drawdown 단계별 배수나 임의 de-risking
속도는 없다.

## tail-risk 제약

Stage36이 이미 사용하던 역사적 CDaR 정의와 90% 신뢰수준을 그대로 계승한다.
target month보다 앞선 자산수익률 행렬을 `R_hist`라고 하면 후보 비중의 과거
시나리오는 `R_hist w`다. 이 수익률로 누적 NAV와 drawdown을 만든 뒤 가장 나쁜
drawdown 10%의 평균을 구한다.

```text
historical_CDaR90(R_hist w) >= L_t
```

현재 drawdown이 깊어질수록 오른쪽의 `L_t`가 0에 가까워져 허용되는 과거 tail
drawdown이 좁아진다. 실행한 19개월의 최소 constraint slack은 약 `-9.7e-10`으로,
모든 달에서 동적 tail budget을 지켰다.

다만 이 제약은 **확률적 위험추정치**이지 다음 달 수익의 hard floor가 아니다.
역사적 CDaR은 과거에 관측한 경로의 평균 tail이고, 다음 한 달에 아직 보지 못한
충격이 그보다 클 가능성은 남는다.

## 2008-10에 정확히 무슨 일이 있었나

2008-10 비중을 정하기 직전 실제 drawdown은 `-6.762%`였다.

```text
남은 손실예산 = 0.90 / (1 - 0.067615) - 1
              = -3.4733%
```

Optimizer가 고른 비중과 forecast는 다음과 같다.

| 항목 | 값 |
|---|---:|
| KODEX200 | 0.000% |
| 채권 | 51.961% |
| GLD | 46.421% |
| USO | 1.618% |
| ex-ante Sharpe | 1.268 |
| 역사적 CDaR90 | -3.4733% |
| 허용 tail budget | -3.4733% |
| 기대 월수익 | 1.597% |
| ex-ante 연변동성 | 14.083% |

두 절대 제약을 모두 만족했고 tail 제약은 사실상 binding이었다. 그러나 실제
2008-10 순수익은 `-5.102%`였다. 허용예산보다 약 `1.63%p` 더 큰 손실이 발생하면서
drawdown이 `-11.519%`가 됐다.

2008-11 시작 시점에는 다음 값이 된다.

```text
0.90 / (1 - 0.115198) - 1 = +1.7167%
```

양수라는 것은 “추가 손실을 허용할 수 없다”를 넘어 이미 peak의 90% 위로 NAV를
회복해야 한다는 뜻이다. 이 상태에서 새 비중을 구해도 이미 발생한 MDD 위반을
되돌릴 수 없다. 코드는 `DrawdownFloorAlreadyBreached`를 기록하고 중단한다.

## 왜 강제 손실보장이 아니었나

세 값을 구분해야 한다.

1. `L_t`: 현재 실제 NAV에서 계산한 정확한 회계적 손실여유
2. `historical_CDaR90`: 미래 손실을 가늠하기 위한 과거 기반 위험추정치
3. `r_{t+1}`: 아직 모르는 다음 달의 실제 수익률

첫 번째 값은 정확하지만 두 번째는 추정치다. 제약은
`historical_CDaR90 >= L_t`만 강제할 수 있고 `r_{t+1} >= L_t`를 강제할 수 없다.
후자는 미래수익을 미리 알아야 가능하다.

정말로 -10%를 기계적으로 넘지 않으려면 월중 거래가 가능한 현금성 자산,
daily/intraday stop, 보호 put처럼 손실함수가 계약적으로 제한되는 수단이 필요하다.
현재 연구 범위는 월 1회, KODEX200·채권·GLD·USO, 무레버리지·완전투자이므로 그런
수단을 사후에 추가하지 않았다.

## 제약과 불가능 해 처리

매월 최적화 문제는 다음과 같다.

```text
maximize  w' mu_t - 0.5*w' Sigma_t*w - C(w,w_pre)

subject to
    Sharpe_exante(w) >= 1.10
    historical_CDaR90(R_hist*w) >= 0.90/(1+d_t)-1
    sum(w) = 1
    0 <= w_i <= 1
```

단일자산 과반금지 조건은 없다. 레버리지와 공매도도 없다.

수치적인 국소해 오판을 줄이기 위해 전월 drift 비중, Stage36 기준 비중, 동일비중,
네 개 단일자산 꼭짓점에서 SLSQP를 시작한다. 이는 서로 다른 전략 parameter를
탐색하는 것이 아니라 동일 문제의 deterministic multi-start다. `success=True`이고
실제 slack을 다시 계산해 모두 통과한 해만 후보로 인정한다.

해가 없거나 이미 MDD 바닥을 넘었을 때 하지 않는 일은 다음과 같다.

- Sharpe 문턱을 1.05나 1.0으로 낮추기
- MDD 문턱을 -12%로 낮추기
- CDaR confidence나 lookback을 바꾸기
- Stage36 비중을 조용히 끼워 넣어 전체경로처럼 표시하기
- 결과가 좋아지는 달만 제외하기

이 때문에 2008-11 이후 성과가 비어 있다.

## Stage36에서 보존한 것과 바꾼 것

보존:

- 월간 자산수익 데이터와 원화 환산
- 거시국면·VIX6 stress/recovery·기술·펀더멘털 정보
- 신용, GVZ, OVX 위험축 조정
- 월말 신호와 다음 달 수익의 인과적 정렬
- drift 후 pre-trade 비중과 실제 거래비용
- 무레버리지, long-only, 비중합 1
- CDaR 신뢰수준 90%

변경:

- Stage36 utility를 parameter-free expected log-growth로 교체
- ex-ante Sharpe 하한을 1.10으로 강화
- 고정 -16% CDaR guard를 실제 NAV 기반 동적 손실예산으로 교체
- 최종 실현 gate를 MDD -10%, Sharpe 1.10으로 강화

Stage36의 연변동성 13% catastrophe cap은 새 문제에 넣지 않았다. 사용자가 제시한
Stage42 제약은 ex-ante Sharpe와 동적 tail budget이며, 13% cap까지 넣으면 무엇이
효과를 냈는지 달라지기 때문이다. 이 선택의 결과로 2008-10 ex-ante 변동성이
14.083%까지 허용됐다는 사실도 결과표에 그대로 남겼다.

## 과최적화 방지

- 후보 전략은 한 개
- MDD -10%, Sharpe 1.10은 요청값 고정
- CDaR 90%는 Stage36 값 계승
- lookback·threshold·목적함수 가중치 grid search 없음
- 기대 log-growth의 `0.5`는 Taylor 전개 계수
- 미래 실현수익을 objective나 constraint에 넣지 않음
- 결과 실패 후 문턱·부호·자산범위를 변경하지 않음
- Stage36 소스·월별 결과·검증 JSON의 실행 전후 SHA-256 확인

## 실행

프로젝트 루트에서 다음 명령을 실행한다.

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m strategies.stage42_dynamic_drawdown_budget.dynamic_drawdown_budget_slsqp
```

## 파일

- `dynamic_drawdown_budget_slsqp.py`: forecast 재구성, 목적함수, 동적 제약,
  워크포워드, 최종 gate 전체 구현
- `outputs/stage42_dynamic_dd_budget_monthly.csv`: 실제 실행 가능했던 19개월의
  비중·forecast·제약 slack·수익
- `outputs/performance_comparison.csv`: 완전한 기간만 성과로 인정한 비교표
- `outputs/partial_failure_window_performance.csv`: 실패 전 19개월 진단 비교
- `outputs/infeasible_events.csv`: 중단 시점과 원인
- `outputs/validation_report.json`: 인과성·제약·최종판정·불변 hash 감사
- `tests/test_stage42_dynamic_drawdown_budget.py`: 공식과 저장결과 회귀 테스트

전체 구현을 읽을 때는 `remaining_loss_budget` → `build_stage36_forecast` →
`portfolio_forecast_statistics` → `solve_weights` → `run_backtest` →
`run_research` 순서가 가장 이해하기 쉽다.

## 최종 해석

Stage42의 아이디어가 무의미했던 것은 아니다. Stage41과 달리 실제 drawdown 상태가
다음 비중 결정에 들어갔고, 손실이 깊어질수록 tail constraint가 자동으로 강해졌다.
문제는 위험 **예산**을 정확히 계산하는 것과 미래 손실을 **보장**하는 것이 다른
일이라는 데 있다.

현재 4자산·월간·완전투자 구조에서는 `MDD <= 10%`, `Sharpe >= 1.1`, 긴 기간의
높은 CAGR을 동시에 달성했다는 증거를 얻지 못했다. 이 결론을 유지하는 것이 실패한
경로 뒤에 임의의 방어규칙을 덧붙여 성과를 완성하는 것보다 연구적으로 타당하다.
