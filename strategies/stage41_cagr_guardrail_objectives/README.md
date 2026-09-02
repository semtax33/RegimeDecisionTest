# Stage41 — CAGR 최대화 + MDD·Sharpe 절대 가드레일

## 결론

Stage36 원본을 수정하지 않고 새 폴더에서 다음 두 SLSQP 경로를 구현했다.

```text
A. maximize causal historical CAGR
   subject to historical MDD >= -14%, historical Sharpe >= 1

B. maximize causal historical CAGR
   subject to historical CDaR(90%) >= -16%, historical Sharpe >= 1
   final validation: realized MDD >= -14%, realized Sharpe >= 1
```

SLSQP는 최소화 알고리즘이므로 실제 코드는 `return -historical_cagr`다. 가중
ratio나 Calmar·Sortino 목적함수는 사용하지 않았다. 비중합은 1이고 각 자산은
0~1이며 공매도·레버리지·단일자산 과반금지 조건이 없다.

결과는 중요한 음의 결과다. 월별 의사결정 때 계산한 역사적 제약은 두 후보 모두
232개월 전부 통과했지만, 실제 비중이 매달 바뀐 동적 백테스트 경로에서는 MDD
-14%와 Sharpe 1 조건을 모두 위반했다. 따라서 Stage41 후보는 채택하지 않고 두
조건을 이미 만족하면서 CAGR도 더 높은 `Stage36_Frozen`을 유지한다.

## 성과

### 전체구간 2007-04~2026-07

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| **Stage36 동결** | **10.499%** | **9.472%** | **1.105** | **2.202** | **-12.407%** | **0.846** |
| A · Hard MDD -14% | 9.829% | 14.843% | 0.707 | 1.256 | -25.019% | 0.393 |
| B · CDaR -16% | 9.644% | 15.957% | 0.657 | 1.152 | -25.358% | 0.380 |

### 공통구간 2010-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| **Stage36 동결** | **9.232%** | **8.719%** | **1.059** | **2.049** | **-12.407%** | **0.744** |
| A · Hard MDD -14% | 7.684% | 10.953% | 0.732 | 1.299 | -23.548% | 0.326 |
| B · CDaR -16% | 7.744% | 12.386% | 0.665 | 1.163 | -25.358% | 0.305 |

### 잠금구간 2018-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| **Stage36 동결** | **12.626%** | **10.178%** | **1.224** | **2.538** | **-11.931%** | **1.058** |
| A · Hard MDD -14% | 11.857% | 11.313% | 1.050 | 2.077 | -13.779% | 0.861 |
| B · CDaR -16% | 11.922% | 12.884% | 0.941 | 1.732 | -17.276% | 0.690 |

A안은 최근구간에서는 두 절대조건을 만족했다. 하지만 전체 연구구간에서 Sharpe
0.707, MDD -25.02%이므로 최종 전략으로 채택할 수 없다. B안은 최근구간에서도
두 조건을 모두 통과하지 못했다.

## 구현한 목적함수

목표월 `t`에서 사용할 수 있는 것은 `t`보다 앞선 자산 월수익뿐이다.

```text
r_i(w) = R_i w,  i < t

historical_CAGR_t(w)
  = exp[12 × mean(log(1 + r_i(w)))] - 1

SLSQP objective_t(w)
  = -historical_CAGR_t(w)
```

전체 백테스트가 끝난 뒤 확인한 실현 CAGR을 목적함수에 넣지 않는다. 매월
expanding history를 다시 계산하므로 미래자료를 미리 보는 look-ahead는 없다.
거래비용은 목적함수에 가중벌점으로 섞지 않고 Stage36과 같은 비용률로 실제 월별
수익에서 차감한다.

## A안: Hard MDD + Sharpe

후보비중을 목표월 이전 모든 자산수익에 고정 적용해 MDD와 Sharpe를 계산한다.

```text
wealth_i(w) = cumulative_product(1 + R_i w)
MDD_t(w)    = min[wealth_i / running_peak_i - 1]

Sharpe_t(w)
  = sqrt(12) × mean(R_i w) / std(R_i w)
```

제약:

```text
sum(w) = 1
0 <= w_i <= 1
MDD_t(w) >= -0.14
Sharpe_t(w) >= 1.0
```

232개월 중 최적화 시점의 최소 역사적 MDD는 수치허용오차를 포함해 정확히
-14.0000001% 수준이고 최소 역사적 Sharpe도 약 1이다. 즉 코드가 제약을
무시해서 성과가 나빠진 것이 아니다.

## B안: CDaR + Sharpe

MDD 대신 Stage36이 사용하던 역사적 CDaR(90%) -16%를 최적화 제약으로 둔다.

```text
CDaR90_t(w) = worst 10% historical drawdowns average

sum(w) = 1
0 <= w_i <= 1
CDaR90_t(w) >= -0.16
Sharpe_t(w) >= 1.0
```

최종 동적 경로에는 별도로 MDD -14%, Sharpe 1 validation gate를 적용한다.
B안의 월별 최소 역사적 CDaR는 -16.0000000% 수준으로 제약을 지켰지만 최종
MDD는 -25.36%였다.

## 왜 월별 제약을 지켰는데 최종 MDD가 -25%인가

피드백의 MDD·Sharpe 함수는 매월 후보비중 `w_t` 하나를 과거 모든 월에 고정해
적용한다.

```text
optimizer가 검사한 경로:
R_1 w_t, R_2 w_t, ..., R_(t-1) w_t

실제 전략 경로:
R_1 w_1, R_2 w_2, ..., R_(t-1) w_(t-1), R_t w_t
```

두 경로는 다르다. 매월 개별적으로 안전해 보이는 비중을 선택해도, 비중 변경
순서와 새롭게 들어온 수익 때문에 실제 누적경로의 낙폭은 더 커질 수 있다.

또한 시점 `t`의 제약은 `t`월 수익을 알 수 없다. 따라서 과거 MDD 제약은 다음
달 손실을 보장하지 않는다. A안의 실제 최악 낙폭은 2009-04에 -25.02%였고,
B안은 장기 GLD 하락이 누적된 2014-10에 -25.36%였다.

Sharpe도 같은 문제를 가진다. 매월 선택된 고정비중의 과거 Sharpe는 1 이상이지만
서로 다른 비중으로 연결된 실제 수익열의 전체 Sharpe는 1 아래로 내려갈 수 있다.

이것은 SLSQP 오류가 아니라 단일기간 고정비중 제약과 다기간 동적 전략경로의
차이다. 미래 동적 MDD를 확실히 제한하려면 별도의 다기간 상태변수 또는 실제
누적 NAV·현재 drawdown에 반응하는 위험예산이 필요하지만, 이번 Stage에서는
피드백에 없던 규칙을 사후 추가하지 않았다.

## 자산배분

| 후보 | KODEX200 평균 | BOND 평균 | GLD 평균 | USO 평균 | 월평균 turnover |
|---|---:|---:|---:|---:|---:|
| A · Hard MDD | 22.87% | 32.06% | **44.51%** | 0.56% | 3.13% |
| B · CDaR | 20.78% | 31.42% | **47.14%** | 0.66% | 3.26% |

두 목적함수 모두 과거 기하수익률이 높고 MDD·Sharpe 제약 안에 들어왔던 GLD를
많이 선택했다. 하지만 2012~2014년 금 하락에서는 과거 기준이 새로운 장기 하락을
충분히 예상하지 못했다. Stage36의 K-ratio·ATR·GVZ 위험조정이 이 문제를 줄였던
반면, 피드백의 순수 역사적 CAGR 목적함수에는 이 예측층이 들어가지 않는다.

## Stage36에서 보존한 것과 바뀐 것

보존:

- 동일한 KODEX200·국내채권·원화 GLD·원화 USO 월수익
- 동일한 2007-04~2026-07 측정구간
- 직전 drift 비중에서 거래
- 국내 거래비용 15bp와 해외 순비중변화 비용 5bp
- long-only·무레버리지·비중합 1
- Stage36 결과를 수정하지 않은 동결 비교선

바뀜:

- Stage36의 효용 목적함수를 순수 역사적 CAGR 최대화로 교체
- A안은 13% 변동성·CDaR 제약 대신 MDD -14%·Sharpe 1 사용
- B안은 13% 변동성 제약 없이 CDaR -16%·Sharpe 1 사용
- Stage36 거시·기술·펀더멘털 기대수익과 GVZ/OVX 공분산은 새 목적함수에 미사용

마지막 항목은 구현 누락이 아니다. 제공된 피드백이 `historical_returns @ w`로
CAGR·MDD·Sharpe를 정의했기 때문에 그대로 따른 결과다. Stage36은 데이터·실행·
비용·비교 프레임으로 사용됐다.

## 불가능 해 처리

각 월에는 다음 결정론적 초기점을 사용한다.

- 직전 drift 비중
- 해당 월 Stage36 동결 비중
- 동일가중
- 네 개 단일자산 꼭짓점

이는 문턱이나 하이퍼파라미터 탐색이 아니라 비매끄러운 MDD 함수의 국소해 문제를
줄이는 수치적 재시작이다. 성공하면서 실제 제약을 만족하는 해 중 역사적 CAGR이
가장 높은 해를 선택한다.

한 개도 없으면 `InfeasiblePortfolioError`를 발생시키며 Sharpe나 MDD 문턱을
완화하지 않는다. 이번 데이터에서는 A/B 모두 불가능 월이 0개월이었다.

## 최종 선택규칙

Stage36과 두 Stage41 후보를 모두 다음 gate에 넣는다.

```text
full realized MDD >= -14%
full realized Sharpe >= 1
```

gate를 통과한 전략 가운데 전체 CAGR이 가장 높은 전략을 선택한다.

| 전략 | MDD gate | Sharpe gate | 전체 CAGR | 최종 자격 |
|---|---:|---:|---:|---:|
| Stage36 | 통과 | 통과 | **10.499%** | 적격·선택 |
| A · Hard MDD | 실패 | 실패 | 9.829% | 탈락 |
| B · CDaR | 실패 | 실패 | 9.644% | 탈락 |

따라서 최종 선택은 `Stage36_Frozen`이다.

## Bootstrap

Stage36 대비 12개월 블록·2,000회 paired bootstrap의 전체구간 개선확률:

| 후보 | ΔCAGR > 0 | ΔSharpe > 0 | ΔMDD > 0 |
|---|---:|---:|---:|
| A · Hard MDD | 32.40% | 0.30% | 1.75% |
| B · CDaR | 29.00% | 0.05% | 0.95% |

두 후보의 Sharpe·MDD 열화는 특정 한두 달의 우연만으로 보기 어렵다.

## 과최적화 방지

- A/B 두 구조만 사전 고정
- MDD -14%, Sharpe 1, 기존 CDaR -16%를 그대로 사용
- 문턱·lookback·목적함수 가중치 탐색 없음
- 전체 과거를 expanding window로 사용
- 결과를 보고 자산상한·최소주식비중을 추가하지 않음
- infeasible 시 문턱 완화 없음
- 최종 실현 MDD·Sharpe는 validation gate에만 사용
- Stage36 소스·월별 결과·검증 JSON의 SHA-256 불변 확인

## 실행

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m strategies.stage41_cagr_guardrail_objectives.cagr_guardrail_slsqp
```

## 파일

- `cagr_guardrail_slsqp.py`: 두 목적함수·제약·워크포워드·gate 구현
- `outputs/stage41_hard_mdd_monthly.csv`: A안 월별 비중·제약·성과
- `outputs/stage41_cdar_monthly.csv`: B안 월별 비중·제약·성과
- `outputs/performance_comparison.csv`: 전체·공통·잠금구간 성과
- `outputs/paired_block_bootstrap_vs_stage36.csv`: Stage36 대비 재표집
- `outputs/validation_report.json`: 불가능 월·인과성·제약·최종 gate 감사
- `tests/test_stage41_cagr_guardrail_objectives.py`: 공식·성과·제약 회귀검증

과거 백테스트는 미래성과를 보장하지 않는다.
