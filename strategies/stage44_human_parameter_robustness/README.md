# Stage44 — Stage36 Human Parameter Robustness

## 한 줄 결론

Stage36의 성과는 downside semivariance 앞의 임의 계수 `1`, CDaR 기준 `-16%`,
CDaR confidence `90%`에 크게 의존하지 않았다. 반면 **연변동성 13% cap은 실제
위험성과를 만드는 핵심 risk-governance 장치**였다. 다만 13%가 국소적인 성과
최적점은 아니었고, 10~15%에서 CAGR·Sharpe·MDD가 매끄러운 trade-off를 보였다.

따라서 가장 정확한 발표 문장은 다음과 같다.

> Stage36의 forecast edge와 parameter-free portfolio objective는 별도로 재현됐고,
> downside penalty 및 CDaR 세부값에 대한 의존성은 작았습니다. 변동성 cap은
> 성과를 만드는 숨은 알파계수가 아니라 수익과 위험의 위치를 정하는 명시적
> governance policy이며, 13%에서만 결과가 튀는 최적화 흔적은 발견되지 않았습니다.

“모든 human parameter가 없다”라고 주장하면 안 된다. 13%는 여전히 투자정책상
선택이다. 이번 Stage의 성과는 그 사실을 숨기는 대신 **forecast/portfolio core와
risk policy를 실증적으로 분리했다**는 데 있다.

## 연구 질문

다음 네 질문을 사전에 고정했다.

1. downside semivariance 계수 `1`을 제거해도 Stage36 결과가 유지되는가?
2. 변동성 13%와 CDaR -16% 제약 중 무엇이 실제 성과를 만들었는가?
3. 13%, -16%, confidence 90%가 주변값보다 유난히 좋아 보이는가?
4. 모든 risk guard를 제거한 core에도 return edge가 남는가?

대체 threshold 중 성과가 가장 좋은 경로를 채택하지 않았다. 모든 sensitivity는
보고용이며 Stage36 또는 Stage44의 새 운영전략으로 승격되지 않는다.

## Human-choice 목록

| 요소 | 분류 | Stage44 처리 |
|---|---|---|
| 분산항 계수 0.5 | 수학적 유도 | 유지 |
| downside semivariance 계수 1 | 연구자의 위험선호 | 모든 Stage44 경로에서 제거 |
| 연변동성 cap 13% | risk-governance 선택 | 제거 ablation + 10~15% 민감도 |
| CDaR limit -16% | risk-governance 선택 | 제거 ablation + -12~-20% 민감도 |
| CDaR confidence 90% | tail 정의 선택 | 80·85·90·95% 민감도 |
| 거래비용 15bp + 5bp | 실행 가정 | 유지, 실제 수익에서도 차감 |

## 무엇을 동결했나

Stage44는 Stage36의 edge model을 바꾸지 않는다.

- 네 가지 거시국면 확률
- VIX6 stress와 recovery 조정
- 일간 기술 신호와 confidence
- EPS revision 및 valuation 조정
- 신용 spread 확인과 KODEX200 분산 조정
- GVZ→GLD, OVX→USO variance-only mapping
- 월말 신호와 다음 달 수익의 인과적 정렬
- KODEX200·채권·GLD·USO 원화수익률
- 장기전용, 무레버리지, 완전투자, 비중합 1
- 단일자산 과반금지 없음
- drift 후 pre-trade 비중 및 거래비용

232개월 동안 Stage44가 재구성한 Stage36 forecast를 저장된 Stage36 비중에 다시
적용했다. 최대 절대오차는 다음과 같다.

```text
expected monthly return:   9.80e-17
expected monthly variance: 9.95e-17
```

부동소수점 오차 수준이다. 즉 Stage44의 차이는 `mu_t`, `Sigma_t` 또는 입력 edge가
달라져서 생긴 것이 아니다.

## 목적함수 정화

Stage36 목적함수:

```text
J36(w)
  = w' mu_t
    - 0.5 * w' Sigma_t w
    - downside_semivariance(R_hist w)
    - C(w,w_pre)
```

Stage44의 모든 경로:

```text
J44(w)
  = w' mu_t
    - 0.5 * w' Sigma_t w
    - C(w,w_pre)
```

downside semivariance 앞의 암묵적 계수 `1`만 제거했다. 나머지 `0.5`는 작은
수익률에서 다음 Taylor 근사로부터 나온다.

```text
E[log(1+r)] ≈ E[r] - 0.5 Var(r)
```

따라서 Stage44 목적함수에는 연구자가 조절하는 위험선호 가중치가 없다.
거래비용은 utility를 예쁘게 만들기 위한 penalty가 아니라 실제 백테스트에도
동일하게 차감되는 비용이다.

## 핵심 A/B/C/D 설계

| 경로 | 13% Vol cap | -16% CDaR | 의미 |
|---|---:|---:|---|
| Stage44 PF BothGuards | 유지 | 유지 | 목적함수만 정화한 clean ablation |
| Stage44 PF CDaROnly | 제거 | 유지 | volatility cap의 독립 효과 |
| Stage44 PF VolOnly | 유지 | 제거 | CDaR guard의 독립 효과 |
| Stage44 PF NoGuards | 제거 | 제거 | 완전 parameter-free portfolio core |

## 전체구간 2007-04~2026-07 결과

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Stage36 동결 | 10.499% | 9.472% | 1.105 | 2.202 | -12.407% | 0.846 |
| **PF + 두 guard** | **10.658%** | **9.655%** | **1.101** | **2.180** | **-12.326%** | **0.865** |
| PF + CDaR만 | 13.708% | 17.455% | 0.825 | 1.516 | -25.248% | 0.543 |
| PF + Vol만 | 10.660% | 9.655% | 1.101 | 2.180 | -12.326% | 0.865 |
| PF + guard 없음 | 15.328% | 20.171% | 0.809 | 1.485 | -26.836% | 0.571 |

### 해석 1: downside semivariance 계수에 의존하지 않았다

Stage36과 `PF + 두 guard`의 차이는 작다.

- CAGR: `+0.159%p`
- Sharpe: `-0.004`
- MDD: `+0.081%p`, 즉 소폭 개선
- 월수익 상관: `0.9972`
- 자산비중 평균 절대차: `1.171%p`

12개월 paired block bootstrap에서 PF+두 guard의 Stage36 대비 개선확률은
CAGR `76.7%`, Sharpe `40.6%`, MDD `34.6%`였다. 차이가 통계적으로 뚜렷하다고
주장할 수 없다. 이것이 오히려 목적함수 robustness에는 좋은 결과다.

> Stage36의 결과가 downside semivariance 계수 `1`을 잘 맞춰 얻은 결과라는 증거를
> 찾지 못했다.

### 해석 2: CDaR guard는 13% vol cap 아래에서 거의 중복된다

`PF + 두 guard`와 `PF + Vol만`은 사실상 같은 경로다.

- CAGR 차이: 약 `0.002%p`
- Sharpe 차이: 약 `0.0002`
- MDD: 소수점 이하에서 동일
- 월수익 상관: `0.9999995`
- 비중 평균 절대차: `0.0082%p`

두 guard를 쓸 때 CDaR constraint는 232개월 중 2개월만 binding이었고, CDaR를
제거한 VolOnly 결과도 거의 바뀌지 않았다. 따라서 Stage36의 낮은 MDD가
`-16%`를 절묘하게 고른 덕분이라는 설명은 데이터와 맞지 않는다.

### 해석 3: volatility cap은 실질적인 governance layer다

13% vol cap을 제거하고 CDaR만 남기면:

- CAGR `10.658% → 13.708%`
- Sharpe `1.101 → 0.825`
- MDD `-12.326% → -25.248%`
- 변동성 `9.655% → 17.455%`

두 guard를 모두 제거하면 CAGR은 `15.328%`까지 올라가지만 Sharpe `0.809`,
MDD `-26.836%`가 된다. 즉 forecast edge의 공격적인 return 성향은 남아 있지만,
Stage36의 위험조정 성과는 13% volatility governance에 크게 의존한다.

이 사실은 숨길 약점이라기보다 모델 역할을 분리하는 근거다.

```text
forecast / portfolio core → 기대수익과 위험의 상대가격
13% volatility policy     → 허용할 절대 위험수준
```

13%는 알파를 생성하는 coefficient가 아니라 투자자가 선택한 risk budget이다.

## 평균 자산배분

| 전략 | KODEX200 | 채권 | GLD | USO | 90% 초과 집중 개월 |
|---|---:|---:|---:|---:|---:|
| PF + 두 guard | 26.06% | 43.19% | 30.56% | 0.19% | 2 |
| PF + CDaR만 | 36.20% | 15.40% | 47.81% | 0.60% | 29 |
| PF + Vol만 | 26.08% | 43.18% | 30.55% | 0.19% | 2 |
| PF + guard 없음 | 42.30% | 8.90% | 48.21% | 0.60% | 60 |

vol cap을 제거하면 채권이 크게 줄고 KODEX200·GLD 및 단일자산 집중이 증가한다.
MDD 악화가 단순한 성과지표 계산 차이가 아니라 실제 자산배분 변화에서 나왔음을
보여준다.

## Volatility cap 10~15% 민감도

CDaR -16%, confidence 90%와 parameter-free objective를 고정했다.

| Vol cap | CAGR | Sharpe | MDD |
|---:|---:|---:|---:|
| 10% | 8.956% | 1.192 | -9.437% |
| 11% | 9.524% | 1.157 | -10.361% |
| 12% | 10.076% | 1.125 | -11.349% |
| **13%** | **10.658%** | **1.101** | **-12.326%** |
| 14% | 11.244% | 1.084 | -13.288% |
| 15% | 11.752% | 1.062 | -14.068% |

이 표에는 13%만 튀는 peak가 없다.

- cap을 높이면 CAGR은 대체로 매끄럽게 상승한다.
- cap을 낮추면 Sharpe가 상승하고 MDD가 개선된다.
- 13%는 CAGR 최고도, Sharpe 최고도, MDD 최고도 아니다.

따라서 “12와 14를 돌린 뒤 가장 좋은 13을 골랐다”는 형태의 흔적은 없다. 어떤
위험수준을 원하는지에 따라 연속적인 frontier에서 한 지점을 선택하는 정책 문제다.
민감도 범위 전체에서 CAGR 범위는 `2.796%p`, Sharpe 범위는 `0.131`, MDD 범위는
`4.631%p`로 크다. 즉 exact cap 값이 중요하지 않다고 주장해서도 안 된다.

## CDaR limit -12~-20% 민감도

volatility cap 13%와 confidence 90%를 고정했다.

| CDaR limit | CAGR | Sharpe | MDD |
|---:|---:|---:|---:|
| -12% | 10.634% | 1.098 | -12.968% |
| -14% | 10.785% | 1.117 | -12.326% |
| **-16%** | **10.658%** | **1.101** | **-12.326%** |
| -18% | 10.660% | 1.101 | -12.326% |
| -20% | 10.660% | 1.101 | -12.326% |

전체 범위에서 CAGR은 `0.151%p`, Sharpe는 `0.018`, MDD는 `0.641%p`만 움직였다.
-16%는 최고 CAGR이나 최고 Sharpe가 아니다. -18%부터는 guard가 한 번도 binding
되지 않아 VolOnly와 같아진다.

## CDaR confidence 80~95% 민감도

volatility cap 13%와 CDaR limit -16%를 고정했다.

| Confidence | CAGR | Sharpe | MDD |
|---:|---:|---:|---:|
| 80% | 10.660% | 1.101 | -12.326% |
| 85% | 10.660% | 1.101 | -12.326% |
| **90%** | **10.658%** | **1.101** | **-12.326%** |
| 95% | 10.823% | 1.121 | -12.448% |

CAGR 범위 `0.165%p`, Sharpe 범위 `0.020`, MDD 범위 `0.122%p`다. 90% confidence에
결과가 예민하게 맞춰져 있다는 증거는 없다. 95%가 오히려 CAGR과 Sharpe가 더
높지만, sensitivity 결과를 보고 95%로 교체하지 않는다.

## Binding 분석

| 경로 | Vol binding | CDaR binding |
|---|---:|---:|
| PF + 두 guard | 123/232개월 | 2/232개월 |
| PF + CDaR만 | 0 | 103/232개월 |
| PF + Vol만 | 125/232개월 | 0 |
| PF + guard 없음 | 0 | 0 |

Stage36 저장경로 자체에서는 vol cap이 106개월, CDaR가 3개월 binding이었다.
목적함수 정화 뒤에도 vol cap은 123개월에서 작동했다. 반면 CDaR guard는 vol cap이
있을 때 대부분 중복이지만, vol cap을 제거하면 103개월에서 위험을 제한한다.

따라서 CDaR가 원천적으로 무의미한 것이 아니라 **13% vol cap과 함께 있을 때
증분 영향이 작다**고 해석해야 한다.

## 과최적화 방지 장치

- Stage36 소스·월별 결과·검증 JSON 실행 전후 SHA-256 동일
- Stage36 forecast 232개월 수치 재현
- 모든 경로에서 target month 이전 데이터만 사용
- 4개 core ablation 사전고정
- sensitivity grid를 전략선택에 사용하지 않음
- sensitivity 최고 경로 승격 없음
- 입력 edge, `mu`, `Sigma`, 거래비용 변경 없음
- 중복 기준경로를 제외한 3,712개 월별 최적화에서 solver 성공, fallback 0회
- 전체·2010 공통·2018 잠금구간 모두 공개
- Stage36 대비 12개월 블록 paired bootstrap 공개

Sensitivity는 사후 robustness 진단이다. 10~15% 중 가장 좋은 metric을 찾아 운영
parameter를 교체하는 절차가 아니다.

## 발표에서 가능한 주장과 불가능한 주장

가능:

> downside semivariance의 임의 계수를 제거해도 Stage36 결과가 사실상 유지됐다.

> CDaR -16%와 confidence 90%의 주변값에서 결과 변화가 작았으며, 기존값이 성과
> 최고점도 아니었다.

> volatility cap 결과는 10~15%에서 매끄러운 risk-return frontier를 보였고 13%만
> 유난히 좋은 국소 최적점은 아니었다.

> threshold 대체값 중 최고 성과를 재채택하지 않았다.

불가능:

> Stage36에는 human choice가 전혀 없다.

> 13% volatility cap을 제거해도 위험조정 성과가 유지된다.

> 완전 parameter-free core가 Stage36과 같은 MDD·Sharpe를 만든다.

정확한 결론은 **return edge의 존재는 guard 제거 뒤에도 남지만, Stage36 수준의
위험통제는 명시적인 volatility governance를 필요로 한다**는 것이다.

## 실행

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m strategies.stage44_human_parameter_robustness.human_parameter_robustness
```

## 파일

- `human_parameter_robustness.py`: 전체 연구 구현
- `outputs/core_ablation_performance.csv`: Stage36과 A/B/C/D 성과
- `outputs/sensitivity_performance.csv`: 15개 sensitivity 경로 성과
- `outputs/sensitivity_summary.csv`: family별 metric 범위
- `outputs/constraint_binding_summary.csv`: guard binding과 solver 감사
- `outputs/weight_path_comparison.csv`: 비중차·수익률 상관
- `outputs/allocation_summary.csv`: 평균비중과 집중도
- `outputs/paired_block_bootstrap_vs_stage36.csv`: core 4경로 재표집 비교
- `outputs/human_parameter_inventory.csv`: 선택값 분류표
- `outputs/validation_report.json`: 전체 검증결과와 불변성 감사
- `tests/test_stage44_human_parameter_robustness.py`: 회귀 테스트

월별 CSV는 각 core 및 sensitivity 이름을 소문자로 바꾼 파일명으로 `outputs`에
저장된다.

## 최종 판단

이번 테스트로 가장 강하게 차단된 공격은 다음 두 가지다.

1. “downside semivariance 가중치 1을 잘 맞춰서 Stage36 성과가 나왔다.”
2. “CDaR -16%나 confidence 90%가 우연히 가장 좋은 값이라 골랐다.”

반면 volatility cap 공격은 말로 없애는 대신 성격을 바로잡아야 한다. 13%는 숨은
알파 parameter가 아니라 risk budget이며, 값에 따라 위험과 수익이 실제로 움직인다.
다만 결과가 13%에서만 튀지 않고 매끄럽기 때문에 backtest 최고점을 고른 흔적은
없다. 앞으로도 13%를 유지한다면 “수익 최적화값”이 아니라 “사전 위험정책”으로
명시하고 sensitivity 전체를 함께 공개하는 것이 가장 방어력이 높다.
