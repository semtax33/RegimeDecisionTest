# Stage 33 — WingAsym 미래위험·VIX6 맥락 검증

## 결론

Stage 33의 사전 판정 기준은 모두 실패했다. `WingAsym`은 VIX6와 최근 수익률, 직전 실현변동성, 거시 취약도를 통제한 뒤 다음 달 변동성·1개월 최대낙폭·3개월 최대낙폭·왼쪽꼬리 중 어느 것도 추가로 예측하지 못했다. `VIX6 × WingAsym`도 Stage20의 느린 재진입을 독립적으로 설명하지 못했다.

따라서 이 연구 가지의 최종 판정은 다음과 같다.

> `close_wingasym_branch_move_to_independent_information_source`

Stage20의 수익률 전망, 위험회피계수, 자산 비중에는 아무 변화도 주지 않았다. 다음 연구부터는 신용, 유동성, 실질금리, 기업이익처럼 옵션·VIX6와 경제적 원천이 다른 정보를 검토하는 편이 맞다.

## 고정한 질문

1. `Future Risk = VIX6 + WingAsym`에서 WingAsym이 VIX6 이후에도 미래위험을 설명하는가?
2. `Future Return/Defense Cost = VIX6 + WingAsym + VIX6×WingAsym`에서 상호작용이 Stage20의 false positive 또는 느린 재진입을 설명하는가?

결과를 본 뒤 Residual WingAsym을 만들거나, FearTermSlope의 부호를 뒤집거나, 분위수·문턱·기간을 다시 고르는 실험은 하지 않았다.

## 표본과 시간 정렬

- 연구기간: 2007-04~2026-07, 232개월
- WingAsym: 직전 월 마지막 관측일의 최근월물 `OTM2 Put IV - OTM2 Call IV`
- VIX6: Stage20에 저장된 직전 월말 스트레스 점수
- 미래위험 가격: Stage20이 사용하는 KODEX200 일별 종가. 2009년 이전은 기존 전략과 같은 KOSPI200 연결 프록시
- 왼쪽꼬리: 해당 월 수익률이 그 달 직전까지 관측된 월수익률의 5% 분위수 이하인지 여부. 최소 과거 60개월
- 2×2 고·저 상태: 각 신호의 직전 60개월 이상 expanding median. 전체표본 중앙값을 쓰지 않음
- 방어 기회비용: `Stage14 향후 복리수익 - Stage20 향후 복리수익`. 양수이면 Stage20의 방어가 수익을 놓친 것

예전에 언급된 `192개 방어월 / 110개 false positive`는 Stage14의 동적 λ를 별도 기준선과 비교한 Stage16 정의다. Stage33은 Stage20을 직접 다루므로 그 숫자를 재사용하지 않고, 저장된 Stage14와 Stage20의 실제 KODEX200 비중·수익률을 같은 달에 비교했다. 이 정의에서는 Stage20의 주식 비중이 더 낮은 달이 204개월이고, 그중 1개월 수익이 Stage14보다 낮은 달은 106개월이다.

## 미래위험 증분 검증

모든 설명변수는 회귀 표본 안에서 표준화했다. 연속 위험은 OLS-HAC, 왼쪽꼬리는 Binomial GLM-HAC를 사용했다. 3개월 MDD에는 HAC lag 3, 나머지는 lag 1을 적용했다.

| 미래위험 목표 | 표본 | WingAsym beta | p값 | 증분 적합도 | 판정 |
|---|---:|---:|---:|---:|---|
| 다음 1개월 실현변동성 | 231 | -0.00108 | 0.941 | -0.00177 Adj. R² | 실패 |
| 다음 1개월 최대낙폭 | 231 | +0.00036 | 0.962 | -0.00334 Adj. R² | 실패 |
| 다음 3개월 최대낙폭 | 231 | -0.00110 | 0.835 | -0.00344 Adj. R² | 실패 |
| 다음 달 왼쪽꼬리 | 232 / 사건 8 | -0.0682 | 0.842 | +0.00066 McFadden R² | 실패 |

양(+)의 부호는 앞으로 위험이 커진다는 뜻이다. 네 목표 중 세 개는 오히려 음수였고, 유일하게 양수인 1개월 MDD도 사실상 0이며 p=0.962다. “옵션 보험료가 VIX6보다 미래위험을 더 잘 알려준다”는 설명은 이 표본에서 지지되지 않는다.

## 왼쪽꼬리 expanding OOS

분류기는 무가중 로지스틱 회귀이며, 매월 과거 자료만으로 다시 적합했다. 결과를 보고 class weight나 규제강도는 조정하지 않았다. 첫 예측은 훈련표본에 꼬리사건이 3개 이상 쌓인 2018-11이고, 2026-07까지 93개월 중 사건은 5개뿐이다.

| 모형 | AUC | Brier | Log loss |
|---|---:|---:|---:|
| 과거 발생률 | 0.402 | 0.05176 | 0.22392 |
| VIX6 단독 | **0.570** | **0.05096** | **0.21277** |
| VIX6 + WingAsym | 0.430 | 0.05231 | 0.25140 |
| 전체 통제 | 0.452 | 0.05405 | 0.28292 |

WingAsym을 더하자 AUC가 0.570에서 0.430으로 낮아지고 Brier도 나빠졌다. 다만 OOS 사건이 5개뿐이므로 이 표 하나만으로 강한 결론을 내리지는 않았다. 회귀계수, 적합도, 변동성·낙폭 결과가 모두 같은 방향이라는 점을 함께 보고 실패로 판정했다.

## VIX6×WingAsym 재진입 가설

원시 2×2 평균만 보면 `VIX6 High / Wing High` 58개월은 매력적으로 보인다.

| 상태 | 월수 | 향후 1M | 향후 3M | 향후 6M | Stage20 방어비용 3M | 방어비용 6M |
|---|---:|---:|---:|---:|---:|---:|
| VIX6 Low / Wing Low | 72 | +0.45% | +0.16% | -0.26% | -0.35% | -0.53% |
| VIX6 High / Wing Low | 64 | +1.04% | +3.91% | +8.68% | +0.91% | +1.60% |
| VIX6 Low / Wing High | 28 | -0.81% | -0.34% | +0.53% | +0.09% | -0.12% |
| **VIX6 High / Wing High** | **58** | **+2.39%** | **+7.69%** | **+15.04%** | **+1.23%** | **+2.64%** |

하지만 이 표는 VIX6 자체가 높은 효과와 WingAsym 자체가 높은 효과를 분리하지 못한다. 그래서 연속형 상호작용 회귀를 사전 주검정으로 정했다. 통제 후 상호작용 beta는 3개월 `-0.0121 (p=0.022)`, 6개월 `-0.0135 (p=0.079)`였다. 가설이 요구한 양(+)이 아니라 유의한 음(-)이다.

Stage20 방어 기회비용에 대한 상호작용도 1개월만 `+0.00148 (p=0.040)`였고, 3개월 `-0.00208 (p=0.197)`, 6개월 `-0.00310 (p=0.086)`였다. 1개월 false-positive 로짓의 상호작용은 `+0.110 (p=0.454)`로 유의하지 않았다.

즉 high-high 구간의 높은 이후 수익과 방어비용은 실제로 관찰되지만, 그것이 “두 공포계가 동시에 높아서 생기는 추가 정보”라는 증거는 없다. VIX6와 WingAsym의 개별 수준, 최근 수익률, 실현변동성, 거시 취약도를 빼고 나면 상호작용은 반대 부호로 돌아선다. 이 차이는 사후적으로 문턱 규칙을 만드는 것을 멈춰야 하는 이유다.

## 사전 게이트 결과

### 독립 위험센서

- 4개 목표 중 양의 부호가 3개 이상: 실패
- 양의 부호이면서 p<10%가 2개 이상: 실패
- MDD 또는 꼬리 중 하나가 유의: 실패

### VIX6 맥락·재진입

- 3·6개월 상호작용이 모두 양수: 실패
- 둘 중 하나가 p<10%: 실패
- high-high 상태의 3·6개월 방어비용이 모두 양수: 통과
- high-high 인과표본 15개월 이상: 통과

핵심 통계 검정이 실패했으므로 보조적인 2×2 평균만으로 전략을 승격하지 않았다.

## 실행

프로젝트 루트에서 다음 명령을 실행한다.

```powershell
$py = 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe'
& $py -m strategies.stage33_wingasym_risk_context_validation.wingasym_risk_context_validation
```

주요 파일:

- `wingasym_risk_context_validation.py`: 데이터 정렬, 미래위험, 회귀, expanding OOS, 2×2 상태, 게이트 판정
- `stage33_wingasym_risk_context_report.html`: 해석과 연구 종료 판단
- `outputs/validation_report.json`: 모든 설정·판정·무결성 manifest
- `outputs/future_risk_incremental_regressions.csv`: 미래위험 회귀
- `outputs/left_tail_expanding_oos_scores.csv`: AUC·Brier·log loss
- `outputs/vix6_wing_context_interactions.csv`: 수익률·방어비용·false-positive 상호작용
- `outputs/causal_2x2_state_diagnostic.csv`: 인과적 2×2 기술통계

## 다음 연구 방향

WingAsym, FearTermSlope, OTM 버킷, Residual WingAsym을 더 파지 않는다. 다음 후보는 옵션 공포와 겹치지 않는 경제적 원천이어야 한다. 우선순위는 신용 스프레드·자금시장 유동성, 실질금리·금융여건, 기업이익 수정·수출이익 사이클이다. 각 원천도 하나의 가설, 고정된 변환, expanding 검증, Stage20 동결이라는 같은 규율로 시작한다.

