# Stage15 — Stage14 채권 슬롯의 HYG 교체 실험

## 결론

HYG 교체는 구현 방식에 따라 결과가 반대로 나왔다.

- **실행자산만 교체**: Stage14가 계산한 채권 목표비중은 유지하되 실제 보유자산을 HYG로 바꾸면 CAGR과 Sharpe가 상승했다. 대신 변동성과 MDD는 소폭 나빠졌다.
- **HYG로 완전 재최적화**: 조건부 평균·공분산까지 HYG 수익률로 다시 추정하면 Sharpe·MDD·변동성은 개선됐지만 CAGR은 하락했다.

따라서 “HYG를 쓰면 전반적으로 좋아진다”는 결론은 아니다. 수익률을 우선하면 실행자산 교체가 낫고, 위험을 낮추려면 완전 재최적화가 낫다.

## 성과 비교

모든 수치는 거래비용과 원화 환산을 반영했다. MDD 변화가 양수면 낙폭이 줄어든 것이다.

### 공통 전체 구간: 2008-04~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD | 최종 배수 |
|---|---:|---:|---:|---:|---:|
| Stage14 BOND 동적 λ | 10.15% | 12.89% | 0.816 | -22.97% | 5.88배 |
| HYG 실행자산 교체 | **11.02%** | 13.63% | 0.838 | -23.15% | **6.80배** |
| HYG 고정 λ 재최적화 | 9.74% | 11.42% | **0.874** | -21.99% | 5.50배 |
| HYG 동적 λ 재최적화 | 9.64% | **11.37%** | 0.870 | **-21.50%** | 5.41배 |

Stage14 대비 변화:

| HYG 구현 | CAGR | Sharpe | MDD | 변동성 |
|---|---:|---:|---:|---:|
| 실행자산 교체 | **+0.87%p** | **+0.021** | -0.18%p | +0.74%p |
| 동적 λ 완전 재최적화 | -0.51%p | **+0.054** | **+1.47%p** | **-1.52%p** |

### 고정 검증 구간: 2018-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD | 최종 배수 |
|---|---:|---:|---:|---:|---:|
| Stage14 BOND 동적 λ | 15.78% | 13.87% | 1.130 | -15.90% | 3.52배 |
| HYG 실행자산 교체 | **17.30%** | 14.52% | 1.176 | -16.51% | **3.93배** |
| HYG 고정 λ 재최적화 | 14.18% | **11.06%** | **1.261** | **-14.70%** | 3.12배 |
| HYG 동적 λ 재최적화 | 13.88% | 11.07% | 1.235 | -14.84% | 3.05배 |

Stage14 대비 실행자산 교체는 CAGR **+1.52%p**, Sharpe **+0.047**이지만 MDD는 0.62%p 나빠졌다. 완전 재최적화는 Sharpe **+0.106**, MDD **+1.06%p**, 변동성 **-2.80%p**이지만 CAGR은 **-1.90%p**다.

## 두 구현의 차이

### 1. HYG 실행자산 교체

Stage14의 모든 신호와 목표비중을 그대로 사용한다. 목표비중의 두 번째 슬롯이 20%라면 기존에는 국내 채권지수를 20% 보유했지만 이 경로에서는 HYG를 20% 보유한다.

Stage14의 목표비중은 모두 과거 정보만으로 계산되므로 이 경로도 인과적이다. 다만 비중 계산에는 여전히 기존 채권 수익률의 조건부 모멘트가 사용되므로, HYG 자체의 위험을 최적화에 직접 입력한 모델은 아니다. HYG의 추가 수익을 가장 순수하게 확인하기 위한 실행 귀속 경로에 가깝다.

### 2. HYG 완전 재최적화

두 번째 입력 열 자체를 HYG 원화 총수익률로 바꾼다. 각 월의 SLSQP는 다음을 HYG 자료로 다시 추정한다.

- 거시국면 확률로 가중한 조건부 기대수익과 공분산
- VKOSPI·VIX6 스트레스에 대한 조건부 수익률 반응
- 스트레스 가중 공분산
- 과거 하방 준분산과 CDaR

목적함수와 제약은 Stage14와 같다.

```text
maximize expected return - 0.5 × variance
         - (1 + stress) × downside semivariance
         - transaction cost

0 <= each weight <= 1
sum(weights) = 1
no cash, no leverage
ex-ante volatility guard, historical CDaR guard
```

## 왜 재최적화의 CAGR이 낮아졌나

공통 구간 평균 비중은 다음과 같이 바뀌었다.

| 전략 | KODEX200 | 채권/HYG | GLD | USO |
|---|---:|---:|---:|---:|
| Stage14 BOND | 32.46% | 19.56% | 47.66% | 0.32% |
| Stage15 HYG | 21.99% | 36.32% | 40.67% | 1.02% |

HYG 단독 CAGR은 같은 구간에서 7.40%로 기존 채권의 3.14%보다 높았다. 그러나 HYG는 국채형 방어자산이 아니라 신용위험을 가진 위험자산이다. SLSQP는 HYG 위험을 직접 학습한 뒤 KODEX200 비중을 약 10.5%p 줄이고 HYG 비중을 크게 높였다. 이 보수적인 재배치가 포트폴리오 변동성과 MDD를 낮췄지만 상승장에서의 주식 참여율과 CAGR도 낮췄다.

기존 채권과 원화 환산 HYG의 월수익률 상관계수는 0.036이었다. 두 자산은 이름에 모두 채권이 들어가지만 포트폴리오에서 거의 같은 역할을 하지 않았다.

## 데이터 처리

- HYG 공식 설정일: 2007-04-04
- Yahoo Finance 첫 가격 관측일: 2007-04-11
- 로컬 스냅숏 마지막 관측일: 2026-08-27
- `auto_adjust=True`로 배당과 분할을 반영한 일별 Open 사용
- 매월 첫 HYG 조정 Open에 같은 날 USD/KRW를 곱해 원화 수준 생성
- 다음 달 첫 거래일 원화 수준과 비교해 월수익률 생성
- HYG·GLD·USO 사이 이동은 순외화 흐름으로 환전비용 계산

공식 상품 설명에 따르면 HYG는 미국 달러 표시 투기등급 회사채 지수를 추종하며 월배당 상품이다.

- 공식 페이지: https://www.ishares.com/us/products/239565/HYG

## 평가 시작일과 2008년 제약 완화

조건부 모멘트에는 최소 12개월의 인과적 수익률이 필요하다. HYG 이전 자료를 임의의 대체지수로 채우지 않았기 때문에 첫 매매월은 2008-04이고, Stage14도 같은 시작일로 잘라 비교했다.

2008-12에는 네 자산으로 만들 수 있는 최소 예상 변동성이 13.0827%여서 Stage14의 13% 상한이 수학적으로 불가능했다. 이 한 달에만 상한을 13.0828%로 올렸다. 다른 219개월은 원래 13% 상한을 유지했다. 상한 완화는 검색된 파라미터가 아니라 해당 월의 장기전용 최소분산 해에서 직접 계산한다.

## 검증 결과

- 동적 HYG 재최적화 220개월 모두 SLSQP 성공
- fallback 0회
- 현금·레버리지 사용 0회
- 비중 합계 최대 오차 `3.33e-16`
- 최대 단일자산 비중 64.24%
- 50% 초과 집중 56개월, 90% 초과 0개월
- 거시 및 스트레스 신호가 매매월보다 앞서는지 확인
- 하이퍼파라미터 탐색 없음

## 실행

프로젝트 루트에서 다음 명령을 실행한다.

```powershell
$py = 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe'
& $py -m strategies.stage15_hyg_substitution.hyg_substitution
```

저장된 HYG 스냅숏을 최신 자료로 명시적으로 교체하려면 다음을 사용한다.

```powershell
& $py -c "from strategies.stage15_hyg_substitution.hyg_substitution import run_research; run_research(save=True, refresh_hyg=True)"
```

## 파일

- `hyg_substitution.py`: 데이터 처리, 두 HYG 경로, SLSQP, 비교 및 검증
- `data/hyg_daily_auto_adjusted.csv`: 고정된 HYG 조정가격 스냅숏
- `outputs/hyg_execution_stage14_weights_monthly.csv`: 실행자산만 HYG로 교체한 경로
- `outputs/hyg_dynamic_lambda_monthly.csv`: HYG 완전 재최적화 최종 경로
- `outputs/hyg_static_lambda_monthly.csv`: 고정 λ 비교 경로
- `outputs/performance_comparison.csv`: 절대 성과
- `outputs/stage15_minus_stage14.csv`: Stage14 대비 증감
- `outputs/bond_vs_hyg_asset_statistics.csv`: 채권과 HYG 단독 통계
- `outputs/validation_report.json`: 데이터·제약·검증 보고서

