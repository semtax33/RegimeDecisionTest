# Stage 20 — 일별 K-Ratio·ATR·주식 상대강도 입력 전략

## 결론

Stage 14 고정 λ 전략을 출발점으로 삼아 네 자산의 일별 K-Ratio와 ATR을 추가하고, KODEX200에는 가격 RSI와 거래량 RSI를 함께 넣었다. Stage 14 원본은 수정하지 않았다.

결과는 명확한 위험·수익 교환관계다.

- 전체 Sharpe: **0.851 → 0.987**
- 전체 MDD: **-23.19% → -14.01%**
- 전체 변동성: **13.34% → 9.60%**
- 전체 CAGR: **11.00% → 9.40%**
- 단일자산 50% 초과: **128개월 → 34개월**
- 단일자산 90% 초과: **2개월 → 0개월**

따라서 Stage 20은 Stage 14보다 손실과 집중도를 크게 낮춘 위험관리형 대안이지만, CAGR 10% 이상을 요구한다면 자동 대체할 수 없다.

## 일별 자료

| 자산 | 가격 자료 | 최초 일자 | 기술지표 최초 일자 | 처리 |
|---|---|---:|---:|---|
| KODEX200 | KOSPI200 proxy + KODEX200 ETF OHLCV | 2000-01-04 | 2000-07-07 | 2009년 4월부터 ETF로 연결 |
| BOND | KRX 채권 총수익지수 종가 | 2006-03-02 | 2006-07-05 | 고가·저가가 없어 close-to-close ATR 사용 |
| GLD | GLD ETF OHLCV | 2004-11-18 | 2005-05-19 | USD/KRW를 곱해 원화 가격으로 계산 |
| USO | USO ETF OHLCV | 2006-04-10 | 2006-10-06 | USD/KRW를 곱해 원화 가격으로 계산 |

모든 자산은 2007-04 첫 백테스트 이전부터 126개 이상의 일별 관측치를 확보한다.

KODEX200의 2009년 이전 proxy 거래량과 이후 ETF 거래량은 단위가 다르다. 원 거래량을 직접 이어 붙이지 않고 각 구간에서 거래량 RSI를 따로 계산한 뒤 0~100 지표만 연결했다.

## K-Ratio

각 자산의 최근 126거래일 로그가격을 시간에 회귀한다.

```text
log(price_t) = intercept + slope × t + residual_t

K-Ratio
  = slope / [standard_error(slope) × sqrt(126)]
```

126일은 약 6거래월이다. 단순히 시작가격과 종료가격만 비교하는 모멘텀과 달리 K-Ratio는 상승·하락 기울기와 그 경로의 일관성을 함께 본다.

optimizer 입력에는 다음과 같이 `-1~1`로 유계화한다.

```text
KScore = K-Ratio / [1 + abs(K-Ratio)]
```

K-Ratio의 정의는 Kestner의 1996년 정의를 사용했다.

- <https://m.cdn.blog.hu/vi/vilagbagoly/nonimage/MEASURI.pdf>

## ATR

일별 True Range는 다음 세 값의 최댓값이다.

```text
high - low
abs(high - previous_close)
abs(low - previous_close)
```

이를 Wilder 방식으로 14일 평활하고 가격으로 나눠 NATR을 만든다.

```text
NATR = ATR(14) / close
```

각 자산의 NATR을 그날까지 알려진 자기 과거 안에서만 백분위화한다. 최신 일별 변동성이 높을수록 Stage 14 조건부 공분산의 해당 자산 위험을 다음과 같이 높인다.

```text
variance_scale_i = 1 + causal_ATR_percentile_i

D_i = sqrt(variance_scale_i)

Sigma_technical = D × Sigma_Stage14 × D
```

상관계수 구조는 유지하면서 각 자산 분산을 최대 2배까지 올릴 수 있다. ATR의 기본 기간 14일과 True Range 계산은 TA-Lib의 공식 구현과 같다.

- <https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ATR.c>

채권 총수익지수에는 장중 고가·저가가 없다. `high=low=close`로 두므로 채권 ATR은 일별 절대 가격변화와 갭을 측정하는 대용치다. ETF OHLC ATR과 완전히 같은 정보량을 가진다고 해석하면 안 된다.

## KODEX200 가격·거래량 상대강도

가격 상대강도는 Wilder 가격 RSI(14)를 사용한다.

```text
price_strength = [price_RSI(14) - 50] / 50
```

거래량 상대강도는 상승일 거래량과 하락일 거래량을 각각 Wilder 방식으로 평활한다.

```text
volume_RSI
  = 100 × smoothed_up_volume
    / [smoothed_up_volume + smoothed_down_volume]

volume_strength = [volume_RSI - 50] / 50
```

KODEX200 방향 점수는 세 지표를 동일 가중한다.

```text
equity_technical_direction
  = mean(KScore, price_strength, volume_strength)
```

다른 자산은 KScore가 기술적 방향 점수다. RSI 역시 TA-Lib의 기본 14일 Wilder 정의를 따른다.

- <https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_RSI.c>

## 매크로 기대수익과 결합

네 매크로 기대수익의 단순 평균을 중립값으로 둔다.

```text
macro_direction_i = sign(macro_mu_i - neutral_mu)

confidence_i
  = [1 + macro_direction_i × technical_direction_i] / 2

filtered_macro_mu_i
  = neutral_mu
    + confidence_i × [macro_mu_i - neutral_mu]
```

기술적 방향이 매크로 상대 전망과 일치하면 전망을 더 보존하고, 충돌하면 중립값으로 축소한다. K-Ratio나 RSI 수익률을 알파로 직접 더하지 않는다.

VKOSPI/VIX6 스트레스 기대수익 조정은 기존 Stage 14와 동일하게 마지막에 더한다. λ=1, 거래비용, 하방 semivariance, 연 변동성 13% 및 CDaR 16% guard, 롱온리·무레버리지·가중치 합 100% 조건도 유지한다.

## 성과

거래비용 차감 후 월별 성과다. 2026년 8월은 월간 자산수익률이 완결되지 않아 2026년 7월까지 측정했다.

### 전체 구간: 2007-04~2026-07, 232개월

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Stage 14 고정 λ | 11.0041% | 13.3417% | 0.8512 | 1.5713 | -23.1921% | 0.4745 |
| Stage 20 일별 기술지표 | 9.3974% | 9.5981% | **0.9865** | **1.8567** | **-14.0148%** | **0.6705** |
| 변화 | -1.6067%p | -3.7436%p | +0.1353 | +0.2854 | +9.1773%p | +0.1961 |

### 고정 검증 구간: 2018-01~2026-07, 103개월

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD |
|---|---:|---:|---:|---:|---:|
| Stage 14 고정 λ | 16.1276% | 14.2741% | 1.1229 | 2.3290 | -15.8980% |
| Stage 20 일별 기술지표 | 12.9392% | 10.1861% | **1.2503** | **2.5849** | **-11.4190%** |

## 입력변수 자체 진단

직전 월말 신호와 다음 투자월 수익률의 순위상관이다. 이 수치로 지표나 기간을 선택하지 않았으며 사후 설명용이다.

| 자산 | KScore → 다음 월 수익률 IC | ATR 백분위 → 다음 월 절대수익률 IC |
|---|---:|---:|
| KODEX200 | +0.059 | +0.291 |
| BOND | +0.155 | +0.198 |
| GLD | +0.104 | +0.237 |
| USO | +0.045 | +0.198 |

K-Ratio는 네 자산 모두 전체 구간에서 양의 방향 IC였다. ATR 백분위도 네 자산 모두 다음 달 절대수익률과 양의 관계를 보여 일별 위험 보정의 방향을 지지한다.

반면 KODEX200 가격강도 IC는 -0.020, 거래량강도 IC는 -0.007이었다. 두 RSI 입력은 요청대로 포함했지만 월간 방향예측력이 있다고 주장할 수 없다. 이 결과를 보고 RSI 부호나 기간을 바꾸지 않았다.

## 낙폭과 집중도

2012-10~2014-10 금 하락 구간에서:

| 항목 | Stage 14 | Stage 20 |
|---|---:|---:|
| 평균 GLD 비중 | 49.47% | 30.61% |
| 해당 낙폭 | -23.19% | -14.01% |
| GLD 손실 기여 | -23.69% | -12.11% |

전체 집중도도 완화됐다.

| 지표 | Stage 14 | Stage 20 |
|---|---:|---:|
| 평균 최대 단일자산 비중 | 51.85% | 45.39% |
| 50% 초과 월 | 128 | 34 |
| 90% 초과 월 | 2 | 0 |
| 관측 최대 비중 | 99.58% | 87.87% |

## 해석과 판단

Stage 20은 Stage 19의 단순 6·12개월 방향 필터보다 경로의 일관성을 보는 K-Ratio와 최신 일별 변동성을 보는 ATR을 사용한다. 그 결과 채권 80~100% 집중 문제도 나타나지 않았고 Sharpe·MDD·집중도가 동시에 개선됐다.

그러나 CAGR은 9.40%로 Stage 14보다 1.61%p 낮다. 따라서:

- 손실 제한과 위험조정 성과가 우선이면 Stage 20이 더 적합하다.
- CAGR 10% 이상 보존이 필수라면 Stage 14를 유지해야 한다.
- 이번 결과를 이용해 ATR 배율이나 K-Ratio 기간을 재조정하면 사후 최적화가 되므로 수행하지 않았다.

백테스트는 미래 실현성과를 보장하지 않는다.

## 파일

- `daily_technical_confidence_slsqp.py`: 일별 데이터, 지표, SLSQP와 전체 검증
- `outputs/daily_technical_features_KODEX200.csv`: 주식 K-Ratio·ATR·가격/거래량 강도
- `outputs/daily_technical_features_BOND.csv`: 채권 K-Ratio·ATR
- `outputs/daily_technical_features_GLD.csv`: 원화 GLD K-Ratio·ATR
- `outputs/daily_technical_features_USO.csv`: 원화 USO K-Ratio·ATR
- `outputs/monthly_technical_signals.csv`: 실제 투자월에 사용한 직전 월말 신호
- `outputs/daily_technical_confidence_monthly.csv`: 월별 비중·수익률·모든 입력값
- `outputs/performance_comparison.csv`: 전체·2018년 이후 성과
- `outputs/feature_diagnostics.csv`: 지표별 IC 진단
- `outputs/drawdown_episode_attribution.csv`: 낙폭별 자산 기여
- `outputs/validation_report.json`: 데이터·공식·성과·제약조건 감사
- `tests/test_stage20_daily_technical_confidence.py`: 지표 공식·일별 자료·인과성·성과 회귀 테스트

## 실행

```powershell
& D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe -m strategies.stage20_daily_technical_confidence.daily_technical_confidence_slsqp
```

```powershell
& D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe -m pytest tests/test_stage20_daily_technical_confidence.py -q
```

