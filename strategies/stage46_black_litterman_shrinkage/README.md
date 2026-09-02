# Stage 46 — Guarded Black–Litterman + Ledoit–Wolf

Stage 36을 기준선으로 고정하고 Black–Litterman(BL)을 작은 의견 슬리브로만
사용한 4자산 배분 연구다. Stage 45처럼 기대수익을 새 ML 모델이 직접
예측하지 않는다. Stage 36의 기존 기대수익을 BL view로 번역하고, view가
위험제약을 통과할 때만 Stage 36 비중에 작게 섞는다.

최종 `Stage46_GuardedBlackLitterman_LW`는 사전 정의한 전체·공통·잠금
세 구간에서 요청된 CAGR·Sharpe·MDD·turnover 조건을 모두 통과했다.

## 고정 실험

피드백대로 성과표에는 세 경로만 둔다.

| 실험 | 기대수익 | 공분산·배치 | 역할 |
| --- | --- | --- | --- |
| A | Stage 36 | Stage 36 | 동결 기준선 |
| B | Stage 36 | 252일 constant-correlation LW | LW 단독 ablation |
| C | BL posterior | LW risk veto + Stage 36 fallback | 최종 guarded BL |

BL 계산은 다음과 같다.

```text
과거 Stage36 비중의 인과적 확장평균 w0
                  +
       252일 Ledoit–Wolf Sigma
                  ↓
          Pi = delta Sigma w0

Stage36 기대수익 view Q
레짐 확률 집중도 confidence
                  ↓
      Black–Litterman posterior
                  ↓
      Stage36 주변의 작은 BL tilt
```

- 전략적 prior는 현재월을 제외한 과거 Stage 36 비중의 확장평균이다.
- 위험회피계수는 과거 Stage 36 gross return의 평균/분산으로 인과적으로
  계산한다.
- view confidence는 네 레짐 확률의 정규화 Herfindahl 집중도다. 균등한
  `25/25/25/25`이면 0, 한 레짐이 100%이면 1이다.
- `Omega`는 Idzorek confidence ratio로 정한다.
- 새 신호, MLP, target 수익률 학습, parameter grid는 없다.

## Guarded 배치 규칙

원시 BL 비중은 2018년 이후에는 우수했지만 2012~2013년 채권·금 집중으로
전체 MDD를 악화시켰다. 따라서 원시 BL을 그대로 배치하지 않는다.

최대 BL tilt는 자산 수 `N=4`에서 `1/(N×(N+1))=5%`다. 현재 전략이
drawdown에 들어가면 이 5%를 연 13% 위험한도의 월 환산값에 걸쳐 0까지
선형 축소한다. 최종 tilt가 LW 연변동성 13% 또는 과거 CDaR -16%를 넘으면
그 달은 검증된 Stage 36 비중을 그대로 사용한다. 전체 220개 활성월 중
20개월에서 이 veto가 작동했다.

Stage 36의 long-only, 완전투자, GVZ→GLD, OVX→USO, 거래비용과 CDaR
방어선은 그대로 유지한다.

## 성과

거래비용 차감 후 2007-04~2026-07 결과다.

| 구간 | 전략 | CAGR | Sharpe | MDD | 월평균 turnover |
| --- | --- | ---: | ---: | ---: | ---: |
| 전체 | Stage 36 | 10.50% | 1.1049 | -12.41% | 3.70% |
| 전체 | Guarded BL+LW | **10.52%** | **1.1053** | **-12.34%** | 3.78% |
| 2010+ | Stage 36 | 9.23% | 1.0592 | -12.41% | 2.79% |
| 2010+ | Guarded BL+LW | **9.26%** | **1.0602** | **-12.34%** | 2.89% |
| 2018+ | Stage 36 | 12.63% | 1.2235 | -11.93% | 2.62% |
| 2018+ | Guarded BL+LW | **12.73%** | **1.2296** | **-11.82%** | 2.74% |

요청 조건은 다음과 같이 판정한다.

- Sharpe: 세 구간 모두 Stage 36 이상
- MDD: 세 구간 모두 Stage 36보다 덜 나쁨
- CAGR: 세 구간 모두 Stage 36 이상이며, 허용한 -50bp보다 충분히 높음
- turnover: Stage 36 대비 전체 +2.2%, 2010+ +3.5%, 2018+ +4.8%로
  사전 상한 +10% 이내
- 모든 비중은 long-only이고 합계 1
- Stage 36·45 동결 파일은 실행 전후 SHA-256 동일

## Paired block bootstrap

12개월 paired block, 2,000회 재표집에서 `delta > 0` 확률이다.

| 구간 | CAGR 개선확률 | Sharpe 개선확률 | MDD 개선확률 |
| --- | ---: | ---: | ---: |
| 전체 | 70.40% | 55.15% | 81.30% |
| 2010+ | 73.80% | 59.35% | 82.35% |
| 2018+ | 95.45% | 85.45% | 83.55% |

세 구간 모두 Sharpe와 MDD 개선확률이 50%를 넘지만, 개선폭 자체는 작다.
즉 BL이 Stage 36을 대체했다기보다 보수적인 tilt가 위험조정성과를 소폭
보완한 결과로 해석해야 한다.

## 실행

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
(& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\Activate.ps1')
python -m strategies.stage46_black_litterman_shrinkage.black_litterman_shrinkage_slsqp
python -m pytest tests\test_stage46_black_litterman_shrinkage.py -q
```

## 산출물

- `black_litterman_shrinkage_slsqp.py`: BL·LW·guard·백테스트 전체 코드
- `outputs/stage46_guarded_blacklitterman_lw_monthly.csv`: 최종 월별 경로
- `outputs/stage46_stage36mu_lw_monthly.csv`: LW-only ablation
- `outputs/stage46_blacklitterman_lw_shadow_monthly.csv`: 배치 전 BL shadow
- `outputs/performance_comparison.csv`: 세 경로 성과표
- `outputs/paired_block_bootstrap_vs_stage36.csv`: paired bootstrap
- `outputs/validation_report.json`: 게이트·인과성·solver·동결 해시 감사

연 13%는 사전 추정 위험한도이지 실현 변동성 보장이 아니다. 과거
백테스트 결과는 미래 성과를 보장하지 않는다.

