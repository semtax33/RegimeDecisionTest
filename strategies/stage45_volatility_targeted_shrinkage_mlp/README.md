# Stage 45 — Stage 36 호환 Volatility-Targeted Shrinkage Tiny MLP

이 폴더는 `volatility-targeted-shrinkage-mlp.py`의 아이디어를 Stage 36의
4자산 원화 투자 엔진에 포팅한 독립 연구 브랜치다. 결론부터 말하면 구현은
정상 작동했지만 성과가 Stage 36보다 명확히 나빠서 **실전략으로 승격하지
않고 연구용으로만 보존**한다.

## 무엇을 이식했고 무엇을 버렸나

원본에서 유지한 부분은 다음 세 가지다.

1. 작은 MLP로 자산별 다음 달 기대수익 `mu` 예측
2. 직전 252거래일 일수익률의 constant-correlation Ledoit–Wolf 공분산
3. `mu'w`를 최대화하되 연 10% 예상 변동성을 넘지 않는 최적화

개별주 횡단면 전용 구조는 제거했다.

| 원본 | Stage 45 포트 |
| --- | --- |
| 월별 횡단면 rank | 연 1회 재학습 구간의 time-series 표준화 |
| 월 횡단면 평균을 뺀 상대수익 target | 사전에 알려진 KTB 3년물 수익률 대용치 대비 실제 다음 달 초과수익 |
| 30종목 이상 universe | KODEX200·BOND·GLD·USO 고정 4자산 |
| 롱숏·gross 4배 | long-only, 위험자산 합계 100% 이하, 잔여 현금, 무레버리지 |
| 4층 128-wide MLP | 은닉층 1개·노드 4개·3 seed 평균 |
| 주식 pooled context | 레짐 확률·스트레스·금리/크레딧·자산별 모멘텀/변동성 |

Stage 36과의 연결은 다음과 같다.

- GVZ는 GLD 공분산 축만, OVX는 USO 공분산 축만 확대한다.
- 국내 거래비용과 해외자산 비중변화 비용은 Stage 36과 같다.
- 과거 90% CDaR -16% 방어선을 유지한다.
- 36개월 학습자료가 모이기 전에는 Stage 36 비중을 그대로 사용한다.
- 모든 학습 target, 특징, 공분산 관측치는 해당 투자월보다 앞선 정보만 쓴다.

## 성과

백테스트는 2007-04~2026-07이다. MLP는 최소 36개월 학습 후인 2011-01부터
활성화된다. 아래 수익률은 거래비용 차감 후다.

| 구간 | 전략 | CAGR | 변동성 | Sharpe | MDD | 월평균 회전율 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2011-01~2026-07 | Stage 36 | 8.71% | 8.80% | 1.00 | -12.41% | 2.68% |
| 2011-01~2026-07 | Tiny MLP + LW | 3.78% | 9.45% | 0.44 | -18.30% | 23.00% |
| 2018-01~2026-07 | Stage 36 | 12.63% | 10.18% | 1.22 | -11.93% | 2.62% |
| 2018-01~2026-07 | Tiny MLP + LW | 5.53% | 11.16% | 0.54 | -18.30% | 22.90% |
| 2007-04~2026-07 | Stage 36 | 10.50% | 9.47% | 1.10 | -12.41% | 3.70% |
| 2007-04~2026-07 | 포트(초기 Stage 36 포함) | 6.44% | 10.02% | 0.67 | -18.30% | 20.08% |

MLP 활성구간의 748개 자산-월 예측은 방향 정확도 51.74%였지만, 예측과
실현수익의 상관은 자산별로 거의 0이거나 음수였다. paired 12개월 block
bootstrap에서 Stage 36 대비 Sharpe 개선확률은 활성구간과 2018년 이후 모두
0%였다. 모델이 만든 불안정한 `mu`가 잦은 비중교체를 유발해 누적 거래비용도
활성구간 13.99%로 커졌다.

따라서 이 결과는 “신경망을 더 크게 만들자”는 근거가 아니라, 4자산 월별
표본에서 비제약적 MLP 기대수익 예측을 직접 최적화에 연결하면 추정오차와
회전율이 Stage 36의 안정적 레짐·위험 구조를 훼손할 수 있다는 반증이다.

## 재현

프로젝트 루트에서 다음을 실행한다.

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
(& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\Activate.ps1')
python -m strategies.stage45_volatility_targeted_shrinkage_mlp.volatility_targeted_shrinkage_mlp
python -m pytest tests\test_stage45_volatility_targeted_shrinkage_mlp.py -q
```

전체 실행은 하이퍼파라미터 탐색 없이 고정 설계 한 경로만 평가한다.

## 산출물

- `volatility_targeted_shrinkage_mlp.py`: 포팅·학습·최적화·백테스트 코드
- `outputs/stage45_monthly.csv`: 월별 예측, 비중, 수익, 제약 및 공분산 감사값
- `outputs/oos_forecasts.csv`: 748개 out-of-sample 자산-월 예측
- `outputs/performance_comparison.csv`: Stage 36과의 기간별 성과 비교
- `outputs/paired_block_bootstrap_vs_stage36.csv`: paired block bootstrap
- `outputs/validation_report.json`: 인과성·재현·solver·동결파일 해시 감사

## 해석상 한계

- KTB 3년물 금리를 12로 나눈 값을 현금수익 대용치로 사용했으며 실제 MMF나
  CD 수익률은 아니다.
- GLD·USO는 Stage 36과 동일하게 원화 환산 수익률이라 환율효과가 포함된다.
- 연 10%는 사전 추정 변동성 상한이며 실현 변동성이 반드시 10% 이하라는
  뜻은 아니다.
- 결과는 과거 백테스트이고 미래 성과를 보장하지 않는다.

