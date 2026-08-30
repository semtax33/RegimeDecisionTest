# Stage 10 — VIX6 조건부 위기 라우터

이 폴더는 VIX6를 단순한 수익률 예측 변수나 ‘다섯 번째 자산’으로 쓰지 않는다.
기존 거시 국면이 만든 4자산 기본 비중을 출발점으로 삼고, KOSPI200·VKOSPI가
위험을 얼마나 줄일지 정한 다음, VIX6가 그 위험의 성격과 피난 방향을 정한다.

## 계층 구조

1. **Macro base allocation**: 성장·물가 확률과 기존 SLSQP 경로가
   KODEX200·채권·금·원유의 월별 기본 비중을 만든다.
2. **Risk budget**: 기존 Robust VKOSPI의 스트레스 강도가 주식과 원유에서 줄일
   비중을 정한다. VIX6는 이 총량을 무제한으로 키울 수 없다.
3. **Crisis type**: 직전 관측치만 사용한 VIX6 left tail, surface breadth,
   VKOSPI 수준, KOSPI 5일·21일 추세, 거시 물가확률을 고정 규칙으로 결합해
   여섯 상태를 분류한다.
4. **Execution**: 상태별로 4자산 이동 방향과 Put Spread·Call Spread·
   Covered Call 중 하나를 선택한다.

| 상태 | 의미 | 4자산 방향 | 옵션 구조 |
|---|---|---|---|
| RiskOn | tail·변동성 압력이 낮음 | 기존 Macro/VKOSPI 유지 | 없음 |
| HighIVRange | IV는 높지만 방향성과 tail이 약함 | 기존 비중 유지 | Covered Call |
| PreCrash | left tail 선행, 현물 하락은 미확인 | 주식·원유 축소, 금/채권 후보 | Put Spread |
| DeflationCrisis | left tail과 하락 확인, 물가확률 낮음 | 채권·금 방향 | Put Spread |
| InflationCrisis | left tail과 하락 확인, 물가확률 높음 | 금·원자재 방향 | 소규모 Put Spread |
| Recovery | left tail 완화와 현물 반등 | 기존 방어 축소 | Call Spread |

분류 임계값은 경제적 가설에 따라 코드에 고정했다. 자산 이동 후보만
`3 routing presets × 2 confirmation scales × 4 recovery relief = 24개`로 제한하며,
2007~2012 보정과 2013~2017 검증을 모두 통과해야 채택한다. 2018년 이후 성과는
후보 선택에 사용하지 않는다.

## 옵션 위험 관리

옵션은 자산 비중이 아니라 별도 위험장부로 관리한다.

- Put/Call Spread: 상태별 NAV 0.25~0.75%의 premium budget, 최대손실은 지급 프리미엄
- Covered Call: 당시 KODEX200 비중의 20%까지만 담보부 매도
- 공통 기록: `delta_equivalent`, `gamma_pnl_for_1pct_move`,
  `vega_pnl_for_1vol_point`, `max_loss_nav`
- 30~60 DTE, 목표 45 DTE, 거래량 1 이상, VKOSPI 스트레스 슬리피지와 최소 1틱 반영
- 원자료가 월중 연속 시계열이 아니라 일부 인접일 표본이므로, 같은 월의 다음
  공통 호가부터 청산할 수 있게 했다. 따라서 옵션 결과는 실행 가능성 진단이지
  실거래 성과 확정치가 아니다.

## 현재 검증 결론

24개 4자산 라우팅 후보는 사전 두 구간에서 CAGR·Sharpe·MDD 동시 개선 조건을
통과하지 못했다. 옵션 후보도 17건만 체결됐고 세 지표 개선에 실패했다. 따라서
선택 결과는 현재 Robust VKOSPI 경로를 그대로 복제하는 안전 폴백이다.

| 2007-04~2026-07 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| 선택된 4자산 경로 | 15.64% | 1.133 | -12.96% |
| 옵션 진단 후보 | 15.59% | 1.130 | -12.96% |

즉, 새 구조는 구현됐지만 성과가 나빠진 후보를 현재 전략으로 승격시키지는 않았다.

## 실행과 파일

```powershell
python -m strategies.stage10_vix6_router.vix6_conditional_router_strategy
python -m strategies.stage10_vix6_router.option_structure_overlay
```

- `vix6_conditional_router_strategy.py`: 6상태 분류, 4자산 라우팅, 24개 사전 후보,
  성과 gate와 안전 폴백
- `option_structure_overlay.py`: 실제 KOSPI200 옵션 호가에서 스프레드·커버드콜
  계약 선택, 슬리피지, Greeks와 위험예산, 채택 gate
- `results/vix6_router_validation.json`: 분류·선택·look-ahead·잠금 성과 감사
- `results/vix6_router_calibration.csv`: 24개 후보의 두 사전 구간 성과
- `results/vix6_router_daily.csv`: 일별 상태, 신호일, 자산비중과 진단점수
- `results/vix6_router_option_trades.csv`: 옵션별 계약·체결·위험장부
- `results/vix6_router_option_validation.json`: 옵션 성과와 선택 근거

재현 테스트는 `tests/test_vix6_conditional_router_strategy.py`와
`tests/test_vix6_router_option_overlay.py`에 있다.
