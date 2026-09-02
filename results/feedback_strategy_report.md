# Feedback strategy experiment report

## Decision

- Do **not** add the Regime-aware LightGBM signal to FinalBlend. Calibration selected a zero allocation shift, and the locked 2018-2026 directional AUC was 0.438.
- Do **not** promote the Jump Model, duration-aware HSMM approximation, engineered technical score, or non-LSTM technical ML ensemble.
- General HMM is only a weak research candidate: it improved locked Calmar and MDD, but its Sharpe improvement was small and disappeared under higher cost assumptions.
- A 15% volatility target is the only operationally robust candidate. It is a risk overlay rather than a predictive alpha factor, and it remained slightly better than FinalBlend at 3x transaction costs.

## Korean-market adaptation

The primary regime detector uses KOSPI200 20-day return, KOSPI200 20-day realized volatility, USDKRW 20-day return, and the KOSPI200 positive-day fraction. US variables are not used as the primary Korean regime. In the global-feature variants, all US observations are lagged by one Korean trading session.

The factor walk-forward starts in January 2005. The portfolio comparison starts in April 2007 because all four existing FinalBlend sleeves and signals do not share usable history before then. Model/shift selection uses data only through December 2017; January 2018 onward is locked.

## Regime-aware LightGBM

| Feature variant | AUC 2005-2026 | AUC 2018-2026 | Selected shift |
|---|---:|---:|---:|
| Korean only | 0.525 | 0.438 | 0.000 |
| Korean + lagged global | 0.495 | 0.435 | 0.000 |
| Original US-paper style | 0.477 | 0.393 | 0.000 |

Every non-zero LightGBM allocation shift reduced calibration CAGR, Sharpe, and MDD quality. Consequently, the selected LightGBM portfolio is identical to FinalBlend.

## Portfolio results

All values include 15 bps turnover cost, an additional 5 bps FX turnover cost, and leverage financing. `Selected shift = 0` means calibration rejected the factor, so its portfolio equals FinalBlend.

| Strategy | Selected shift | CAGR 2007-2026 | Sharpe | MDD | Calmar | CAGR 2018-2026 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FinalBlend | — | 14.30% | 1.024 | -14.90% | 0.960 | 17.07% | 1.164 | -14.90% | 1.145 |
| Regime LightGBM | 0.000 | 14.30% | 1.024 | -14.90% | 0.960 | 17.07% | 1.164 | -14.90% | 1.145 |
| Jump Model | 0.200 | 14.31% | 1.012 | -13.63% | 1.051 | 15.22% | 1.044 | -13.63% | 1.117 |
| General HMM | 0.075 | 14.45% | 1.034 | -14.23% | 1.016 | 17.55% | 1.178 | -12.47% | 1.407 |
| Duration-aware HSMM approximation | 0.000 | 14.30% | 1.024 | -14.90% | 0.960 | 17.07% | 1.164 | -14.90% | 1.145 |
| Engineered technical factor | 0.000 | 14.30% | 1.024 | -14.90% | 0.960 | 17.07% | 1.164 | -14.90% | 1.145 |
| Technical ML ensemble (no LSTM) | 0.000 | 14.30% | 1.024 | -14.90% | 0.960 | 17.07% | 1.164 | -14.90% | 1.145 |
| 15% volatility target | fixed | 14.67% | 1.037 | -13.04% | 1.126 | 17.67% | 1.181 | -12.74% | 1.388 |

## Robustness

The paired six-month block bootstrap did not establish Sharpe improvement: the probability of a positive Sharpe delta was 42.4% for General HMM and 26.5% for volatility targeting. The observed benefit is therefore better characterized as drawdown/Calmar improvement than statistically established alpha.

At 3x trading costs, full-period results were:

| Strategy | CAGR | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| FinalBlend | 13.56% | 0.978 | -15.56% | 0.872 |
| General HMM | 13.32% | 0.962 | -14.29% | 0.932 |
| 15% volatility target | 13.77% | 0.982 | -14.45% | 0.953 |

Volatility targeting was not uniformly superior: it lagged in 2013-2017 and 2018-2021, but was materially stronger in 2007-2012 and 2022-2026. General HMM also lacked cost robustness.

## Implemented variants

- Statistical Jump Model using 10-day downside deviation and 20/60-day Sortino features.
- General causal three-state Gaussian HMM.
- Duration-aware, minimum-duration/hysteresis approximation over causal HMM posteriors. This is not a fitted parametric HSMM package.
- Engineered MA-distance, RSI/reversal, and realized-volatility factor.
- Quarterly walk-forward ExtraTrees, histogram gradient boosting, and logistic regression ensemble; LSTM excluded.
- Fifteen-percent ex-ante volatility target with leverage clipped to 0.5-1.5.

