# Short-regime prediction improvement report

## Conclusion

The most useful improvement is to replace symmetric 10-day direction prediction with a **meta-label for a large loss in the portfolio that is actually being traded**. The model should only reduce risk when the predicted loss risk is unusually high; it should not increase equity exposure when the signal is quiet.

The best predictive specification was a balanced logistic regression for a next-month FinalBlend loss below 3%, using domestic portfolio state plus lagged global stress auxiliaries. Its locked 2018-2026 ROC AUC was 0.814 and it captured 50% of loss months in the highest-risk 20% of observations. The best economic specification was the domestic-only loss-below-4% overlay, but its weak calibration-period classifier AUC makes it a research candidate rather than a production-ready rule.

## Why the old target failed

- Ten-day KODEX200 up/down is close to a noisy coin flip and is not the same objective as protecting FinalBlend.
- A symmetric score increases exposure on a low-risk reading even though the empirical benefit is almost entirely drawdown protection.
- A ten-day forecast used only once per month has a horizon/rebalance mismatch.
- Complex tree models did not add value over simple persistent-state or regularized linear models.

## Tested improvements

### Volatility-scaled daily tail event

Target: whether the next 10 or 20 trading days contain a maximum adverse excursion worse than 1.25 times current volatility. The model was a quarterly-refit, five-year rolling Korean-only LightGBM. Allocation was reduced only above the causal trailing 80th risk percentile.

| Model | Locked AUC | AP | Recall at top 20% | Locked Sharpe | Locked MDD |
|---|---:|---:|---:|---:|---:|
| Original 10-day direction LightGBM | 0.438 | — | — | 1.164 | -14.90% |
| 10-day tail-risk | 0.499 | 0.198 | 15.0% | 1.164 | -14.90% |
| 20-day tail-risk | 0.561 | 0.244 | 30.0% | 1.166 | -14.90% |

The target redesign improved prediction, but monthly execution made its economic effect negligible.

### FinalBlend loss meta-label

Target: a loss in the next monthly FinalBlend return. Features use only information available before the target month, and training applies a conservative one-month label embargo. Risk is reduced only when the causal probability percentile exceeds 80%.

| Model | Locked AUC | Recall at top 20% | Selected max shift | CAGR | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|
| FinalBlend baseline | — | — | — | 17.07% | 1.164 | -14.90% | 1.145 |
| Loss below 3%, stress auxiliaries | 0.814 | 50.0% | 10% | 16.47% | 1.173 | -13.47% | 1.223 |
| Loss below 4%, domestic only | 0.688 | 28.6% | 15% | 16.82% | 1.195 | -12.39% | 1.357 |
| Loss below 4%, stress auxiliaries | 0.707 | 42.9% | 10% | 16.81% | 1.189 | -13.35% | 1.259 |

The domestic-only loss-4% variant had the best portfolio result, but its calibration AUC was only 0.483. The stress-assisted loss-4% version had calibration AUC 0.455. The loss-3% stress model was more consistent as a predictor (calibration AUC 0.662 and locked AUC 0.814), but its protection reduced CAGR more.

The six-month block bootstrap probability of a positive Sharpe delta was 62.1% for the domestic loss-4% overlay and 72.7% for the stress-assisted version. Neither clears a strong statistical-evidence threshold.

## Robustness

At three times normal costs, full-period results were:

| Strategy | CAGR | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| FinalBlend | 13.56% | 0.978 | -15.56% | 0.872 |
| Loss-4% domestic | 13.54% | 0.992 | -14.34% | 0.944 |
| Loss-4% stress | 13.49% | 0.986 | -14.34% | 0.941 |

The overlay remained useful for drawdown control but not as return alpha. It lagged during 2018-2021 and worked best during 2022-2026, so the improvement is not uniform across regimes.

## Recommended architecture

1. Predict `P(next-month FinalBlend return < -3% or -4%)`, not KODEX200 direction.
2. Use a regularized logistic model as the primary estimator. Keep LightGBM only as a challenger.
3. Use domestic momentum, downside volatility, correlation, drawdown, and current portfolio state as primary features.
4. Treat lagged VIX and credit/financial-stress series as auxiliary inputs only. They improved loss-4% locked AUC from 0.688 to 0.707, but did not improve the portfolio result.
5. Convert raw probability to a causal 36-60 month percentile and act only in the highest-risk 20%.
6. Apply a one-sided 5-10% de-risking tilt initially. Avoid binary 75-100% protection and avoid risk-on boosts.
7. Keep 15% volatility targeting as a separate sizing layer; the classifier decides when to brake, and volatility targeting decides how hard.

## Highest-value missing data

The local database currently contains price series but not the following Korean market internals. These are more likely to add incremental short-regime information than another technical indicator:

- KOSPI/KOSDAQ advance-decline and percentage above 20/60-day moving averages.
- New highs/lows and up-volume/down-volume breadth.
- Foreign investor spot/futures net flow.
- KOSPI200 futures basis, put-call ratio, and options open interest/skew.
- VKOSPI level and term structure.

True 10-20 day operation should also move from monthly to weekly inference, but only with a soft exposure change. Earlier hard daily protection tests in this repository materially over-traded and lost return.

