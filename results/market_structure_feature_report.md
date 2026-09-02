# Market-structure feature experiment

## Decision

Pairwise sector correlation, KOSPI200 volume stress, sector breadth/dispersion, and KOSPI200 skewness are economically sensible additions when they are compressed into a few causal composites. Adding all raw variables directly caused severe rare-event overfitting; five pre-specified core economic composites worked materially better.

The MDD/Calmar-first candidate remains:

`FinalBlend loss-3% meta-label + domestic market-structure composites + one-sided 15% allocation shift + 15% volatility target`

It dominated both FinalBlend and the standalone 15% volatility target on full-period and locked CAGR, Sharpe, and MDD. An `index level x volume` activity proxy was subsequently tested. It produces a nearby Pareto alternative with slightly higher CAGR and Sharpe, but a worse MDD than the MDD-first candidate.

## Feature construction

All features use data available at least two calendar days before the target month.

- Systemic correlation stress: sector mean pairwise correlation, first correlation-matrix eigenvalue share, and inverse correlation dispersion.
- Breadth/dispersion stress: fraction of KRX sectors with positive 5/20-day returns and cross-sectional return dispersion.
- Volume stress: KOSPI200 63-day volume z-score, 20/63-day volume ratio, down-volume share, and signed volume imbalance.
- Tail-shape stress: economically fixed combination of negative 63/126-day skewness and positive 63-day excess kurtosis.
- Sector-linkage stress: KOSPI200-semiconductor, KOSPI200-bank, and cyclical-defensive 63-day correlations.

Each raw feature is standardized using only the preceding 60 monthly observations. The five composite signs were fixed by economic interpretation rather than optimized on the locked sample.

## Component ablation

Normal trading costs, 2007-04 to 2026-07:

| Strategy | CAGR | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| FinalBlend | 14.30% | 1.024 | -14.90% | 0.960 |
| Correlation composite | 14.32% | 1.057 | -13.99% | 1.023 |
| Breadth/dispersion composite | 14.19% | 1.046 | -13.99% | 1.014 |
| Volume composite | 14.21% | **1.063** | -13.99% | 1.016 |
| Skew/kurtosis composite | 14.27% | 1.035 | -13.99% | 1.019 |
| All five composites | **14.42%** | 1.056 | -13.99% | **1.030** |

Volume was the strongest standalone short-loss predictor: calibration AUC 0.645, locked AUC 0.844, and locked Sharpe 1.241. Its selected 20% exposure reduction sacrificed too much CAGR, so it is better used inside the balanced composite.

Pairwise correlation added stable risk-adjusted value and preserved full-period CAGR. The skew/kurtosis block helped, but was the weakest standalone addition. Raw KOSPI200 skewness should therefore not receive a large independent weight.

## Pareto comparison

| Strategy | Calibration CAGR | Sharpe | MDD | Locked CAGR | Sharpe | MDD | Full CAGR | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FinalBlend | 12.14% | 0.904 | -13.99% | 17.07% | 1.164 | -14.90% | 14.30% | 1.024 | -14.90% |
| 15% volatility target | 12.33% | 0.913 | -13.04% | 17.67% | 1.181 | -12.74% | 14.67% | 1.037 | -13.04% |
| Structure composite | 12.54% | 0.928 | -13.99% | 16.81% | **1.216** | -13.59% | 14.42% | **1.056** | -13.99% |
| Structure + 15% vol target | **12.60%** | 0.926 | **-13.04%** | **17.74%** | 1.196 | **-12.56%** | **14.85%** | 1.050 | **-13.04%** |

The composite alone maximizes Sharpe. The combined overlay is the better multi-objective solution because it improves CAGR, Sharpe, and MDD simultaneously relative to the baseline and standalone volatility target.

## Cost and subperiod robustness

At three times normal transaction costs:

| Strategy | Full CAGR | Sharpe | MDD | Locked CAGR | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|---:|
| FinalBlend | 13.56% | 0.978 | -15.56% | 16.16% | 1.109 | -15.56% |
| Structure composite | 13.59% | 1.004 | -14.34% | 15.82% | 1.151 | -14.14% |
| Structure + 15% vol target | **13.84%** | **0.988** | **-13.84%** | **16.53%** | **1.123** | **-12.78%** |

The combined model lagged during 2013-2017 and 2018-2021, but was substantially better during 2007-2012 and 2022-2026. The six-month block-bootstrap probability of a positive Sharpe delta was 81.8% for the composite alone but only 35.6% for the combined volatility-target portfolio. This is not sufficient for a strong statistical claim, especially after testing multiple variants.

## Index-level times volume proxy test

True market-cap features were not tested. The local database contains KRX index price and volume history but no point-in-time KOSPI200 constituent membership or free-float market capitalization. The current KRX endpoint also requires credentials in this environment. At the user's request, `KOSPI200 index level x index volume` was tested as an **activity/turnover proxy**, not as market capitalization.

To reduce the nonstationary price-level trend, the model did not use the raw product directly. It used 5/20-day and 20/63-day mean ratios, a 63-day log z-score, 20-day momentum, and the 20-day down-day share. The fixed economic composite averages the causally standardized 63-day z-score, 20/63-day ratio, and down-day share. Every monthly standardization uses only the preceding 60 monthly observations.

Normal trading costs:

| Strategy | Locked CAGR | Sharpe | MDD | Full CAGR | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|---:|
| FinalBlend | **17.07%** | 1.164 | -14.90% | 14.30% | 1.024 | -14.90% |
| Existing volume composite | 16.42% | **1.241** | -13.14% | 14.21% | 1.062 | -13.99% |
| Index-volume five raw features | 16.44% | 1.235 | **-12.33%** | 14.32% | 1.067 | -13.99% |
| Index-volume fixed composite | 16.37% | 1.229 | -12.89% | 14.31% | 1.066 | -13.99% |
| Five structure composites | 16.81% | 1.216 | -13.59% | **14.42%** | 1.056 | -13.99% |
| Five structure composites + index-volume | 16.69% | 1.235 | -13.18% | 14.41% | **1.067** | -13.99% |

The raw proxy's locked prediction AUC was 0.843 and average precision was 0.435, versus 0.844 and 0.353 for the existing volume composite. Its loss classification ranking is therefore comparable, with better precision-recall ranking, but portfolio Sharpe is marginally lower than the volume composite. Adding the proxy composite to the five structure composites raised full-period Sharpe from 1.056 to 1.067 while leaving CAGR and MDD essentially unchanged. Locked Sharpe rose from 1.216 to 1.235, but CAGR fell from 16.81% to 16.69%.

The proxy composite is 0.853 correlated with the existing volume-stress composite over the full sample (0.866 through 2017 and 0.843 from 2018 onward). It therefore mostly repackages trading-volume information; the index-level term adds some state dependence but not an independent capitalization signal.

Six-month block bootstrap evidence is suggestive, not decisive. The probability that the proxy-augmented structure model improves locked Sharpe over the structure model was 60.9%; its 5th-to-95th percentile Sharpe-delta interval was -0.046 to +0.075. The raw proxy was less likely than the existing volume composite to improve locked Sharpe (28.5%). These results do not justify replacing the volume signal.

### Volatility-targeted Pareto choice

| Strategy | Locked CAGR | Sharpe | MDD | Full CAGR | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|---:|
| Structure + 15% vol target | 17.74% | 1.196 | **-12.56%** | 14.85% | 1.050 | **-13.04%** |
| Structure + index-volume + 15% vol target | **17.75%** | **1.210** | -12.95% | **14.90%** | **1.060** | -13.60% |

The proxy-augmented portfolio is the Sharpe/CAGR-leaning Pareto alternative; the original structure-plus-volatility-target portfolio remains preferable when MDD and Calmar are prioritized. At three times normal costs, the proxy-augmented version still beat FinalBlend on all three objectives in both the full period and locked period, although it did not dominate the original structure-plus-volatility-target portfolio.

The raw proxy improved Sharpe versus FinalBlend in all four subperiods (2007-2012, 2013-2017, 2018-2021, and 2022-2026). It did not improve CAGR in the latter two subperiods, which confirms that its main contribution is risk reduction rather than return enhancement.

## True market capitalization

When point-in-time constituent data becomes available, the economically valid features are:

- Top-10 constituent free-float market-cap share.
- Herfindahl concentration and effective constituent count (`1 / HHI`).
- Cap-weighted minus equal-weighted KOSPI200 return.
- Large-cap versus small-cap constituent leadership.
- Cap-weighted breadth minus equal-weighted breadth.
- One- and three-month changes in concentration.

Membership and shares must be reconstructed as they were known on each historical date. Using today's KOSPI200 members historically would introduce survivorship bias.

## Recommendation

- Keep the five-composite domestic model as the primary short-loss classifier.
- Keep the index-volume activity composite as an optional sixth input when Sharpe is prioritized; do not label or interpret it as market capitalization.
- Use raw volume, correlation, or skew measures only through the composites, not as 23 independent inputs.
- Apply the classifier only as an asymmetric de-risking signal above its causal top-20% probability threshold.
- Use the original structure + 15% volatility-target portfolio for the MDD-first allocation, or the proxy-augmented version for the slightly higher CAGR/Sharpe Pareto alternative.
- Treat the result as a research candidate until point-in-time market-cap/breadth data and another untouched test period are available.
