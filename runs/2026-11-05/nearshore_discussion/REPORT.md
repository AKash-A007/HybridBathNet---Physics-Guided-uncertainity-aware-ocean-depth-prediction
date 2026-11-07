# Nearshore cutoff and uncertainty discussion

Exploratory analysis of saved expanded_smooth seed-42 predictions; no training, clipping or sample deletion. Reference-depth cutoffs are not deployable optical masks. All scores use the original GEBCO targets.

95% intervals use 2,000 approximately 2 km spatial-cluster bootstrap resamples (seed 42), not independent-pixel resampling. Intervals are conditional on this model/split, do not include reference uncertainty and are unreliable for sparse bins/few clusters. No cutoff is selected from these test scores.

## validation

| Bin (m) | n | RMSE | 95% cluster CI | MAPE (%) | Delta <1.25 (%) | Clusters |
|---|---:|---:|---|---:|---:|---:|
| 0–5 | 2 | 1.435 | 0.120–2.026 | 73.2 | 50.0 | 2 |
| 5–10 | 6 | 1.590 | 1.156–1.983 | 20.4 | 66.7 | 6 |
| 10–15 | 9 | 1.791 | 1.278–2.202 | 13.1 | 100.0 | 9 |
| 15–+ | 48 | 2.848 | 2.312–3.369 | 8.9 | 97.9 | 43 |

| Maximum reference depth | n | Coverage (%) | RMSE | 95% cluster CI | MAPE (%) |
|---:|---:|---:|---:|---|---:|
| 15 | 17 | 26.2 | 1.683 | 1.351–1.997 | 22.8 |
| 20 | 26 | 40.0 | 2.081 | 1.429–2.853 | 18.5 |
| 25 | 39 | 60.0 | 2.335 | 1.817–2.832 | 16.1 |

Spearman sigma vs reference depth: 0.391; sigma vs absolute error: 0.327.

| Lowest-uncertainty fraction retained | n | RMSE | Mean sigma (m) |
|---:|---:|---:|---:|
| 25% | 17 | 1.519 | 1.710 |
| 50% | 33 | 2.107 | 1.927 |
| 75% | 49 | 2.431 | 2.193 |
| 100% | 65 | 2.594 | 2.619 |

## test

| Bin (m) | n | RMSE | 95% cluster CI | MAPE (%) | Delta <1.25 (%) | Clusters |
|---|---:|---:|---|---:|---:|---:|
| 0–5 | 2 | 1.709 | 0.882–2.250 | 45.9 | 50.0 | 2 |
| 5–10 | 7 | 3.810 | 2.306–4.857 | 46.3 | 28.6 | 7 |
| 10–15 | 8 | 2.247 | 1.075–3.183 | 13.7 | 75.0 | 8 |
| 15–+ | 48 | 3.745 | 2.940–4.694 | 11.1 | 91.7 | 40 |

| Maximum reference depth | n | Coverage (%) | RMSE | 95% cluster CI | MAPE (%) |
|---:|---:|---:|---:|---|---:|
| 15 | 17 | 26.2 | 2.949 | 2.006–3.731 | 30.9 |
| 20 | 26 | 40.0 | 2.871 | 2.208–3.455 | 24.4 |
| 25 | 38 | 58.5 | 2.850 | 2.345–3.325 | 20.3 |

Spearman sigma vs reference depth: 0.143; sigma vs absolute error: 0.432.

| Lowest-uncertainty fraction retained | n | RMSE | Mean sigma (m) |
|---:|---:|---:|---:|
| 25% | 17 | 2.851 | 1.800 |
| 50% | 33 | 2.309 | 1.936 |
| 75% | 49 | 3.140 | 2.178 |
| 100% | 65 | 3.554 | 2.963 |

## What can be claimed

Uncertainty associated with depth/error and lower error in low-uncertainty subsets would be evidence of useful ranking on this dataset, not proof that the network detects an invisible seafloor. Depth, location, bottom type and label quality are confounded. Use within-depth-bin ranking, calibrated coverage, a geographic holdout and an independent bottom-detectability reference before making that claim.

A future four-model table requires Stumpf, RF, CNN and Hybrid trained/evaluated on identical new labels, spatial groups, optical eligibility and depth bins. Existing scalar summaries or mismatched historical splits cannot supply that table. The user now authorizes considering limited tabular comparison, but no new training has been run because training is stopped pending ICESat-2 discussion.
