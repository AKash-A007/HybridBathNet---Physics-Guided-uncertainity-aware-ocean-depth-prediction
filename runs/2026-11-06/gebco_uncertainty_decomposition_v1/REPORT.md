# GEBCO uncertainty-head and deep-ensemble decomposition

Three independently trained full HybridBathNet models (seeds 42, 1, 7). Physics, depth and uncertainty remain active. This is a read-only analysis of their preserved predictions; no retraining or model promotion.

For each location: aleatoric variance = average head variance; epistemic variance = population variance of member depth means; total variance = their sum. This is the law of total variance for the equal-weight Gaussian mixture. Epistemic disagreement is not divided by the ensemble size a second time.

| Split, 0–20 m | n | RMSE (m) | Mean aleatoric SD (m) | Mean epistemic SD (m) | Mean total SD (m) |
|---|---:|---:|---:|---:|---:|
| validation | 208 | 2.219 | 1.413 | 0.452 | 1.506 |
| test | 228 | 2.372 | 1.538 | 0.434 | 1.612 |

| Test depth bin (m) | n | RMSE (m) | Mean aleatoric SD (m) | Mean epistemic SD (m) | Mean total SD (m) |
|---|---:|---:|---:|---:|---:|
| 0-5 | 17 | 2.136 | 1.414 | 0.526 | 1.520 |
| 5-10 | 58 | 2.173 | 1.983 | 0.513 | 2.066 |
| 10-15 | 64 | 2.687 | 1.650 | 0.470 | 1.731 |
| 15–20 | 89 | 2.295 | 1.190 | 0.339 | 1.249 |

variances in m^2; square roots in m. Reported mean sigmas are averaged after per-point decomposition; their squares do not add after averaging. RMS sigmas do combine in quadrature.
The RMS component SDs on the test set are 1.613 m aleatoric, 0.507 m epistemic and 1.690 m total. Ensemble disagreement contributes 9.0% of aggregate predictive variance.

Head variance is a model-based estimate of conditional residual noise against GEBCO, not measured pure sensor noise. Ensemble disagreement is an approximation to epistemic uncertainty; three members may miss shared bias.
All components are raw, before conformal scaling. Interval calibration does not identify an aleatoric/epistemic attribution.
Raw Gaussian moment-based nominal 90% intervals cover only 70.6% of primary test locations. Small ensemble disagreement does not establish accuracy: all members can share bias. These values therefore do not certify calibrated true-seabed uncertainty.
The preserved v1 remains the validation-selected GEBCO reference. Its checkpoint is not an ensemble member: it used a different training/calibration allocation and fitted feature references. No models or normalizations were mixed.

Checks: same dataset hash across members and against the prepared NPZ; exact member target/order matching; finite positive sigma; independent second-moment identity; exact agreement with the existing saved ensemble components. Test labels were used only for reporting errors/bins, never fitting or masking.

[All metrics](components.json) | [Per-location test values](test_components.csv) | [Validation values](validation_components.csv)
[Original three-model training protocol](../recovery_v2/gebco_uq/seed_42/protocol.json) | [Deep ensembles reference](https://arxiv.org/abs/1612.01474)
