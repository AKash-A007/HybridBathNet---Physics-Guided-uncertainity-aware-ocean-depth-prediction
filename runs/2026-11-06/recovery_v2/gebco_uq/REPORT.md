# Full-model uncertainty recovery study

new447 training/123 independent calibration/208 validation/343 fixed test centres. All models retain encoder/depth/physics/uncertainty.
The spatial variance head uses detached shared features/mean and separate gradient clipping. The15 calibration groups were excluded from training and all feature fitting. Validation selects EMA checkpoints; calibration is used only after every member finishes.

| Model | Validation RMSE | Test RMSE | Test raw90% coverage | Test calibrated90% coverage | Raw/calibrated mean interval width |
|---|---:|---:|---:|---:|---:|
| single_seed42 | 2.254m | 2.501m | 68.0% | 100.0% | 5.41/81.67m |
| ensemble3 | 2.219m | 2.372m | 70.6% | 100.0% | 5.31/17.12m |

| Ensemble test bin | n | RMSE | Raw90% coverage | Calibrated90% coverage | Calibrated mean width |
|---|---:|---:|---:|---:|---:|
| 0-5m | 17 | 2.136m | 88.2% | 100.0% | 16.14m |
| 5-10m | 58 | 2.173m | 89.7% | 100.0% | 21.93m |
| 10-15m | 64 | 2.687m | 59.4% | 100.0% | 18.38m |
| 15-20.0001m | 89 | 2.295m | 62.9% | 100.0% | 13.26m |

Coverage guarantees require exchangeable groups; geographic shift/GEBCO reference error remain.
Same test coordinates/targets, but v1 used637 training centres and different feature fitting; not an isolated head ablation.
Calibrated coverage and interval width must be assessed together; interval scaling cannot improve RMSE. GEBCO agreement does not establish independent seabed accuracy. Full primary test set stays included; optical masking remains disabled.

[All metrics](uncertainty_comparison.json) | [Dataset audit](features/metadata.json)
