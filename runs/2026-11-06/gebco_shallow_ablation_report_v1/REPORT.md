# GEBCO ablation and uncertainty tables — shallow0–10m

No retraining or recalibration. Scores are recomputed from saved prediction files, using reference depth0<=z<=10m. The overall0–20m benchmark remains preserved. Validation and test remain separate.
Uncertainty means average predicted per-location standard deviation (SD), not RMSE or a guaranteed error bound. Interval width is the full lower-to-upper90% Gaussian range. Uncertainty-off arms have no valid sigma; N/A never means zero error.

## Matched component ablations

All four arms use the same637 training points, fixed validation/test targets, seed42 and input tensor shape. Physics/uncertainty configurations differ as specified in the original ablation protocol. Single-model epistemic uncertainty was not estimated.

### validation,0–10m (n=72)

| Model | RMSE(m) | MAE(m) | Mean head SD(m) |90% coverage |90% width(m) |
|---|---:|---:|---:|---:|---:|
| Full HybridBathNet | 1.778 | 1.309 | 1.763 | 86.111 | 5.800 |
| No physics | 1.941 | 1.386 | 1.969 | 91.667 | 6.476 |
| No uncertainty | 2.058 | 1.563 | N/A | N/A | N/A |
| Neither physics nor uncertainty | 2.104 | 1.646 | N/A | N/A | N/A |
### test,0–10m (n=75)

| Model | RMSE(m) | MAE(m) | Mean head SD(m) |90% coverage |90% width(m) |
|---|---:|---:|---:|---:|---:|
| Full HybridBathNet | 1.264 | 1.035 | 1.802 | 98.667 | 5.930 |
| No physics | 1.373 | 1.183 | 2.071 | 98.667 | 6.813 |
| No uncertainty | 1.665 | 1.386 | N/A | N/A | N/A |
| Neither physics nor uncertainty | 1.887 | 1.474 | N/A | N/A | N/A |

### Test RMSE by bin and original full domain

| Model |0–5m(n17) |5–10m(n58) |0–10m(n75) |10–15m(n64) |15–20m(n89) |0–20m(n228) |
|---|---:|---:|---:|---:|---:|---:|
| Full HybridBathNet | 1.094 | 1.310 | 1.264 | 3.255 | 2.800 | 2.561 |
| No physics | 1.378 | 1.372 | 1.373 | 3.266 | 2.585 | 2.495 |
| No uncertainty | 1.228 | 1.772 | 1.665 | 2.875 | 2.705 | 2.467 |
| Neither physics nor uncertainty | 1.184 | 2.048 | 1.887 | 3.006 | 2.530 | 2.491 |

## Later ensemble and uncertainty-head experiments

These use447 training points and123 independent calibration points rather than the original637 training points. They share test labels/locations, but are not matched component ablations. Do not attribute their RMSE differences solely to the physics/uncertainty heads.
Raw aleatoric/epistemic decomposition applies only to the original ensemble. Postprocessed total uncertainty does not identify separately corrected components; original raw component files remain preserved.

### validation,0–10m

| Variant | n | RMSE(m) | Mean total SD(m) |90% coverage |90% width(m) |
|---|---:|---:|---:|---:|---:|
| Three-member ensemble, raw | 72 | 2.362 | 1.873 | 76.39% | 6.162 |
| Ensemble + global calibration | 72 | 2.362 | 3.294 | 95.83% | 10.837 |
| Ensemble + OOF risk head | 72 | 2.362 | 2.849 | 94.44% | 9.373 |
| Ensemble + OOF head + calibration | 72 | 2.362 | 3.470 | 95.83% | 11.414 |
### test,0–10m

| Variant | n | RMSE(m) | Mean total SD(m) |90% coverage |90% width(m) |
|---|---:|---:|---:|---:|---:|
| Three-member ensemble, raw | 75 | 2.165 | 1.942 | 89.33% | 6.389 |
| Ensemble + global calibration | 75 | 2.165 | 3.416 | 94.67% | 11.237 |
| Ensemble + OOF risk head | 75 | 2.165 | 2.951 | 94.67% | 9.709 |
| Ensemble + OOF head + calibration | 75 | 2.165 | 3.594 | 98.67% | 11.824 |

### Raw ensemble decomposition,0–10m

| Split | n | Mean aleatoric SD(m) | Mean epistemic SD(m) | Mean total SD(m) |
|---|---:|---:|---:|---:|
| validation | 72 | 1.741 | 0.576 | 1.873 |
| test | 75 | 1.854 | 0.516 | 1.942 |

The component ablations were not rerun with the OOF head. No new matched physics-off/uncertainty-off deep ensembles exist; unavailable epistemic values must not be fabricated. This is an exploratory requested subgroup, not an optical mask or a newly selected reporting cutoff.

[All domains/metrics JSON](results.json) | [All tables CSV](all_results.csv) | [Original ablation study](../gebco_component_ablation_v1/REPORT.md) | [OOF uncertainty study](../gebco_oof_uncertainty_v1/REPORT.md)
