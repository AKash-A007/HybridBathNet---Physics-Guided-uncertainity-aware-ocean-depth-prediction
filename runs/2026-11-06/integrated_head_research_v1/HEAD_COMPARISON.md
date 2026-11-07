# Integrated physics and uncertainty experiment

One seed42 candidate on the same GEBCO nearshore data and geographic splits. All modules remain active. Variance NLL is detached from the mean/features; effective per-band attenuation is shared within patch water.

| Model | Validation RMSE 0–20m | Test RMSE 0–20m | Test 90% coverage | Mean 90% interval width | Test Gaussian NLL |
|---|---:|---:|---:|---:|---:|
| full | 1.963m | 2.561m | 71.05% | 5.285m | 2.596 |
| candidate | 1.961m | 2.527m | 68.42% | 5.154m | 2.634 |

| Test depth bin | n | Full RMSE | Candidate RMSE | Full 90% coverage | Candidate 90% coverage |
|---|---:|---:|---:|---:|---:|
| 0-5m | 17 | 1.094m | 1.110m | 100.0% | 94.1% |
| 5-10m | 58 | 1.310m | 1.705m | 98.3% | 93.1% |
| 10-15m | 64 | 3.255m | 3.166m | 60.9% | 62.5% |
| 15-20m | 89 | 2.800m | 2.650m | 55.1% | 51.7% |

Lowest numerical validation RMSE: **candidate**. This is not evidence of a reliable joint improvement. Paired ~2km spatial-bootstrap95% interval for candidate minus full test RMSE: [-0.217, +0.153]m (2000 draws).

The candidate is not promoted: its small depth differences are inconclusive, validation Gaussian NLL worsened, and held-out uncertainty coverage/ranking also worsened. Preserve the existing full model as the reference while developing the next optical/uncertainty design.

Candidate test reflectance reconstruction RMSE: 0.001609. Reloaded predictions match the recorded predictions when its model options are used. This does not validate physical Kd, bottom visibility or an operational optical mask.

Single-seed exploratory test comparison; the test has been inspected before. Coverage alone can improve by widening intervals: assess coverage, width, NLL and bin behaviour together. Calibration does not change depth RMSE. The patch-homogeneous water assumption and shallow-percentile background remain unvalidated. No test sample was masked or clipped.

[Literature and recovery design](../../../docs/HEADS_OPTICAL_LIMIT_RESEARCH_2026-11-06.md) | [Full metrics](head_comparison.json) | [Protocol](protocol.json)
