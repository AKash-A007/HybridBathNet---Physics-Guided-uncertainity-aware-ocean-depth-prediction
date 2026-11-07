# Official ATL24 V002 paired-input experiment

Same official reference/coordinates/geography and full-model settings in both arms. Validation selects each checkpoint.
Matched clear scenes receive tide-adjusted scene-relative labels; predictions are transformed back to the identical ATL24 acquisition surface for all scores below.
Cloudy/unavailable scenes retain the2024 input as an explicitly flagged fallback; all eligible test examples remain.

| Arm | Validation RMSE | Test RMSE | Test90% raw coverage |
|---|---:|---:|---:|
| composite_control | 6.027m | 6.512m | 88.3% |
| matched_tide | 6.713m | 6.093m | 90.1% |

| Test depth bin | n | Composite control RMSE | Matched/tide RMSE |
|---|---:|---:|---:|
| 0-5 | 28 | 10.727m | 10.166m |
| 5-10 | 5 | 8.276m | 0.920m |
| 10-15 | 25 | 2.979m | 2.121m |
| 15-+ | 113 | 5.537m | 5.435m |

Per-bin spatial-bootstrap intervals,15/20/25m sensitivity, relative errors and fallback strata are in[comparison.json](comparison.json).
This changes reference version/coordinates relative to the old public ATL24 and GEBCO studies. Absolute RMSE differences across those benchmarks cannot establish model improvement.
Product sigma and horizontal footprints are audited, not yet propagated into the likelihood. V002 classifications and coastal tide residuals remain imperfect.
Optical visibility is uncalibrated and no held-out point was masked based on model error.
