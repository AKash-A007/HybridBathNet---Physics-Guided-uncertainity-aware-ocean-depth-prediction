# GEBCO nearshore HybridBathNet component ablation

Same GEBCO nearshore dataset, geographic splits and test regions. The full model is the preserved selected v1 checkpoint.
Physics-off zeros the seven engineered feature channels and removes physics conditioning and losses; uncertainty-off removes NLL supervision. All arms use the same 16-channel backbone tensor shape.

| Model | Validation RMSE 0-20m | Test RMSE 0-20m | Test MAE 0-20m |
|---|---:|---:|---:|
| full model | 1.963 | 2.561 | 1.894 |
| no physics | 2.121 | 2.495 | 1.902 |
| no uncertainty | 2.033 | 2.467 | 1.905 |
| no both | 2.068 | 2.491 | 1.874 |

| Ablation | Test RMSE difference vs full (m) | Paired ~2km spatial-bootstrap 95% CI (m) |
|---|---:|---:|
| no physics | -0.067 | [-0.220, +0.094] |
| no uncertainty | -0.094 | [-0.290, +0.146] |
| no both | -0.070 | [-0.378, +0.248] |

| Model | 0-5m | 5-10m | 10-15m | 15-20m |
|---|---:|---:|---:|---:|
| full model | 1.094 (n=17) | 1.310 (n=58) | 3.255 (n=64) | 2.800 (n=89) |
| no physics | 1.378 (n=17) | 1.372 (n=58) | 3.266 (n=64) | 2.585 (n=89) |
| no uncertainty | 1.228 (n=17) | 1.772 (n=58) | 2.875 (n=64) | 2.705 (n=89) |
| no both | 1.184 (n=17) | 2.048 (n=58) | 3.006 (n=64) | 2.530 (n=89) |

RMSE is computed from depth prediction error against GEBCO, not from composite loss. All test centres remain included in the 0-20m primary score. Paired intervals resample approximate 2km spatial groups 2000 times and are conditional on these models and GEBCO. Single seed; prior test inspection makes this an exploratory component comparison. The optical cutoff is uncalibrated and is not used for masking. Uncertainty metrics from uncertainty-off arms are invalid and should not be interpreted.

Full-model source: [selected v1](../../2026-11-05/nearshore_hybrid_v1/REPORT.md). Per-arm results and validation-selected checkpoints are in the named subdirectories.
