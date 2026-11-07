# Recovery experiment progress

Updated UTC: 2026-11-06T14:02:59.420955+00:00

Three GEBCO uncertainty seeds are complete. Official ATL24 paired arms use426 train/102 validation/171 test; all modules active.
Validation metrics below are intermediate, not test results. Historical clear imagery is used for86/426 training and29/171 test points; other inputs are explicit composite fallbacks.

## composite_control

Latest epoch117; best validation6.027m at87.

| Epoch | Validation RMSE | Depth loss | NLL | Extinction | Reconstruction |
|---|---:|---:|---:|---:|---:|
| 10 | 13.170m | 0.4743 | 2.4668 | 7.0720 | 0.0028 |
| 20 | 11.368m | 0.4470 | 2.3887 | 7.0334 | 0.0025 |
| 30 | 9.130m | 0.4862 | 2.4286 | 7.1232 | 0.0022 |
| 40 | 7.747m | 0.4396 | 2.6955 | 6.9315 | 0.0025 |
| 50 | 6.823m | 0.4355 | 2.4448 | 6.7508 | 0.0020 |
| 60 | 6.693m | 0.3965 | 2.7397 | 7.3173 | 0.0017 |
| 70 | 7.146m | 0.3703 | 2.3246 | 7.3557 | 0.0016 |
| 80 | 6.656m | 0.3704 | 4.5788 | 7.0407 | 0.0015 |
| 90 | 6.296m | 0.3535 | 2.1258 | 7.4646 | 0.0016 |
| 100 | 6.408m | 0.3498 | 10.1125 | 7.1116 | 0.0014 |
| 110 | 6.738m | 0.2933 | 2.4213 | 7.4240 | 0.0013 |
| 117 | 7.804m | 0.3001 | 6.7420 | 7.0277 | 0.0013 |

## matched_tide

Latest epoch96; best validation6.713m at66.

| Epoch | Validation RMSE | Depth loss | NLL | Extinction | Reconstruction |
|---|---:|---:|---:|---:|---:|
| 10 | 11.810m | 0.4569 | 2.4887 | 7.4169 | 0.0053 |
| 20 | 9.389m | 0.4645 | 2.4565 | 6.9588 | 0.0056 |
| 30 | 8.416m | 0.4593 | 2.3954 | 7.2224 | 0.0051 |
| 40 | 7.301m | 0.4432 | 2.3943 | 7.2412 | 0.0047 |
| 50 | 7.005m | 0.4488 | 2.3568 | 7.0958 | 0.0055 |
| 60 | 6.992m | 0.3648 | 2.3455 | 7.1673 | 0.0040 |
| 70 | 7.599m | 0.3769 | 2.4497 | 7.6784 | 0.0042 |
| 80 | 7.221m | 0.3841 | 2.3940 | 7.2781 | 0.0039 |
| 90 | 7.042m | 0.4148 | 2.5146 | 7.5872 | 0.0043 |
| 96 | 7.681m | 0.3379 | 3.4331 | 7.3286 | 0.0039 |

