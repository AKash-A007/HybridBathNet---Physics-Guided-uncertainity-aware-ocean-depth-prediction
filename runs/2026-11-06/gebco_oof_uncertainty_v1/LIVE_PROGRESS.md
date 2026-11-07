# Spatial OOF uncertainty progress

Fold validation selects checkpoints. OOF errors are evaluated only after each fold finishes.

## Fold 0

Best validation 2.758m at epoch113.

| Epoch | Validation RMSE (m) |
|---|---:|
| 10 | 11.753 |
| 20 | 10.909 |
| 30 | 10.368 |
| 40 | 9.204 |
| 50 | 8.364 |
| 60 | 7.216 |
| 70 | 6.132 |
| 80 | 4.671 |
| 90 | 4.149 |
| 100 | 3.386 |
| 110 | 2.849 |
| 120 | 2.861 |
| 130 | 3.061 |
| 140 | 3.289 |
| 143 | 3.314 |

## Fold 1

Best validation 2.410m at epoch153.

| Epoch | Validation RMSE (m) |
|---|---:|
| 10 | 11.956 |
| 20 | 10.668 |
| 30 | 9.487 |
| 40 | 7.950 |
| 50 | 6.916 |
| 60 | 6.589 |
| 70 | 6.005 |
| 80 | 6.410 |
| 90 | 4.393 |
| 100 | 4.441 |
| 110 | 3.178 |
| 120 | 2.615 |
| 130 | 2.495 |
| 140 | 2.569 |
| 150 | 2.470 |
| 160 | 2.499 |
| 170 | 2.455 |
| 180 | 2.495 |

## Fold 2

Best validation 2.340m at epoch136.

| Epoch | Validation RMSE (m) |
|---|---:|
| 10 | 11.577 |
| 20 | 10.699 |
| 30 | 9.426 |
| 40 | 7.649 |
| 50 | 7.962 |
| 60 | 7.238 |
| 70 | 5.922 |
| 80 | 5.415 |
| 90 | 4.288 |
| 100 | 3.718 |
| 110 | 3.632 |
| 120 | 2.572 |
| 130 | 2.577 |
| 140 | 2.366 |
| 150 | 2.475 |
| 160 | 2.427 |
| 166 | 2.429 |

## Uncertainty correction head

| Epoch | Validation NLL, constant omitted |
|---|---:|
| 0 | 1.5803 |
| 10 | 1.3475 |
| 20 | 1.3754 |
| 30 | 1.3615 |
| 34 | 1.4463 |
