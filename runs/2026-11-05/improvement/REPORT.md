# Longer training and expanded-data results

Selected by validation: **expanded_smooth**.

| Run | Validation RMSE (m) | Test RMSE (m) | Test MAE (m) | Test R2 | Best epoch |
|---|---:|---:|---:|---:|---:|
| long_original | 3.474 | 4.186 | 3.401 | 0.884 | 269 |
| expanded_restart | 2.876 | 4.195 | 3.108 | 0.884 | 167 |
| expanded_smooth | 2.594 | 3.554 | 2.826 | 0.917 | 175 |
| selected_seed_1 | 2.768 | 3.783 | 2.741 | 0.906 | 178 |

Previous selected run: validation RMSE 5.118 m; test RMSE 5.335 m.

Expanded training: 2176 patches versus 306; validation/test retain the same 65 locations each. Features refitted using training pixels only. Held-out footprints have a 32-pixel exclusion buffer. These are additional overlapping training patches from the same composite, not independent new scenes.

The original-data run continues from epoch 90 up to 300. Expanded runs start fresh with up to 180 epochs. All use patience 60, EMA 0.999, spatial conditioning and extinction weight 0.03. The smooth run uses a single 180-epoch cosine cycle instead of restarting every 30 epochs. The selected configuration is repeated from scratch with seed 1.

The existing test set has already been inspected in prior work; results are exploratory comparisons, not a new untouched generalization estimate. Adjacent GEBCO cells and overlapping training patches limit independence. More patches do not establish more independent depth observations.

The optical attenuation heuristic remains uncalibrated. Validation depth errors and bound mismatch are saved in [data audit](data_audit.json). No test-dependent depth cutoff or formula was introduced.

[Full metrics](test_summary.json) | [Validation results](validation_summary.json) | [10-epoch progress](LIVE_PROGRESS.md)
