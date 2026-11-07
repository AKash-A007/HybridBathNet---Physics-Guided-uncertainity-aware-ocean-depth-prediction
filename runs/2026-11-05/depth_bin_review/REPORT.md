# Requested depth-bin RMSE diagnostics

Bins use reference depths: [0,5), [5,10), [10,15), and [15,infinity) metres. All saved examples are retained. Small-bin estimates are shown explicitly with counts and are exploratory.

| Run | Depth (m) | n | Samples (%) | RMSE (m) | MAE (m) | Bias (m) | Squared error (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| selected_validation | 0–5 | 2 | 3.1 | 1.435 | 1.073 | 0.953 | 0.9 |
| selected_validation | 5–10 | 6 | 9.2 | 1.590 | 1.492 | 0.859 | 3.5 |
| selected_validation | 10–15 | 9 | 13.8 | 1.791 | 1.605 | 0.232 | 6.6 |
| selected_validation | 15+ | 48 | 73.8 | 2.848 | 2.354 | -0.310 | 89.0 |
| selected_test | 0–5 | 2 | 3.1 | 1.709 | 1.566 | 1.566 | 0.7 |
| selected_test | 5–10 | 7 | 10.8 | 3.810 | 3.153 | 3.006 | 12.4 |
| selected_test | 10–15 | 8 | 12.3 | 2.247 | 1.723 | -0.874 | 4.9 |
| selected_test | 15+ | 48 | 73.8 | 3.745 | 3.014 | -1.811 | 82.0 |
| repeat_seed1_test | 0–5 | 2 | 3.1 | 3.214 | 3.126 | 3.126 | 2.2 |
| repeat_seed1_test | 5–10 | 7 | 10.8 | 3.213 | 2.486 | 2.261 | 7.8 |
| repeat_seed1_test | 10–15 | 8 | 12.3 | 2.049 | 1.532 | -0.866 | 3.6 |
| repeat_seed1_test | 15+ | 48 | 73.8 | 4.091 | 2.963 | -1.034 | 86.4 |
| grid_pilot_direct_validation | 0–5 | 2 | 3.1 | 1.438 | 1.211 | 0.777 | 0.9 |
| grid_pilot_direct_validation | 5–10 | 6 | 9.2 | 1.803 | 1.719 | 0.915 | 4.1 |
| grid_pilot_direct_validation | 10–15 | 9 | 13.8 | 1.906 | 1.601 | 0.621 | 6.9 |
| grid_pilot_direct_validation | 15+ | 48 | 73.8 | 2.943 | 2.381 | -0.117 | 88.1 |

## Interpretation

Squared-error share must be read with sample share: an abundant deep bin can dominate total error without establishing an abrupt optical cutoff. These bins diagnose error against GEBCO references, not an independently measured physical observability limit. A one-time aggregate cannot establish that reference quality is irrelevant.

Any shallower operating domain should be justified beforehand using physics, measured water properties, bottom detectability and/or training-region evidence. Report both full-domain and restricted-domain coverage/accuracy when such a domain is established; never tune the depth cutoff to test scores or imply excluded regions are accurate.

The current heuristic optical bound is not a verified exclusion mask. Its average near 6 m is inconsistent with most reference depths, and the existing signal-ratio/attenuation constants are not calibrated locally. Masking by that bound now would change the task substantially and could discard usable observations.

No data, predictions, reference labels, training indices or test indices were changed by this analysis. Historical six-channel scalar metrics alone do not permit its per-depth-bin reconstruction.
