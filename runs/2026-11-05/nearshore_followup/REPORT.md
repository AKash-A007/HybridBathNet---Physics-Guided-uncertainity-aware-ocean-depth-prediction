# HybridBathNet nearshore follow-up, 5 November 2026

## Decision

Keep the original **nearshore_hybrid_v1** checkpoint selected by its lower 0–20 m validation RMSE (1.963 m). Two full, matched HybridBathNet-only runs tested a physics-penalty change and a depth-loss change. Neither produced a convincing overall improvement. Every model was selected on validation before test evaluation. All used the identical feature NPZ (SHA256 `5a15b02acf58ce7cada2b168bca30cca934dc85b0ac9af89531aa11619609cf8`), split coordinates, seed 42, joint physics/depth/uncertainty/reconstruction network, 180-epoch maximum, and original geographic test blocks. The old checkpoint and historical result JSON are untouched.

| Model | Change from selected v1 | Best EMA / stopped epoch | Validation 0–20 m RMSE | Test 0–20 m RMSE |
|---|---|---:|---:|---:|
| [v1](../nearshore_hybrid_v1/REPORT.md) | original joint composite | 118 / 148 | **1.963** | 2.561 |
| [v2](../nearshore_no_ext_v2/REPORT.md) | extinction penalty .03 → 0 | 135 / 165 | 2.095 | 2.550 |
| [v3](../nearshore_squared_depth_v3/REPORT.md) | normalized depth Huber → squared error | 133 / 163 | 2.100 | 2.590 |

The 0.011 m v2 test difference has a paired 2 km spatial-bootstrap 95% interval **−0.250 to +0.252 m** and worse validation, so it does not justify switching. The v3 test difference is +0.029 m, interval **−0.209 to +0.316 m**. These are exploratory intervals after prior test-set inspection, not confirmatory evidence.

## All test depth bins, same locations and corrected GEBCO labels

| Reference depth | Test patch centres | v1 RMSE | v2 RMSE | v3 RMSE |
|---|---:|---:|---:|---:|
| 0–5 m | 17 | **1.094** | 0.986 | 1.399 |
| 5–10 m | 58 | **1.310** | 1.806 | 2.088 |
| 10–15 m | 64 | **3.255** | 3.220 | 3.270 |
| 15–20 m | 89 | **2.800** | 2.625 | 2.507 |
| 0–20 m including exact-20 m labels | 228 | **2.561** | 2.550 | 2.590 |

The bolded v1 values are the selected model, rather than per-bin cherry-picked models. Bin improvements are not uniform: both changes reduce 15–20 m error, while both substantially worsen 5–10 m. The 10–15 m test bin scarcely changes. Its baseline mean prediction bias is −1.863 m; four approximate 2 km groups account for 76% of its squared error. Validation 10–15 m RMSE is only 1.665 m on 64 centres; geographic/optical or reference differences are plausible hypotheses, not diagnosed causes. None of these test groups was excluded or used to tune parameters.

Cutoff sensitivity (test RMSE v1/v2/v3): ≤15 m **2.386/2.487/2.626** on143 centres; ≤20 m **2.561/2.550/2.590** on228; ≤25 m **3.434/3.361/3.365** on343. The 20–25 m part is beyond the supervised 0–20 m depth range. The original v1 0–20 m test spatial-bootstrap 95% interval is **1.840–3.261 m**. See [v2](../nearshore_no_ext_v2/matched_comparison.json) and [v3](../nearshore_squared_depth_v3/matched_comparison.json) paired comparison details.

## What “samples” means here

One sample is a **64×64-pixel Sentinel-2 feature patch** with one GEBCO depth label at its centre. The dataset has **637 training**, **208 validation**, and **343 test** patch centres. Training and validation centres are ≤20 m; 228 test centres are ≤20 m and the other115 are 20–25 m sensitivity cases. Training bins 0–5/5–10/10–15/15–20 contain **56/131/191/258** centres. Patches overlap within a split, and native GEBCO is ~460 m, interpolated to a 10 m imagery grid. These counts are not independent 10 m seabed survey measurements. Native GEBCO interpolation-support cells are disjoint between splits, while geographic test blocks stay unchanged. [Dataset audit](../nearshore_hybrid_v1/features/metadata.json).

## Uncertainty action and remaining limitation

The model's sigma is trained jointly with depth using beta-NLL. We fitted multiplicative 68% and 90% interval widths using **validation residual / predicted sigma only**, with a finite-sample split-conformal quantile; no test residual entered the calibration. On the selected v1 test 0–20 m domain, raw nominal-90% coverage is **71.05%**, and the validation-calibrated interval covers **78.07%** (median half-width **3.02 m**). V2 goes 64.47%→82.02%; v3 75.44%→82.46%. None reaches nominal90% on the geographic test region. Spatial correlation and distribution shift invalidate a blanket coverage guarantee here. Do not report these as calibrated 90% confidence bounds or hide poorly covered pixels. [v1/v2 calibration](../nearshore_no_ext_v2/uncertainty_calibration.json), [v3 calibration](../nearshore_squared_depth_v3/uncertainty_calibration.json). A future independent, geographically representative calibration set and ensemble or out-of-domain evaluation are needed.

## Does the physics module work?

**Numerically, yes; scientifically, partially established.** The architecture uses mask-aware attenuation/prior conditioning at all encoder scales and an optical reconstruction loss jointly with the centre-depth and uncertainty losses. Training stayed finite and gradients reach those branches. The selected model reconstructs held-out valid-water reflectance with RMSE **0.00182** on validation and **0.00153** on the full ≤25 m test set. Its learned attenuation median is **0.058 m⁻¹** on test, and median reconstructed centre bottom-signal transmission is **0.154**. These show the branch fits image reflectance; because attenuation and bottom albedo are free to compensate each other, reconstruction fit alone does **not** prove physically correct depths or attenuation.

The separate hardcoded extinction heuristic is **not locally calibrated**. Its mean is ~6.19 m on primary test centres, while **89.9%** of their GEBCO labels exceed it. It is unsuitable as a hard validity mask. Removing its weak training penalty did not improve validation, but this single ablation does not prove the FiLM prior/reconstruction adds accuracy. A matched same-budget HybridBathNet ablation of conditioning and reconstruction, plus measured optical properties or independent depth reference, is needed for that claim. [Physics audit](../nearshore_squared_depth_v3/physics_audit.json).

## Literature and practical next step

[Qian et al. (2025)](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1636124/full) used an explicitly supervised water-depth-index branch and feature fusion, and tested physical components with ablations. Their 1.1425 m result uses airborne LiDAR labels in other locations and is not an apples-to-apples target for this GEBCO experiment. Their documented poorer transfer across different optical/topographic conditions supports investigating the four difficult 10–15 m geographic groups. [Christofilakos et al. (2025)](https://elib.dlr.de/216314/1/remotesensing-17-03060.pdf) also found only weak associations between some uncertainty indices and absolute error, reinforcing the need to measure coverage rather than equate an uncertainty output with reliability. Split-conformal interval guarantees require exchangeability ([Angelopoulos and Bates](https://arxiv.org/abs/2107.07511)); this geographic test does not establish that assumption.

Next: inspect **training/validation** imagery quality, bottom type and native GEBCO support around high-error optical regimes, then obtain independent ICESat-2/airborne/survey depths or new scenes for better geographic coverage. Keep the current test blocks fixed and untouched. Predefine a new validation-driven physical-component experiment, repeat across seeds, and validate optical coefficients before using any extinction cutoff. More overlapping patches from the same coarse GEBCO cells will not supply independent depth truth. Do not claim a sub-metre result or select a model from the test-bin minima.

All **21 regression tests**, script compilation and `git diff --check` passed. No training job is active. Protocols, selected models, histories, predictions, reports and source remain in distinct v1/v2/v3 folders.
