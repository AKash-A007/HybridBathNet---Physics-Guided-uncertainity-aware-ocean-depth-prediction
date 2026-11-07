# HybridBathNet physics and uncertainty improvement plan

Status: **proposal, not a launched experiment** (5 November 2026). Read [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) and the [nearshore follow-up report](../runs/2026-11-05/nearshore_followup/REPORT.md) for completed results. Keep the original geographic test blocks and saved v1/v2/v3 comparators. Train only HybridBathNet under the current user scope.

## What the present model actually computes

`src/hybridbathnet.py:compute_kd_and_extinction` forms one combined heuristic attenuation map from glint-corrected Sentinel-2-like reflectance channels B02/B03/B04:

```text
K_bg = clamp(0.0166 + 0.156 (B/G)^(-1.14), 0.01, 5)
K_red_term = clamp(0.430 R/(G + epsilon), 0, 3)
K_map = clamp(K_bg + K_red_term, 0.01, 6)
z_ext = clamp(ln(100)/(2 K_map[patch centre]), 1, 50) metres
z_prior = clamp(ln(G_surface95/G)/(2 K_map), 0, 50) metres
```

The centre Kd and zext are used in the weak extinction loss; `K_map`, `z_prior`, and the water mask condition FiLM layers in the encoder. `G_surface95` is the green 95th percentile over valid water within **that patch**. The current reconstructed reflectance is `R_hat = R_inf + (A_lambda - R_inf) exp(-2 K_lambda z_pred)`. Its optical head learns **three per-band `K_lambda` and albedo `A_lambda` maps**, but no measured Kd labels constrain them. `R_inf` is a training-patch 2nd percentile, not a verified optically deep-water observation. The original input is a temporal composite; the first three optical channels have glint correction, so these are not calibrated water-leaving remote-sensing reflectances. The heuristic blue/green coefficients and red term have not been locally validated or shown to be a physically consistent Kd retrieval for these input units. The model's `log_var` head predicts metre-squared aleatoric variance and trains jointly with centre-depth Huber, beta-NLL, weak extinction and masked reconstruction.

Evidence: v1 valid-water optical reconstruction RMSE0.00153 on the full test sensitivity set, but 89.9% of 0–20m test reference depths exceed mean heuristic zext~6.19m; removing the extinction penalty worsened validation1.963→2.095m and barely changed test2.561→2.550m (paired test difference CI crosses zero). Reconstruction fit shows a numerical path, not correct depth/Kd or a proven accuracy contribution. The 10–15m bin has validation/test RMSE1.665/3.255m on64/64 centres; four test spatial groups contribute ~76% of its squared error. Nominal90% uncertainty coverage is71.05%, and validation-only conformal scaling reaches78.07% on test. More than one failure mode is plausible.

## Ranked work packages

### 1. Audit the optical inputs and geographic failure modes before retraining

- Record distributions and clipping rates of B02/B03/B04, B/G, R/G, current Kd, zprior, zext, learned per-band K, albedo and `exp(-2Kz)` by **training/validation** depth bin and spatial block. Review representative raw Sentinel scenes, cloud/glint/turbidity, bottom-type proxies, temporal variability, shore adjacency and native GEBCO support. Keep test-group maps diagnostic; never remove or fit against their residuals.
- Test whether reflectance reconstruction error, bottom-signal transmission, Kd and temporal standard deviation actually predict absolute depth error on validation groups. If they do not, do not present them as uncertainty or optical-support estimates.
- Verify water mask against imagery, the preprocessing reflectance units, and whether the study area contains genuinely optically deep water in training geography. Use train-only or external areas for background estimation.

### 2. Replace the uncalibrated Kd/optical-limit heuristic with a testable physics module

- Calibrate Sentinel reflectance to an appropriate water-leaving reflectance or develop a **site-specific empirical relationship** using training-region water-quality/attenuation observations or independent depth transects. NASA's blue/green Kd490 product is fitted to in-situ Kd and remote-sensing reflectance; its coefficients cannot simply be applied to our glint-corrected L2A composite.
- Estimate `R_inf` from verified deep-water control pixels outside depth supervision but within training geography, with turbidity/temporal matching. Do not use a shallow-patch percentile as a physical deep-water measurement without validation.
- Model separate wavelength-dependent `K_B`, `K_G`, `K_R` with positive, bounded parameters and an explicit downward/upward attenuation path. Allow water-type or scene-level variation but prevent arbitrary per-pixel K and bottom-albedo maps from absorbing any depth error. Avoid a universal fixed ordering of blue and green attenuation across turbid regimes. Validate K against independent observations before naming it measured Kd.
- Derive optical support from bottom contrast, noise, and `K_down + K_up` with uncertainty margins. Until calibrated, output a flag and never hard-clip depth or exclude test pixels. `src/optical_limit.py` is an analytical helper, not a completed local retrieval.

### 3. Make physics carry identifiable information into the depth prediction

- Compare the present FiLM prior and reconstruction to matched HybridBathNet **internal ablations**: conditioning disabled, reconstruction disabled, and the proposed calibrated/shared-parameter optical branch. These are diagnostic variants within HybridBathNet, not Stumpf/RF/CNN comparisons. Use identical data, budgets and at least multiple seeds, select using validation only.
- Constrain an explicit water-depth-index or band-difference branch using training-region reference depths and/or independent optical observations. [Qian et al.](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1636124/full) used a supervised WDI branch and feature fusion, with component ablations; its LiDAR-based score is not directly comparable to our GEBCO score.
- Use native GEBCO-cell support and weighting if adding dense/low-resolution depth supervision. Do not treat hundreds of interpolated 10m pixels from one ~460m GEBCO cell as independent labels. An independent ICESat-2 ATL24, airborne LiDAR or survey depth dataset would be a stronger way to identify the physics parameters and assess metre RMSE.

### 4. Improve uncertainty for **coverage and depth accuracy separately**

- Keep an aleatoric variance head, but compare beta-NLL settings on validation groups and inspect whether the mean head can benefit from uncertainty-weighted loss. Evaluate proper interval scores/NLL, coverage at68/90%, sigma–absolute-error rank correlation and RMSE-versus-retained-fraction by depth bin. Never reduce reported RMSE by discarding high-sigma test examples.
- Add **epistemic uncertainty** from three to five independently seeded HybridBathNet models or a checked MC-dropout protocol. Combine total variance as mean predicted aleatoric variance plus variance of model depth means. Check whether the ensemble mean improves **full-set** validation RMSE; uncertainty alone cannot reduce full-set RMSE without changing the depth predictor.
- Calibrate intervals on a geographically separate **validation/calibration** partition grouped by native support/spatial blocks. Split-conformal residual/sigma scaling is implemented as a diagnostic, but current test 90% coverage78.07% shows geographic shift remains. Report observed test coverage honestly; exchangeability cannot be assumed for spatial transfer. Consider a lower bound on variance from reference uncertainty and a distinct reference-noise head if independent truth becomes available.
- Test whether cloud/glint proxies, temporal spread, water-type and physics residual provide useful uncertainty features on validation groups. A good reflectance reconstruction by itself may be too flexible to reveal depth error.

### 5. Improve independent information in the training data

- Assess actual ATL24 coverage and uncertainty for this AOI before committing to it; align tide/vertical datum and only use quality-filtered, refraction-corrected observations. Keep separate survey/ICESat tracks or geographic blocks for validation/test. [ATL24 v2](https://nsidc.org/data/atl24/versions/2) provides nearshore bathymetry rather than a full continuous seafloor grid.
- Preserve per-date Sentinel-2 reflectance and quality masks instead of relying only on the 16-scene median/std composite. Multiple observations may help separate stable bottom signal from transient turbidity/glint. All dates of one geographic footprint must stay in one split.
- Add **new independent shallow-depth observations**, especially 10–15m optical regimes represented poorly by training geography. More overlapping patches over the same GEBCO cells increase optimization examples but not independent depth truth.

## Suggested next controlled sequence

1. Read-only training/validation optical and reference audit; decide whether any measured optical/depth controls are available. Review test clusters only as a descriptive failure map.
2. Implement a constrained per-band physics candidate and matched HybridBathNet internal ablations, with the same fixed test geography. Predeclare validation overall RMSE plus a guardrail on each validation depth bin, uncertainty coverage and reconstruction diagnostics; do not choose by test-bin minima.
3. Train sequentially on the existing CUDA venv in new run directories; at least three seeds for the promising candidate. Preserve the v1 checkpoint and original features. Report validation and test overall/bin RMSE, paired spatial-bootstrap differences, relative error, uncertainty coverage and physical-parameter plausibility.
4. Seek independent ATL24/survey depth and region-matched calibration. Because the current test region has already been inspected repeatedly, treat further improvements on it as exploratory until a fresh independent holdout is available.

No run, dataset download, or model modification was launched by this planning note. The selected checkpoint remains `runs/2026-11-05/nearshore_hybrid_v1/hybridbathnet.pt`.
