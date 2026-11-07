# Improving GEBCO uncertainty: papers, implementation and findings

The user requested action to improve uncertainty, using shared references and current internet research. [Completed experiment](../runs/2026-11-06/gebco_uncertainty_calibration_v2/REPORT.md) | [Predefined protocol](../runs/2026-11-06/gebco_uncertainty_calibration_v2/PROTOCOL.md).

## What the papers support

| Primary source | Relevant evidence | Applied here |
|---|---|---|
| [Christofilakos et al., SDB uncertainty, 2025](https://elib.dlr.de/216314/1/remotesensing-17-03060.pdf), from the user's literature list | Their spatial uncertainty index has weak error/depth correlation and limited optimization effectiveness. An uncertainty map is not itself a validated error bound. | Keep all held-out examples and assess coverage, sharpness, proper interval scores and depth bins; no uncertainty-based removal to lower RMSE. |
| [Ferchichi et al., environmental remote-sensing UQ review, 2025](https://www.sciencedirect.com/science/article/pii/S0925231225009142), from the literature list | Reviews deep-learning uncertainty across environmental remote-sensing tasks and emphasizes multiple sources of uncertainty. It is not a local bathymetry calibration recipe. | Preserve separate raw head/ensemble components and explicitly evaluate calibration of their combined predictive uncertainty. |
| [Levi et al., Evaluating and Calibrating Uncertainty Prediction in Regression Tasks](https://pmc.ncbi.nlm.nih.gov/articles/PMC9330317/) | Proposes variance scaling and warns that marginal calibration alone can hide uninformative uncertainty. | Primary candidate: analytic Gaussian sigma scaling fitted on independent calibration residuals. Equal geographic-group weights are our adaptation. Scaling cannot improve uncertainty ranking. |
| [Zelikman et al., CRUDE](https://arxiv.org/html/2005.12496v2) | Empirical standardized-residual quantiles relax the Gaussian error shape. | Equal-group signed-residual quantiles; no correction of point-depth means. This is a spatially weighted adaptation. |
| [Lee, Barber and Willett, hierarchical inference](https://arxiv.org/html/2306.06342v4) | Calibrates grouped observations with a finite-group correction. | Pointwise hierarchical conformal comparison, with the infinity atom retained. Geographic exchangeability remains unverified. |

## What changed

Added `src/uncertainty_calibration.py` with variance scaling, empirical residual intervals, hierarchical intervals and inference-only state reload. The three full GEBCO ensemble models, physics and uncertainty heads remain unchanged. Aleatoric and epistemic raw variances are preserved; a calibrated total variance does not identify which component caused an error.

The existing 123 calibration locations in 15 approximately 2 km geographic groups were excluded from network training and feature fitting. Fits use these locations only, with equal mass per group. Whole-group cross-validation precedes final fits, which are saved before new validation/test evaluation. No test-based choice of multiplier, operating coverage, depth cutoff, model or mask occurs. Previously inspected test results remain exploratory.

The predefined primary multiplier is `sigma_calibrated = 1.758801 * sigma_total_raw`. Calibration-only maximum-likelihood fitting determines it analytically. Inference requires no reference depths. Source hashes, calibrator states, example-level exports and reload checks are preserved under the new experiment directory.

## Actual outcomes

| Primary metric | Raw | Scaled |
|---|---:|---:|
| Validation nominal90% coverage | 80.3% | 93.8% |
| Validation90% mean interval width | 4.95 m | 8.71 m |
| Test nominal90% coverage | 70.6% | 96.5% |
| Test90% mean interval width | 5.30 m | 9.33 m |
| Test90% interval score, lower better | 11.037 m | 9.817 m |
| Test Gaussian CRPS, lower better | 1.345 m | 1.321 m |
| Validation Gaussian CRPS, lower better | 1.197 m | 1.219 m |

The earlier simultaneous group-max intervals were 17.12 m wide on test; that stronger coverage objective is not equivalent to pointwise intervals. Empirical CRUDE reaches96.1% test coverage at9.85m width; hierarchical conformal reaches99.6% at10.94m, both more conservative than nominal90%.

The primary candidate improves severe undercoverage, but overshoots90%. Its paired spatial95% interval-score improvement interval is[-3.614,+0.866]m and includes zero. Leave-one-calibration-group-out coverage is only80.7%, exposing geographic instability. Validation CRPS slightly worsens. Do not call this uniformly better or perfectly calibrated. Fifteen calibration groups cannot support bounded95% hierarchical conformal intervals; unbounded sets are explicitly reported.

Validation/test depth RMSE remains2.219/2.372m. The mean total test sigma becomes2.836m, while raw mean aleatoric/epistemic SDs remain1.538/0.434m. Scaling does not improve selective error ranking, physical identifiability, or surveyed seabed truth. No optical masking was enabled.

Keep this as an explicit uncertainty-calibration research option for its source GEBCO ensemble, not a replacement depth model or a calibrator for the differently trained v1 checkpoint. Next work should preserve shallow training support and add independent calibration geography. To improve conditional error ranking, study variance learning on spatially out-of-fold residuals while keeping depth/physics fixed; that training experiment is proposed, not completed.
