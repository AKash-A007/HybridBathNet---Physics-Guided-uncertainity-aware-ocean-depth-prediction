# Optical-limit masking audit, 5 November 2026

**Decision: operational masking stays disabled.** This audit improves the numerical safety of optical-limit calculation and tests a candidate against the fixed validation/test regions. It does not establish a reliable local bottom-detection limit with the data currently available. The selected HybridBathNet checkpoint, predictions, full-set RMSE, geographic test blocks, and all previous results remain unchanged. No retraining was run.

## Why the original ~6 m limit cannot be a mask

The trained model's existing Kd map combines a blue/green reflectance-ratio power law and a red/green term; its `z_ext=ln(100)/(2 K)` is a declared heuristic, not a calibrated detection threshold. Its mean is ~6.19 m on primary test points, although ~89.9% of their GEBCO reference depths exceed it. Applying it as a hard mask would remove most eligible predictions and would be driven by an unverified optical parameterization. NASA's [Kd490 description](https://modis.gsfc.nasa.gov/data/dataprod/kd_490.php) specifies a relationship calibrated to *in-situ Kd and remote-sensing reflectance*, not the model's glint-corrected Sentinel L2A composite.

## New conservative computation and diagnostic

`src/optical_limit.py` now supports a **lower bound** on bottom-detection depth from matched upper bounds on downwelling/upwelling attenuation and noise and a lower bound on bottom contrast:

```text
z_lower = max[0, ln(C_lower / (SNR × noise_upper)) / (K_down_upper + K_up_upper)]
```

If the contrast is already below the threshold at zero depth, the limit is zero. Invalid inputs stay invalid. The calculation never claims to be calibrated by itself. `optical_support_flags` now refuses a simple `calibrated=True` switch unless the supplied limit also explicitly carries operational-calibration state. No present output carries that state. This is a software guard, not a substitute for independent measurements.

The frozen v1 model's learned per-band K and A were tested as a *candidate* with K used for both attenuation directions, contrast `|A-R_inf|`, blue/green bands, SNR 3, and per-pixel Sentinel temporal standard deviation as noise. These are deliberately labelled proxies: learned K and A can trade off with depth, `R_inf` comes from shallow training-patch percentiles, and temporal variation includes water-condition changes rather than just measurement noise. [Inputs and results](diagnostic.json).

| 0–20 m split | All samples, RMSE | Point-limit retained | Point-limit RMSE | Conservative upper-depth retained |
|---|---:|---:|---:|---:|
| Validation | 208, 1.963 m | 7 (3.37%) | 1.312 m | 4 (1.92%) |
| Test | 228, 2.561 m | 4 (1.75%) | 2.048 m | 1 (0.44%) |

Median estimated limit is **0 m** in both splits; 93.75%/97.38% of validation/test limits are zero. The candidate fails the predeclared requirement of at least 20% retained coverage and 10 independent approximate-2-km groups. The retained validation examples span only four groups; one of four retained test points has >3 m error. Small-subset RMSE is therefore **not** a reliable accuracy claim. [Spatial risk audit](risk_validation.json). It also cannot independently verify optical bottom visibility, because GEBCO residuals measure model/reference agreement rather than a bottom-detection ground truth. All eligible predictions remain in the official RMSE.

Training-geography deep-water controls are abundant by pixel count but not optically homogeneous. With corrected GEBCO depth≥30 m there are 1.60 million pixels across121 original training blocks; at≥40 m, 0.88 million across67 blocks. Median raw B02/B03 reflectance changes from about **0.0450/0.0285** at≥30 m to **0.0436/0.0273** at≥40 m. The tiny≥60 m subset spans only three blocks and is much brighter (~**0.0578/0.0625**), so simply selecting the deepest GEBCO pixels does not establish a stable `R_inf` or noise distribution. This may be spatial water-type/quality variation; the cause has not been identified. Centre-patch temporal B/G standard deviations are commonly around **0.01**, high compared with the learned bottom-water contrast. [Deep-control audit](deep_controls.json).

## What would make an operational mask defensible

1. Use **individual Sentinel acquisition dates** with per-scene quality/cloud/glint/water masks. Identify optically deep control water within training geography and the same water type/observation period; verify that its blue/green background is stable rather than choosing pixels by GEBCO depth alone. Estimate sensor/scene noise separately from across-date environmental variability.
2. Obtain local independent attenuation/reflectance observations for compatible B02/B03 wavelengths, or calibrate a constrained optical inversion against independent high-quality depth transects with explicit bottom-type handling. Derive **joint** uncertainty bounds for K, bottom contrast and noise; inserting unrelated marginal quantiles into the conservative formula is not a guaranteed bound. ICESat-2 ATL24 or survey data can provide independent depth checks but do not directly measure Kd. Validate vertical datum/tide and quality before use.
3. Before enabling `optical_support_flags`, fix an acceptable error tolerance and minimum useful map coverage. Validate the mask on geographically independent groups and a new independent depth reference, with a confidence bound on unsupported predictions and retained-error risk. Report full-region RMSE, retained-region RMSE and retention fraction together. The current test region has already been inspected, so further tuning against it would make the score exploratory.

The IHO [Satellite-Derived Bathymetry Best Practice Guide](https://iho.int/uploads/user/pubs/bathy/B_13_Ed100_032024.pdf) emphasizes water/seabed heterogeneity and high-quality reference data. No reliable operational mask can be certified from this median composite, current heuristic/learned optical parameters and coarse GEBCO labels alone.

**Checks:** 22 regression tests pass, including the new conservative-limit and anti-bypass checks; relevant scripts compile; `git diff --check` passes. Saved data: [diagnostic](diagnostic.json), [risk validation](risk_validation.json), [deep controls](deep_controls.json), [validation flags](val_idx_flags.npz), [test flags](test_idx_flags.npz). The flags are diagnostic and carry `operationally_calibrated=False`.
