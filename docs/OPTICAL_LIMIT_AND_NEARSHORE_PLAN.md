# Optical support, shallow GEBCO training, and current decisions

Updated 2026-11-05. Latest user decisions supersede the earlier stop/discussion scope: **train only HybridBathNet**, keep the existing test regions, sample more real nearshore locations, use a predefined shallow scope, report 15/20/25 m sensitivity and bootstrap per-bin intervals. Do not run Stumpf, RF, CNN, ICESat-2 acquisition or the separate Sentinel-only model under this latest scope.

## Current run

**Completed20:01IST on Nov5.** Best EMA epoch118; early stopping at148. Validation <=20m RMSE1.963m; test <=20m2.561m (228, CI1.840–3.261). Test cutoff15/20/25m sensitivity2.386/2.561/3.434m. Primary90% uncertainty coverage71.1%, so further calibration work is required. An exploratory same-coordinate old/new comparison at26 shallow points gives2.898/2.552m, but its paired improvement CI crosses zero. See the completed report for per-bin confidence intervals, relative errors and figures. No training is active, and no follow-up run is automatically authorized.

[Nearshore job](../runs/2026-11-05/nearshore_hybrid_v1/status.json), [dataset audit](../runs/2026-11-05/nearshore_hybrid_v1/features/metadata.json), [protocol](../runs/2026-11-05/nearshore_hybrid_v1/protocol.json), [report when complete](../runs/2026-11-05/nearshore_hybrid_v1/REPORT.md).

- Test-region block IDs are unchanged. Additional distinct patch coordinates are sampled within those blocks; no locations are duplicated.
- The 0–25 m shelf is masked before patch extraction; the fixed primary supervision/validation domain is 0–20 m. The test 20–25 m subset is explicitly an extrapolation sensitivity check, not in-domain supervised accuracy.
- Split counts: 637 train, 208 validation, 343 test. This replaces neither the historical dataset nor its saved scores.
- New test bins [0,5), [5,10), [10,15), [15,20), [20,25] have 17, 58, 64, 89 and 115 examples respectively; exact 20 m values are counted in the final bin but included in the primary <=20 m score. The primary test count is reported from exact inclusive thresholds, not inferred by summing half-open bins.
- Distinct GEBCO bilinear support-cell counts are 828/261/413. These are support counts, not independent survey counts. Within-split patch footprints and source cells overlap; train/validation/test support sets are disjoint. Training footprints also have a 32-pixel buffer from held-out footprints. Validation candidates that share test source support and training candidates that share held-out support are excluded by this predefined reference-independence rule, never by prediction error.
- Features/glint/statistical references fit training footprints only. Augmentation falls back to the untransformed patch if a flip would move centre supervision beyond 20 m or onto invalid water.
- Fresh seed-42 model, all parameters trainable, maximum 180 epochs, batch 8, lr .001 single cosine cycle, EMA .999, early-stop patience 30. One composite: normalized depth Huber + beta-NLL (variance in metres squared) + .03 normalized extinction violation + .1 masked optical reconstruction Huber on reflectance/.05. Shared encoder/physics conditioning/depth/uncertainty/reconstruction receive gradients.
- Select the checkpoint by 0–20 m validation RMSE. After selection, evaluate the unchanged-region test sample, report reference-depth cutoff sensitivity at 15/20/25 m and approximately 2 km spatial-cluster bootstrap intervals (2,000 draws). Keep all eligible samples in the primary result; heuristic optical flags are separate.

Original full-depth 3.554 m test RMSE is not a matched comparator to new shallow samples/targets. A lower shallow-only score cannot by itself establish a better architecture. No independent measured bathymetry is introduced. The older four-arm job stopped at control epoch 8 and must not be resumed automatically.

## What the existing optical bound actually does

`compute_kd_and_extinction` estimates Kd from a heuristic blue/green power law plus an optional red contribution, then returns z = log(100)/(2 Kd) by default. The chosen signal ratio 100, added red term and exact coefficients are not established local Sentinel-2 calibration. A precise numerical implementation is not the same as an accurate optical detection limit.

The previous selected model's mean heuristic test bound is 6.234 m. Masking every prediction above that bound would retain only **3/65 (4.62%)** test examples. This is a major coverage collapse, not a demonstrated accuracy improvement. The 0–15/20/25 m reference-based sensitivity results also do not establish where the bottom is optically detectable.

NASA's [Kd product description](https://modis.gsfc.nasa.gov/data/dataprod/kd_490.php) describes an empirical attenuation relationship calibrated on in-situ Kd and remote-sensing reflectance. Those coefficients/input conventions cannot simply be substituted for this model's glint-corrected L2A surface-reflectance inputs.

## More defensible detection-limit function

[src/optical_limit.py](../src/optical_limit.py) implements the explicit contrast/noise model:

S_bottom(z) = C_bottom exp(-(Kd + Ku) z)

z_limit = max(0, log(C_bottom / (tau sigma_noise)) / (Kd + Ku)).

Inputs: positive downwelling/upwelling attenuation in inverse metres; bottom-versus-water contrast and noise in the same reflectance units; declared detection SNR tau. Invalid inputs produce NaN plus validity flags; a surface-undetectable contrast produces zero depth support. Separate support flags can include a one-sided depth-uncertainty margin. Operational flagging requires an explicit assertion that inputs were calibrated; the helper does not clip depths or alter evaluation.

Three numerical tests verify the detection-threshold equality, invalid/undetectable handling, and expected response to lower attenuation/noise. **The function is physically explicit but not locally calibrated**. No measured Kd, Ku, bottom contrast or effective noise has been provided. Temporal image standard deviation is not automatically sensor noise: turbidity/tides/glint variability contribute too. Neither predicted optical parameters nor the current heuristic alone establish an accurate metric cutoff.

This independent helper was added after the old joint job had launched, so it did not alter that job. The current nearshore run retains the declared heuristic bound as a weak loss and reports flags; it does not hard-mask outputs using an unvalidated estimate.

The reconstruction background `r_inf` is fitted from the second percentile of training-region reflectance. In a shallow-only dataset that is a statistical reference, not a confirmed optically deep-water reflectance. The same applies to the Lyzenga feature offsets. This run tests the existing heuristic convention; a future physics calibration should assess suitable unlabeled deep-water optical controls inside training regions, or measured water properties, without importing test-region statistics or changing the current frozen experiment.

## Is the AOI responsible?

Potentially, but not established. Water clarity, bottom reflectance, waves/glint, atmospheric correction and depth all affect detectability. The [IHO best-practice guide](https://iho.int/uploads/user/pubs/bathy/B_13_Ed100_032024.pdf) recommends examining water clarity and suitable satellite records when choosing an SDB site. No local in-situ clarity/detection measurements were analyzed here. Do not label the Kerala AOI unsuitable based only on a model's residuals or a heuristic 6 m bound.

A useful site assessment would examine individual-date bottom visibility, cloud/glint/clear-observation quality, locally appropriate attenuation estimates, and whether distinct shallow-reference locations cover the domain. If selecting another site later, establish its reference availability and holdout design before tuning the model.

## Existing results: scope and uncertainty evidence

[Read-only sensitivity report](../runs/2026-11-05/nearshore_discussion/REPORT.md) uses saved expanded_smooth predictions on old targets, preserving every example. Test RMSE is 2.949 m at <=15 m (17 samples), 2.871 m at <=20 m (26), 2.850 m at <=25 m (38), and 3.554 m over all 65. These are exploratory subgroup scores, not new trained-model results. Per-bin bootstrap intervals, MAPE and delta <1.25 are included. The 2 km clusters are an approximation; intervals do not include model-training or GEBCO reference uncertainty.

Selected test sigma has Spearman correlation 0.143 with depth and 0.432 with absolute error. Lowest-sigma 50% has RMSE 2.309 m versus 3.554 overall; lowest 25% is 2.851, so risk is not perfectly monotonic. This supports investigating useful uncertainty ranking, but does not prove the model recognizes an invisible bottom. Such a claim needs calibrated coverage, within-depth comparisons, independent optical detectability and a new geographic evaluation.

## GEBCO rather than ICESat-2

GEBCO can support a coarse-reference shallow-water benchmark. Use more distinct native shallow-reference locations, source-quality information if available, grouped spatial splits and honest reference-scale uncertainty. Oversampling/augmentation is training-only; duplicated held-out rows never add independent evidence.

GEBCO cells are centre-registered elevation values, not guaranteed exact area averages. A future aggregated loss must explicitly justify its support assumption; it is not automatically correct to label every 10 m pixel—or its exact mean—with a native cell value. Native-centre/coarse-support supervision and physics-constrained fine detail are possible model choices, but fine-scale truth remains unverified.

If later reconsidered, [ATL24 v2](https://nsidc.org/data/atl24/versions/2) supplies refraction-corrected ICESat-2 coastal/nearshore heights/depths and uncertainties. Usable AOI tracks, bottom returns, flags, acquisition timing and height datums must be inspected before promising better labels. No ATL24 was downloaded or used in the current run.

OSM may supply coastline features and individual soundings with chart datums ([OSM documentation](https://wiki.openstreetmap.org/wiki/OpenSeaMap/Seamark_Attributes)); it is not an assumed complete bathymetric replacement. Paper depth RMSE compares predictions with external LiDAR/survey/reference depths, independently of composite training loss.

## Startup issue and preserved state

The older joint job initially stopped before training because corrected native-coverage gaps remained NaN in masked-out Y storage. All centre and valid-water labels were finite. The producer now fills only missing masked-out storage with zero and asserts no valid-water missing values. The original failed NPZ remains preserved. After repair the old control ran eight epochs, then the user explicitly stopped it; no integrated/full/Sentinel-only training from that job completed. Current nearshore data implement the safe storage rule from creation.

All 19 earlier tests passed, plus the new shallow-augmentation regression test (20 total distinct checks). No change claims true-depth sub-metre accuracy. Use the current nearshore job status rather than the historical stop marker to identify active work.
