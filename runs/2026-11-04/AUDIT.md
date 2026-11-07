# HybridBathNet audit and experiment interpretation

## Confirmed implementation issues

| Location | Evidence | Repair |
|---|---|---|
| `src/hybridbathnet.py`, `HybridBathNetModel` reconstruction head | Unmodified HEAD forward pass raises a 64-versus-32 input-channel error; captured in `original/reproduction.json` | Match the head to the full-resolution 32-channel encoder skip |
| `src/cnn_baseline.py`, `EMA.update` | Runtime check: live BatchNorm mean was 0.5 while the EMA copy remained 0 | Copy trained buffers when updating the EMA |
| `src/cnn_baseline.py`, `train_hybridbathnet` | Code restored ordinary weights before saving the allegedly best EMA state | Capture the EMA state actually evaluated and save it as best checkpoint |
| `src/cnn_baseline.py`, `PatchDataset` | A water-only mask statistic has mean 1 and near-zero standard deviation; land then becomes about -1,000,000 | Preserve binary mask values and handle constant channels |
| `src/hybridbathnet.py`, `CompositePhysicsNLLLoss` | Normalized squared depth residuals were used with a variance later reported as metres² | Compute the residual term in metres² and keep variance arithmetic in float32 |
| Half-precision variance arithmetic | Focused example returned infinity for a float16 squared-error/variance calculation | Float32 loss arithmetic and explicit finite-loss/gradient checks; this reproduces a failure mode, not proof of the friend's exact failure |
| Existing `data/features/patches.npz` | 6 channels, spectral means roughly 1,200–1,500; current code expects 16 channels and surface reflectance | Rebuild preprocessing and features; reject stale channel layouts |
| Spectral augmentation / physics data flow | Inverting normalization after spectral jitter also jitters the physical input | Pass separate raw spectral inputs and apply matching spatial transforms, without spectral jitter |
| Feature-reference fitting | Glint slopes, deep-water offsets and surface-reference percentiles used scene-wide pixels | Fit those statistics on training patch footprints only in the controlled workflow |
| Validation/test fallback | Empty validation or test indices silently fell back to training indices | Fail if any split is empty |
| Patch-centre labels | A patch could be over 50% water while its supervised centre was invalid | Require a valid water centre |
| Stumpf ratio | `log(1000*green)` vanishes at green=0.001 | Explicit neutral imputation at the singularity; no undocumented equation replacement |
| `src/compare_baselines.py` | The compatibility result file duplicated HybridBathNet as a separate CNN baseline | Avoid the duplicate; the study trains a genuine CNN-only control |
| Sparse depth-bin reporting | Bins with fewer than 10 samples were deliberately reported as NaN | Report null plus sample counts, distinguishing insufficient data from numerical failure |

Ten focused tests passed before full GPU training. The source snapshot and package versions are saved under `study/`. Checkpoints now also preserve optimizer, scheduler, EMA, scaler, sampler and random-number states for resumption.

## Original implementation and historical result

The original HEAD source is preserved as `original/source_HEAD.zip`. The user's pre-existing modified checkpoint and metrics were copied without replacing their source files. Historical metrics are RMSE 3.5066 m, MAE 2.6732 m, R² 0.9163, ±1 m accuracy 23.1589%, and ±2 m accuracy 44.6705%. The checkpoint lacks seed, split, timing and input-normalization provenance, and the old features have a different channel/scaling layout. These numbers cannot establish an improvement or deterioration against the new controlled split.

Unmodified HEAD cannot train with the default reconstruction head: its first forward pass fails. The study's compatibility reference necessarily includes mandatory numerical/execution repairs and common training-only feature fitting. It is a controlled repaired reference, not an unmodified historical reproduction.

## Physics conclusions

The current v3 implementation already stores exact spectral normalization parameters and reverses them before computing Kd. The attachment's claim that normalized channels are necessarily fed directly into Kd does not hold for this version. Spectral jitter still changes the inverted input, which the explicit-raw ablation addresses. V3 also already removes the cited latent-feature clipping and applies Softplus to predicted depth, not merely to hidden features. The study therefore does not invent duplicate ablations of changes already present.

The exact blue-green power law `0.0166 + 0.156*(B/G)^(-1.14)` and the additive `0.430*R/(G+eps)` term could not be established as a calibrated Sentinel-2 retrieval from the primary sources reviewed. They remain explicitly labelled heuristics. [Wang et al. (2009)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009JC005286) describes empirical normalized water-leaving-radiance algorithms and semianalytical alternatives; this is not blanket justification for the repository's coefficients or additive red term. [NASA's SeaDAS training material](https://appliedsciences.nasa.gov/sites/default/files/2024-02/SeaDAS_Final_1Feb.pdf) shows a sensor-specific log-polynomial KD2 expression using remote-sensing reflectance, differing from the implemented power law. Sentinel-2 L2A surface reflectance is not automatically a validated aquatic remote-sensing-reflectance product.

The implementation tests prove arithmetic and shape correctness, not empirical validity of the optical equation. No independent optical profiles, bottom-detection thresholds or surveyed depths were available to calibrate the formula. Signal ratios 10, 100 and 1000 are sensitivity assumptions. Epsilon is numerical stabilization, distinct from this ratio. Bounds are never hard ceilings on predictions. The study measures bound clipping, reference depths above the bound, prediction excess and accuracy. An early completed repaired reference put 95.38% of validation targets above the heuristic bound, a reason to test removing the penalty rather than claim physical validity.

## Evaluation protocol and remaining limitations

- Fresh preprocessing of the 16 real SAFE products completed on 2026-11-05. Product names match the live catalogue, and all 1,376 locally manifest-listed files pass size and checksum validation. Original ZIPs are absent, so equality with current remote archive bytes has not been established. Existing GEBCO_2020 labels were reused.
- All controlled experiments use the same 306 training, 65 validation and 65 test patches, 16 channels, seed 42 for paired comparisons, and a 90-epoch budget. Train/validation/test pixel footprints have no overlap; coarse GEBCO reference cells and regional spatial correlation remain limitations.
- Normalization and fitted feature statistics use training pixels only. Model/configuration selection uses validation RMSE. All choices are frozen before test evaluation. Selected configuration variability is assessed with seeds 42, 0 and 1; seeds are not selected using test scores.
- Multi-stage validation selection can itself overfit 65 validation samples. A new geographic region and independent depth reference are necessary before broad generalization claims.
- The current training objective supervises patch centres. Dense reconstruction and GEBCO pooling losses are not active, despite broader architecture comments.
- Uncertainty evaluation is aleatoric Gaussian calibration/coverage. MC dropout, deep ensembles and conformal calibration are not invoked and will not be claimed as evaluated.
- The CNN-only control bypasses FiLM and removes engineered physics channels 4–10 from the network input, with extinction loss disabled. It shares spectral, mask and temporal inputs, optimizer and split with the hybrid models.
- Patch-mean/centre/spatial conditioning, red correction, signal-ratio sensitivity, unclipped bounds, weaker/no extinction penalty and linear-depth/nonnegativity-loss control are changed in documented stages. Compare each variant to its stated parent configuration, not as though every pair differs by only one factor.

See [manual commands](../../RUN_TRAINING_MANUALLY.md), [study progress](study/status.json), and the final `study/REPORT.md` once the study completes.
