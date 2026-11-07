# Authorized joint composite and Sentinel-only experiments

User authorizes joint physics-depth-uncertainty training, the four proposed comparisons and a separate Sentinel-only experiment. Preserve all older artifacts. Questions about paper RMSE, OSM and depth bins steer interpretation without cancelling the runs.

## Objective and checks

JointCompositeLoss = normalized depth Huber + metre-variance beta-NLL + 0.03 normalized extinction + 0.1 masked optical reconstruction Huber on reflectance scaled by 0.05. RMSE is computed separately from predictions and reference depths. Reconstruction gradients reach depth and learned attenuation/bottom-reflectance heads. Optical scale remains non-identifiable without depth references.

Mask-aware conditioning computes water-only surface percentiles and zeros invalid conditioning. It is opt-in to preserve old model behavior. Optical-head initialization avoids exponential saturation. Full-coverage sampling uses every training index once with resumable RNG. Sixteen tests pass, including invalid-pixel invariance and explicit gradients to encoder, conditioning, depth, logvariance and reconstruction heads. Sentinel-only backward smoke check also passes without updating weights. Compilation and diff checks pass.

## Supervised data and runs

Native GEBCO labels and train-fitted features are rebuilt on the historical imagery grid, preserving all 65 validation and 65 test locations. Six training patches fail corrected centre/full-patch water eligibility: **2170 training** remain. No held-out examples are removed. This differs from the prior centre-only pilot's five exclusions.

Four seed-42 matched warm-start pilots, 15 epochs each: control; mask-aware only; mask-aware joint reconstruction; joint reconstruction with sampling without replacement. Fixed data, initialization, batch 8, lr 1e-4 cosine schedule, augmentation, depth/NLL/extinction weights, and identical optical-head initialization. Reconstruction weight is zero in the first two arms.

Then train full_joint from random initialization for up to 180 epochs: mask-aware physics, reconstruction weight .1, full coverage, lr .001 single cosine cycle, batch 8, EMA .999, patience 30. This full run is explicitly requested regardless of inconclusive pilot ordering. Pilot-versus-full scores do not isolate one factor because budgets/initializations differ. Select checkpoints by validation RMSE; report Gaussian NLL and interval coverage separately. Freeze choices before exploratory test evaluation.

## Sentinel-only run

After supervised training, run the independent `scripts/run_sentinel_only.py` from random initialization. Read only Sentinel spectral/std and SCL validity bands, never GEBCO band 9. Independently saved optical NPZ has no depth target field. NDWI/clear-water mask, training-only feature fitting, contiguous latitude groups with gaps: **1097/237/290**. All dates in the existing composite remain grouped.

Self-supervised composite: optical reconstruction + local delta-method propagation of latent depth variance into optical NLL + weak heuristic optical-depth-prior/extinction consistency. Up to 30 epochs, patience 10, select by validation reflectance RMSE. Test reflectance RMSE is available; true-depth RMSE is null. These are composite/std inputs, not individual-scene inference. Do not call the latent depth or its variance independently validated/calibrated.

## Monitoring and recovery

Status: status.json. Supervised report: REPORT.md; optical report: sentinel_only/REPORT.md. Reports appear on completion. The runner updates marked status blocks in PROJECT_CONTEXT.md and README.md at each stage. History under each supervised checkpoints directory records component losses each epoch; watch_study_progress.ps1 displays ten-epoch milestones. Sentinel-only history records optical metrics separately.

Resume only if no existing job is active:

```powershell
Set-Location 'C:\Amrita\FInal Year Proj\hybridbathnet_baseline'
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8='1'
python -u scripts/run_joint_study.py
```

Completed validation arms are skipped; incomplete training restores optimizer/scheduler/EMA/RNG. Failure writes failure.json. Sentinel-only checkpoints also restore optimizer/scheduler/Torch/CUDA/sampler RNG. No independent survey depths are added and the existing test set has already been inspected.

## Depth-domain steering

[Requested analysis](../depth_bin_review/REPORT.md): selected test 0–5/5–10/10–15/15+ m RMSE **1.709/3.810/2.247/3.745 m**, counts **2/7/8/48**. Deep samples account for 82.0% of squared error but 73.8% of samples. No abrupt optical cutoff is shown; no test-dependent sample removal is performed. A restricted operating domain needs an independently justified fixed rule and coverage reporting.
