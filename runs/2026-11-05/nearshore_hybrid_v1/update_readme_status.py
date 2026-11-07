from pathlib import Path
p=Path('README.md')
s=p.read_text(encoding='utf-8')
a=s.index('## Current experimental status')
b=s.index('It includes:',a)
section='''## Current experimental status — updated 2026-11-05

**Nearshore HybridBathNet training is complete. No training is running.** Only HybridBathNet was trained, with one active composite loss joining depth Huber, uncertainty beta-NLL, extinction and masked optical reconstruction. Mask-aware conditioning and full-coverage training sampling are enabled. Test-region block IDs are unchanged; shallow sampling adds distinct locations without duplication.

Selected EMA checkpoint: epoch **118**, early stopping at **148**. Primary 0–20m validation RMSE **1.963m**; test RMSE **2.561m**, MAE **1.894m**, on **228** examples. The test RMSE 95% spatial-bootstrap interval is **1.840–3.261m**. Sensitivity at maximum reference depth15/20/25m is **2.386/2.561/3.434m**; 20–25m is outside supervised training depth. [Full report, per-bin intervals and uncertainty](runs/2026-11-05/nearshore_hybrid_v1/REPORT.md) · [Selected model](runs/2026-11-05/nearshore_hybrid_v1/hybridbathnet.pt).

Dataset: **637 train /208 validation /343 test sensitivity locations**, with shelf masking before extraction, train-fitted features, and disjoint native GEBCO interpolation-support sets. Overlapping patches and interpolated labels are not independent10m survey observations. [Dataset audit](runs/2026-11-05/nearshore_hybrid_v1/features/metadata.json).

Uncertainty remains under-calibrated: nominal90% intervals cover **71.1%** of primary test examples. On26 identical shallow test coordinates with identical corrected targets, old/new RMSE is **2.898/2.552m**, but the paired improvement interval crosses zero. Historical full-depth results are not matched shallow-domain comparisons. The optical limit is a tested mathematical helper with uncalibrated local inputs; no hard clipping or test-error-based exclusions were applied.

**Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** for the maintained LLM handoff, exact run state, checks and future work. Update it and this section after every modification; see [AGENTS.md](AGENTS.md). All20 regression tests pass. Source snapshots, checkpoints and original results are preserved. [Current nearshore/optical design](docs/OPTICAL_LIMIT_AND_NEARSHORE_PLAN.md).

The older joint_v1 job was stopped at control epoch8; its full joint and Sentinel-only training never began. Prepared prototype code/data remain available but are not automatically authorized for new runs. Earlier [depth-bin](runs/2026-11-05/depth_bin_review/REPORT.md), [cutoff/relative-error/uncertainty](runs/2026-11-05/nearshore_discussion/REPORT.md), and [GEBCO audit](runs/2026-11-05/grid_audit/REPORT.md) reports are retained.

Broader commands and repository capabilities below include tabular/dense/ensemble features that are not all active in the reported experiments. Current user scope is HybridBathNet only with the existing venv. Depth supervision is at patch centres; the new run actively supervises optical reconstruction over masked patches. Dense depth/pooled losses, MC-dropout ensembling and conformal calibration are not active.

'''
p.write_text(s[:a]+section+s[b:],encoding='utf-8')
