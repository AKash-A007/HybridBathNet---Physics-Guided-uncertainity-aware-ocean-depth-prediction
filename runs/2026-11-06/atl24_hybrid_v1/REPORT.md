# ATL24-supervised HybridBathNet results

A distinct, from-scratch full joint HybridBathNet trained on ATL24 depth labels.
The Jan–Mar 2024 Sentinel-2 composite is the input; no GEBCO depths are targets or features.
Original geographic test blocks are kept. Check [protocol](protocol.json) and [data audit](features/metadata.json).

Validation 0–20 m RMSE: **3.443 m** (n=112).
Test 0–20 m RMSE: **4.741 m** (n=233).
The test 95% approximate 2 km cluster-bootstrap interval is **3.460–5.832 m**;
test MAE is **3.285 m**, mean bias **+0.867 m**, and delta <1.25 is **26.2%**.
At ≤15 m, test RMSE is **3.997 m** on 218 centres; the 15–20 m bin has only 15 centres.

The best EMA checkpoint was selected at epoch **60** by validation RMSE; early stopping
ended training at epoch **90**. Training remained finite. At epoch 60 the logged
depth, beta-NLL, extinction and reflectance-reconstruction losses were
0.2893, 4.1190, 1.1208 and 0.00169 respectively. The shared encoder,
physics conditioning/reconstruction, depth and uncertainty branches were trained
jointly; the separate optional `w_phys` term was zero, as in the selected GEBCO
v1 setup. Physics accuracy is not established by a finite reconstruction loss.

| Test ATL24 depth bin | n | RMSE (m) | 95% approx. 2 km group bootstrap CI |
|---|---:|---:|---|
| 0–5 | 129 | 4.287 | 2.531–5.990 |
| 5–10 | 55 | 2.356 | 1.498–3.114 |
| 10–15 | 34 | 4.873 | 2.706–6.582 |
| 15–20 | 15 | 10.811 | 7.861–12.448 |

The model overpredicts the 0–5 m bin by **2.929 m** on average and underpredicts
15–20 m by **10.321 m**. It compresses depths toward the middle: test target
median is 2.950 m, predicted median is 5.960 m. The 10–15/15–20 m bins span
only 9/8 approximate 2 km groups, so their bootstrap intervals are fragile.

Raw Gaussian-style 68% and 90% intervals cover **65.24%** and **84.98%** of test
centres, respectively (validation: 67.86% and 87.50%). Mean predicted sigma is
**3.156 m** on test and its correlation with absolute error is **0.211**. These
are under nominal 90% coverage and are not geographically calibrated
uncertainty bounds. [Full results](results.json) include the other metrics.

**Interpretation:** ATL24 points along repeat tracks are correlated and some public-service granules may be absent.
Optical conditions/seabed may have changed between 2018–2024 reference dates and 2024 imagery.
The existing GEBCO test RMSE 2.561 m is a different benchmark, not a matched comparator.
The optical-limit heuristic is uncalibrated; no pixels were masked to improve reported RMSE.
The four official NSIDC V002 2024 AOI-intersecting files had only **32** confidence≥0.6,
0–20 m bathymetry photons, so this run uses a public SlideRule archive subset
that advertises **V003** inputs. The returned subset had 3,189 bathymetry-class
AOI photons across 2018–November 2024; per-pixel aggregation, quality screening,
fixed block assignment and 96-pixel split buffers left **570/112/233** train/
validation/test labels. Source errors during the public request mean its coverage
may be incomplete. The source and split checks are in [the data metadata](features/metadata.json)
and [AOI audit](../../2026-11-05/nearshore_error_reference_audit_v1/REPORT.md).

**Decision:** This is a completed independent-label feasibility run, not a replacement
for the selected GEBCO v1 model. The ATL24 score does not establish an accuracy
improvement. Before another ATL24 training iteration, verify the reference across
time and quality flags, expand independent deep/shallow track coverage, and
predeclare a validation-driven change. Do not tune to these inspected test bins.
