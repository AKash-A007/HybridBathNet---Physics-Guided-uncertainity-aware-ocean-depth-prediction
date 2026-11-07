# Integrated physics/depth/uncertainty candidate results

Candidate: decoupled Gaussian variance supervision and patch-shared effective per-band attenuation. Full encoder, depth, physics and uncertainty remain active. Optical masking remains disabled.

Original geographic test blocks retained. Only HybridBathNet trained. Shelf eligibility <=25m applied before patch extraction; depth training/selection limited to <=20m. No sample duplication or predicted-depth clipping.

Validation RMSE (0–20m): 1.961 m.

| Test maximum reference depth | n | RMSE (m) | 95% spatial bootstrap CI | MAPE (%) | Delta <1.25 (%) |
|---|---:|---:|---|---:|---:|
| 15 | 143 | 2.433 | 1.633–3.235 | 18.8 | 69.2 |
| 20 | 228 | 2.527 | 1.847–3.208 | 16.4 | 74.1 |
| 25 | 343 | 3.363 | 2.630–3.994 | 17.2 | 67.1 |

| Primary 0–20m depth bin | n | RMSE (m) | 95% spatial bootstrap CI |
|---|---:|---:|---|
| 0-5 | 17 | 1.110 | 0.570–3.296 |
| 5-10 | 58 | 1.705 | 1.281–2.078 |
| 10-15 | 64 | 3.166 | 1.895–4.432 |
| 15-+ | 89 | 2.650 | 1.836–3.519 |

Heuristic optical-support flags cover 89.5% of test predictions. These are diagnostic flags, not a calibrated mask; all eligible samples remain scored.

The 20–25m test subset is outside the supervised training-depth range. Bootstrap intervals are conditional on the model and GEBCO, and do not include reference error. Source-cell support is disjoint across splits; within-split patches overlap. Prior 3.554m full-depth test RMSE is not a matched comparator for the new shallow samples and scope. No sub-metre real-seabed accuracy is established.

[Protocol](protocol.json) | [All results](results.json) | [Frozen GEBCO comparator](../../2026-11-05/nearshore_hybrid_v1/REPORT.md)
