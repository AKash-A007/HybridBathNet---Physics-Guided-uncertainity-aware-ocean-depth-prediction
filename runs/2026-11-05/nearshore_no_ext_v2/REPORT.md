# Nearshore HybridBathNet extinction-penalty ablation results

Original geographic test blocks retained. Only HybridBathNet trained. Shelf eligibility <=25m applied before patch extraction; depth training/selection limited to <=20m. No sample duplication or predicted-depth clipping.

Validation RMSE (0–20m): 2.095 m.

| Test maximum reference depth | n | RMSE (m) | 95% spatial bootstrap CI | MAPE (%) | Delta <1.25 (%) |
|---|---:|---:|---|---:|---:|
| 15 | 143 | 2.487 | 1.680–3.345 | 19.1 | 67.8 |
| 20 | 228 | 2.550 | 1.893–3.234 | 16.6 | 72.8 |
| 25 | 343 | 3.361 | 2.625–3.980 | 17.3 | 66.8 |

| Primary 0–20m depth bin | n | RMSE (m) | 95% spatial bootstrap CI |
|---|---:|---:|---|
| 0-5 | 17 | 0.986 | 0.508–3.145 |
| 5-10 | 58 | 1.806 | 1.361–2.231 |
| 10-15 | 64 | 3.220 | 1.921–4.550 |
| 15-+ | 89 | 2.625 | 1.845–3.504 |

Heuristic optical-support flags cover 90.4% of test predictions. These are diagnostic flags, not a calibrated mask; all eligible samples remain scored.

The 20–25m test subset is outside the supervised training-depth range. Bootstrap intervals are conditional on the model and GEBCO, and do not include reference error. Source-cell support is disjoint across splits; within-split patches overlap. Prior 3.554m full-depth test RMSE is not a matched comparator for the new shallow samples and scope. No sub-metre real-seabed accuracy is established.

[Protocol](protocol.json) | [All results](results.json) | [Dataset audit](features/metadata.json)
