# Nearshore HybridBathNet squared-depth-objective results

Original geographic test blocks retained. Only HybridBathNet trained. Shelf eligibility <=25m applied before patch extraction; depth training/selection limited to <=20m. No sample duplication or predicted-depth clipping.

Validation RMSE (0–20m): 2.100 m.

| Test maximum reference depth | n | RMSE (m) | 95% spatial bootstrap CI | MAPE (%) | Delta <1.25 (%) |
|---|---:|---:|---|---:|---:|
| 15 | 143 | 2.626 | 1.826–3.491 | 21.6 | 62.9 |
| 20 | 228 | 2.590 | 1.910–3.277 | 17.8 | 71.5 |
| 25 | 343 | 3.365 | 2.680–3.947 | 18.0 | 65.3 |

| Primary 0–20m depth bin | n | RMSE (m) | 95% spatial bootstrap CI |
|---|---:|---:|---|
| 0-5 | 17 | 1.399 | 0.759–3.994 |
| 5-10 | 58 | 2.088 | 1.569–2.538 |
| 10-15 | 64 | 3.270 | 1.884–4.656 |
| 15-+ | 89 | 2.507 | 1.683–3.383 |

Heuristic optical-support flags cover 89.5% of test predictions. These are diagnostic flags, not a calibrated mask; all eligible samples remain scored.

The 20–25m test subset is outside the supervised training-depth range. Bootstrap intervals are conditional on the model and GEBCO, and do not include reference error. Source-cell support is disjoint across splits; within-split patches overlap. Prior 3.554m full-depth test RMSE is not a matched comparator for the new shallow samples and scope. No sub-metre real-seabed accuracy is established.

[Protocol](protocol.json) | [All results](results.json) | [Dataset audit](features/metadata.json)
