# Improvement experiment protocol

Preserve the completed 2026-11-04 study and the historical six-channel results.

- Continue the selected weak-extinction checkpoint from epoch 90 to at most 300 on the original dataset.
- Train the same configuration from scratch on 2176 training patches for at most 180 epochs, with the original 30-epoch cosine restarts.
- Train the expanded-data configuration using one 180-epoch cosine cycle, with no intermediate restarts.
- Select by validation RMSE, repeat the selected configuration from scratch with seed 1, then evaluate the fixed test set and write REPORT.md.
- Use patience 60, EMA decay 0.999, batch size 8, and extinction penalty weight 0.03. Preserve all sixteen input channels.

The training expansion recovers patches entirely contained in existing training blocks. Original validation/test coordinates and labels are unchanged. Every training footprint has at least a 32-pixel buffer from held-out footprints. Feature fitting is repeated using only training pixels. Overlapping patches within training are allowed; this increases samples, not independent scenes or bathymetric surveys.

The original split retained one of sixteen patch positions per complete 128-pixel block. Its boundary exclusion was more restrictive than required for pixel separation. The expanded builder does not alter the historical splitter or existing data files.

The original 306 samples give approximately 39 optimizer steps per epoch; EMA 0.999 averages roughly 1000 steps, about 26 epochs. Expanded data gives 272 steps per epoch, about 3.7 epochs for the same EMA horizon. Holding decay fixed isolates data expansion from an additional EMA change.

Validation errors are saved by depth in data_audit.json. The original model overestimates shallow depths and underestimates deeper depths. Its optical bound remains an uncalibrated heuristic; no formula or test-dependent depth cutoff is introduced in this comparison.

Previously inspected test scores are exploratory comparisons. A future independent geographic holdout is needed. The proposed all-scene Sentinel temporal dataset is deferred until these results are available; dates from the same spatial footprint must not be randomly scattered across splits.

## Resume and reporting

Do not launch another copy while the VS Code task is running. If interrupted:

```powershell
Set-Location 'C:\Amrita\FInal Year Proj\hybridbathnet_baseline'
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8='1'
python -u scripts/run_improvement_study.py
```

The runner resumes checkpoints, skips completed validation runs, and produces REPORT.md and comparison.csv after final evaluation. LIVE_PROGRESS.md records every ten epochs. A failure writes failure.json and stops the study.
