# Run or resume HybridBathNet on Windows

## Spatial OOF uncertainty study (latest authorized experiment)

Use only after the existing job exits; do not run a second GPU copy. Prepared fold features are reused, and completed folds are skipped. Fold training resumes its saved optimizer/EMA/RNG state. The short correction-head stage restarts if interrupted before completion; completed head/results are preserved.

```bat
cd /d "C:\Amrita\FInal Year Proj\hybridbathnet_baseline"
call venv\Scripts\activate.bat
python -u scripts\run_gebco_oof_models.py
python -u scripts\train_gebco_oof_head.py
python scripts\report_gebco_oof_uncertainty.py
```

For progress without training: `python scripts\progress_gebco_oof.py`. Outputs: `runs/2026-11-06/gebco_oof_uncertainty_v1/REPORT.md`, `results.json`, `test_predictions.csv`, full fold checkpoints and `risk_head/best.pt`. This study uses GEBCO only, all physics/depth/uncertainty source modules and unchanged final ensemble depth means.

## Current recovery study (6 November2026)

Read PROJECT_CONTEXT.md first. Run the training command only after the existing job has exited; a second copy would duplicate GPU work. Completed arms are skipped and incomplete arms resume `last_training.pt`, including optimizer/EMA/RNG state. Existing downloaded inputs and immutable prepared features are reused. This command runs only the two full HybridBathNet official-ATL24 recovery arms.

In **Command Prompt**:

```bat
cd /d "C:\Amrita\FInal Year Proj\hybridbathnet_baseline"
call venv\Scripts\activate.bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u scripts\run_official_recovery.py
python scripts\report_recovery_v2.py
python scripts\verify_recovery_checkpoints.py
```

For progress only, without starting training:

```bat
python scripts\progress_recovery.py
```

Outputs: `runs/2026-11-06/recovery_v2/REPORT.md`, `LIVE_PROGRESS.md`, `atl24_paired/comparison.json`, each arm's checkpoints, and `verification.json`. The GEBCO uncertainty seeds are already complete; their report is under `gebco_uq/REPORT.md`. Do not rebuild completed features or run the historical commands below to resume the recovery study.

## Historical study commands

Use PowerShell. The existing environment already contains CUDA PyTorch. This workflow trains HybridBathNet ablations and a CNN-only control; it does not train tabular methods.

## 1. Activate the existing environment

```powershell
Set-Location 'C:\Amrita\FInal Year Proj\hybridbathnet_baseline'
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = '1'
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If PowerShell prevents activation, use `& '.\venv\Scripts\python.exe'` in place of `python` in each command below. Activation is not required when invoking the interpreter directly.

## 2. Check whether training is still running

Closing a chat does not necessarily stop its background process. Do not launch a second training process while one is still using the same study directory.

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Select-Object ProcessId, CommandLine
Get-Content '.\runs\2026-11-04\study\status.json'
nvidia-smi
```

Look for `scripts/run_physics_study.py` or `-m src.cnn_baseline`. The status file alone does not prove a process is alive.

## 3. Resume the current study (recommended after interruption)

Preprocessing and training-only feature engineering have already completed. To continue without repeating them:

```powershell
python -u scripts/run_physics_study.py --config runs/2026-11-04/config.yaml
```

The runner skips experiments with completed validation results. An interrupted experiment reloads `last_training.pt`, including model, optimizer, scheduler, EMA, scaler, sampler and random-number states, and resumes at the next epoch. A partially completed epoch is repeated. Keep the configuration, source and dataset unchanged when resuming.

The study runs the configured 90-epoch budget per variant. It compares explicit raw reflectance, patch-mean/centre/spatial physics, red correction, signal ratios, clipping, extinction penalties and output nonnegativity. It selects using validation RMSE, repeats the selected configuration with additional seeds, then evaluates held-out test data. The compatibility reference contains mandatory execution/stability repairs; it is not the unmodified historical implementation.

## 4. Alternatively, run the full workflow starting at preprocessing

Only use this when no study process is running. The 16 local Sentinel-2 products match the catalogue selection and all 1,376 manifest-listed files passed checksums, so downloading them again is unnecessary.

```powershell
python -u run_baseline_pipeline.py --config runs/2026-11-04/config.yaml --skip-download --hybrid-only --physics-study
```

This starts at preprocessing, rebuilds training-fitted features and spatial splits, then runs/resumes the controlled study. It uses the existing GEBCO file. Do not overwrite features under a running experiment. For an intentionally changed dataset or configuration, create a separate run directory instead of resuming these checkpoints.

For only a single HybridBathNet run, without the ablation study:

```powershell
python -u run_baseline_pipeline.py --config runs/2026-11-04/config.yaml --skip-download --hybrid-only
```

That single-run command saves its checkpoint and metrics to the paths in the configuration; it does not generate the study comparison report.

## 5. Monitor progress

```powershell
Get-Content '.\runs\2026-11-04\study\status.json'
$studyStatus = Get-Content '.\runs\2026-11-04\study\status.json' | ConvertFrom-Json
Get-Content (Join-Path '.\runs\2026-11-04\study' ($studyStatus.experiment + '\training.log')) -Tail 20 -Wait
```

Press Ctrl+C to stop following the log; that command does not stop the training process. Each experiment also saves `checkpoints/training_history.json` and `checkpoints/best_ema.pt`. If the runner fails, inspect `study/failure.json` and the active experiment's training log. Numerical failures stop explicitly rather than continuing with NaNs.

## 6. Generate and inspect results

The study automatically creates its report when all training and final evaluation stages succeed. To regenerate it afterwards:

```powershell
python scripts/run_physics_study.py --config runs/2026-11-04/config.yaml --report-only
Get-Content '.\runs\2026-11-04\study\REPORT.md'
Import-Csv '.\runs\2026-11-04\study\comparison.csv' | Format-Table
$selection = Get-Content '.\runs\2026-11-04\study\selection.json' | ConvertFrom-Json
Write-Output ('Selected checkpoint: runs/2026-11-04/study/' + $selection.selected_hybrid + '/checkpoints/hybridbathnet.pt')
```

Report-only requires completed `selection.json`, `validation_summary.json` and `test_summary.json`. If these do not exist, resume the study using step 3 first.

Outputs:

- [Comparison report](runs/2026-11-04/study/REPORT.md)
- [All metrics](runs/2026-11-04/study/comparison.csv)
- [Validation curves](runs/2026-11-04/study/validation_curves.png)
- [Selected configuration record](runs/2026-11-04/study/selection.json)
- [Multiple-seed variability](runs/2026-11-04/study/seed_variability.json)
- Per-experiment configuration, checkpoints, logs, predictions and uncertainty metrics under `runs/2026-11-04/study/<experiment>/`.

Historical results are preserved under [original](runs/2026-11-04/original/). Their different features and incomplete experiment provenance prevent a direct improvement claim against the new study. Scores use GEBCO reference depths, not independent surveyed depths. Sparse depth bins are explicitly unavailable rather than treated as zero error.
