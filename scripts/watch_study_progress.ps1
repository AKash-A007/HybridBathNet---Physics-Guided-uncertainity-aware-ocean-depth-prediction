param(
    [string]$StudyRoot = (Join-Path $PSScriptRoot '../runs/2026-11-04/study'),
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
$StudyRoot = (Resolve-Path -LiteralPath $StudyRoot).Path
do {
    try {
        $status = Get-Content -LiteralPath (Join-Path $StudyRoot 'status.json') -Raw | ConvertFrom-Json
        $lines = @('# Live training progress', '', "Updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')", '',
            "Stage: $($status.stage). Active experiment: $($status.experiment).", '',
            'Validation metrics only; final test results are reported after selection.', '')
        $rows = @()
        foreach ($folder in (Get-ChildItem -LiteralPath $StudyRoot -Directory | Sort-Object CreationTime)) {
            $historyPath = Join-Path $folder.FullName 'checkpoints/training_history.json'
            if (!(Test-Path -LiteralPath $historyPath)) { continue }
            $history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
            if (!$history.Count) { continue }
            $latest = $history[-1]
            $best = $history | Sort-Object val_rmse | Select-Object -First 1
            $lines += @("## $($folder.Name)", '',
                ('Latest epoch: {0}; validation RMSE: {1:F3} m. Best: {2:F3} m at epoch {3}.' -f $latest.epoch, $latest.val_rmse, $best.val_rmse, $best.epoch), '',
                '| Epoch | Validation RMSE (m) | MAE (m) | R2 |', '|---:|---:|---:|---:|')
            foreach ($entry in $history) {
                if ($entry.epoch % 10 -ne 0) { continue }
                $lines += ('| {0} | {1:F3} | {2:F3} | {3:F3} |' -f $entry.epoch, $entry.val_rmse, $entry.val_mae, $entry.val_r2)
                $rows += [pscustomobject]@{ experiment=$folder.Name; epoch=$entry.epoch; val_rmse=$entry.val_rmse; val_mae=$entry.val_mae; val_r2=$entry.val_r2 }
            }
            $lines += ''
        }
        $lines | Set-Content -LiteralPath (Join-Path $StudyRoot 'LIVE_PROGRESS.md') -Encoding UTF8
        $rows | Export-Csv -LiteralPath (Join-Path $StudyRoot 'epoch_milestones.csv') -NoTypeInformation
    } catch {
        # History files can be between writes; retry without touching training.
        Write-Warning $_
    }
    if ($Once -or $status.stage -in @('complete', 'failed')) { break }
    Start-Sleep -Seconds 15
} while ($true)
