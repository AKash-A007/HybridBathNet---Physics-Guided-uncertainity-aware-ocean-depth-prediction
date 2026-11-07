"""Create a standalone report from the isolated HybridBathNet run."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    run = Path('runs/2026-11-04')
    results = json.loads((run / 'results/hybridbathnet_results.json').read_text())
    history = json.loads((run / 'checkpoints/training_history.json').read_text())
    best = min(history, key=lambda row: row['val_rmse'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [row['loss'] for row in history])
    axes[0].set(xlabel='Epoch', ylabel='Training loss', title='HybridBathNet training')
    axes[1].plot(epochs, [row['val_rmse'] for row in history])
    axes[1].set(xlabel='Epoch', ylabel='Validation RMSE (m)', title='EMA validation')
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(run / 'results/training_curves.png', dpi=150)
    plt.close(fig)
    lines = [
        '# HybridBathNet real-data rerun', '',
        'The run uses the existing virtual environment and rebuilt real Sentinel-2 inputs.', '',
        f'Completed epochs: {len(history)}. Best validation epoch: {best["epoch"]}. '
        f'Best validation RMSE: {best["val_rmse"]:.4f} m.', '',
        '| Test metric | Value |', '|---|---:|',
    ]
    for name in ('rmse', 'mae', 'r2', 'acc_within_1m_%', 'acc_within_2m_%',
                 'mean_aleatoric_std_m', 'coverage_68pct', 'coverage_90pct'):
        lines.append(f'| {name} | {results[name]:.4f} |')
    lines += ['', '## Data and protocol', '',
        '- Current Copernicus catalogue selection exactly matches all 16 existing SAFE product names.',
        '- All 1,376 files listed in local SAFE manifests passed size and checksum verification.',
        '- ZIP files are unavailable, so equality with current remote ZIP checksums was not established.',
        '- Raw imagery and GEBCO were reused; preprocessing and 16-channel features were rebuilt.',
        '- Buffered patch splits: 306 training, 65 validation, 65 test. Seed 42.',
        '- Metrics use patch-centre predictions against resampled GEBCO, not independent surveyed depths.',
        '- This run supervises patch centres. Dense reconstruction and pooled-depth losses are not active in the current trainer.',
        '- Reported uncertainty is aleatoric; MC-dropout ensembles and conformal recalibration are not applied.',
        '- Glint and feature-reference statistics currently use the full scene; this is a limitation for strict held-out generalization claims.',
        '- The 65 test patches are a small spatial sample. Results do not establish performance at new coastal sites.',
        '- Tabular benchmarks were excluded at the user\'s request.', '',
        '## Repairs verified before training', '',
        '- EMA now carries BatchNorm statistics and saves the weights actually evaluated.',
        '- Variance loss uses float32 arithmetic and consistent metre-squared units.',
        '- Reconstruction-head input width matches its feature tensor.',
        '- The binary mask is preserved during normalization; constant channels cannot amplify invalid pixels.',
        '- Stale channel layouts, empty splits, non-finite inputs, losses, gradients and validation RMSE fail explicitly.',
        '- Five focused regression tests passed. Depth bins with fewer than 10 observations are explicitly unavailable.', '',
        '![Training curves](results/training_curves.png)', '',
        'See [configuration](config.yaml), [metrics](results/hybridbathnet_results.json), '
        '[epoch history](checkpoints/training_history.json), and [manifest checks](manifest_verification.json).',
    ]
    (run / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
