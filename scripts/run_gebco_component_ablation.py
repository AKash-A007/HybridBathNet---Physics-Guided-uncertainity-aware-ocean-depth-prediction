"""Matched GEBCO nearshore HybridBathNet component ablations."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.run_nearshore_hybrid import ShallowAugDataset, interval, write
from src.cnn_baseline import PatchDataset, train_hybridbathnet, evaluate_hybridbathnet
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import JointCompositeLoss, CoverageSampler, initialize_optical_head

ROOT = Path('runs/2026-11-06/gebco_component_ablation_v1')
DATA = Path('runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz')
ARMS = ('no_physics', 'no_uncertainty', 'no_both')


def update_status(arm, stage, extra=None):
    folder = ROOT / arm
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'status.json', {'arm': arm, 'stage': stage,
          'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **(extra or {})})
    marker = '<!-- GEBCO_ABLATION_STATUS -->'
    end = '<!-- /GEBCO_ABLATION_STATUS -->'
    states = []
    for name in ARMS:
        path = ROOT / name / 'status.json'
        state = json.loads(path.read_text(encoding='utf-8'))['stage'] if path.exists() else 'pending'
        states.append(f'{name}: {state}')
    note = (marker + '\nGEBCO nearshore component ablation: **' + ', '.join(states) + '**. '
            '[Run directory](runs/2026-11-06/gebco_component_ablation_v1/) and '
            '[report](runs/2026-11-06/gebco_component_ablation_v1/REPORT.md).\n' + end)
    for path in (Path('PROJECT_CONTEXT.md'), Path('README.md')):
        content = path.read_text(encoding='utf-8')
        if marker in content:
            start = content.index(marker)
            stop = content.index(end, start) + len(end)
            content = content[:start] + note + content[stop:]
        else:
            first = content.index('\n') + 1
            content = content[:first] + '\n' + note + '\n' + content[first:]
        path.write_text(content, encoding='utf-8')


def summarize(path, coords):
    with np.load(path) as d:
        target = d['target'].reshape(-1).astype('float64')
        pred = d['prediction'].reshape(-1).astype('float64')
    _, groups = np.unique((coords + 32) // 200, axis=0, return_inverse=True)

    def metric(mask):
        n = int(mask.sum())
        if not n:
            return {'n': 0}
        truth, estimate = target[mask], pred[mask]
        error = estimate - truth
        return {'n': n, 'rmse': float(np.sqrt(np.mean(error ** 2))),
                'mae': float(np.mean(np.abs(error))), 'bias': float(np.mean(error)),
                **interval(truth, estimate, groups[mask])}

    return {'primary_0_20': metric((target >= 0) & (target <= 20)),
            'bins': {f'{lo}-{hi}': metric((target >= lo) & (target < hi) & (target <= 20))
                     for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20))},
            'cutoffs': {str(limit): metric((target >= 0) & (target <= limit))
                        for limit in (15, 20, 25)}}


def run(arm):
    folder = ROOT / arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'results.json').exists():
        print(f'{arm}: already complete; skipping duplicate training', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU training')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with DATA.open('rb') as f:
        datahash = hashlib.file_digest(f, 'sha256').hexdigest()
    with np.load(DATA) as data:
        X, Y, M, coords = (data[k] for k in ('X', 'Y', 'M', 'coords'))
        split = {k: data[k] for k in ('train_idx', 'val_idx', 'test_idx')}
        support = data['native_support']
    for left, right in (('train_idx', 'val_idx'), ('train_idx', 'test_idx'), ('val_idx', 'test_idx')):
        assert not set(support[split[left]].ravel()) & set(support[split[right]].ravel())
    assert len(np.unique(coords, axis=0)) == len(coords)
    assert all(np.max(Y[split[k], 32, 32]) <= 20.0001 for k in ('train_idx', 'val_idx'))
    physics = arm == 'no_uncertainty'
    uncertainty = arm == 'no_physics'
    base = PatchDataset(X, Y, M, split['train_idx'])

    def loader(name, training=False):
        dataset = base if training else PatchDataset(X, Y, M, split[name],
            mean=base.mean, std=base.std, y_mean=base.y_mean, y_std=base.y_std)
        return DataLoader(ShallowAugDataset(dataset, augment=training,
                          noise_std=.02, use_rotations=False), batch_size=8,
                          sampler=CoverageSampler(len(dataset), 42) if training else None)

    train, val, test = loader('train_idx', True), loader('val_idx'), loader('test_idx')
    rinf = np.array([np.percentile(base.X[:, b][base.M > .5], 2) for b in range(3)], dtype='float32')
    model = HybridBathNetModel(in_channels=16, mean=base.mean.squeeze(), std=base.std.squeeze(),
        r_inf=rinf, mask_aware_physics=True, kd_aggregation='center',
        physics_conditioning='spatial', physics_enabled=physics,
        uncertainty_enabled=uncertainty,
        use_recon_head=physics).cuda()
    if physics:
        initialize_optical_head(model)
    criterion = JointCompositeLoss(w_mse=1, w_nll=int(uncertainty),
        w_ext=.03 if physics else 0, w_phys=0, w_recon=.1 if physics else 0,
        use_depth_weighting=False, beta_nll=.5, huber_delta=.5)
    protocol = {'reference': 'GEBCO centre depths; 0-20m primary', 'arm': arm,
        'physics_enabled': physics, 'uncertainty_enabled': uncertainty,
        'physics_off_definition': 'Zero normalized engineered channels 4:11; no FiLM conditioning, extinction loss or optical reconstruction loss/head',
        'uncertainty_off_definition': 'NLL loss and curriculum base weight zero; log-variance output has no supervised gradient and must not be interpreted',
        'unchanged': 'Same 16-channel tensor shape/backbone, dataset, split, augmentation, seed42, 180-epoch cosine schedule, optimizer and validation checkpoint rule as GEBCO v1',
        'dataset_sha256': datahash, 'split_sizes': {k: len(v) for k, v in split.items()},
        'weights': {'depth_huber': 1, 'beta_nll': int(uncertainty),
                    'extinction': .03 if physics else 0, 'reconstruction': .1 if physics else 0},
        'epochs': 180, 'patience': 30, 'lr': .001, 'batch': 8, 'ema': .999,
        'cosine_period': 180, 'nll_warmup': 20, 'selection': 'EMA validation RMSE',
        'test_policy': 'Evaluate all original test centres once after training; report 0-20m and depth bins',
        'caveat': 'GEBCO is a coarse reference, not independent seabed survey truth; test already inspected in earlier studies.'}
    write(folder / 'protocol.json', protocol)
    update_status(arm, 'training', {'split_sizes': protocol['split_sizes']})
    model, _ = train_hybridbathnet(model, criterion, train, val, epochs=180, lr=.001,
        device='cuda', y_mean=base.y_mean, y_std=base.y_std, nll_warmup_epochs=20,
        base_nll_weight=float(uncertainty), lr_restart_period=180, ema_decay=.999,
        early_stop_patience=30, checkpoint_dir=str(folder / 'hybridbathnet/checkpoints'))
    torch.save({'model': model.state_dict(), 'protocol': protocol, 'ch_mean': base.mean,
        'ch_std': base.std, 'y_mean': base.y_mean, 'y_std': base.y_std}, folder / 'hybridbathnet.pt')
    update_status(arm, 'evaluating')
    vm = evaluate_hybridbathnet(model, val, 'cuda', base.y_mean, base.y_std,
        compute_calibration=uncertainty, predictions_path=folder / 'validation_predictions.npz')
    tm = evaluate_hybridbathnet(model, test, 'cuda', base.y_mean, base.y_std,
        compute_calibration=uncertainty, predictions_path=folder / 'test_predictions.npz')
    vd = summarize(folder / 'validation_predictions.npz', coords[split['val_idx']])
    td = summarize(folder / 'test_predictions.npz', coords[split['test_idx']])
    write(folder / 'results.json', {'validation': vd, 'test': td,
          'validation_metrics': vm, 'test_metrics': tm,
          'uncertainty_metrics_valid': uncertainty, 'physics_diagnostics_valid': physics})
    update_status(arm, 'complete', {'validation_rmse_0_20': vd['primary_0_20']['rmse'],
                                   'test_rmse_0_20': td['primary_0_20']['rmse']})


def report():
    baseline = json.loads(Path('runs/2026-11-05/nearshore_hybrid_v1/results.json').read_text(encoding='utf-8'))
    rows = [('full model', baseline)]
    for arm in ARMS:
        path = ROOT / arm / 'results.json'
        if path.exists():
            rows.append((arm.replace('_', ' '), json.loads(path.read_text(encoding='utf-8'))))
    labels = ('0-5', '5-10', '10-15', '15-20')
    source = Path('runs/2026-11-05/nearshore_hybrid_v1')
    with np.load(DATA) as data:
        coords = data['coords'][data['test_idx']]
    with np.load(source / 'test_predictions.npz') as data:
        reference = data['target'].reshape(-1).astype('float64')
        baseline_pred = data['prediction'].reshape(-1).astype('float64')
    primary = (reference >= 0) & (reference <= 20)
    _, groups = np.unique((coords[primary] + 32) // 200, axis=0, return_inverse=True)
    unique = np.unique(groups)
    comparison = {}
    for arm in ARMS:
        path = ROOT / arm / 'test_predictions.npz'
        if not path.exists():
            continue
        with np.load(path) as data:
            truth = data['target'].reshape(-1).astype('float64')
            pred = data['prediction'].reshape(-1).astype('float64')
        if not np.allclose(truth, reference, rtol=0, atol=1e-5):
            raise ValueError(f'{arm}: test reference/order differs from full model')
        truth, pred, full = truth[primary], pred[primary], baseline_pred[primary]
        delta = float(np.sqrt(np.mean((pred-truth)**2)) - np.sqrt(np.mean((full-truth)**2)))
        rng = np.random.RandomState(42)
        draws = []
        for _ in range(2000):
            ids = np.concatenate([np.flatnonzero(groups == group)
                                  for group in rng.choice(unique, len(unique), replace=True)])
            draws.append(np.sqrt(np.mean((pred[ids]-truth[ids])**2))
                         - np.sqrt(np.mean((full[ids]-truth[ids])**2)))
        comparison[arm] = {'delta_test_rmse_m': delta,
                           'paired_spatial_bootstrap_ci95_m': np.percentile(draws, [2.5, 97.5]).tolist(),
                           'clusters': len(unique), 'resamples': 2000}
    write(ROOT / 'comparison.json', comparison)
    lines = ['# GEBCO nearshore HybridBathNet component ablation', '',
        'Same GEBCO nearshore dataset, geographic splits and test regions. The full model is the preserved selected v1 checkpoint.',
        'Physics-off zeros the seven engineered feature channels and removes physics conditioning and losses; uncertainty-off removes NLL supervision. All arms use the same 16-channel backbone tensor shape.',
        '', '| Model | Validation RMSE 0-20m | Test RMSE 0-20m | Test MAE 0-20m |',
        '|---|---:|---:|---:|']
    for name, result in rows:
        v, t = result['validation']['primary_0_20'], result['test']['primary_0_20']
        lines.append(f"| {name} | {v['rmse']:.3f} | {t['rmse']:.3f} | {t['mae']:.3f} |")
    lines += ['', '| Ablation | Test RMSE difference vs full (m) | Paired ~2km spatial-bootstrap 95% CI (m) |',
              '|---|---:|---:|']
    for arm, item in comparison.items():
        lo, hi = item['paired_spatial_bootstrap_ci95_m']
        lines.append(f"| {arm.replace('_', ' ')} | {item['delta_test_rmse_m']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '| Model | 0-5m | 5-10m | 10-15m | 15-20m |',
              '|---|---:|---:|---:|---:|']
    for name, result in rows:
        bins = result['test'].get('bins', result['test'].get('primary_bins'))
        if '15-20' not in bins and '15-+' in bins:
            bins = {**bins, '15-20': bins['15-+']}
        lines.append('| ' + name + ' | ' + ' | '.join(
            f"{bins[k]['rmse']:.3f} (n={bins[k]['n']})" for k in labels if k in bins) + ' |')
    lines += ['', 'RMSE is computed from depth prediction error against GEBCO, not from composite loss. All test centres remain included in the 0-20m primary score. Paired intervals resample approximate 2km spatial groups 2000 times and are conditional on these models and GEBCO. Single seed; prior test inspection makes this an exploratory component comparison. The optical cutoff is uncalibrated and is not used for masking. Uncertainty metrics from uncertainty-off arms are invalid and should not be interpreted.',
              '', 'Full-model source: [selected v1](../../2026-11-05/nearshore_hybrid_v1/REPORT.md). Per-arm results and validation-selected checkpoints are in the named subdirectories.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=(*ARMS, 'all'), default='all')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    for arm in ARMS if args.arm == 'all' else (args.arm,):
        try:
            run(arm)
            report()
        except Exception as exc:
            write(ROOT / arm / 'failure.json', {'error': repr(exc)})
            update_status(arm, 'failed')
            raise
