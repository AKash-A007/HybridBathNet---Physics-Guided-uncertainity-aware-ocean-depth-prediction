"""Compare the single integrated candidate with frozen GEBCO v1."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from src.cnn_baseline import PatchDataset
from src.hybridbathnet import HybridBathNetModel, calibration_metrics

ROOT = Path('runs/2026-11-06/integrated_head_research_v1')
BASE = Path('runs/2026-11-05/nearshore_hybrid_v1')
DATA = BASE / 'features/patches.npz'
OPTIONS = {'detach_uncertainty_features': True, 'reconstruction_attenuation': 'patch'}


def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def uncertainty(root, split):
    with np.load(root / f'{split}_predictions.npz') as d:
        t, p, sigma = [d[key].reshape(-1).astype('float64') for key in ('target','prediction','sigma')]
    keep = (t >= 0) & (t <= 20)
    t, p, sigma = t[keep], p[keep], sigma[keep]
    error = p-t
    result = calibration_metrics(t, p, sigma)
    result.update(n=len(t), mean_sigma_m=float(sigma.mean()),
        interval90_mean_width_m=float((2*1.645*sigma).mean()),
        sigma_error_spearman=float(spearmanr(sigma, np.abs(error)).statistic),
        lower_sigma_half_rmse_m=float(np.sqrt(np.mean(error[np.argsort(sigma)[:len(t)//2]]**2))))
    result['bins'] = {}
    for lo, hi in ((0,5),(5,10),(10,15),(15,20.0001)):
        mask = (t >= lo) & (t < hi)
        result['bins'][f'{lo}-{hi}'] = {'n': int(mask.sum()),
            'coverage90_pct': float(100*np.mean(np.abs(error[mask]) <= 1.645*sigma[mask])),
            'mean_sigma_m': float(sigma[mask].mean())}
    return result


@torch.no_grad()
def physics_check():
    saved = torch.load(ROOT / 'hybridbathnet.pt', map_location='cpu', weights_only=False)
    # This candidate started before structured options were added to its runner.
    # Supplement metadata only, without altering trained tensors.
    protocol = saved['protocol']
    protocol['model_options'] = OPTIONS
    write(ROOT / 'protocol.json', protocol)
    saved['protocol'] = protocol
    torch.save(saved, ROOT / 'hybridbathnet.pt')
    model = HybridBathNetModel(in_channels=16, mean=saved['ch_mean'].squeeze(),
        std=saved['ch_std'].squeeze(), r_inf=saved['model']['r_inf_buf'].numpy().reshape(-1),
        mask_aware_physics=True, kd_aggregation='center', physics_conditioning='spatial', **OPTIONS)
    model.load_state_dict(saved['model'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for candidate physics verification')
    model.cuda().eval()
    with np.load(DATA) as d:
        dataset = PatchDataset(d['X'], d['Y'], d['M'], d['test_idx'],
            mean=saved['ch_mean'], std=saved['ch_std'], y_mean=saved['y_mean'], y_std=saved['y_std'])
    squared = 0.
    count = 0
    predictions = []
    for x, y, mask, raw in DataLoader(dataset, batch_size=8):
        mask = mask.cuda().unsqueeze(1)
        out = model(x.cuda(), mask, raw.cuda())
        valid = mask.bool().expand_as(raw.cuda())
        squared += float(((out['r_hat']-raw.cuda())[valid]**2).sum())
        count += int(valid.sum())
        torch.testing.assert_close(out['kl_map'], out['kl_map'][:,:,:1,:1].expand_as(out['kl_map']))
        predictions.extend(out['depth'][:,0,32,32].cpu().tolist())
    with np.load(ROOT / 'test_predictions.npz') as d:
        np.testing.assert_allclose(predictions, d['prediction'].reshape(-1), rtol=1e-4, atol=1e-4)
    return {'test_water_reflectance_rmse': float(np.sqrt(squared/count)),
            'reload_predictions_match': True, 'attenuation_patch_shared': True,
            'physical_optical_calibration_established': False}


def main():
    if not (ROOT / 'results.json').exists():
        raise RuntimeError('Wait for the candidate training and evaluation to complete')
    full = json.loads((BASE / 'results.json').read_text())
    candidate = json.loads((ROOT / 'results.json').read_text())
    with np.load(DATA) as d:
        coords = d['coords'][d['test_idx']] + 32
    with np.load(BASE / 'test_predictions.npz') as d:
        t, p0 = [d[k].reshape(-1).astype('float64') for k in ('target','prediction')]
    with np.load(ROOT / 'test_predictions.npz') as d:
        np.testing.assert_allclose(d['target'].reshape(-1), t, atol=1e-5)
        p1 = d['prediction'].reshape(-1).astype('float64')
    mask = (t >= 0) & (t <= 20)
    t, p0, p1 = t[mask], p0[mask], p1[mask]
    _, groups = np.unique(coords[mask]//200, axis=0, return_inverse=True)
    unique = np.unique(groups)
    rng = np.random.RandomState(42)
    delta = []
    for _ in range(2000):
        idx = np.concatenate([np.flatnonzero(groups == key) for key in rng.choice(unique,len(unique),replace=True)])
        delta.append(np.sqrt(np.mean((p1[idx]-t[idx])**2)) - np.sqrt(np.mean((p0[idx]-t[idx])**2)))
    uq = {name: {split: uncertainty(path, split) for split in ('validation','test')}
          for name,path in (('full',BASE),('candidate',ROOT))}
    result = {'full': full, 'candidate': candidate, 'uncertainty_0_20': uq,
              'paired_test_rmse_difference_ci95_m': np.percentile(delta,[2.5,97.5]).tolist(),
              'physics': physics_check(),
              'validated_joint_improvement_established': False,
              'selected_by_validation': 'candidate' if candidate['validation']['primary_0_20']['rmse'] < full['validation']['primary_0_20']['rmse'] else 'full'}
    write(ROOT / 'head_comparison.json', result)
    lines = ['# Integrated physics and uncertainty experiment', '',
        'One seed42 candidate on the same GEBCO nearshore data and geographic splits. All modules remain active. Variance NLL is detached from the mean/features; effective per-band attenuation is shared within patch water.', '',
        '| Model | Validation RMSE 0–20m | Test RMSE 0–20m | Test 90% coverage | Mean 90% interval width | Test Gaussian NLL |',
        '|---|---:|---:|---:|---:|---:|']
    for name, values in (('full',full),('candidate',candidate)):
        v, tmetric = [values[k]['primary_0_20'] for k in ('validation','test')]
        u = uq[name]['test']
        lines.append(f"| {name} | {v['rmse']:.3f}m | {tmetric['rmse']:.3f}m | {u['coverage_90pct']:.2f}% | {u['interval90_mean_width_m']:.3f}m | {u['gaussian_nll']:.3f} |")
    lines += ['', '| Test depth bin | n | Full RMSE | Candidate RMSE | Full 90% coverage | Candidate 90% coverage |',
              '|---|---:|---:|---:|---:|---:|']
    for key, uk in (('0-5','0-5'),('5-10','5-10'),('10-15','10-15'),('15-+','15-20.0001')):
        a, b = full['test']['primary_bins'][key], candidate['test']['primary_bins'][key]
        lines.append(f"| {key.replace('15-+','15-20')}m | {a['n']} | {a['rmse']:.3f}m | {b['rmse']:.3f}m | {uq['full']['test']['bins'][uk]['coverage90_pct']:.1f}% | {uq['candidate']['test']['bins'][uk]['coverage90_pct']:.1f}% |")
    lo,hi = result['paired_test_rmse_difference_ci95_m']
    lines += ['', f"Lowest numerical validation RMSE: **{result['selected_by_validation']}**. This is not evidence of a reliable joint improvement. Paired ~2km spatial-bootstrap95% interval for candidate minus full test RMSE: [{lo:+.3f}, {hi:+.3f}]m (2000 draws).",
        '', 'The candidate is not promoted: its small depth differences are inconclusive, validation Gaussian NLL worsened, and held-out uncertainty coverage/ranking also worsened. Preserve the existing full model as the reference while developing the next optical/uncertainty design.',
        '', f"Candidate test reflectance reconstruction RMSE: {result['physics']['test_water_reflectance_rmse']:.6f}. Reloaded predictions match the recorded predictions when its model options are used. This does not validate physical Kd, bottom visibility or an operational optical mask.",
        '', 'Single-seed exploratory test comparison; the test has been inspected before. Coverage alone can improve by widening intervals: assess coverage, width, NLL and bin behaviour together. Calibration does not change depth RMSE. The patch-homogeneous water assumption and shallow-percentile background remain unvalidated. No test sample was masked or clipped.',
        '', '[Literature and recovery design](../../../docs/HEADS_OPTICAL_LIMIT_RESEARCH_2026-11-06.md) | [Full metrics](head_comparison.json) | [Protocol](protocol.json)']
    (ROOT / 'HEAD_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'selected':result['selected_by_validation'], 'candidate_validation_rmse':candidate['validation']['primary_0_20']['rmse'],
                      'candidate_test_rmse':candidate['test']['primary_0_20']['rmse'],
                      'coverage90_pct':uq['candidate']['test']['coverage_90pct'],
                      'gaussian_nll':uq['candidate']['test']['gaussian_nll']},indent=2))


if __name__ == '__main__':
    main()
