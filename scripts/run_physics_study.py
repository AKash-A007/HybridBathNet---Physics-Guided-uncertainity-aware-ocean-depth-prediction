"""Sequential, resumable ablations. Freeze selection before any test evaluation."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import yaml


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='runs/2026-11-04/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).parent / ('pilot' if args.pilot else 'study')
    root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        tests = read(root/'test_summary.json')
        runtimes = {name:read(root/name/'run_metadata.json')['runtime_seconds'] for name in tests}
        report(root, read(root/'selection.json'), read(root/'validation_summary.json'), tests, runtimes)
        print(f'Report written to {root / "REPORT.md"}', flush=True)
        return
    env = dict(os.environ, PYTHONUTF8='1', MPLBACKEND='Agg', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently launch an overnight CPU study')
    features = Path(args.config).parent / 'study/features'
    if not (features / 'patches.npz').exists():
        subprocess.run([sys.executable, 'scripts/build_study_data.py', '--config', args.config,
                        '--output', str(features)], check=True, env=env)
    with (features / 'patches.npz').open('rb') as stream:
        data_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    base = copy.deepcopy(cfg)
    base['features']['features_dir'] = str(features)
    base['training']['epochs'] = args.epochs or (2 if args.pilot else cfg['training']['epochs'])
    base['training']['batch_size'] = 8
    base['hybridbathnet']['early_stop_patience'] = base['training']['epochs']
    base['hybridbathnet']['loss_weights']['w_phys'] = 0.0  # softplus already guarantees nonnegativity
    base['hybridbathnet']['use_explicit_reflectance'] = True
    write(root / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
          'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
          'features_sha256': data_hash, 'epoch_budget': base['training']['epochs'],
          'protocol': 'fixed budget; choose by validation RMSE, then freeze and evaluate test'})
    configs, results, runtimes = {}, {}, {}

    def status(stage, experiment=None):
        write(root / 'status.json', {'stage': stage, 'experiment': experiment,
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'completed_validation': list(results)})
        print(stage, experiment or '', flush=True)

    def train(name, parent=None, changes=None, loss=None, seed=42):
        current = copy.deepcopy(parent or base)
        current['hybridbathnet'].update(changes or {})
        current['hybridbathnet']['loss_weights'].update(loss or {})
        folder = root / name
        folder.mkdir(exist_ok=True)
        current['training']['checkpoint_dir'] = str(folder / 'checkpoints')
        current['evaluation']['results_dir'] = str(folder / 'results')
        config_file = folder / 'config.yaml'
        if config_file.exists() and yaml.safe_load(config_file.read_text()) != current:
            raise RuntimeError(f'Refusing to resume changed experiment configuration: {name}')
        config_file.write_text(yaml.safe_dump(current, sort_keys=False), encoding='utf-8')
        configs[name] = current
        metrics_file = folder / 'results/validation_results.json'
        if not metrics_file.exists():
            status('training', name)
            start = time.perf_counter()
            with (folder / 'training.log').open('a', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-u', '-m', 'src.cnn_baseline', '--config',
                                str(config_file), '--seed', str(seed), '--validation-only'],
                               check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            runtimes[name] = time.perf_counter()-start
            write(folder / 'run_metadata.json', {'seed': seed, 'runtime_seconds': runtimes[name],
                  'features_sha256': data_hash})
        else:
            runtimes[name] = read(folder / 'run_metadata.json')['runtime_seconds']
        results[name] = read(metrics_file)
        if not np.isfinite(results[name]['rmse']):
            raise RuntimeError(f'Invalid validation RMSE: {name}')
        write(root / 'validation_summary.json', results)
        return name

    def best(names):
        return min(names, key=lambda name: results[name]['rmse'])

    try:
        reference = train('compatibility_reference', changes={'use_explicit_reflectance': False})
        if args.pilot:
            status('pilot_complete', reference)
            return
        candidates = [train('explicit_raw_reference')]
        candidates += [train('patch_mean', changes={'kd_aggregation':'mean', 'physics_conditioning':'mean'}),
                       train('center', changes={'kd_aggregation':'center', 'physics_conditioning':'center'}),
                       train('spatial_center', changes={'kd_aggregation':'center', 'physics_conditioning':'spatial'})]
        selected_kd = best(candidates)
        bg = train('blue_green_only', parent=configs[selected_kd], changes={'use_red_kd':False})
        candidates.append(bg)
        selected_optics = best(candidates)
        optics = configs[selected_optics]
        for name, changes, loss in [
            ('signal_ratio_10', {'i0_over_epsilon':10.0}, {}),
            ('signal_ratio_1000', {'i0_over_epsilon':1000.0}, {}),
            ('unclipped_bound', {'z_ext_min':None,'z_ext_max':None}, {}),
            ('weak_extinction', {}, {'w_ext':.03}),
            ('no_extinction', {}, {'w_ext':0.0}),
            ('linear_depth', {'nonnegative_depth':False}, {'w_phys':.3}),
        ]:
            candidates.append(train(name, parent=optics, changes=changes, loss=loss))
        cnn = train('cnn_only', parent=optics, changes={'physics_enabled':False}, loss={'w_ext':0.0})
        selected = best(candidates)
        selected_cfg = configs[selected]
        selection = {'selected_hybrid':selected, 'validation_rmse':results[selected]['rmse'],
                     'selection_basis':'validation RMSE only; test not evaluated yet',
                     'kd_stage':selected_kd, 'optics_stage':selected_optics,
                     'cnn_control':cnn, 'features_sha256':data_hash}
        write(root / 'selection.json', selection)
        for seed in (0,1):
            train(f'selected_seed_{seed}', parent=selected_cfg, seed=seed)
        # This point freezes all model and configuration choices. No subsequent tuning.
        test_results = {}
        for name in configs:
            folder = root/name
            output = folder/'results/hybridbathnet_results.json'
            if not output.exists():
                status('final_test_evaluation', name)
                seed = read(folder/'run_metadata.json')['seed']
                with (folder/'evaluation.log').open('a',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-u','-m','src.cnn_baseline','--config',
                                    str(folder/'config.yaml'),'--seed',str(seed),'--evaluate-only'],
                                   check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            test_results[name] = read(output)
        write(root/'test_summary.json',test_results)
        report(root, selection, results, test_results, runtimes)
        status('complete', selected)
    except Exception as error:
        status('failed')
        write(root/'failure.json',{'error':repr(error)})
        raise


def report(root, selection, validation, test, runtimes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    rows=[]
    fig, ax=plt.subplots(figsize=(11,6))
    for name, metrics in test.items():
        history=read(root/name/'checkpoints/training_history.json')
        row={'experiment':name,'val_rmse':validation[name]['rmse'],
             **{key:metrics[key] for key in ('rmse','mae','r2','acc_within_1m_%','acc_within_2m_%',
                'coverage_68pct','coverage_90pct','mean_z_ext_m','reference_above_bound_pct','bound_clipped_pct')},
             'best_epoch':min(history,key=lambda item:item['val_rmse'])['epoch'],
             'runtime_seconds':runtimes[name]}
        rows.append(row)
        ax.plot([r['epoch'] for r in history],[r['val_rmse'] for r in history],label=name)
    ax.set(xlabel='Epoch',ylabel='Validation RMSE (m)',title='Controlled HybridBathNet experiments')
    ax.legend(fontsize=7,ncol=2); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(root/'validation_curves.png',dpi=160); plt.close(fig)
    table=pd.DataFrame(rows)
    table.to_csv(root/'comparison.csv',index=False)
    selected=selection['selected_hybrid']
    seed_names=[selected,'selected_seed_0','selected_seed_1']
    variability={metric:{'mean':float(np.mean([test[n][metric] for n in seed_names])),
                         'sample_std':float(np.std([test[n][metric] for n in seed_names],ddof=1))}
                 for metric in ('rmse','mae','r2')}
    write(root/'seed_variability.json',variability)
    lines=['# HybridBathNet controlled study','',f'Selected using validation only: **{selected}**.','',
           '| Experiment | Validation RMSE | Test RMSE | Test MAE | Test R² |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['val_rmse']:.3f} | {row['rmse']:.3f} | {row['mae']:.3f} | {row['r2']:.3f} |")
    lines += ['',f"Selected configuration over seeds 42, 0, 1: RMSE {variability['rmse']['mean']:.3f} ± {variability['rmse']['sample_std']:.3f} m (sample SD).",'',
        f'[Final configuration]({selected}/config.yaml) · [Checkpoint]({selected}/checkpoints/hybridbathnet.pt) · [All metrics](comparison.csv)', '',
        '## Original comparison', '',
        'The preserved historical result reports RMSE 3.5066 m, MAE 2.6732 m and R² 0.9163. '
        'Its checkpoint lacks seed/split/runtime provenance and used old six-channel features. '
        'It is not an apples-to-apples comparator. Unmodified current HEAD fails its first forward pass '
        'with a 64-versus-32-channel reconstruction-head error. The compatibility reference is therefore '
        'a repaired reference on the controlled dataset, not a reproduction of the historical score.', '',
        '## Scientific interpretation and limits', '',
        'The existing Kd expressions are tested heuristics; their exact blue-green coefficients and additive red term '
        'were not verified as a calibrated Sentinel-2 attenuation retrieval. Reflectance inversion was already valid '
        'for stored spectral statistics, but spectral jitter altered those inputs; the explicit-raw variants isolate '
        'physics from jitter. Signal ratios 10/100/1000 are sensitivity assumptions, not measured detection thresholds. '
        'The bound is a soft penalty, never a hard ceiling on predictions.', '',
        'All variants use the same training-fitted features and 306/65/65 spatial patch split. Pixel footprints do not '
        'overlap, but adjacent patches can share coarse GEBCO reference cells. GEBCO is not independent surveyed '
        'bathymetry. Validation selection across many variants can overfit this small validation set. '
        'A new geographic site and independent optical/depth measurements are still needed.', '',
        'The trainer supervises centre-pixel depths; dense reconstruction/pooling losses are not active. '
        'Uncertainty results are aleatoric Gaussian coverage; MC dropout, ensembles and conformal calibration '
        'were not evaluated. Empty/small depth bins are null with counts, not silently converted to zero.', '',
        'Sources: [Wang et al. (2009)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009JC005286), '
        '[NASA SeaDAS KD2 example](https://appliedsciences.nasa.gov/sites/default/files/2024-02/SeaDAS_Final_1Feb.pdf).', '',
        '![Validation curves](validation_curves.png)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
