"""Longer training and expanded training-area patches; held-out locations fixed."""
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path('runs/2026-11-05/improvement')
OLD = Path('runs/2026-11-04/study')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def audit_data():
    with np.load(OLD/'features/patches.npz') as old, np.load(ROOT/'features/patches.npz') as new:
        for split in ('val_idx', 'test_idx'):
            np.testing.assert_array_equal(old['coords'][old[split]], new['coords'][new[split]])
            np.testing.assert_array_equal(old['Y'][old[split]], new['Y'][new[split]])
        coords, train = new['coords'], new['train_idx']
        for held in np.concatenate([new['val_idx'], new['test_idx']]):
            dr = coords[train] - coords[held]
            assert not np.any((np.abs(dr[:,0]) < 96) & (np.abs(dr[:,1]) < 96)), 'Holdout buffer violated'
        audit = {'train':len(train), 'validation':len(new['val_idx']), 'test':len(new['test_idx']),
                 'heldout_coordinates_and_labels_unchanged':True, 'holdout_buffer_pixels':32,
                 'note':'Additional patches from existing training blocks, not new independent scenes. Training patches may overlap each other.'}
        for name in ('train_idx','val_idx','test_idx'):
            depths = new['Y'][new[name],32,32]
            audit[name] = {'min':float(depths.min()), 'max':float(depths.max()),
                          'mean':float(depths.mean()), 'count_over_20m':int((depths>20).sum())}
    with (ROOT/'features/patches.npz').open('rb') as stream:
        audit['features_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
    # Diagnose physics mismatch on validation only; do not tune to test errors.
    with np.load(OLD/'weak_extinction/results/validation_predictions.npz') as pred:
        bins=[]
        for lo, hi in ((0,10),(10,20),(20,10000)):
            mask=(pred['target']>=lo)&(pred['target']<hi)
            n=int(mask.sum())
            bins.append({'lower_m':lo,'upper_m':hi,'n':n,
                         'rmse_m':float(np.sqrt(np.mean((pred['prediction'][mask]-pred['target'][mask])**2))) if n else None,
                         'bias_m':float(np.mean(pred['prediction'][mask]-pred['target'][mask])) if n else None,
                         'mean_bound_m':float(pred['z_ext'][mask].mean()) if n else None,
                         'sparse':n<10})
        audit['baseline_validation_depth_errors']=bins
    write(ROOT/'data_audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    audit_data()
    if '--audit-only' in sys.argv:
        return
    base=yaml.safe_load((OLD/'weak_extinction/config.yaml').read_text())
    env=dict(os.environ,PYTHONUTF8='1',MPLBACKEND='Agg',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    results, configs = {}, {}

    def status(stage, name=None):
        write(ROOT/'status.json',{'stage':stage,'experiment':name,
              'completed_validation':list(results),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

    def train(name, expanded=False, smooth=False, seed=42, parent=None):
        cfg=copy.deepcopy(parent or base)
        folder=ROOT/name
        folder.mkdir(exist_ok=True)
        cfg['training']['checkpoint_dir']=str(folder/'checkpoints')
        cfg['evaluation']['results_dir']=str(folder/'results')
        if parent is None:
            cfg['training']['epochs']=180 if expanded else 300
            cfg['hybridbathnet']['early_stop_patience']=60
            if expanded:
                cfg['features']['features_dir']=str(ROOT/'features')
            if smooth:
                # A single cosine cycle: no restarts before the final epoch.
                cfg['hybridbathnet']['lr_restart_period']=180
        cfgfile=folder/'config.yaml'
        if cfgfile.exists() and yaml.safe_load(cfgfile.read_text()) != cfg:
            raise RuntimeError(f'Configuration changed for {name}')
        cfgfile.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        configs[name]=(cfg,seed)
        if name=='long_original' and not (folder/'checkpoints').exists():
            shutil.copytree(OLD/'weak_extinction/checkpoints',folder/'checkpoints')
        metrics=folder/'results/validation_results.json'
        if not metrics.exists():
            status('training',name)
            start=time.perf_counter()
            with (folder/'training.log').open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-u','-m','src.cnn_baseline','--config',str(cfgfile),
                                '--seed',str(seed),'--validation-only'],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(folder/'run_metadata.json',{'seed':seed,'runtime_seconds':time.perf_counter()-start})
        results[name]=read(metrics)
        if not np.isfinite(results[name]['rmse']):
            raise RuntimeError(f'Invalid validation score: {name}')
        write(ROOT/'validation_summary.json',results)

    try:
        train('long_original')
        train('expanded_restart',expanded=True)
        train('expanded_smooth',expanded=True,smooth=True)
        selected=min(results,key=lambda n:results[n]['rmse'])
        write(ROOT/'selection.json',{'selected':selected,'basis':'validation RMSE only',
                                    'selected_validation_rmse':results[selected]['rmse']})
        train('selected_seed_1',seed=1,parent=configs[selected][0])
        tests={}
        for name in configs:
            folder=ROOT/name
            path=folder/'results/hybridbathnet_results.json'
            if not path.exists():
                status('final_test_evaluation',name)
                with (folder/'evaluation.log').open('a',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-u','-m','src.cnn_baseline','--config',str(folder/'config.yaml'),
                        '--seed',str(configs[name][1]),'--evaluate-only'],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            tests[name]=read(path)
        write(ROOT/'test_summary.json',tests)
        rows=[]
        lines=['# Longer training and expanded-data results','',f'Selected by validation: **{selected}**.','',
               '| Run | Validation RMSE (m) | Test RMSE (m) | Test MAE (m) | Test R2 | Best epoch |',
               '|---|---:|---:|---:|---:|---:|']
        for name, metrics in tests.items():
            history=read(ROOT/name/'checkpoints/training_history.json')
            best=min(history,key=lambda e:e['val_rmse'])
            row={'experiment':name,'validation_rmse':results[name]['rmse'],
                 'test_rmse':metrics['rmse'],'test_mae':metrics['mae'],'test_r2':metrics['r2'],
                 'best_epoch':best['epoch'],'epochs_completed':history[-1]['epoch']}
            rows.append(row)
            lines.append(f"| {name} | {row['validation_rmse']:.3f} | {row['test_rmse']:.3f} | {row['test_mae']:.3f} | {row['test_r2']:.3f} | {row['best_epoch']} |")
        with (ROOT/'comparison.csv').open('w',newline='',encoding='utf-8') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        lines += ['', 'Previous selected run: validation RMSE 5.118 m; test RMSE 5.335 m.', '',
            'Expanded training: 2176 patches versus 306; validation/test retain the same 65 locations each. '
            'Features refitted using training pixels only. Held-out footprints have a 32-pixel exclusion buffer. '
            'These are additional overlapping training patches from the same composite, not independent new scenes.', '',
            'The original-data run continues from epoch 90 up to 300. Expanded runs start fresh with up to 180 epochs. '
            'All use patience 60, EMA 0.999, spatial conditioning and extinction weight 0.03. '
            'The smooth run uses a single 180-epoch cosine cycle instead of restarting every 30 epochs. '
            'The selected configuration is repeated from scratch with seed 1.', '',
            'The existing test set has already been inspected in prior work; results are exploratory comparisons, '
            'not a new untouched generalization estimate. Adjacent GEBCO cells and overlapping training patches '
            'limit independence. More patches do not establish more independent depth observations.', '',
            'The optical attenuation heuristic remains uncalibrated. Validation depth errors and bound mismatch '
            'are saved in [data audit](data_audit.json). No test-dependent depth cutoff or formula was introduced.', '',
            '[Full metrics](test_summary.json) | [Validation results](validation_summary.json) | [10-epoch progress](LIVE_PROGRESS.md)']
        (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status('complete',selected)
    except Exception as error:
        write(ROOT/'failure.json',{'error':repr(error)})
        status('failed')
        raise


if __name__=='__main__':
    main()
