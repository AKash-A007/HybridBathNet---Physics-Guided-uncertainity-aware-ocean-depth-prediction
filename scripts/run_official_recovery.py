"""Two predefined whole-model arms on identical official V002 label locations."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import hashlib
import json
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.cnn_baseline import PatchDataset,SpectralAugDataset,train_hybridbathnet,evaluate_hybridbathnet
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import JointCompositeLoss,CoverageSampler,initialize_optical_head
from scripts.run_nearshore_hybrid import diagnostics
from scripts.recovery_state import update
ROOT=Path('runs/2026-11-06/recovery_v2/atl24_paired')
ARMS=('composite_control','matched_tide')
OPTIONS={'uncertainty_head':'spatial','detach_uncertainty_features':True,'reconstruction_attenuation':'pixel'}


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not (ROOT/'dataset_complete.json').exists():raise RuntimeError('Dataset audit incomplete')
    if not (ROOT/'source_snapshot.zip').exists():
        with zipfile.ZipFile(ROOT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in [*Path('src').glob('*.py'),Path(__file__),Path('scripts/build_official_recovery_data.py')]:z.write(path,str(path))
    for arm in ARMS:
        root=ROOT/arm
        if (root/'complete.json').exists():print('Already complete',arm,flush=True);continue
        torch.manual_seed(42);np.random.seed(42)
        data_path=root/'features/patches.npz'
        with np.load(data_path) as d:
            x,y,m,coords=(d[k] for k in ('X','Y','M','coords'))
            split={k:d[f'{k}_idx'] for k in ('train','val','test')}
        base=PatchDataset(x,y,m,split['train'])
        def loader(key,training=False):
            ds=base if training else PatchDataset(x,y,m,split[key],mean=base.mean,std=base.std,y_mean=base.y_mean,y_std=base.y_std)
            # Constant per-patch scalar labels preserve the centre under flips.
            return DataLoader(SpectralAugDataset(ds,augment=training,noise_std=.02,use_rotations=False),
                batch_size=8,sampler=CoverageSampler(len(ds),42) if training else None)
        rinf=np.array([np.percentile(base.X[:,b][base.M>.5],2) for b in range(3)],dtype='float32')
        model=HybridBathNetModel(in_channels=16,mean=base.mean.squeeze(),std=base.std.squeeze(),r_inf=rinf,
            mask_aware_physics=True,kd_aggregation='center',physics_conditioning='spatial',**OPTIONS).cuda()
        initialize_optical_head(model)
        criterion=JointCompositeLoss(w_mse=1,w_nll=1,w_ext=.03,w_phys=0,w_recon=.1,
            beta_nll=0,detach_nll_mean=True,use_depth_weighting=False,huber_delta=.5)
        protocol={'arm':arm,'reference':'official ATL24 V002','seed':42,'epochs':180,'patience':30,
            'lr':.001,'batch':8,'cosine_period':180,'ema':.999,'nll_warmup':20,
            'modules':'full encoder/depth/physics/uncertainty, all modules active',
            'model_options':OPTIONS,'dataset_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
            'split_sizes':{k:len(v) for k,v in split.items()},'selection':'validation RMSE only',
            'loss':'Huber1 + detached-mean Gaussian NLL1 + extinction.03 + reconstruction.1',
            'prediction_datum':'scene-relative for matched data; subtract stored tide_delta for common original-ATL24 scoring',
            'test_coverage':'all product-eligible original-geography points; unmatched/cloudy samples use explicit2024 fallback',
            'optical_limit':'heuristic diagnostic; no masking/clipping',
            'reference_uncertainty':'sigma_tvu/sigma_thu recorded, not yet propagated into likelihood',
            'device':torch.cuda.get_device_name(0)}
        (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        update('atl24','training_paired_arm',arm=arm,split_sizes=protocol['split_sizes'])
        model,_=train_hybridbathnet(model,criterion,loader('train',True),loader('val'),epochs=180,lr=.001,
            device='cuda',y_mean=base.y_mean,y_std=base.y_std,nll_warmup_epochs=20,lr_restart_period=180,
            ema_decay=.999,early_stop_patience=30,separate_uncertainty_clip=True,
            checkpoint_dir=str(root/'checkpoints'))
        torch.save({'model':model.state_dict(),'protocol':protocol,'ch_mean':base.mean,'ch_std':base.std,
                    'y_mean':base.y_mean,'y_std':base.y_std},root/'hybridbathnet.pt')
        # Preserve model/datum metrics, then separately transform predictions to
        # common source-relative targets for honest identical-reference bins.
        metrics={}
        for key,stem in (('val','validation'),('test','test')):
            metrics[stem]=evaluate_hybridbathnet(model,loader(key),'cuda',base.y_mean,base.y_std,
                compute_calibration=True,predictions_path=root/f'{stem}_scene_predictions.npz')
        (root/'complete.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print('Completed',arm,'validation',metrics['validation']['rmse'],flush=True)
    # Both best checkpoints are fixed before common-reference comparison.
    results={}
    for arm in ARMS:
        root=ROOT/arm
        with np.load(root/'features/patches.npz') as d:
            coords=d['coords'];acquisition=d['acquisition_target'];delta=d['tide_delta'];matched=d['matched']
            split={s:d[f'{s}_idx'] for s in ('val','test')}
        results[arm]={}
        for key,stem in (('val','validation'),('test','test')):
            idx=split[key]
            with np.load(root/f'{stem}_scene_predictions.npz') as d:pred={k:d[k].copy() for k in d.files}
            np.testing.assert_allclose(pred['target'],acquisition[idx]+delta[idx],atol=2e-6)
            pred['target']=acquisition[idx];pred['prediction']=pred['prediction']-delta[idx]
            # Extinction bound lives in scene depth; keep it diagnostic there.
            pred['z_ext']=pred['z_ext']-delta[idx]
            path=root/f'{stem}_acquisition_predictions.npz';np.savez(path,**pred)
            row=diagnostics(path,coords[idx])
            error=pred['prediction']-pred['target'];sigma=pred['sigma']
            row['raw_coverage90_pct']=float(100*np.mean(abs(error)<=1.645*sigma))
            row['mean_sigma_m']=float(sigma.mean())
            row['clear_historical_n']=int(matched[idx].sum())
            for label,mask in (('historical_clear',matched[idx]),('fallback',~matched[idx])):
                row[label]={'n':int(mask.sum()),'rmse':float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None}
            results[arm][stem]=row
    (ROOT/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    lines=['# Official ATL24 V002 paired-input experiment','',
        'Same official reference/coordinates/geography and full-model settings in both arms. Validation selects each checkpoint.',
        'Matched clear scenes receive tide-adjusted scene-relative labels; predictions are transformed back to the identical ATL24 acquisition surface for all scores below.',
        'Cloudy/unavailable scenes retain the2024 input as an explicitly flagged fallback; all eligible test examples remain.',
        '', '| Arm | Validation RMSE | Test RMSE | Test90% raw coverage |', '|---|---:|---:|---:|']
    for arm,res in results.items():
        lines.append(f"| {arm} | {res['validation']['primary_0_20']['rmse']:.3f}m | {res['test']['primary_0_20']['rmse']:.3f}m | {res['test']['raw_coverage90_pct']:.1f}% |")
    lines+=['','| Test depth bin | n | Composite control RMSE | Matched/tide RMSE |','|---|---:|---:|---:|']
    for b in results['composite_control']['test']['primary_bins']:
        a=results['composite_control']['test']['primary_bins'][b];c=results['matched_tide']['test']['primary_bins'][b]
        lines.append(f"| {b} | {a['n']} | {a.get('rmse',float('nan')):.3f}m | {c.get('rmse',float('nan')):.3f}m |")
    lines+=['','Per-bin spatial-bootstrap intervals,15/20/25m sensitivity, relative errors and fallback strata are in[comparison.json](comparison.json).',
        'This changes reference version/coordinates relative to the old public ATL24 and GEBCO studies. Absolute RMSE differences across those benchmarks cannot establish model improvement.',
        'Product sigma and horizontal footprints are audited, not yet propagated into the likelihood. V002 classifications and coastal tide residuals remain imperfect.',
        'Optical visibility is uncalibrated and no held-out point was masked based on model error.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    update('atl24','paired_training_complete',results={a:{s:r[s]['primary_0_20']['rmse'] for s in ('validation','test')} for a,r in results.items()})
    print(json.dumps({a:r['test']['primary_0_20'] for a,r in results.items()},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        update('atl24','paired_experiment_failed',error=type(exc).__name__);raise
