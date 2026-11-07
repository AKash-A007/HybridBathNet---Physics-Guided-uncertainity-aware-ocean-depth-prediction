"""Matched HybridBathNet run with a squared supervised depth term."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datetime
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.cnn_baseline import PatchDataset,SpectralAugDataset,train_hybridbathnet,evaluate_hybridbathnet
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import SquaredDepthJointLoss,CoverageSampler,initialize_optical_head

ROOT=Path('runs/2026-11-05/nearshore_squared_depth_v3')
DATA=Path('runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz')


class ShallowAugDataset(SpectralAugDataset):
    """Do not let flips move the supervised centre outside the stated depth domain."""
    def __getitem__(self,index):
        sample=super().__getitem__(index)
        _,y,m,_=sample; cy,cx=y.shape[0]//2,y.shape[1]//2
        if self.augment and not (m[cy,cx]>.5 and 0<y[cy,cx]<=20):
            return self.base[index]
        return sample


def write(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); tmp.replace(path)


def status(stage,extra=None):
    state={'stage':stage,'experiment':'hybridbathnet_only',
           'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**(extra or {})}
    write(ROOT/'status.json',state)
    marker='<!-- SQUARED_JOB_STATUS -->'; end='<!-- /SQUARED_JOB_STATUS -->'
    note=f'{marker}\nSquared-depth HybridBathNet ablation: **{stage}**. '
    note+='Same nearshore data/splits and all other joint losses. '
    note+='[Status](runs/2026-11-05/nearshore_squared_depth_v3/status.json) | '
    note+='[Report](runs/2026-11-05/nearshore_squared_depth_v3/REPORT.md).\n'+end
    for path in (Path('README.md'),Path('PROJECT_CONTEXT.md')):
        text=path.read_text(encoding='utf-8')
        if marker in text:
            a=text.index(marker); b=text.index(end,a)+len(end); text=text[:a]+note+text[b:]
        else:
            a=text.index('\n'); text=text[:a+1]+'\n'+note+'\n'+text[a+1:]
        path.write_text(text,encoding='utf-8')


def interval(t,p,groups):
    unique=np.unique(groups); rng=np.random.RandomState(42); errors=(p-t)**2
    boot=[]
    for _ in range(2000):
        ids=np.concatenate([np.flatnonzero(groups==u) for u in rng.choice(unique,len(unique),replace=True)])
        boot.append(np.sqrt(errors[ids].mean()))
    return {'rmse_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'clusters':len(unique)}


def diagnostics(pred_path,coords):
    with np.load(pred_path) as d:
        t,p,sigma,zext=(d[k].reshape(-1).astype('float64') for k in ('target','prediction','sigma','z_ext'))
    _,groups=np.unique((coords+32)//200,axis=0,return_inverse=True)
    def summary(m):
        n=int(m.sum())
        if not n:return {'n':0}
        tm,pm=t[m],p[m]; error=pm-tm
        return {'n':n,'rmse':float(np.sqrt((error**2).mean())), 'mae':float(np.abs(error).mean()),
                'bias':float(error.mean()),'mape_pct':float(100*np.mean(np.abs(error)/np.maximum(tm,1e-6))),
                'delta_lt_1_25_pct':float(100*np.mean(np.maximum(pm/np.maximum(tm,1e-6),tm/np.maximum(pm,1e-6))<1.25)),
                'coverage_pct':float(100*m.mean()),'small_sample':n<10,**interval(tm,pm,groups[m])}
    primary=(t>=0)&(t<=20)
    bins={f'{lo}-{hi if np.isfinite(hi) else "+"}':summary(primary&(t>=lo)&(t<hi))
          for lo,hi in ((0,5),(5,10),(10,15),(15,float('inf')))}
    cutoff={str(c):summary((t>=0)&(t<=c)) for c in (15,20,25)}
    flags=(p>zext)
    np.savez(pred_path.with_name(pred_path.stem+'_optical_flags.npz'),
             predicted_depth=p,predicted_sigma=sigma,heuristic_zext=zext,
             outside_heuristic_optical_support=flags,calibrated_optical_limit=np.array(False))
    return {'primary_0_20':summary(primary),'primary_bins':bins,'cutoff_sensitivity':cutoff,
            'optical_flagged_pct':float(100*flags.mean()),'optical_limit_calibrated':False,
            'bootstrap':'2000 resamples of approximately 2km spatial groups; seed42; conditional on this model/reference',
            'no_clipping_or_test_error_based_exclusion':True}


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if (ROOT/'results.json').exists(): print('Already complete; no duplicate training'); return
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    path=DATA
    with path.open('rb') as f: datahash=hashlib.file_digest(f,'sha256').hexdigest()
    with np.load(path) as data:
        X,Y,M,coords=data['X'],data['Y'],data['M'],data['coords']
        groups={k:data[k] for k in ('train_idx','val_idx','test_idx')}
        supports=data['native_support']
    for name in ('train_idx','val_idx'):
        assert Y[groups[name],32,32].max()<=20.0001
    assert not set(supports[groups['train_idx']].ravel())&set(supports[groups['test_idx']].ravel())
    assert not set(supports[groups['train_idx']].ravel())&set(supports[groups['val_idx']].ravel())
    assert not set(supports[groups['val_idx']].ravel())&set(supports[groups['test_idx']].ravel())
    assert len(np.unique(coords,axis=0))==len(coords),'Duplicate sample coordinates'
    base=PatchDataset(X,Y,M,groups['train_idx'])
    def loader(name,training=False):
        d=base if training else PatchDataset(X,Y,M,groups[name],mean=base.mean,std=base.std,
                                             y_mean=base.y_mean,y_std=base.y_std)
        return DataLoader(ShallowAugDataset(d,augment=training,noise_std=.02,use_rotations=False),
                          batch_size=8,sampler=CoverageSampler(len(d),42) if training else None)
    train,val,test=loader('train_idx',True),loader('val_idx'),loader('test_idx')
    rinf=np.array([np.percentile(base.X[:,b][base.M>.5],2) for b in range(3)],dtype='float32')
    model=HybridBathNetModel(in_channels=16,mean=base.mean.squeeze(),std=base.std.squeeze(),r_inf=rinf,
                            mask_aware_physics=True,kd_aggregation='center',physics_conditioning='spatial').cuda()
    initialize_optical_head(model)
    criterion=SquaredDepthJointLoss(w_mse=1,w_nll=1,w_ext=.03,w_phys=0,w_recon=.1,
                                 use_depth_weighting=False,beta_nll=.5,huber_delta=.5)
    protocol={'model':'HybridBathNet only; all modules jointly optimized','seed':42,'initialization':'random',
        'epochs':180,'patience':30,'lr':.001,'batch':8,'ema':.999,'cosine_period':180,
        'depth_supervision':'centre-pixel GEBCO; 0–20m','reconstruction':'masked optical map; shelf <=25m',
        'weights':{'depth_mse_normalized':1,'beta_nll':1,'extinction':.03,'reconstruction':.1},
        'reflectance_loss_scale':.05,'nll_warmup':20,'rinf_training_only':rinf.tolist(),
        'dataset_sha256':datahash,'dataset_path':str(DATA),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'comparison':'Single-change ablation vs nearshore_hybrid_v1: normalized Huber depth term to normalized squared error; physics/uncertainty/reconstruction unchanged.',
        'test_regions':'original block IDs retained; denser distinct shallow coordinates',
        'selection':'validation RMSE on 0–20m; test examined after training only',
        'test_20_to_25m':'extrapolation beyond supervised training depth, sensitivity only',
        'optical_limit':'heuristic flag only, not an exclusion mask',
        'augmentation':'spectral jitter, flips; optical reconstruction targets remain unjittered',
        'warning':'Overlapping patches and coarse GEBCO are not independent 10m survey truths.'}
    write(ROOT/'protocol.json',protocol)
    status('training',{'split_sizes':{k:len(v) for k,v in groups.items()}})
    # All heads are optimized together through this single composite objective.
    model,_=train_hybridbathnet(model,criterion,train,val,epochs=180,lr=.001,device='cuda',
        y_mean=base.y_mean,y_std=base.y_std,nll_warmup_epochs=20,lr_restart_period=180,
        ema_decay=.999,early_stop_patience=30,checkpoint_dir=str(ROOT/'hybridbathnet/checkpoints'))
    torch.save({'model':model.state_dict(),'protocol':protocol,'ch_mean':base.mean,'ch_std':base.std,
                'y_mean':base.y_mean,'y_std':base.y_std},ROOT/'hybridbathnet.pt')
    status('evaluating')
    vm=evaluate_hybridbathnet(model,val,'cuda',base.y_mean,base.y_std,compute_calibration=True,
                             predictions_path=ROOT/'validation_predictions.npz')
    tm=evaluate_hybridbathnet(model,test,'cuda',base.y_mean,base.y_std,compute_calibration=True,
                             predictions_path=ROOT/'test_predictions.npz')
    write(ROOT/'validation_metrics.json',vm);write(ROOT/'test_metrics_0_25.json',tm)
    vd=diagnostics(ROOT/'validation_predictions.npz',coords[groups['val_idx']])
    td=diagnostics(ROOT/'test_predictions.npz',coords[groups['test_idx']])
    write(ROOT/'results.json',{'validation':vd,'test':td,'uncertainty_0_25':tm})
    lines=['# Nearshore HybridBathNet squared-depth-objective results','',
        'Original geographic test blocks retained. Only HybridBathNet trained. Shelf eligibility <=25m applied '
        'before patch extraction; depth training/selection limited to <=20m. No sample duplication or predicted-depth clipping.', '',
        f"Validation RMSE (0–20m): {vd['primary_0_20']['rmse']:.3f} m.",'',
        '| Test maximum reference depth | n | RMSE (m) | 95% spatial bootstrap CI | MAPE (%) | Delta <1.25 (%) |',
        '|---|---:|---:|---|---:|---:|']
    for cut,m in td['cutoff_sensitivity'].items():
        ci=m['rmse_ci95'];lines.append(f"| {cut} | {m['n']} | {m['rmse']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | {m['mape_pct']:.1f} | {m['delta_lt_1_25_pct']:.1f} |")
    lines+=['','| Primary 0–20m depth bin | n | RMSE (m) | 95% spatial bootstrap CI |',
            '|---|---:|---:|---|']
    for b,m in td['primary_bins'].items():
        if m['n']:
            ci=m['rmse_ci95'];lines.append(f"| {b} | {m['n']} | {m['rmse']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} |")
    lines+=['',f"Heuristic optical-support flags cover {td['optical_flagged_pct']:.1f}% of test predictions. These are diagnostic flags, not a calibrated mask; all eligible samples remain scored.",
        '', 'The 20–25m test subset is outside the supervised training-depth range. Bootstrap intervals are '
        'conditional on the model and GEBCO, and do not include reference error. Source-cell support is disjoint '
        'across splits; within-split patches overlap. Prior 3.554m full-depth test RMSE is not a matched comparator '
        'for the new shallow samples and scope. No sub-metre real-seabed accuracy is established.', '',
        '[Protocol](protocol.json) | [All results](results.json) | [Dataset audit](features/metadata.json)']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status('complete',{'validation_rmse_0_20':vd['primary_0_20']['rmse'],
                       'test_rmse_0_20':td['primary_0_20']['rmse']})


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    try:main()
    except Exception as e:
        write(ROOT/'failure.json',{'error':repr(e)});status('failed');raise
