"""Matched joint-loss comparisons, full training, then Sentinel-only prototype."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.cnn_baseline import (PatchDataset,SpectralAugDataset,DepthStratifiedSampler,
    train_hybridbathnet,evaluate_hybridbathnet)
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import JointCompositeLoss,CoverageSampler,initialize_optical_head

ROOT=Path('runs/2026-11-05/joint_v1')
BASE=Path('runs/2026-11-05/improvement/expanded_smooth/checkpoints/hybridbathnet.pt')


def write(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2),encoding='utf-8'); tmp.replace(path)


def status(stage,experiment):
    write(ROOT/'status.json',{'stage':stage,'experiment':experiment,
                            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    # Keep the requested handoff current as the sequential job advances.
    for path in (Path('PROJECT_CONTEXT.md'),Path('README.md')):
        text=path.read_text(encoding='utf-8')
        marker='<!-- JOINT_JOB_STATUS -->'; end='<!-- /JOINT_JOB_STATUS -->'
        note=(f'{marker}\nJoint composite study: **{stage}**, experiment **{experiment}**. '
              f'UTC update: {datetime.datetime.now(datetime.timezone.utc).isoformat()}. '
              'See [live status](runs/2026-11-05/joint_v1/status.json), '
              '[supervised report](runs/2026-11-05/joint_v1/REPORT.md), and '
              '[Sentinel-only report](runs/2026-11-05/joint_v1/sentinel_only/REPORT.md). '
              'Pending reports appear only when their experiments finish.\n'+end)
        if marker in text:
            a=text.index(marker); b=text.index(end,a)+len(end); text=text[:a]+note+text[b:]
        else:
            parts=text.split('\n',1); text=parts[0]+'\n\n'+note+'\n'+parts[1]
        path.write_text(text,encoding='utf-8')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if not (ROOT/'features/patches.npz').exists():
        subprocess.run([sys.executable,'scripts/build_joint_data.py'],check=True)
    with np.load(ROOT/'features/patches.npz') as d:
        X,Y,M=d['X'],d['Y'],d['M']; groups={k:d[k] for k in ('train_idx','val_idx','test_idx')}
    base=PatchDataset(X,Y,M,groups['train_idx'])
    def ds(indices):
        return PatchDataset(X,Y,M,indices,mean=base.mean,std=base.std,y_mean=base.y_mean,y_std=base.y_std)
    val=DataLoader(SpectralAugDataset(ds(groups['val_idx']),augment=False),batch_size=8)
    test=DataLoader(SpectralAugDataset(ds(groups['test_idx']),augment=False),batch_size=8)
    water=base.M>.5
    rinf=np.array([np.percentile(base.X[:,b][water],2) for b in range(3)],dtype='float32')
    with (ROOT/'features/patches.npz').open('rb') as stream:
        datahash=hashlib.file_digest(stream,'sha256').hexdigest()
    write(ROOT/'protocol.json',{'dataset_sha256':datahash,'cuda':torch.version.cuda,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0),'seed':42,'batch_size':8,'pilot_epochs':15,'full_epochs':180,
        'composite':'1 Huber(normalized depth) + 1 betaNLL(metre variance) + .03 extinction(normalized metres) + .1 masked optical Huber(reflectance/.05)',
        'selection':'validation RMSE; uncertainty calibration reported separately; test after pilot choices frozen',
        'rinf_training_only':rinf.tolist(),'label_reference':'direct native GEBCO; held-out coordinates preserved',
        'initialization':'matched pilots warm-start from same historical selected state; full_joint starts from scratch',
        'weights':{'w_depth':1,'w_nll':1,'w_ext':.03,'w_recon':.1},
        'sentinel_only':'separate fresh model/optical-only splits; no depth truth RMSE'})
    results={}
    def run(name,mask_aware,recon,coverage,epochs,warm):
        folder=ROOT/name; folder.mkdir(exist_ok=True)
        metrics=folder/'validation.json'
        if metrics.exists():
            results[name]=json.loads(metrics.read_text()); return
        torch.manual_seed(42); np.random.seed(42)
        model=HybridBathNetModel(in_channels=16,mean=base.mean.squeeze(),std=base.std.squeeze(),
            kd_aggregation='center',physics_conditioning='spatial',mask_aware_physics=mask_aware,r_inf=rinf).cuda()
        if warm:
            model.load_state_dict(torch.load(BASE,map_location='cuda',weights_only=False)['model'])
            model.mu.copy_(torch.from_numpy(base.mean[:,:16]).cuda())
            model.sig.copy_(torch.from_numpy(base.std[:,:16]).cuda())
            model.r_inf_buf.copy_(torch.from_numpy(rinf).view(1,3,1,1).cuda())
        initialize_optical_head(model)
        train=SpectralAugDataset(base,augment=True,noise_std=.02,use_rotations=True)
        sampler=CoverageSampler(len(base),42) if coverage else DepthStratifiedSampler(base.Y[:,32,32],seed=42)
        loader=DataLoader(train,batch_size=8,sampler=sampler)
        objective=JointCompositeLoss(w_recon=.1 if recon else 0,w_mse=1,w_nll=1,w_phys=0,w_ext=.03,
                                     use_depth_weighting=False,beta_nll=.5,huber_delta=.5)
        status('training',name)
        settings={'mask_aware':mask_aware,'reconstruction_weight':objective.w_recon,
                  'sampler':'without_replacement' if coverage else 'depth_quantile_replacement',
                  'epochs':epochs,'warm_start':warm}
        write(folder/'settings.json',settings)
        model,_=train_hybridbathnet(model,objective,loader,val,epochs=epochs,lr=1e-4 if warm else .001,
            device='cuda',y_mean=base.y_mean,y_std=base.y_std,nll_warmup_epochs=1 if warm else 20,
            lr_restart_period=epochs,early_stop_patience=30 if not warm else epochs,
            checkpoint_dir=str(folder/'checkpoints'))
        torch.save({'model':model.state_dict(),'settings':settings,'y_mean':base.y_mean,'y_std':base.y_std,
                    'ch_mean':base.mean.tolist(),'ch_std':base.std.tolist()},folder/'best_model.pt')
        res=evaluate_hybridbathnet(model,val,'cuda',base.y_mean,base.y_std,compute_calibration=True,
                                   predictions_path=folder/'validation_predictions.npz')
        write(metrics,res); results[name]=res; write(ROOT/'validation_summary.json',results)

    try:
        for name,mask,recon,coverage in [('control',False,False,False),('mask_aware',True,False,False),
                                      ('joint_composite',True,True,False),('joint_full_coverage',True,True,True)]:
            run(name,mask,recon,coverage,15,True)
        # The full integrated run is mandated regardless of short-pilot ordering.
        write(ROOT/'pilot_selection.json',{'best_validation_pilot':min(results,key=lambda n:results[n]['rmse']),
              'full_run':'full_joint','reason':'user-requested full integrated training; both masking and full coverage enabled'})
        run('full_joint',True,True,True,180,False)
        status('final_evaluation','supervised')
        tests={}
        for name in results:
            folder=ROOT/name
            path=folder/'test.json'
            if path.exists(): tests[name]=json.loads(path.read_text()); continue
            saved=torch.load(folder/'best_model.pt',map_location='cuda',weights_only=False)
            model=HybridBathNetModel(in_channels=16,mask_aware_physics=saved['settings']['mask_aware'],
                                     kd_aggregation='center',physics_conditioning='spatial').cuda()
            model.load_state_dict(saved['model'])
            tests[name]=evaluate_hybridbathnet(model,test,'cuda',base.y_mean,base.y_std,
                compute_calibration=True,predictions_path=folder/'test_predictions.npz')
            write(path,tests[name])
        write(ROOT/'test_summary.json',tests)
        lines=['# Joint composite-loss study results','','Depth RMSE is a reference-based evaluation metric, not the composite loss.','',
               '| Run | Validation RMSE (m) | Test RMSE (m) | Test MAE (m) | 68% coverage | 90% coverage |',
               '|---|---:|---:|---:|---:|---:|']
        for name,m in tests.items():
            lines.append(f"| {name} | {results[name]['rmse']:.3f} | {m['rmse']:.3f} | {m['mae']:.3f} | {m['coverage_68pct']:.1f}% | {m['coverage_90pct']:.1f}% |")
        lines+=['','Four 15-epoch matched warm-start pilots, then one fresh full integrated model up to 180 epochs. '
                'Pilot versus full-run scores do not isolate a single intervention because initialization and budget differ. '
                'Original expanded_smooth reported 2.594/3.554 m validation/test against old targets; these scores use corrected targets/features.',
                '', 'No independent measured bathymetry was added. Existing held-out coordinates have already been inspected; '
                'results remain exploratory. Learned optical reconstruction is not proof of uniquely identifiable depth.',
                '', '[Protocol](protocol.json) | [Validation](validation_summary.json) | [Test](test_summary.json)']
        (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status('training','sentinel_only')
        subprocess.run([sys.executable,'-u','scripts/run_sentinel_only.py'],check=True)
        status('complete','all_requested_runs')
    except Exception as e:
        write(ROOT/'failure.json',{'error':repr(e)}); status('failed','inspect_failure'); raise


if __name__=='__main__': main()
