"""Three independently trained full models with untouched calibration groups."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
import json
import hashlib
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.recovery_state import update
from scripts.run_nearshore_hybrid import ShallowAugDataset
from src.cnn_baseline import PatchDataset,train_hybridbathnet,evaluate_hybridbathnet
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import CoverageSampler,JointCompositeLoss,initialize_optical_head

ROOT=Path('runs/2026-11-06/recovery_v2/gebco_uq')
DATA=ROOT/'features/patches.npz'
SEEDS=(42,1,7)
OPTIONS={'uncertainty_head':'spatial','detach_uncertainty_features':True,
         'reconstruction_attenuation':'pixel'}


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (ROOT/'source_snapshot.zip').exists():
        with zipfile.ZipFile(ROOT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in ('src/hybridbathnet.py','src/cnn_baseline.py','src/joint_objective.py',
                         'scripts/run_recovery_uncertainty.py','scripts/build_recovery_calibration_data.py'):
                archive.write(name)
    with np.load(DATA) as d:
        x,y,m=(d[k] for k in ('X','Y','M'))
        split={k:d[k] for k in ('train_idx','calib_idx','val_idx','test_idx')}
    base=PatchDataset(x,y,m,split['train_idx'])
    for seed in SEEDS:
        root=ROOT/f'seed_{seed}';root.mkdir(exist_ok=True)
        if (root/'complete.json').exists():
            print('Completed seed; skipping',seed,flush=True);continue
        torch.manual_seed(seed);np.random.seed(seed)
        def loader(key,training=False):
            ds=base if training else PatchDataset(x,y,m,split[key],mean=base.mean,std=base.std,
                                                  y_mean=base.y_mean,y_std=base.y_std)
            return DataLoader(ShallowAugDataset(ds,augment=training,noise_std=.02,use_rotations=False),
                batch_size=8,sampler=CoverageSampler(len(ds),seed) if training else None)
        train,val=loader('train_idx',True),loader('val_idx')
        rinf=np.array([np.percentile(base.X[:,b][base.M>.5],2) for b in range(3)],dtype='float32')
        model=HybridBathNetModel(in_channels=16,mean=base.mean.squeeze(),std=base.std.squeeze(),
            r_inf=rinf,mask_aware_physics=True,kd_aggregation='center',physics_conditioning='spatial',**OPTIONS).cuda()
        initialize_optical_head(model)
        criterion=JointCompositeLoss(w_mse=1,w_nll=1,w_ext=.03,w_phys=0,w_recon=.1,
            beta_nll=0,detach_nll_mean=True,use_depth_weighting=False,huber_delta=.5)
        protocol={'seed':seed,'model_options':OPTIONS,'modules':'full encoder/depth/physics/uncertainty',
            'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
            'split_sizes':{k:len(v) for k,v in split.items()},'epochs':180,'patience':30,
            'lr':.001,'batch':8,'ema':.999,'cosine_period':180,'nll_warmup':20,
            'loss':'Huber1 + detached-mean Gaussian NLL1 + extinction.03 + reconstruction.1',
            'separate_variance_gradient_clipping':True,
            'selection':'validation RMSE; calibration untouched until all members complete',
            'comparison':'Old v1 has more training points; same fixed test coordinates/labels but not a single-change comparison',
            'optical_limit':'heuristic only; no clipping or masking'}
        (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        update('uncertainty','training',seed=seed,seeds=list(SEEDS),split_sizes=protocol['split_sizes'])
        model,_=train_hybridbathnet(model,criterion,train,val,epochs=180,lr=.001,device='cuda',
            y_mean=base.y_mean,y_std=base.y_std,nll_warmup_epochs=20,lr_restart_period=180,
            ema_decay=.999,early_stop_patience=30,separate_uncertainty_clip=True,
            checkpoint_dir=str(root/'checkpoints'))
        torch.save({'model':model.state_dict(),'protocol':protocol,'ch_mean':base.mean,'ch_std':base.std,
            'y_mean':base.y_mean,'y_std':base.y_std},root/'hybridbathnet.pt')
        metrics={}
        for key,stem in (('val_idx','validation'),('calib_idx','calibration'),('test_idx','test')):
            metrics[stem]=evaluate_hybridbathnet(model,loader(key),'cuda',base.y_mean,base.y_std,
                compute_calibration=True,predictions_path=root/f'{stem}_predictions.npz')
        (root/'complete.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print('Completed seed',seed,'validation RMSE',metrics['validation']['rmse'],flush=True)
    update('uncertainty','models_complete_pending_calibration',seeds=list(SEEDS))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        update('uncertainty','failed',error=type(exc).__name__);raise
