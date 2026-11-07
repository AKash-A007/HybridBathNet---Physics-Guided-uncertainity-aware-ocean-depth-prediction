"""Fit fresh full-model folds; OOF labels never select model checkpoints."""
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
from scripts.gebco_oof_common import ROOT,update
from scripts.run_nearshore_hybrid import ShallowAugDataset
from src.cnn_baseline import PatchDataset,train_hybridbathnet,evaluate_hybridbathnet
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import CoverageSampler,JointCompositeLoss,initialize_optical_head
OPTIONS={'uncertainty_head':'spatial','detach_uncertainty_features':True,'reconstruction_attenuation':'pixel'}


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not (ROOT/'source_folds.zip').exists():
        with zipfile.ZipFile(ROOT/'source_folds.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in [*Path('src').glob('*.py'),Path(__file__),Path('scripts/build_gebco_oof_data.py'),Path('scripts/gebco_oof_common.py')]:z.write(p,str(p))
    for fold in range(3):
        root=ROOT/f'fold_{fold}';data=root/'features.npz'
        if (root/'complete.json').exists():continue
        seed=42+fold;torch.manual_seed(seed);np.random.seed(seed)
        with np.load(data) as d:
            x,y,m=(d[k] for k in ('X','Y','M'))
            split={k:d[k] for k in ('train_idx','val_idx','held_idx')}
        base=PatchDataset(x,y,m,split['train_idx'])
        def loader(key,training=False):
            ds=base if training else PatchDataset(x,y,m,split[key],mean=base.mean,std=base.std,y_mean=base.y_mean,y_std=base.y_std)
            return DataLoader(ShallowAugDataset(ds,augment=training,noise_std=.02,use_rotations=False),
                batch_size=8,sampler=CoverageSampler(len(ds),seed) if training else None)
        rinf=np.array([np.percentile(base.X[:,b][base.M>.5],2) for b in range(3)],dtype='float32')
        model=HybridBathNetModel(in_channels=16,mean=base.mean.squeeze(),std=base.std.squeeze(),r_inf=rinf,
            mask_aware_physics=True,kd_aggregation='center',physics_conditioning='spatial',**OPTIONS).cuda()
        initialize_optical_head(model)
        loss=JointCompositeLoss(w_mse=1,w_nll=1,w_ext=.03,w_phys=0,w_recon=.1,beta_nll=0,
            detach_nll_mean=True,use_depth_weighting=False,huber_delta=.5)
        protocol={'fold':fold,'seed':seed,'model_options':OPTIONS,'dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
            'split_sizes':{k:len(v) for k,v in split.items()},'epochs':180,'patience':30,'lr':.001,'batch':8,
            'cosine_period':180,'ema':.999,'selection':'external validation RMSE only; OOF region never used',
            'initialization':'fresh random; no checkpoint trained on OOF labels','loss':'Huber1 + NLL1 + extinction.03 + reconstruction.1'}
        (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        update('training_spatial_fold',fold=fold,train=len(split['train_idx']),held=len(split['held_idx']))
        model,_=train_hybridbathnet(model,loss,loader('train_idx',True),loader('val_idx'),epochs=180,lr=.001,
            device='cuda',y_mean=base.y_mean,y_std=base.y_std,nll_warmup_epochs=20,lr_restart_period=180,
            ema_decay=.999,early_stop_patience=30,separate_uncertainty_clip=True,checkpoint_dir=str(root/'checkpoints'))
        torch.save({'model':model.state_dict(),'protocol':protocol,'ch_mean':base.mean,'ch_std':base.std,
            'y_mean':base.y_mean,'y_std':base.y_std},root/'hybridbathnet.pt')
        metrics={s:evaluate_hybridbathnet(model,loader(k),'cuda',base.y_mean,base.y_std,
            compute_calibration=True,predictions_path=root/f'{s}_predictions.npz')
            for k,s in (('val_idx','validation'),('held_idx','held'))}
        (root/'complete.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print('Completed fold',fold,'validation',metrics['validation']['rmse'],'OOF',metrics['held']['rmse'],flush=True)
        del model,base,x,y,m;torch.cuda.empty_cache()
    update('spatial_folds_complete')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        update('spatial_folds_failed',error=type(exc).__name__);raise
