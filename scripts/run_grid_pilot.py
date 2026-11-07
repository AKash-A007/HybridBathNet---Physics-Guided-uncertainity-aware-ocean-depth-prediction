"""Matched 10-epoch label-only fine-tuning diagnostic; validation only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
import yaml
from src.preprocess import resample_gebco
from src.cnn_baseline import (PatchDataset, SpectralAugDataset, DepthStratifiedSampler,
                              train_hybridbathnet, evaluate_hybridbathnet)
from src.hybridbathnet import HybridBathNetModel, CompositePhysicsNLLLoss

ROOT=Path('runs/2026-11-05/grid_audit')
BASE=Path('runs/2026-11-05/improvement/expanded_smooth')


def save(path, data):
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    cfg=yaml.safe_load((BASE/'config.yaml').read_text())
    saved=torch.load(BASE/'checkpoints/hybridbathnet.pt',map_location='cpu',weights_only=False)
    with np.load('runs/2026-11-05/improvement/features/patches.npz') as data:
        X,Y,M=data['X'],data['Y'],data['M']; coords=data['coords']
        train_idx,val_idx,test_idx=data['train_idx'],data['val_idx'],data['test_idx']
    with rasterio.open('runs/2026-11-04/processed/stack.tif') as stack:
        corrected=resample_gebco('data/raw/gebco/gebco_subset.tif',stack.transform,stack.crs,stack.shape)
    targets=-corrected[coords[:,0]+32,coords[:,1]+32]
    keep=np.isfinite(targets)&(targets>0)
    assert keep[val_idx].all() and keep[test_idx].all(), 'Do not remove held-out locations'
    original_n=len(train_idx); train_idx=train_idx[keep[train_idx]]
    directY=Y.copy(); directY[:,32,32]=targets
    # Only centre labels are consumed by the scalar training/evaluation path.
    # Unused non-centre targets and all imagery/features are deliberately fixed.
    directY[~keep,32,32]=Y[~keep,32,32]
    mean=np.asarray(saved['ch_mean'],dtype='float32')[None,:,None,None]
    std=np.asarray(saved['ch_std'],dtype='float32')[None,:,None,None]
    ym,ys=saved['y_mean'],saved['y_std']
    def dataset(labels,indices):
        return PatchDataset(X,labels,M,indices,mean=mean,std=std,y_mean=ym,y_std=ys)
    # No rotations: they move the supervised centre and would need full corrected patches.
    val_loaders={name:DataLoader(SpectralAugDataset(dataset(labels,val_idx),augment=False),batch_size=8)
                 for name,labels in [('old',Y),('direct',directY)]}
    results={}
    save(ROOT/'pilot_protocol.json',{'epochs':10,'seed':42,'lr':1e-5,'ema':.999,
        'training_patches':len(train_idx),'excluded_new_land_training_centres':original_n-len(train_idx),
        'validation_locations':len(val_idx),'test_evaluated':False,
        'initial_checkpoint':str(BASE/'checkpoints/hybridbathnet.pt'),
        'fixed':'features, masks, normalization, split coordinates, seed and initialization',
        'augmentation':'disabled; centre-label-only diagnostic','selection_metric':'direct-label validation RMSE'})
    try:
        for name,labels in [('old',Y),('direct',directY)]:
            folder=ROOT/f'pilot_{name}'; folder.mkdir(exist_ok=True)
            torch.manual_seed(42); np.random.seed(42)
            model=HybridBathNetModel(in_channels=16,n_spectral=3,mean=mean.squeeze(),std=std.squeeze(),
                feature_dim=128,dropout_rate=.1,i0_over_epsilon=100,use_red_kd=True,use_recon_head=True,
                n_recon_bands=3,kd_aggregation='center',physics_conditioning='spatial',
                use_explicit_reflectance=True).to('cuda')
            model.load_state_dict(saved['model'])
            before={n:evaluate_hybridbathnet(model,l,'cuda',ym,ys,compute_calibration=True) for n,l in val_loaders.items()}
            # Identical sample order across arms: stratify on the same original depths.
            sampler=DepthStratifiedSampler(Y[train_idx,32,32],seed=42)
            loader=DataLoader(SpectralAugDataset(dataset(labels,train_idx),augment=False),batch_size=8,sampler=sampler)
            criterion=CompositePhysicsNLLLoss(w_mse=1,w_nll=1,w_phys=0,w_ext=.03,
                                             use_depth_weighting=False,beta_nll=.5,huber_delta=.5)
            if '--evaluate-only' in sys.argv:
                model.load_state_dict(torch.load(folder/'pilot_best.pt',map_location='cuda',weights_only=False)['model'])
            else:
                save(ROOT/'status.json',{'stage':'training','experiment':f'pilot_{name}',
                                        'completed_validation':list(results)})
                model,_=train_hybridbathnet(model,criterion,loader,val_loaders['direct'],
                    epochs=10,lr=1e-5,device='cuda',y_mean=ym,y_std=ys,nll_warmup_epochs=1,
                    lr_restart_period=10,ema_decay=.999,early_stop_patience=10,
                    checkpoint_dir=str(folder/'checkpoints'))
            after={n:evaluate_hybridbathnet(model,l,'cuda',ym,ys,
                compute_calibration=True, predictions_path=folder/f'validation_{n}.npz') for n,l in val_loaders.items()}
            results[name]={'before':before,'after':after}
            save(ROOT/'pilot_results.json',results)
            if '--evaluate-only' not in sys.argv:
                torch.save({'model':model.state_dict(),'configuration':cfg,'normalization_source':str(BASE),
                            'protocol':json.loads((ROOT/'pilot_protocol.json').read_text())},folder/'pilot_best.pt')
        save(ROOT/'status.json',{'stage':'complete','experiment':'grid_pilot','completed_validation':list(results)})
        print('PILOT COMPLETE',flush=True)
        for name,res in results.items():
            print(name,'before/direct',res['before']['direct']['rmse'],'after/direct',res['after']['direct']['rmse'],flush=True)
    except Exception as error:
        save(ROOT/'status.json',{'stage':'failed','error':repr(error)})
        raise


if __name__=='__main__': main()
