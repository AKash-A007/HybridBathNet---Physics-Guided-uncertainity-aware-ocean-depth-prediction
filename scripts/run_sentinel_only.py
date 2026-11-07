"""Separate Sentinel-only self-supervised prototype. No depth ground truth loaded."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import json
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from src.features import extract_patches
from src.cnn_baseline import PatchDataset,SpectralAugDataset
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import CoverageSampler,initialize_optical_head

ROOT=Path('runs/2026-11-05/joint_v1/sentinel_only')


def save(path,data): path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def build():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'patches.npz').exists(): return
    # Explicit band list: never read band 9 (GEBCO). Band 10 is clear-scene validity from SCL.
    with rasterio.open('runs/2026-11-04/processed/stack.tif') as src:
        a=src.read([1,2,3,4,5,6,7,8,10])
    bands=['B02','B03','B04','B08']
    arrays={b:a[i] for i,b in enumerate(bands)}
    arrays.update({f'{b}_std':a[i+4] for i,b in enumerate(bands)})
    ndwi=(a[1]-a[3])/(a[1]+a[3]+1e-6)
    valid=(a[8]>.5)&(ndwi>0)&np.isfinite(a[:8]).all(0)&(a[0]>0)
    arrays['valid_mask']=valid.astype('float32')
    # Constant placeholder only satisfies the existing feature-builder interface.
    # It is not loaded bathymetry and is NEVER used as depth supervision.
    arrays['gebco_depth']=-np.ones_like(a[0])
    height,width=valid.shape
    cutoff1=int(height*.7)//64*64; cutoff2=int(height*.85)//64*64
    fit=np.zeros_like(valid); fit[:cutoff1-32]=True
    X,Y,M,coords=extract_patches(arrays,bands,64,64,fit_mask=fit)
    groups=[[],[],[]]
    for i,(r,c) in enumerate(coords):
        if r+64<=cutoff1-32: groups[0].append(i)
        elif r>=cutoff1+32 and r+64<=cutoff2-32: groups[1].append(i)
        elif r>=cutoff2+32: groups[2].append(i)
    if any(not g for g in groups): raise RuntimeError('Optical spatial split empty')
    # Save optical arrays without any Y/GEBCO field, independently of supervised dataset.
    keep=np.sort(np.concatenate(groups)); remap={int(i):j for j,i in enumerate(keep)}
    np.savez(ROOT/'patches.npz',X=X[keep],M=M[keep],coords=np.asarray(coords)[keep],
             **{name:np.array([remap[i] for i in group]) for name,group in zip(('train_idx','val_idx','test_idx'),groups)})
    save(ROOT/'data_protocol.json',{'source_bands_read':[1,2,3,4,5,6,7,8,10],
        'depth_reference_loaded':False,'water_mask':'NDWI > 0 and optical clear-scene validity',
        'splits':'contiguous latitude bands with at least 64-pixel gaps; all composite dates grouped',
        'split_sizes':[len(g) for g in groups],'feature_fit':'training region only',
        'reference':'none; no verified depth accuracy or calibrated metre-depth uncertainty',
        'input':'Sentinel-2 median and temporal standard deviation, not individual-scene model'})


def optical_loss(out,raw,mask):
    valid=mask.bool().expand_as(raw)
    error=out['r_hat'].float()-raw.float()
    reconstruction=torch.nn.functional.smooth_l1_loss(out['r_hat'].float()/.05,raw.float()/.05,reduction='none')[valid].mean()
    # Local delta-method propagation of latent depth variance into reflectance.
    # This is model-implied uncertainty, NOT calibrated true-depth uncertainty.
    slope=2*out['kl_map'].float()*(out['al_map'].float()-out['r_inf'].float()).abs()*torch.exp(-2*out['kl_map'].float()*out['depth'].float())
    variance=(slope.detach().square()*out['log_var'].float().exp()+.005**2).clamp_min(1e-8)
    nll=(.5*(error.square()/variance+variance.log()))[valid].mean()
    water=mask.bool()
    prior=((out['depth'].float()-out['z_prior'].detach())/50).abs()[water].mean()
    extinction=torch.relu(out['depth'].float()-out['z_ext'].view(-1,1,1,1)-1)[water].mean()/50
    total=reconstruction+.01*nll+.01*prior+.01*extinction
    return total,{'reconstruction':float(reconstruction.detach()),'nll':float(nll.detach()),
                  'prior':float(prior.detach()),'extinction':float(extinction.detach())}


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    build()
    if (ROOT/'results.json').exists(): return
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    with np.load(ROOT/'patches.npz') as data:
        X,M=data['X'],data['M']; groups={k:data[k] for k in ('train_idx','val_idx','test_idx')}
    dummy=np.zeros_like(M) # Ignored; no target enters optical_loss.
    base=PatchDataset(X,dummy,M,groups['train_idx'],y_mean=0,y_std=1)
    def loader(name):
        d=PatchDataset(X,dummy,M,groups[name],mean=base.mean,std=base.std,y_mean=0,y_std=1)
        return DataLoader(SpectralAugDataset(d,augment=False),batch_size=8,
                          sampler=CoverageSampler(len(d),42) if name=='train_idx' else None)
    train,val,test=(loader(n) for n in ('train_idx','val_idx','test_idx'))
    rinf=np.array([np.percentile(base.X[:,b][base.M>.5],2) for b in range(3)],dtype='float32')
    model=HybridBathNetModel(in_channels=16,mean=base.mean.squeeze(),std=base.std.squeeze(),
        mask_aware_physics=True,kd_aggregation='center',r_inf=rinf).cuda()
    initialize_optical_head(model)
    if '--smoke' in sys.argv:
        x,_,m,raw=next(iter(train))
        out=model(x.cuda(),m[:,None].cuda(),raw.cuda())
        loss,parts=optical_loss(out,raw.cuda(),m[:,None].cuda()); loss.backward()
        for name in ('encoder','decoder.depth_conv','decoder.logvar_conv','recon_head.kl_head','recon_head.al_head'):
            grads=[p.grad for n,p in model.named_parameters() if n.startswith(name) and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads)>0, name
        print('Sentinel-only finite gradient smoke check:',parts,flush=True)
        return
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,30,eta_min=3e-6)
    history=[]; best=float('inf'); stalled=0; start=1
    last=ROOT/'last_training.pt'
    if last.exists():
        state=torch.load(last,map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        history=state['history']; best=state['best']; stalled=state['stalled']; start=state['epoch']+1
        torch.set_rng_state(state['rng'].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        train.sampler.rng.set_state(state['sampler_rng'])
    def evaluate(data):
        model.eval(); err=0.; count=0; losses=[]; depths=[]
        with torch.no_grad():
            for x,_,m,raw in data:
                x,m,raw=x.cuda(),m[:,None].cuda(),raw.cuda()
                out=model(x,m,raw); loss,_=optical_loss(out,raw,m)
                valid=m.bool().expand_as(raw)
                err+=float(((out['r_hat'].float()-raw)**2)[valid].sum()); count+=int(valid.sum())
                losses.append(float(loss)); depths.extend(out['depth'][:,0,32,32].cpu().tolist())
        return {'composite_loss':float(np.mean(losses)),'reflectance_rmse':float(np.sqrt(err/count)),
                'depth_rmse_m':None,'latent_depth_range':[min(depths),max(depths)],
                'note':'Latent depth is optically non-identifiable; not validated physical depth.'}
    for epoch in range(start,31):
        model.train(); total=0.; n=0
        for x,_,m,raw in train:
            x,m,raw=x.cuda(),m[:,None].cuda(),raw.cuda()
            optimizer.zero_grad(); out=model(x,m,raw); loss,parts=optical_loss(out,raw,m)
            if not torch.isfinite(loss): raise FloatingPointError(parts)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step(); total+=float(loss.detach()); n+=1
        scheduler.step(); metrics=evaluate(val)
        history.append({'epoch':epoch,'train_loss':total/n,**metrics})
        if metrics['reflectance_rmse']<best:
            best=metrics['reflectance_rmse']; stalled=0
            torch.save({'model':model.state_dict(),'mean':base.mean,'std':base.std},ROOT/'best_model.pt')
        else: stalled+=1
        save(ROOT/'history.json',history)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'history':history,'best':best,'stalled':stalled,'rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'sampler_rng':train.sampler.rng.get_state()},last)
        print('Sentinel-only',epoch,metrics,flush=True)
        if stalled>=10: break
    model.load_state_dict(torch.load(ROOT/'best_model.pt',map_location='cuda',weights_only=False)['model'])
    results={'validation':evaluate(val),'test':evaluate(test),'epochs':history[-1]['epoch'],
             'depth_reference_loaded':False,'initialization':'random; no GEBCO-trained checkpoint',
             'selection':'validation reflectance RMSE','objective':'optical reconstruction + propagated optical NLL + heuristic prior + extinction',
             'warning':'Does not demonstrate absolute depth accuracy or calibrated depth uncertainty.'}
    save(ROOT/'results.json',results)
    (ROOT/'REPORT.md').write_text('# Sentinel-only self-supervised prototype\n\n'+
        f"Validation reflectance RMSE: {results['validation']['reflectance_rmse']:.6f}.\n\n"+
        f"Test reflectance RMSE: {results['test']['reflectance_rmse']:.6f}.\n\n"+
        'Depth RMSE in metres is unavailable: no independent depth labels were used. This is an optical consistency experiment. '
        'Heuristic priors constrain latent depth but do not identify its true metric scale. Neither aleatoric depth calibration '
        'nor superiority over the supervised model is established.\n',encoding='utf-8')


if __name__=='__main__': main()
