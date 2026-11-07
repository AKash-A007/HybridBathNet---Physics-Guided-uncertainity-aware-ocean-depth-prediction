"""Teach a small uncertainty correction using only spatial OOF residuals."""
from pathlib import Path
import sys
import json
import hashlib
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.gebco_oof_common import ROOT,SOURCE,update
from src.oof_uncertainty import SpatialVarianceCorrection,variance_nll
from src.uncertainty_calibration import group_weights
CHANNELS=[0,1,2,3,12,13,14,15,11]


def semantics(mean,sigma,kd,zext):
    return np.column_stack([mean,np.log(np.maximum(sigma,1e-6)),np.log(np.maximum(kd,1e-6)),zext]).astype('float32')


def ensemble_predictions(stem):
    means=[];sigmas=[];kds=[];exts=[];target=None
    for seed in (42,1,7):
        with np.load(SOURCE/f'seed_{seed}/{stem}_predictions.npz') as d:
            if target is None:target=d['target'].ravel().astype('float64')
            np.testing.assert_array_equal(target,d['target'].ravel())
            means.append(d['prediction'].ravel());sigmas.append(d['sigma'].ravel());kds.append(d['kd'].ravel());exts.append(d['z_ext'].ravel())
    means=np.array(means,dtype='float64');sigmas=np.array(sigmas,dtype='float64')
    a=np.mean(sigmas**2,axis=0);e=np.var(means,axis=0)
    return {'target':target,'prediction':means.mean(0),'sigma':np.sqrt(a+e),'aleatoric_variance':a,
        'epistemic_variance':e,'kd':np.mean(kds,axis=0),'z_ext':np.mean(exts,axis=0)}


def normalize(patch,scalar,stats):
    water=patch[:,8:9]>.5
    value=(patch-np.asarray(stats['mean'],dtype='float32')[None,:,None,None])/np.asarray(stats['std'],dtype='float32')[None,:,None,None]
    value[:,:8]*=water;value[:,8]=patch[:,8]
    semantic=(scalar-np.asarray(stats['semantic_mean'],dtype='float32'))/np.asarray(stats['semantic_std'],dtype='float32')
    return value.astype('float32'),semantic.astype('float32')


def main():
    out=ROOT/'risk_head';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():print('Head complete; skipping');return
    if not all((ROOT/f'fold_{k}/complete.json').exists() for k in range(3)):raise RuntimeError('OOF models incomplete')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if not (out/'source_snapshot.zip').exists():
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for p in (Path('src/oof_uncertainty.py'),Path(__file__),Path('scripts/report_gebco_oof_uncertainty.py')):archive.write(p,str(p))
    torch.set_num_threads(4);torch.manual_seed(20261107);np.random.seed(20261107)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    patches=[];scalars=[];variances=[];errors=[];ids=[]
    for k in range(3):
        root=ROOT/f'fold_{k}'
        with np.load(root/'features.npz') as d:
            held=d['held_idx'];x=d['X'][held][:,CHANNELS];idx=d['original_index'][held]
            y=d['Y'][held,32,32]
        with np.load(root/'held_predictions.npz') as p:
            np.testing.assert_array_equal(y,p['target'].ravel())
            pred=p['prediction'].ravel();sigma=p['sigma'].ravel()
            scalar=semantics(pred,sigma,p['kd'].ravel(),p['z_ext'].ravel())
        patches.append(x);scalars.append(scalar);variances.append(sigma**2);errors.append(y-pred);ids.append(idx)
    patch=np.concatenate(patches);scalar=np.concatenate(scalars);base=np.concatenate(variances);residual=np.concatenate(errors);ids=np.concatenate(ids)
    with np.load(SOURCE/'features/patches.npz') as d:
        np.testing.assert_array_equal(np.sort(ids),np.sort(d['train_idx']))
        coords=d['coords']+32;val_idx=d['val_idx'];val_patch=d['X'][val_idx][:,CHANNELS]
    # Risk normalization uses only the OOF training rows and their water pixels.
    mask=patch[:,8]>.5
    mean=np.array([patch[:,b][mask].mean() for b in range(9)],dtype='float32')
    std=np.array([patch[:,b][mask].std() for b in range(9)],dtype='float32');std[std<1e-5]=1
    mean[8]=0;std[8]=1
    sem_std=scalar.std(0);sem_std[sem_std<1e-5]=1
    stats={'mean':mean.tolist(),'std':std.tolist(),'semantic_mean':scalar.mean(0).tolist(),'semantic_std':sem_std.tolist()}
    xp,sp=normalize(patch,scalar,stats)
    _,groups=np.unique(coords[ids]//200,axis=0,return_inverse=True)
    weights=(len(ids)*group_weights(groups)).astype('float32')
    dataset=TensorDataset(*[torch.from_numpy(a.astype('float32')) for a in (xp,sp,base,residual,weights)])
    loader=DataLoader(dataset,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(20261107))
    val=ensemble_predictions('validation');vscalar=semantics(val['prediction'],val['sigma'],val['kd'],val['z_ext'])
    vx,vs=normalize(val_patch,vscalar,stats)
    validation=TensorDataset(torch.from_numpy(vx),torch.from_numpy(vs),torch.from_numpy((val['sigma']**2).astype('float32')),
        torch.from_numpy((val['target']-val['prediction']).astype('float32')))
    vloader=DataLoader(validation,batch_size=32)
    model=SpatialVarianceCorrection().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    protocol={'training':'447 spatial OOF single-model errors; three fresh full folds, with strict feature refitting',
        'validation':'existing208 outer-validation ensemble predictions; checkpoint selected by Gaussian variance NLL',
        'architecture':'9-channel small CNN + centre9 +4 semantic features, bounded log total-variance multiplier',
        'loss':'Gaussian NLL, equal2km training group weight','epochs_max':200,'patience':30,'seed':20261107,
        'zero_correction_control':'epoch0 identity is eligible for validation selection; never force a worse raw-NLL head',
        'final_depth_mean':'unchanged source three-member ensemble','component_interpretation':'conditional total-risk correction, not new identifiable aleatoric variance'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    update('training_oof_risk_head',oof_samples=len(ids))
    def validation_score():
        model.eval();values=[]
        with torch.no_grad():
            for x,s,b,r in vloader:
                lv=model(x.cuda(),s.cuda(),b.cuda());values.extend((.5*(lv+r.cuda().square()*torch.exp(-lv))).cpu().numpy().tolist())
        return float(np.mean(values))
    best=validation_score();bad=0
    history=[{'epoch':0,'training_nll_no_constant':None,'validation_nll_no_constant':best}]
    torch.save({'model':model.state_dict(),'stats':stats,'protocol':protocol,'epoch':0,'validation_nll_no_constant':best},out/'best.pt')
    for epoch in range(1,201):
        model.train();total=0
        for x,s,b,r,w in loader:
            x,s,b,r,w=(a.cuda() for a in (x,s,b,r,w))
            opt.zero_grad();lv=model(x,s,b);loss=variance_nll(lv,r,w)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite OOF risk loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(x)
        score=validation_score();history.append({'epoch':epoch,'training_nll_no_constant':total/len(ids),'validation_nll_no_constant':score})
        (out/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if score<best:
            best=score;bad=0
            torch.save({'model':model.state_dict(),'stats':stats,'protocol':protocol,'epoch':epoch,'validation_nll_no_constant':score},out/'best.pt')
        else:bad+=1
        if epoch%10==0:print('Risk head epoch',epoch,'val NLL',score,'best',best,flush=True)
        if bad>=30:break
    (out/'complete.json').write_text(json.dumps({'epochs':epoch,'best_validation_nll_no_constant':best,
        'oof_training_indices':ids.tolist(),'raw_oof_rmse':float(np.sqrt(np.mean(residual**2))),
        'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    update('oof_risk_head_complete',best_validation_nll_no_constant=best)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        update('oof_risk_head_failed',error=type(exc).__name__);raise
