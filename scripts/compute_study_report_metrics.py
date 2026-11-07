"""Recompute full project-study metrics from immutable saved predictions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import ndtri,ndtr

ROOT=Path('reports/2026-11-06_project_study')
FULL=Path('runs/2026-11-05/nearshore_hybrid_v1')
UQ=Path('runs/2026-11-06/recovery_v2/gebco_uq')
OOF=Path('runs/2026-11-06/gebco_oof_uncertainty_v1')
ABL=Path('runs/2026-11-06/gebco_component_ablation_v1')


def domains(y):
    return {'0-5':(y>=0)&(y<5),'5-10':(y>=5)&(y<10),'10-15':(y>=10)&(y<15),
        '15-20':(y>=15)&(y<=20),'20-25':(y>20)&(y<=25),
        '0-10':(y>=0)&(y<=10),'0-15':(y>=0)&(y<=15),'0-20':(y>=0)&(y<=20),'0-25':(y>=0)&(y<=25)}


def metric(y,p,s,mask,groups,alea=None,epi=None,bootstrap=False):
    y,p=y[mask],p[mask];n=len(y)
    if not n:return {'n':0}
    e=p-y;absolute=abs(e);ss=float(np.sum(e*e));sst=float(np.sum((y-y.mean())**2))
    bias=float(e.mean());std=float(e.std(ddof=0));rmse=float(np.sqrt(ss/n))
    np.testing.assert_allclose(rmse**2,bias**2+std**2,rtol=1e-12,atol=1e-12)
    r={'n':n,'small_sample':n<10,'rmse_m':rmse,'mae_m':float(absolute.mean()),
        'r2':float(1-ss/sst) if sst>0 else None,'bias_m':bias,'error_std_population_m':std,
        'reference_std_population_m':float(y.std(ddof=0)),'prediction_std_population_m':float(p.std(ddof=0)),
        'median_absolute_error_m':float(np.median(absolute)),'p90_absolute_error_m':float(np.percentile(absolute,90)),
        'accuracy_within_1m_pct':float(100*np.mean(absolute<=1)),'accuracy_within_2m_pct':float(100*np.mean(absolute<=2)),
        'count_within_1m':int((absolute<=1).sum()),'count_within_2m':int((absolute<=2).sum()),
        'squared_error_sum_m2':ss,'mean_predicted_sigma_m':None,'mean_aleatoric_sigma_m':None,'mean_epistemic_sigma_m':None,
        'gaussian_moment_nll':None,'gaussian_moment_crps_m':None,'interval_score90_m':None}
    if s is not None:
        sigma=s[mask];assert (sigma>0).all() and np.isfinite(sigma).all()
        r['mean_predicted_sigma_m']=float(sigma.mean());r['rms_predicted_sigma_m']=float(np.sqrt(np.mean(sigma*sigma)))
        r['std_of_predicted_sigma_m']=float(sigma.std(ddof=0))
        for level in (.68,.8,.9,.95):
            z=ndtri((1+level)/2);key=str(round(level*100))
            r[f'coverage{key}_pct']=float(100*np.mean(absolute<=z*sigma));r[f'width{key}_m']=float(np.mean(2*z*sigma))
        z=(y-p)/sigma;phi=np.exp(-z*z/2)/np.sqrt(2*np.pi)
        r['gaussian_moment_nll']=float(np.mean(.5*np.log(2*np.pi*sigma*sigma)+.5*z*z))
        r['gaussian_moment_crps_m']=float(np.mean(sigma*(z*(2*ndtr(z)-1)+2*phi-1/np.sqrt(np.pi))))
        radius=ndtri(.95)*sigma
        r['interval_score90_m']=float(np.mean(2*radius+20*np.maximum(absolute-radius,0)))
    for key,value in (('aleatoric',alea),('epistemic',epi)):
        if value is not None:r[f'mean_{key}_sigma_m']=float(np.sqrt(value[mask]).mean())
    if bootstrap:
        g=groups[mask];unique=np.unique(g);rng=np.random.RandomState(42);values=[]
        for _ in range(2000):
            b=np.concatenate([np.flatnonzero(g==group) for group in rng.choice(unique,len(unique),replace=True)])
            values.append(np.sqrt(np.mean(e[b]**2)))
        r['rmse_spatial_ci95_m']=np.percentile(values,[2.5,97.5]).tolist();r['spatial_groups']=len(unique)
    return r


def load(path,valid_sigma=True):
    with np.load(path) as d:
        y=d['target'].ravel().astype('float64');p=d['prediction'].ravel().astype('float64')
        sigma=d['sigma'].ravel().astype('float64') if valid_sigma else None
    assert np.isfinite(y).all() and np.isfinite(p).all()
    return y,p,sigma


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'evaluation_metrics.json').exists():raise RuntimeError('Preserve completed study metrics')
    with np.load(FULL/'features/patches.npz') as d:
        centres=d['coords']+32;split={s:d[k] for s,k in (('validation','val_idx'),('test','test_idx'))}
    flat=[];report={};manifest=[]
    models=[('Full v1','matched637',FULL,True),('No physics','matched637',ABL/'no_physics',True),
        ('No uncertainty','matched637',ABL/'no_uncertainty',False),('Neither','matched637',ABL/'no_both',False),
        ('Integrated head candidate','matched637',Path('runs/2026-11-06/integrated_head_research_v1'),True)]
    for name,path in (('No extinction','runs/2026-11-05/nearshore_no_ext_v2'),('Squared depth loss','runs/2026-11-05/nearshore_squared_depth_v3')):
        if (Path(path)/'test_predictions.npz').exists():models.append((name,'matched637',Path(path),True))
    for seed in (42,1,7):models.append((f'Ensemble member seed{seed}','research447',UQ/f'seed_{seed}',True))
    references={}
    def add(name,family,stem,y,p,sigma,alea=None,epi=None):
        if stem in references:np.testing.assert_allclose(y,references[stem],atol=1e-12,rtol=0)
        else:references[stem]=y.copy()
        _,g=np.unique(centres[split[stem]]//200,axis=0,return_inverse=True)
        summary={domain:metric(y,p,sigma,m,g,alea,epi,bootstrap=(name in ('Full v1','No physics','No uncertainty','Neither','Ensemble raw','OOF raw') and domain in ('0-10','0-20','0-5','5-10','10-15','15-20'))) for domain,m in domains(y).items()}
        primary=summary['0-20']['squared_error_sum_m2']
        for domain in ('0-5','5-10','10-15','15-20','0-10'):
            summary[domain]['share_of_primary_squared_error_pct']=float(100*summary[domain]['squared_error_sum_m2']/primary)
        report.setdefault(name,{'family':family,'splits':{}})['splits'][stem]=summary
        for domain,r in summary.items():
            if r['n']:flat.append({'model':name,'family':family,'split':stem,'domain_m':domain,**{k:v for k,v in r.items() if not isinstance(v,list)}})
    for name,family,path,valid in models:
        for stem in ('validation','test'):
            file=path/f'{stem}_predictions.npz'
            if not file.exists():continue
            y,p,sigma=load(file,valid)
            add(name,family,stem,y,p,sigma,alea=sigma*sigma if valid else None)
            manifest.append({'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    for stem in ('validation','test'):
        file=UQ/f'ensemble3_{stem}_predictions.npz'
        with np.load(file) as d:
            y,p,sigma=(d[k].ravel() for k in ('target','prediction','sigma'));a=d['aleatoric_variance'];e=d['epistemic_variance']
        add('Ensemble raw','research447',stem,y,p,sigma,a,e)
        table=pd.read_csv(OOF/f'{stem}_predictions.csv')
        np.testing.assert_allclose(p,table.prediction_m,atol=1e-12);np.testing.assert_allclose(y,table.target_m,atol=1e-12)
        for name,column in (('Ensemble scaled','original_scaled_sigma_m'),('OOF raw','oof_head_raw_sigma_m'),('OOF scaled','oof_head_scaled_sigma_m')):
            add(name,'research447',stem,y,p,table[column].to_numpy())
        manifest.append({'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        f=OOF/f'{stem}_predictions.csv';manifest.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    (ROOT/'evaluation_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(flat).to_csv(ROOT/'evaluation_metrics.csv',index=False)
    (ROOT/'prediction_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    dataset={}
    for label,path in (('matched637',FULL/'features/patches.npz'),('research447',UQ/'features/patches.npz')):
        with np.load(path) as d:
            y=d['Y'][:,32,32]
            dataset[label]={key:{'n':len(d[key]),'bins':{b:int(mask[d[key]].sum()) for b,mask in domains(y).items() if b in ('0-5','5-10','10-15','15-20','20-25','0-10')},
                'target_mean_m':float(y[d[key]].mean()),'target_sd_m':float(y[d[key]].std())}
                for key in d.files if key.endswith('_idx')}
    (ROOT/'dataset_counts.json').write_text(json.dumps(dataset,indent=2),encoding='utf-8')
    print(pd.DataFrame(flat).query("domain_m=='0-10' and model in ['Full v1','No physics','No uncertainty','Neither','Ensemble raw','OOF raw']")[[
        'model','split','n','rmse_m','mae_m','r2','accuracy_within_1m_pct','accuracy_within_2m_pct','bias_m','error_std_population_m']].to_string(index=False))


if __name__=='__main__':main()
