"""Compute the same metrics for separate ATL24 diagnostic benchmarks."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.compute_study_report_metrics import ROOT,metric,domains,load


def main():
    out={};rows=[];manifest=[]
    for name,root,filename in (
        ('Public ATL24 v1','runs/2026-11-06/atl24_hybrid_v1','{stem}_predictions.npz'),
        ('Official V002 composite','runs/2026-11-06/recovery_v2/atl24_paired/composite_control','{stem}_acquisition_predictions.npz'),
        ('Official V002 matched/tide','runs/2026-11-06/recovery_v2/atl24_paired/matched_tide','{stem}_acquisition_predictions.npz')):
        out[name]={}
        with np.load(Path(root)/'features/patches.npz') as d:coords=d['coords']+32;split={s:d[k] for s,k in (('validation','val_idx'),('test','test_idx'))}
        for stem in split:
            file=Path(root)/filename.format(stem=stem)
            y,p,s=load(file);_,g=np.unique(coords[split[stem]]//200,axis=0,return_inverse=True)
            manifest.append({'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
            out[name][stem]={domain:metric(y,p,s,m,g,alea=s*s) for domain,m in domains(y).items()}
            for domain,a in out[name][stem].items():
                if a['n']:rows.append({'model':name,'split':stem,'domain_m':domain,**a})
    (ROOT/'atl24_appendix_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(rows).to_csv(ROOT/'atl24_appendix_metrics.csv',index=False)
    (ROOT/'atl24_prediction_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('ATL24 diagnostic appendix computed; references remain separate')


if __name__=='__main__':main()
