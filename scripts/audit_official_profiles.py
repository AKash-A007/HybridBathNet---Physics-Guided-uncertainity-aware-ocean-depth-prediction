"""Reference-only profile diagnostics; never filter held-out training results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path('runs/2026-11-06/recovery_v2')


def main():
    raw=pd.read_parquet(ROOT/'official_bottom_audited.parquet')
    good=raw.loc[raw.product_quality_eligible].copy()
    good['neighbor_count_25m']=0;good['local_line_residual_m']=np.nan
    for _,group in good.groupby(['granule','beam']):
        x=group.x_atc.to_numpy();z=group.depth_m.to_numpy()
        for i,index in enumerate(group.index):
            neighbors=(np.abs(x-x[i])<=25)&(np.arange(len(x))!=i)
            good.loc[index,'neighbor_count_25m']=int(neighbors.sum())
            if neighbors.sum()<3:continue
            a=np.column_stack([np.ones(neighbors.sum()),x[neighbors]-x[i]])
            pred=np.linalg.lstsq(a,z[neighbors],rcond=None)[0][0]
            good.loc[index,'local_line_residual_m']=float(z[i]-pred)
    good['sparse_profile']=good.neighbor_count_25m<3
    good['large_profile_residual']=good.local_line_residual_m.abs()>np.maximum(1,3*good.sigma_tvu)
    good.to_csv(ROOT/'profile_diagnostics.csv',index=False)
    report={'rule':'same pass/beam, leave-one-photon-out linear profile within25m; flag abs residual>max(1m,3sigma_tvu) when>=3 neighbors',
        'interpretation':'roughness/slope/classification proxy only; not independently verified bad labels; never used to remove a held point',
        'bins':{}}
    for lo,hi in ((0,5),(5,10),(10,15),(15,20.0001)):
        a=good.loc[(good.depth_m>=lo)&(good.depth_m<hi)]
        report['bins'][f'{lo}-{hi}']={'photons':len(a),'sparse_profile_pct':float(100*a.sparse_profile.mean()),
            'large_residual_pct_all':float(100*a.large_profile_residual.mean()),
            'assessable':int(a.local_line_residual_m.notna().sum())}
    (ROOT/'profile_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
