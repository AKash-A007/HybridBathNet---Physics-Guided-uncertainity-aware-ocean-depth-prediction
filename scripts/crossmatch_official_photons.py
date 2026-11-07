"""Compare immutable photon times across references, including non-bottom classes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path('runs/2026-11-06/recovery_v2')


def main():
    old=pd.read_parquet('runs/2026-11-05/nearshore_error_reference_audit_v1/atl24_archive_with_original_split.parquet')
    old=old.loc[old.class_ph.eq(40)].copy()
    old['beam']=old['gt'].map({10:'gt1l',20:'gt1r',30:'gt2l',40:'gt2r',50:'gt3l',60:'gt3r'})
    old['delta_time']=(old.index-pd.Timestamp('2018-01-01')).total_seconds()
    current=pd.read_parquet(ROOT/'official_atl24_aoi.parquet',columns=['rgt','cycle','beam','delta_time','class_ph','depth_m','confidence','lon_ph','lat_ph','ortho_h','surface_h'])
    matches=[];ambiguity=[]
    for key,a in old.groupby(['rgt','cycle','beam']):
        b=current.loc[current.rgt.eq(key[0])&current.cycle.eq(key[1])&current.beam.eq(key[2])].sort_values('delta_time')
        if not len(b):continue
        t=b.delta_time.to_numpy();u=a.delta_time.to_numpy()
        idx=np.searchsorted(t,u);left=np.clip(idx-1,0,len(t)-1);right=np.clip(idx,0,len(t)-1)
        idx=np.where(abs(t[left]-u)<=abs(t[right]-u),left,right)
        dist=abs(t[idx]-u)
        for original,new,dt,stamp in zip(a.itertuples(),b.iloc[idx].itertuples(),dist,u):
            if dt>1e-5:continue
            candidates=b.iloc[np.searchsorted(t,stamp-1e-5):np.searchsorted(t,stamp+1e-5,side='right')]
            distance=np.hypot((candidates.lat_ph-original.lat_ph)*111132,(candidates.lon_ph-original.lon_ph)*109800)
            candidates=candidates.loc[distance<=2]
            bottom=candidates.loc[candidates.class_ph.eq(40)]
            ambiguity.append({'candidates_within_time_position':len(candidates),'any_official_bottom':bool(len(bottom)),
                'official_bottom_height_difference_min_m':float(abs(bottom.ortho_h-original.ortho_h).min()) if len(bottom) else np.nan})
            matches.append({'rgt':int(key[0]),'cycle':int(key[1]),'beam':key[2],
                'delta_time_difference_s':float(dt),'old_depth_m':original.depth_m,
                'official_depth_m':new.depth_m,'official_class':int(new.class_ph),
                'old_confidence':original.confidence,'official_confidence':new.confidence,
                'approx_position_difference_m':float(np.hypot((original.lat_ph-new.lat_ph)*111132,(original.lon_ph-new.lon_ph)*109800)),
                'old_ortho_h':original.ortho_h,'official_ortho_h':new.ortho_h,
                'old_surface_h':original.surface_h,'official_surface_h':new.surface_h})
    m=pd.DataFrame(matches);m.to_csv(ROOT/'photon_time_crossmatch.csv',index=False)
    result={'scope':'same rgt/cycle/beam and nearest acquisition delta_time<=10microseconds; candidate photon identity, not certified source version',
        'old_bottom_photons':len(old),'matched':len(m),'official_classes':{str(k):int(v) for k,v in m.official_class.value_counts().items()},
        'interpretation':'Multiple photon returns within one pulse can share rounded delta_time and position. Nearest-time class counts cannot establish per-photon classification disagreement. Immutable public photon indices/source version are missing. Official V002 is verified provenance, not established better truth.'}
    both=m.loc[m.official_class.eq(40)&np.isfinite(m.old_depth_m)&np.isfinite(m.official_depth_m)]
    result['both_bottom_finite_n']=len(both)
    result['both_bottom_median_abs_depth_difference_m']=float(abs(both.old_depth_m-both.official_depth_m).median()) if len(both) else None
    close=m.loc[m.approx_position_difference_m<=2]
    result['time_and_position_within2m']={'n':len(close),'official_classes':{str(k):int(v) for k,v in close.official_class.value_counts().items()}}
    result['median_position_difference_m']=float(m.approx_position_difference_m.median())
    result['median_abs_surface_difference_m']=float(abs(close.old_surface_h-close.official_surface_h).median())
    result['median_abs_ortho_height_difference_m']=float(abs(close.old_ortho_h-close.official_ortho_h).median())
    amb=pd.DataFrame(ambiguity)
    result['candidate_ambiguity']={'median_candidates':float(amb.candidates_within_time_position.median()),
        'multiple_candidates_n':int(amb.candidates_within_time_position.gt(1).sum()),
        'any_bottom_candidate_n':int(amb.any_official_bottom.sum()),
        'median_abs_height_difference_when_bottom_candidate_m':float(amb.official_bottom_height_difference_min_m.median())}
    (ROOT/'photon_time_crossmatch.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
