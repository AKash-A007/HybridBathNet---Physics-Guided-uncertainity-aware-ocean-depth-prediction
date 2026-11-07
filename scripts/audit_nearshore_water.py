"""Independent optical-water audit of the frozen nearshore dataset; no filtering."""
from pathlib import Path
import json
import numpy as np
import rasterio

ROOT=Path('runs/2026-11-05/nearshore_hybrid_v1')
with rasterio.open('runs/2026-11-04/processed/stack.tif') as src:
    green,nir=src.read([2,4])
with np.load(ROOT/'features/patches.npz') as d:
    coords=d['coords']+32; depth=d['Y'][:,32,32]
    ndwi=(green[coords[:,0],coords[:,1]]-nir[coords[:,0],coords[:,1]])/(green[coords[:,0],coords[:,1]]+nir[coords[:,0],coords[:,1]]+1e-6)
    report={}
    for name in ('train_idx','val_idx','test_idx'):
        idx=d[name]
        report[name]={'n':len(idx),'nonpositive_ndwi_count':int((ndwi[idx]<=0).sum()),
                     'nonpositive_ndwi_pct':float(100*np.mean(ndwi[idx]<=0)),
                     'bins':{f'{lo}-{hi}':{'n':int(((depth[idx]>=lo)&(depth[idx]<hi)).sum()),
                         'nonpositive_ndwi_count':int(((depth[idx]>=lo)&(depth[idx]<hi)&(ndwi[idx]<=0)).sum())}
                         for lo,hi in ((0,5),(5,10),(10,15),(15,25.01))}}
    report['interpretation']='NDWI is an optical screening proxy, not definitive truth; no samples or settings changed.'
(ROOT/'optical_water_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
