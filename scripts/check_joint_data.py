from pathlib import Path
import sys
import numpy as np
p=Path('runs/2026-11-05/joint_v1/features/patches.npz')
with np.load(p) as d:
    for key in ('X','Y','M'):
        a=d[key]; bad=~np.isfinite(a)
        print(key,a.shape,'nonfinite',int(bad.sum()),flush=True)
        if key=='Y':
            print('nonfinite_on_valid_water',int((bad&(d['M']>.5)).sum()),
                  'nonfinite_centres',int(bad[:,32,32].sum()),flush=True)
    if '--repair-masked-storage' in sys.argv:
        arrays={k:d[k] for k in d.files}
        missing=~np.isfinite(arrays['Y'])
        assert not (missing&(arrays['M']>.5)).any()
        arrays['Y'][missing]=0
if '--repair-masked-storage' in sys.argv:
    backup=p.with_name('patches_before_invalid_label_fill.npz')
    if backup.exists(): raise RuntimeError('Backup already exists; inspect instead of overwriting')
    p.replace(backup)
    np.savez(p,**arrays)
    print('Filled masked-out missing-label storage only; original dataset preserved:',backup)
