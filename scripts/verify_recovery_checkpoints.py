"""Strict reload and same-device validation prediction check without training."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.hybridbathnet import HybridBathNetModel
ROOT=Path('runs/2026-11-06/recovery_v2')


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('Use the original CUDA evaluation device for numerical equivalence')
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    checks=[]
    tasks=[(ROOT/f'gebco_uq/seed_{seed}',ROOT/'gebco_uq/features/patches.npz','validation_predictions.npz') for seed in (42,1,7)]
    tasks += [(ROOT/f'atl24_paired/{arm}',ROOT/f'atl24_paired/{arm}/features/patches.npz','validation_scene_predictions.npz') for arm in ('composite_control','matched_tide')]
    for root,data,prediction_file in tasks:
        checkpoint=root/'hybridbathnet.pt'
        if not checkpoint.exists():raise RuntimeError(f'Incomplete checkpoint: {root}')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        options=saved['protocol']['model_options']
        model=HybridBathNetModel(in_channels=16,mean=saved['ch_mean'].squeeze(),std=saved['ch_std'].squeeze(),
            mask_aware_physics=True,kd_aggregation='center',physics_conditioning='spatial',**options).cuda().eval()
        model.load_state_dict(saved['model'],strict=True)
        with np.load(data) as d:
            indices=d['val_idx'][:8];raw=d['X'][indices];mask=d['M'][indices]
        normalized=(raw-saved['ch_mean'])/saved['ch_std']
        with torch.no_grad():
            out=model(torch.from_numpy(normalized).cuda(),mask=torch.from_numpy(mask[:,None]).cuda(),raw_reflectance=torch.from_numpy(raw[:,:3].copy()).cuda())
        pred=out['depth'][:,0,32,32].cpu().numpy();sigma=torch.exp(.5*out['log_var'][:,0,32,32]).cpu().numpy()
        with np.load(root/prediction_file) as d:
            expected=d['prediction'].reshape(-1)[:len(indices)];expected_sigma=d['sigma'].reshape(-1)[:len(indices)]
        np.testing.assert_allclose(pred,expected,rtol=1e-4,atol=5e-4)
        np.testing.assert_allclose(sigma,expected_sigma,rtol=1e-4,atol=5e-4)
        checks.append({'checkpoint':str(checkpoint),'strict_reload':True,'validation_points':len(indices),
            'max_abs_depth_difference_m':float(abs(pred-expected).max()),'max_abs_sigma_difference_m':float(abs(sigma-expected_sigma).max()),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'dataset_sha256_matches_protocol':hashlib.sha256(data.read_bytes()).hexdigest()==saved['protocol']['dataset_sha256']})
        if not checks[-1]['dataset_sha256_matches_protocol']:raise AssertionError('Dataset hash mismatch')
    packages={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','pandas','rasterio','pyarrow','h5py','pyTMD','timescale','lxml','netCDF4','requests')}
    report={'checks':checks,'packages':packages,'scope':'strict state reload; first8 validation points same CUDA device vs saved evaluation, not retraining or new model selection',
            'cpu_probe':'Initial CPU comparison differed by about1.5mm, exceeding the tight equivalence tolerance; final checks use original evaluation backend.'}
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
