"""Build compact, spatially separated data with training-only feature fitting."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.features import load_stack, extract_patches, stratified_spatial_patch_split
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='runs/2026-11-04/config.yaml')
    parser.add_argument('--output', default=None)
    parser.add_argument('--expand-training', action='store_true',
                        help='Recover patches inside existing training blocks; preserve held-out locations')
    args = parser.parse_args()
    cfg = load_config(args.config)
    arrays, transform, crs = load_stack(str(Path(cfg['preprocessing']['processed_dir']) / 'stack.tif'))
    valid = (arrays['valid_mask'] > .5) & (arrays['gebco_depth'] < 0)
    valid &= np.isfinite(arrays['gebco_depth'])
    for band in cfg['sentinel2']['bands']:
        valid &= np.isfinite(arrays[band])
    arrays['valid_mask'] = valid.astype(np.float32)
    ps, stride = cfg['features']['patch_size'], cfg['features']['patch_stride']
    coords = []
    for r in range(0, valid.shape[0] - ps + 1, stride):
        for c in range(0, valid.shape[1] - ps + 1, stride):
            if valid[r:r+ps, c:c+ps].mean() >= .5 and valid[r+ps//2, c+ps//2]:
                coords.append((r, c))
    center = np.array([-arrays['gebco_depth'][r+ps//2, c+ps//2] for r, c in coords])[:, None, None]
    splits = stratified_spatial_patch_split(coords, center, patch_size=ps,
            train_frac=cfg['features']['train_frac'], val_frac=cfg['features']['val_frac'])
    original_sizes = [len(s) for s in splits]
    if args.expand_training:
        # Keep the original held-out coordinates. Only use blocks already
        # represented in training, with a half-patch exclusion around holdouts.
        train_blocks = {(coords[i][0] // 128, coords[i][1] // 128) for i in splits[0]}
        forbidden = np.zeros_like(valid)
        margin = ps // 2
        for i in np.concatenate(splits[1:]):
            r, c = coords[i]
            forbidden[max(0,r-margin):r+ps+margin, max(0,c-margin):c+ps+margin] = True
        expanded = []
        for i, (r, c) in enumerate(coords):
            block = (r // 128, c // 128)
            if (block in train_blocks and r % 128 + ps <= 128
                    and c % 128 + ps <= 128
                    and not forbidden[r:r+ps,c:c+ps].any()):
                expanded.append(i)
        assert set(splits[0]).issubset(expanded)
        splits = (np.asarray(expanded, dtype=np.int64), splits[1], splits[2])
    fit_mask = np.zeros_like(valid)
    for idx in splits[0]:
        r, c = coords[idx]
        fit_mask[r:r+ps, c:c+ps] = True
    x, y, m, extracted = extract_patches(arrays, cfg['sentinel2']['bands'], ps, stride,
                                        fit_mask=fit_mask)
    assert coords == extracted
    # Assert zero shared pixels across splits, not just different patch IDs.
    occupancy = np.zeros_like(valid, dtype=np.uint8)
    for split_id, indices in enumerate(splits, 1):
        for idx in indices:
            r, c = coords[idx]
            region = occupancy[r:r+ps, c:c+ps]
            assert not np.any((region != 0) & (region != split_id))
            region[:] = split_id
    keep = np.sort(np.concatenate(splits))
    remap = {old: new for new, old in enumerate(keep)}
    out = Path(args.output or cfg['features']['features_dir'])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'patches.npz', X=x[keep], Y=y[keep], M=m[keep],
             coords=np.asarray(coords)[keep],
             **{name: np.array([remap[i] for i in idx], dtype=np.int64)
                for name, idx in zip(('train_idx','val_idx','test_idx'), splits)})
    metadata = {'split_sizes': [len(s) for s in splits], 'feature_fit': 'training pixels only',
                'original_split_sizes': original_sizes,
                'expanded_training': args.expand_training,
                'holdout_buffer_pixels': ps // 2 if args.expand_training else None,
                'shared_pixels_across_splits': 0, 'patch_size': ps, 'stride': stride,
                'channels': 16, 'normalization': 'training water pixels only',
                'limitations': 'GEBCO is spatially coarse; neighbouring patches may share coarse reference cells.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(metadata, flush=True)


if __name__ == '__main__':
    main()
