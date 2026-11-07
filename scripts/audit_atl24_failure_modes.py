"""Audit frozen ATL24 labels/predictions; do not filter a benchmark or retrain."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path('runs/2026-11-06/atl24_hybrid_v1')
OUT = Path('runs/2026-11-06/physics_uncertainty_research_v1')
SOURCE = Path('runs/2026-11-05/nearshore_error_reference_audit_v1/atl24_archive_with_original_split.parquet')


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'n': len(values), **({key: float(value) for key, value in zip(
        ('p10', 'median', 'p90'), np.quantile(values, [.1, .5, .9]))} if len(values) else {})}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(SOURCE)
    good = ((raw.class_ph == 40) & (raw.confidence >= .6) & (raw.quality_ph == 0) &
            np.isfinite(raw.depth_m) & (raw.depth_m > 0) & (raw.depth_m < 100) &
            np.isfinite(raw.sigma_tvu) & (raw.sigma_tvu <= 1) & raw.inside_stack &
            raw.composite_valid_any_mask & raw.original_geo_split.isin(['train', 'val', 'test']))
    points = raw.loc[good].copy()
    points['date'] = points.index.strftime('%Y-%m-%d')
    points['year'] = points.index.year
    dates = json.loads(Path('runs/2026-11-05/nearshore_error_reference_audit_v1/quality_audit.json').read_text())['scene_dates']
    scene_days = pd.to_datetime(dates).to_numpy(dtype='datetime64[D]')
    days = points.index.to_numpy(dtype='datetime64[D]')
    points['closest_scene_days'] = np.abs(days[:, None] - scene_days).astype('timedelta64[D]').astype(int).min(1)
    points['pass_id'] = (points.rgt.astype(str) + ':' + points.cycle.astype(str) + ':' +
                         points['gt'].astype(str) + ':' + points.date)
    points['pixel_key'] = points.grid_row.astype(str) + ':' + points.grid_col.astype(str)
    grouped = points.groupby('pixel_key')
    labels = grouped.agg(depth=('depth_m', 'median'), photons=('depth_m', 'size'),
        dates=('date', 'nunique'), passes=('pass_id', 'nunique'),
        year_min=('year', 'min'), year_max=('year', 'max'),
        min_scene_gap_days=('closest_scene_days', 'min'), median_scene_gap_days=('closest_scene_days', 'median'),
        depth_min=('depth_m', 'min'), depth_max=('depth_m', 'max'),
        sigma_tvu=('sigma_tvu', 'median'), sigma_thu=('sigma_thu', 'median'),
        confidence=('confidence', 'median'), roughness=('surface_roughness', 'median'),
        surface_h=('surface_h', 'median'), ortho_h=('ortho_h', 'median'),
        gebco_depth=('gebco_depth_m', 'median'))
    pass_depth = points.groupby(['pixel_key', 'pass_id']).depth_m.median()
    labels['between_pass_depth_range_m'] = pass_depth.groupby(level=0).agg(lambda a: a.max() - a.min())
    with np.load(ROOT / 'features/patches.npz') as data:
        centres = data['coords'] + 32
        target = data['Y'][:, 32, 32].copy()
        split = {name: data[f'{name}_idx'].copy() for name in ('train', 'val', 'test')}
    keys = [f'{row}:{col}' for row, col in centres]
    sample = labels.loc[keys].reset_index()
    np.testing.assert_allclose(sample.depth, target, atol=1e-5)
    sample['row'], sample['col'] = centres[:, 0], centres[:, 1]
    sample['split'] = ''
    sample['prediction'], sample['sigma'] = np.nan, np.nan
    report = {'scope': 'Frozen ATL24 v1 data/model; descriptive audit only; no exclusions or training',
              'depth_arithmetic_max_abs_error_m': float(np.max(np.abs(raw.depth_m - (raw.surface_h - raw.ortho_h)))),
              'missing_quality_fields': [key for key in ('sensor_depth_exceeded', 'invalid_kd', 'invalid_wind_speed', 'low_confidence_flag') if key not in raw],
              'scene_dates': dates, 'splits': {}}
    for name, idx in split.items():
        sample.loc[idx, 'split'] = name
        if name != 'train':
            stem = 'validation' if name == 'val' else 'test'
            with np.load(ROOT / f'{stem}_predictions.npz') as data:
                np.testing.assert_allclose(data['target'].reshape(-1), target[idx], atol=1e-5)
                sample.loc[idx, 'prediction'] = data['prediction'].reshape(-1)
                sample.loc[idx, 'sigma'] = data['sigma'].reshape(-1)
        sub = sample.iloc[idx]
        used = points.loc[points.pixel_key.isin(sub.pixel_key)]
        entry = {'n': len(sub), 'rgt': sorted(int(v) for v in used.rgt.unique()),
                 'passes': int(used.pass_id.nunique()),
                 'groups_2km': int(len(np.unique(centres[idx] // 200, axis=0))),
                 'photon_count': stats(sub.photons), 'single_photon_pixels': int((sub.photons == 1).sum()),
                 'multiple_date_pixels': int((sub.dates > 1).sum()),
                 'no_2024_photon_pixels': int((sub.year_max < 2024).sum()),
                 'no_photon_within_30_days_any_scene_pixels': int((sub.min_scene_gap_days > 30).sum()),
                 'minimum_scene_gap_days': stats(sub.min_scene_gap_days),
                 'median_photon_scene_gap_days': stats(sub.median_scene_gap_days),
                 'repeat_pass_depth_range_m': stats(sub.loc[sub.passes > 1, 'between_pass_depth_range_m']),
                 'median_sigma_tvu_m': stats(sub.sigma_tvu), 'median_sigma_thu_m': stats(sub.sigma_thu),
                 'bins': {}}
        for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20.0001)):
            b = sub.loc[(sub.depth >= lo) & (sub.depth < hi)]
            row = {'n': len(b), 'target_depth_m': stats(b.depth),
                   'single_photon_pixels': int((b.photons == 1).sum()),
                   'minimum_scene_gap_days': stats(b.min_scene_gap_days)}
            if name != 'train' and len(b):
                error = b.prediction - b.depth
                row.update(rmse_m=float(np.sqrt(np.mean(error**2))), bias_m=float(error.mean()),
                           prediction_depth_m=stats(b.prediction), sigma_m=stats(b.sigma),
                           interval90_coverage=float(np.mean(np.abs(error) <= 1.645 * b.sigma)))
            entry['bins'][f'{lo}-{hi}'] = row
        report['splits'][name] = entry
    sample.to_csv(OUT / 'atl24_sample_audit.csv', index=False)
    used = points.loc[points.pixel_key.isin(sample.pixel_key)].copy()
    # Query manifest only: scene acquisition, quality selection and water-level
    # reconciliation have not yet been performed. Keep every frozen test label.
    manifest = used.groupby('pass_id', as_index=False).agg(
        date=('date','first'), rgt=('rgt','first'), cycle=('cycle','first'),
        beam=('gt','first'), photons=('depth_m','size'), pixel_count=('pixel_key','nunique'),
        splits=('original_geo_split',lambda x: ','.join(sorted(set(x)))),
        lon_min=('lon_ph','min'), lon_max=('lon_ph','max'),
        lat_min=('lat_ph','min'), lat_max=('lat_ph','max'))
    manifest['query_start'] = (pd.to_datetime(manifest.date)-pd.Timedelta(days=10)).dt.strftime('%Y-%m-%d')
    manifest['query_end'] = (pd.to_datetime(manifest.date)+pd.Timedelta(days=10)).dt.strftime('%Y-%m-%d')
    manifest['aoi_wkt'] = 'POLYGON((76.30 8.90,76.55 8.90,76.55 9.20,76.30 9.20,76.30 8.90))'
    manifest['requested_product'] = 'Sentinel-2 L1C for aquatic correction/visibility; no download yet'
    manifest.to_csv(OUT / 'atl24_scene_pairing_manifest.csv', index=False)
    report['pairing_manifest_passes'] = len(manifest)
    report['pairing_window_days'] = 10
    report['pairing_window_status'] = 'initial query hypothesis; no acquired/matched scenes or changed labels'
    (OUT / 'atl24_failure_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
