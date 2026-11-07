"""Validate held-region provenance and saved head reload; export study plots."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.gebco_oof_common import ROOT,SOURCE
from src.oof_uncertainty import SpatialVarianceCorrection
from scripts.train_gebco_oof_head import CHANNELS,normalize,semantics,ensemble_predictions


def main():
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    with np.load(SOURCE/'features/patches.npz') as source:
        support=source['native_support'];centres=source['coords']+32
        reserved=np.r_[source['calib_idx'],source['val_idx'],source['test_idx']]
    audits=[];held_ids=[]
    for k in range(3):
        root=ROOT/f'fold_{k}';meta=json.loads((root/'features.json').read_text())
        held_ids.extend(meta['original_held_indices'])
        assert not set(meta['original_train_indices'])&set(meta['original_held_indices'])
        train=np.asarray(meta['original_train_indices']);held=np.r_[meta['original_held_indices'],reserved]
        assert not set(support[train].ravel())&set(support[held].ravel())
        minimum=float(cKDTree(centres[held]).query(centres[train],p=np.inf)[0].min())
        assert minimum>=96
        sha=hashlib.sha256((root/'features.npz').read_bytes()).hexdigest()
        assert sha==meta['dataset_sha256']
        saved=torch.load(root/'hybridbathnet.pt',map_location='cpu',weights_only=False)
        assert saved['protocol']['dataset_sha256']==sha
        audits.append({'fold':k,'no_held_label_training':True,'dataset_hash_verified':True,
            'training':meta['train'],'held':meta['held'],'reserved_source_support_disjoint':True,
            'minimum_centre_gap_to_all_reserved_px':minimum})
    with np.load(SOURCE/'features/patches.npz') as d:
        np.testing.assert_array_equal(np.sort(held_ids),np.sort(d['train_idx']))
        idx=d['val_idx'][:32];patch=d['X'][idx][:,CHANNELS]
    p=ensemble_predictions('validation');s=semantics(p['prediction'][:32],p['sigma'][:32],p['kd'][:32],p['z_ext'][:32])
    saved=torch.load(ROOT/'risk_head/best.pt',map_location='cpu',weights_only=False)
    xp,sp=normalize(patch,s,saved['stats'])
    model=SpatialVarianceCorrection().cuda().eval();model.load_state_dict(saved['model'],strict=True)
    with torch.no_grad():
        sigma=torch.exp(.5*model(torch.from_numpy(xp).cuda(),torch.from_numpy(sp).cuda(),torch.from_numpy((p['sigma'][:32]**2).astype('float32')).cuda())).cpu().numpy()
    import pandas as pd
    table=pd.read_csv(ROOT/'validation_predictions.csv')
    np.testing.assert_allclose(sigma,table.oof_head_raw_sigma_m.to_numpy()[:32],atol=2e-5,rtol=1e-5)
    source_hashes={str(SOURCE/f'seed_{seed}/hybridbathnet.pt'):hashlib.sha256((SOURCE/f'seed_{seed}/hybridbathnet.pt').read_bytes()).hexdigest() for seed in (42,1,7)}
    previous=json.loads(Path('runs/2026-11-06/recovery_v2/verification.json').read_text())
    old={str(Path(a['checkpoint'])):a['checkpoint_sha256'] for a in previous['checks']}
    assert all(old[key]==sha for key,sha in source_hashes.items())
    report={'folds':audits,'oof_exactly_once':True,'source_ensemble_hashes_unchanged':True,
        'risk_head_strict_reload':True,'validation_points_checked':32,
        'max_abs_sigma_reload_difference_m':float(abs(sigma-table.oof_head_raw_sigma_m.to_numpy()[:32]).max()),
        'head_selected_epoch':saved['epoch'],'source_hashes':source_hashes}
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    a=json.loads((ROOT/'results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,stem in zip(axes,('validation','test')):
        for method in ('original_scaled','oof_head_scaled'):
            levels=a['splits'][stem][method]['primary_0_20']['levels']
            ax.plot([float(k) for k in levels],[v['coverage_pct'] for v in levels.values()],marker='o',label=method)
        ax.plot([65,100],[65,100],'k--',alpha=.4);ax.set(title=stem,xlabel='Nominal coverage (%)',ylabel='Observed coverage (%)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(ROOT/'coverage_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    labels=['0–5','5–10','10–15','15–20'];x=np.arange(4)
    for offset,method in ((-.18,'original_scaled'),(.18,'oof_head_scaled')):
        bins=a['splits']['test'][method]['bins']
        axes[0].bar(x+offset,[v['levels']['90']['width_mean_m'] for v in bins.values()],.36,label=method)
        axes[1].bar(x+offset,[v['levels']['90']['coverage_pct'] for v in bins.values()],.36,label=method)
    for ax in axes:ax.set_xticks(x,labels);ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2);ax.set_xlabel('Reference depth (m)')
    axes[0].set_ylabel('Mean90% interval width (m)');axes[1].set_ylabel('Observed90% coverage (%)');axes[1].axhline(90,color='k',linestyle='--',alpha=.4)
    fig.savefig(ROOT/'bin_width_coverage.png',dpi=180);plt.close(fig)
    path=ROOT/'REPORT.md';text=path.read_text();text+='\nStrict head reload, original ensemble checkpoint hashes, fold dataset hashes and exactly-once OOF coverage verified. [Verification](verification.json).\n\n![Coverage](coverage_comparison.png)\n\n![Width/coverage by bin](bin_width_coverage.png)\n'
    path.write_text(text,encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
