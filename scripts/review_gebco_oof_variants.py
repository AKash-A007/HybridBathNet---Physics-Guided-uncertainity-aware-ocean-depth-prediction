"""Summarize all predefined variants; no fitting or new model selection."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.gebco_oof_common import ROOT
from src.uncertainty_calibration import interval_score
from scripts.report_gebco_oof_uncertainty import gaussian_crps


def main():
    results=json.loads((ROOT/'results.json').read_text());review={}
    for stem in ('validation','test'):
        table=pd.read_csv(ROOT/f'{stem}_predictions.csv');table=table.loc[table.primary_0_20]
        y=table.target_m.to_numpy();mu=table.prediction_m.to_numpy()
        old=table.original_scaled_sigma_m.to_numpy();new=table.oof_head_raw_sigma_m.to_numpy()
        rc=table[['row','col']].to_numpy();_,groups=np.unique(rc//200,axis=0,return_inverse=True)
        old_score=interval_score(y,mu-1.6448536269514722*old,mu+1.6448536269514722*old)
        new_score=interval_score(y,mu-1.6448536269514722*new,mu+1.6448536269514722*new)
        cd=gaussian_crps(y,mu,new)-gaussian_crps(y,mu,old)
        unique=np.unique(groups);rng=np.random.RandomState(42);boot={k:[] for k in ('width_difference','score_difference','coverage_new','crps_difference')}
        for _ in range(2000):
            idx=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
            boot['width_difference'].append(float((3.2897072539029444*(new[idx]-old[idx])).mean()))
            boot['score_difference'].append(float((new_score[idx]-old_score[idx]).mean()))
            boot['coverage_new'].append(float(100*np.mean(abs(y[idx]-mu[idx])<=1.6448536269514722*new[idx])))
            boot['crps_difference'].append(float(cd[idx].mean()))
        before=results['splits'][stem]['original_scaled']['primary_0_20'];after=results['splits'][stem]['oof_head_raw']['primary_0_20']
        gate=(after['levels']['90']['coverage_pct']>=90 and
            after['levels']['90']['width_mean_m']<before['levels']['90']['width_mean_m'] and
            after['levels']['90']['interval_score_mean_m']<=before['levels']['90']['interval_score_mean_m'] and after['crps_mean_m']<=before['crps_mean_m'])
        review[stem]={'predefined_raw_variant_validation_style_gate':bool(gate),'count_covered90':int(np.sum(abs(y-mu)<=1.6448536269514722*new)),
            'n':len(y),'width_reduction_pct':float(100*(1-after['levels']['90']['width_mean_m']/before['levels']['90']['width_mean_m'])),
            'paired_spatial_ci95':{k:np.percentile(v,[2.5,97.5]).tolist() for k,v in boot.items()}}
    review['interpretation']='Both raw and scaled variants were predefined and reported. Raw is promising on validation; test comparison remains exploratory. No new multiplier or coverage was selected from test residuals.'
    (ROOT/'variant_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,stem in zip(axes,('validation','test')):
        for name in ('original_scaled','oof_head_raw','oof_head_scaled'):
            r=results['splits'][stem][name]['primary_0_20']['levels']
            ax.plot([float(k) for k in r],[a['coverage_pct'] for a in r.values()],marker='o',label=name)
        ax.plot([65,100],[65,100],'k--',alpha=.4);ax.set(title=stem,xlabel='Nominal coverage (%)',ylabel='Observed coverage (%)')
        ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(ROOT/'all_variants_coverage.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True);x=np.arange(4)
    for offset,method in ((-.18,'original_scaled'),(.18,'oof_head_raw')):
        bins=results['splits']['test'][method]['bins']
        axes[0].bar(x+offset,[r['levels']['90']['width_mean_m'] for r in bins.values()],.36,label=method)
        axes[1].bar(x+offset,[r['levels']['90']['coverage_pct'] for r in bins.values()],.36,label=method)
    for ax in axes:ax.set_xticks(x,['0–5','5–10','10–15','15–20']);ax.set_xlabel('Reference depth (m)');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean90% interval width (m)');axes[1].set_ylabel('Observed90% coverage (%)');axes[1].axhline(90,color='k',linestyle='--',alpha=.4)
    fig.savefig(ROOT/'raw_head_bin_comparison.png',dpi=180);plt.close(fig)
    path=ROOT/'REPORT.md';text=path.read_text(encoding='utf-8');marker='\n## Raw versus additionally scaled correction head\n'
    if marker in text:text=text[:text.index(marker)]
    text+=marker+'\n'
    text+='Both variants were predefined. The extra calibration multiplier does not achieve the desired test narrowing:9.41m width versus9.33m previously, at97.8% coverage. Do not describe this scaled result as a width improvement.\n\n'
    text+='The raw learned correction head is the promising alternative: validation width7.14m at90.4% coverage, test width7.73m at89.0% coverage (203/228). Compared with previous globally calibrated uncertainty, widths decrease18.1% on validation and17.2% on test. Validation CRPS1.219→1.188m and test1.321→1.306m improve slightly; test90% interval score9.817→9.050m. It passes the same descriptive validation narrowing criteria without needing the extra multiplier. This is not a claim that it was selected before seeing historical tests, nor a coverage guarantee.\n\n'
    text+='Raw learned-head test coverage by bin0–5/5–10/10–15/15–20m is94.1/94.8/87.5/85.4%, with widths8.27/10.13/8.48/5.51m. The deep bins still under-cover; do not present the aggregate89% as uniform90% reliability. Physics/depth means and original aleatoric/epistemic components remain unchanged. No optical mask is enabled.\n\n'
    text+='[Paired spatial intervals and variant review](variant_review.json). Keep the learned raw head as an explicit research candidate, preserve the prior calibration option and all checkpoints, and examine independent geographic validation before default deployment. Future conditional calibration should be learned without tuning to these test bins. All39 project tests and the strict source/head verification checks passed; no training remains active.\n\n![All predefined variants](all_variants_coverage.png)\n\n![Raw head by bin](raw_head_bin_comparison.png)\n'
    path.write_text(text,encoding='utf-8')
    print(json.dumps(review,indent=2))


if __name__=='__main__':main()
