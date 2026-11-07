"""Build a study guide from independently recalculated metrics and layer audit."""
from pathlib import Path
import json
import html
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import rasterio
from pyproj import Transformer

ROOT=Path('reports/2026-11-06_project_study')
METRICS=json.loads((ROOT/'evaluation_metrics.json').read_text())
ARCH=json.loads((ROOT/'architecture_audit.json').read_text())
COUNTS=json.loads((ROOT/'dataset_counts.json').read_text())


def r(model,split='test',domain='0-20'):return METRICS[model]['splits'][split][domain]
def f(value,digits=3):return 'N/A' if value is None else f'{value:.{digits}f}'
def link(path):return '../../'+str(path).replace('\\','/')
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows]+['']


def figures():
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    bins=['0-5','5-10','10-15','15-20'];x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for offset,name in ((-.18,'Full v1'),(.18,'Ensemble raw')):
        axes[0].bar(x+offset,[r(name,domain=b)['rmse_m'] for b in bins],.36,label=name)
    axes[0].set_xticks(x,['0–5','5–10','10–15','15–20']);axes[0].set(xlabel='Reference depth bin (m)',ylabel='Test RMSE (m)',title='Same test locations, different training protocols')
    axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    axes[1].bar(x,[r('Full v1',domain=b)['share_of_primary_squared_error_pct'] for b in bins])
    axes[1].set_xticks(x,['0–5\nn=17','5–10\nn=58','10–15\nn=64','15–20\nn=89'])
    axes[1].set(ylabel='Share of 0–20 m squared error (%)',title='Why the shallow-only RMSE is lower')
    axes[1].grid(axis='y',alpha=.2);fig.savefig(folder/'depth_error_analysis.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    for ax,(name,path) in zip(axes,[('Full v1','runs/2026-11-05/nearshore_hybrid_v1/test_predictions.npz'),('Ensemble raw','runs/2026-11-06/recovery_v2/gebco_uq/ensemble3_test_predictions.npz')]):
        with np.load(path) as d:y=d['target'].ravel();p=d['prediction'].ravel()
        m=(y>=0)&(y<=20);scatter=ax.scatter(y[m],p[m],c=y[m],s=16,cmap='viridis',vmin=0,vmax=20)
        ax.plot([0,22],[0,22],'k--',alpha=.5);ax.set(title=name,xlabel='GEBCO reference depth (m)',ylabel='Predicted depth (m)',xlim=(0,22),ylim=(0,22));ax.grid(alpha=.2)
    fig.colorbar(scatter,ax=axes,label='Reference depth (m)',shrink=.8);fig.savefig(folder/'prediction_scatter.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,(label,path) in zip(axes,[('637-train matched study','runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz'),('447-train uncertainty study','runs/2026-11-06/recovery_v2/gebco_uq/features/patches.npz')]):
        with np.load(path) as d:
            coords=d['coords']+32
            for key,color in (('train_idx','tab:blue'),('calib_idx','tab:orange'),('val_idx','tab:green'),('test_idx','tab:red')):
                if key not in d.files:continue
                rc=coords[d[key]]
                ax.scatter(rc[:,1]*.01,-rc[:,0]*.01,s=7,c=color,label=f'{key[:-4]} n={len(rc)}',alpha=.75)
        ax.set(title=label,xlabel='Eastward grid position (km)',ylabel='Northward relative grid position (km)');ax.legend(fontsize=7);ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.savefig(folder/'sample_geography.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,7),constrained_layout=True);ax.set_xlim(0,13);ax.set_ylim(0,8.6);ax.axis('off')
    def box(x,y,text,color='#e7f0fb',w=3.0,h=.72):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor=color,edgecolor='#4b6179'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(a,b,color='#506070'):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.4})
    box(.2,7.1,'Input: B × 16 × 64 × 64')
    enc=[('Encoder1: 32 × 64 × 64',6),('Encoder2: 64 × 32 × 32',4.8),('Encoder3: 96 × 16 × 16',3.6),('Bottleneck: 128 × 8 × 8',2.4)]
    for text,y in enc:box(.2,y,text)
    for top,bottom in ((7.1,6),(6,4.8),(4.8,3.6),(3.6,2.4)):arrow((1.7,top),(1.7,bottom+.72))
    for text,y in [('Decoder3: 192 × 16 × 16',3.6),('Decoder2: 128 × 32 × 32',4.8),('Decoder1: 64 × 64 × 64',6)]:box(5.1,y,text)
    arrow((3.2,2.75),(6.6,3.6));arrow((6.6,4.32),(6.6,4.8));arrow((6.6,5.52),(6.6,6))
    for y in (3.6,4.8,6):arrow((3.2,y+.36),(5.1,y+.36))
    ax.text(4.15,6.65,'skip connections',ha='center',fontsize=8)
    box(9.1,6.5,'Depth head: 64 → 1\nSoftplus, centre depth',color='#e4f4e5');box(9.1,5.25,'Variance head: 64 → 1\nσ = exp(log_var / 2)',color='#fff1d6')
    arrow((8.1,6.36),(9.1,6.86));arrow((8.1,6.36),(9.1,5.61))
    box(8.7,2.9,'Reconstruction: 3-band K/A\nUses encoder1 skip + depth',color='#e7f6f3',w=3.8)
    ax.plot([12.1,12.7,12.7],[6.86,6.86,3.26],color='#506070',lw=1.4);arrow((12.7,3.26),(12.5,3.26));ax.text(12.8,4.5,'depth',fontsize=8,rotation=90)
    ax.plot([.2,.05,.05,10.6],[6.36,6.36,1.9,1.9],color='#3e8b86',lw=1.1,linestyle='--')
    arrow((10.6,1.9),(10.6,2.9),color='#3e8b86');ax.text(6.5,2.03,'32-channel encoder1 skip',color='#28746f',fontsize=8,ha='center')
    box(3.9,.6,'Physics context: Kd proxy + local prior + mask\nFiLM at all four encoder scales',color='#f1e8fa',w=6,h=1.0)
    arrow((3.9,1.1),(1.7,2.4),color='#8755a0')
    ax.text(6.5,8.3,'Verified full model: spatial uncertainty head replaces the linear variance branch in later runs',ha='center',fontsize=10)
    ax.text(6.5,.1,'Shapes omit batch B inside the network. Full layer inventory and exact active parameter counts accompany this report.',ha='center',fontsize=8)
    fig.savefig(folder/'architecture.png',dpi=180);plt.close(fig)


def evidence():
    quality=json.loads(Path('runs/2026-11-05/nearshore_error_reference_audit_v1/quality_audit.json').read_text())
    with np.load('runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz') as d:coords=d['coords'][d['test_idx']]+32
    with np.load('runs/2026-11-05/nearshore_hybrid_v1/test_predictions.npz') as d:y=d['target'].ravel();p=d['prediction'].ravel()
    m=(y>=10)&(y<15);groups=coords[m]//200;error=(p-y)[m].astype('float64')
    _,inverse=np.unique(groups,axis=0,return_inverse=True);sums=np.array([np.sum(error[inverse==g]**2) for g in np.unique(inverse)])
    ev={'mid_bin_top4_spatial_group_sse_share_pct':float(100*np.sort(sums)[-4:].sum()/sums.sum()),
        'cloud_medians':{key:{field:value['proxy_summary'][field]['median'] for field in ('scl_clear_scene_count','scl_cloud_shadow_scene_count')} for key,value in quality['splits']['test'].items()}}
    (ROOT/'analysis_evidence.json').write_text(json.dumps(ev,indent=2),encoding='utf-8');return ev


def report():
    ev=evidence();lines=[]
    def add(text):lines.extend(text.strip().splitlines());lines.append('')
    add('''# HybridBathNet project study report

Dataset, preprocessing, verified architecture, evaluation and evidence-based interpretation

Prepared from preserved source code, checkpoints and prediction files on 6 November 2026. This is a study guide for the current GEBCO project and the uncertainty research, not a claim of independently surveyed seabed accuracy.

## Reading guide

1. Understand the data and reference limitations before comparing scores.
2. Follow preprocessing and the 16-channel architecture.
3. Read the metric definitions, then the separate validation/test tables.
4. Use the interpretation section to distinguish facts from plausible causes.
5. Review the appendix for complete metrics, layer inventory and reproducibility links.

## 1. What the project does

HybridBathNet predicts coastal depth from Sentinel-2 patches using a U-Net, physics-conditioned features and an uncertainty head. Optical reconstruction adds a physics-related training objective. Independently trained full models provide ensemble disagreement; later spatially out-of-fold residual learning corrects total uncertainty without changing the predicted depth.

Two configurations must remain distinct: **the selected single full model**, trained on 637 patches, and **the uncertainty research ensemble**, trained on 447 patches with a separate 123-point calibration set. The selected single model has better validation depth RMSE; the research ensemble has numerically lower overall test RMSE but worse shallow errors. The new uncertainty head is a research option, not an automatically deployed replacement.

The name full v1 below is the preserved nearshore experiment label, not a claim that it is the older six-channel architecture. Both current compared source configurations have16 input channels.

The numbers called six versus sixteen in earlier discussion refer to input-channel counts, not a verified network-layer count. The present architecture audit strictly reloads the actual saved models and traces active modules.

## 2. Dataset and study area

The AOI is longitude76.30–76.55°E, latitude8.90–9.20°N; processing CRS is EPSG:32643. The rebuilt stack is3329×2736 pixels on an approximately10m grid. This is the same study/test geography across the current GEBCO comparisons.

Sentinel-2 L2A supplies B02/blue, B03/green, B04/red and B08/NIR, all nominally10m bands. The verified acquisition-date audit contains12 dates from13 January to18 March2024;16 tile products were present. A Jan–Apr catalogue request is not16 independent temporal training datasets. The model uses the temporal composite/std channels, not a separate supervised sample for each acquisition. [Official band information](https://sentiwiki.copernicus.eu/web/s2-mission).

Depth labels come from the local GEBCO_2020 NetCDF, whose native coordinate spacing was verified as0.004166667° (15arcsec), roughly460m here. Reprojection to10m interpolates this coarse surface; it does not create measured10m bathymetry. The source contains mixed data origins, and local Type Identifier provenance has not been verified. [GEBCO grid development](https://www.gebco.net/data-products/gridded-bathymetry-data/grid-development).

The nearshore shelf eligibility screen retains negative GEBCO elevations down to−25m before patch extraction. Training/validation targets are limited to20m; test20–25m is explicitly extrapolation sensitivity. A sample is a64×64 image patch with a supervised **centre depth**, not a new independent survey. Patch stride is32 pixels; overlapping training patches and adjacent coarse labels remain correlated. Source-support counts are not counts of independent measured soundings.
''')
    lines+=table(['Dataset/split','n','0–5','5–10','10–15','15–20','20–25'],[
        [family+'/'+key.replace('_idx',''),v['n'],*[v['bins'][b] for b in ('0-5','5-10','10-15','15-20','20-25')]] for family,data in COUNTS.items() for key,v in data.items()])
    add('''Bins are [0,5),[5,10),[10,15),[15,20],(20,25]; requested cutoff0–10 means0≤depth≤10. Exactly10m training labels therefore belong to the10–15 bin but are also included in the cutoff analysis: inclusive shallow counts are189/637 and82/447. There are no exactly10m validation/test labels here. Exactly20m values stay inside the0–20 benchmark; legacy strict-bin metadata can otherwise look as though some are >20m.

The matched study has828/261/413 distinct native interpolation-support cells in train/validation/test. Cross-split support sets are disjoint, with a32-pixel image-footprint gap (96-pixel centre separation). The research calibration study reserves15 approximately2km training-region groups, removes nearby/shared-support training candidates and refits features. It retains the same208 validation and343 total test locations. Only228 test labels are0–20m and75 are0–10m.

Of the original637 training candidates,123 become calibration samples and67 are excluded by additional buffers/source-support separation, leaving447 for learning. Some excluded rows remain stored in the feature archive for provenance but do not enter training/calibration loaders.

![Sample geography](figures/sample_geography.png)

## 3. Preprocessing and feature engineering

1. **Input verification:** existing catalogue/tile files were checked; the recorded manifest audit verified1376 listed files. The original ZIP archive equality was not established, so do not claim a remote archive checksum comparison.
2. **Radiometry:** for the2024 SAFE processing path, reflectance is implemented as `max((DN−1000)/10000,0)`. These are fixed input-product conventions, not fitted depth parameters. Offsets depend on processing metadata for other products; this formula is not a universal instruction for all Sentinel files. [Official product/offset description](https://sentiwiki.copernicus.eu/web/s2-products).
3. **Alignment:** spectral bands use bilinear reprojection onto the AOI grid; SCL uses nearest-neighbour reprojection.
4. **Cloud/validity screen:** source code accepts SCL classes4/5/6/7 plus positive blue reflectance. This is a usable-clear-pixel mask, **not water-only SCL6**. The later GEBCO shelf screen supplies the coastal eligibility mask; coarse shoreline mixing can remain.
5. **Temporal aggregation:** masked median reflectance plus temporal standard deviation, `ddof=1`; std with fewer than two usable observations is filled with0. Such0 values mean unavailable variability estimates, not proven zero sensor noise. A median composite is not guaranteed cloud-free or contemporaneous with seabed reference measurements.
6. **GEBCO alignment:** the repaired path warps native NetCDF directly, bilinearly, preserves missing coverage as NaN, and converts negative underwater elevation into positive depth for labels. It does not perform a tidal/datum validation of GEBCO against local survey truth.
7. **Train-only engineering:** fit visible-band versus NIR glint slopes on training fit pixels, using the5th percentile NIR as the reference. Estimate visible reference offsets at the2nd percentile of training reflectance. These are statistical proxies, not verified deep-water measurements. Compute log/ratio/prior channels from those references.
8. **Patch construction and splitting:** shelf eligibility precedes extraction; retain centre labels and geographic blocks, apply native-support and footprint exclusions, and fit normalization on training water pixels only. New patches are distinct centres; evaluation samples were not duplicated.
9. **Training augmentation:** spectral jitter and flips; unjittered visible reflectance follows the spatial transform for physics reconstruction. Invalid/deep moved centres are rejected by the shallow augmentation wrapper.
''')
    channels=[('0–2','Glint-corrected B02/B03/B04'),('3','B08 NIR'),('4–6','Three Lyzenga-style visible log channels'),('7–8','Two log-channel differences/ratios'),('9','Stumpf log-band ratio feature'),('10','Beer–Lambert-style heuristic depth prior'),('11','Shelf/valid-water eligibility mask'),('12–15','Four spectral temporal-standard-deviation channels')]
    lines+=table(['Zero-based channels','Meaning'],channels)
    add('''The Stumpf ratio is an input feature; no separate Stumpf baseline was trained in this work. The engineered prior uses training reference statistics. The model also computes a separate, local masked95th-percentile green reference for its internal physics context at inference. Local image statistics are not fitting to held-out depth labels.

## 4. Architecture and layer information

![Verified architecture](figures/architecture.png)

The encoder combines two-convolution stem blocks, residual blocks, squeeze-and-excitation channel attention and dropout0.1. Four FiLM blocks modulate encoder features with projected `[Kd proxy, local depth prior, mask]`. Each stem uses Conv3×3→BatchNorm→ReLU→Dropout→Conv3×3→BatchNorm→ReLU→SE; each residual block uses two3×3 convolutions with normalization, dropout, SE and an identity addition. Max pooling halves resolution three times.

Decoder transposed convolutions upsample, concatenate the matching encoder skip, and apply a residual block. A1×1 depth convolution followed by Softplus gives non-negative depth. The variance output is clamped log variance, with `sigma=exp(log_var/2)` in metres. Dense outputs exist, but depth/variance supervision and scalar reporting use the centre pixel; the report does not claim dense surveyed-depth training.
''')
    names=['encoder.stem1','encoder.pool1','encoder.stem2','encoder.pool2','encoder.stem3','encoder.pool3','encoder.bottleneck','decoder.up3','decoder.dec3','decoder.up2','decoder.dec2','decoder.up1','decoder.dec1','decoder.depth_conv','decoder.logvar_conv']
    lines+=table(['Stage','Traced output shape, batch B=1'],[[name,str(ARCH['Full v1']['stages'][name]['output'])] for name in names])
    lines+=table(['Configuration','Registered parameters','Parameters in executed leaves','Active Conv2d','Active ConvTranspose2d','Active Linear'],[
        [name,v['parameters_registered'],v['parameters_in_executed_leaves'],v['executed_leaf_types'].get('Conv2d',0),v['executed_leaf_types'].get('ConvTranspose2d',0),v['executed_leaf_types'].get('Linear',0)] for name,v in ARCH.items()])
    add('''The full model executes33 Conv2d,3 ConvTranspose2d and22 Linear modules:58 parameterized convolution/linear modules, spread across branches rather than58 sequential depth layers. It also executes22 BatchNorm,11 Dropout2d and11 SE global-pooling modules. Functional operations such as interpolation, concatenation, residual additions, Softplus and reconstruction arithmetic are not counted as separate registered layers. The complete leaf-by-leaf type, specification, parameter count and input/output trace is in[layer_inventory.csv](layer_inventory.csv).

The later spatial variance head is64→32(3×3)→GroupNorm8→SiLU→16(3×3)→SiLU→1(1×1). Its old65-parameter linear variance projection remains registered but unexecuted; this explains the difference between registered and active parameters. Three ensemble members are three independently trained full networks, not three added depth layers. Old standalone MLP head classes and dense-loss classes in source are not the active current training path.

The separate4441-parameter OOF correction head uses two small16-channel convolutions with GroupNorm/SiLU, pooling, centre features and four semantic values (depth, log sigma, log Kd, z_ext). It predicts a bounded log multiplier of **total** predictive variance. Source inputs are detached; it cannot alter the original ensemble depth weights.

## 5. Physics: what is implemented and what is not established

There are two different attenuation quantities:

* The empirical context proxy is `K=clip(0.0166+0.156*(B/G)^(-1.14),0.01,5)` plus a red ratio term `clip(0.430*R/G,0,3)`, followed by an overall clamp. With centre aggregation, `z_ext=log(100)/(2*K_centre+epsilon)`, clipped to1–50m. This is one combined empirical map, not independently measured blue/green/red Kd retrievals.
* The reconstruction branch learns three positive effective coefficients and three bounded amplitudes. **It receives the32-channel first encoder skip**, plus predicted depth and fixed training reference reflectance; it does not receive the final64-channel decoder tensor.

Its simplified reconstruction is:

```text
R_hat(lambda) = R_inf(lambda)*(1-exp(-2*K_eff(lambda)*z))
                + A(lambda)*exp(-2*K_eff(lambda)*z)
```

The coefficients are effective half-two-way attenuation parameters. Fitting this expression is not proof that true Kd, bottom reflectance or a metre-valued optical limit has been identified: depth and attenuation can compensate through their product. The training-percentile R_inf is not a measured deep-water spectrum. The extinction ratio100 is an assumption.

A published independent optical-visibility classifier was numerically checked and applied in the recovery study, but retained zero validation/test points at its recommended shallow-support threshold on the tested image. That is a failure of local usable-coverage validation, not proof that every bottom is invisible. No optical mask was enabled. The physical expectation that depth retrieval depends on clarity/bottom visibility motivates investigation; it cannot alone diagnose our model errors. [IHO SDB guide](https://iho.int/uploads/user/pubs/bathy/B_13_Ed100_032024.pdf).

## 6. Training objective and experiment identities

The selected full nearshore run uses:

```text
L = normalized-centre Huber
    + annealed beta-NLL
    + 0.03 * mean(ReLU(z_pred-z_ext-1m)) / training_target_SD
    + 0.1 * masked reflectance reconstruction Huber
```

The normalized Huber delta is0.5; reconstruction is scaled by0.05 reflectance. The variance NLL uses metre residuals and metre-squared variance, with beta0.5 in the selected full model. NLL weight warms up over20 epochs. Non-negativity is provided by Softplus; the separate non-negativity loss weight is0. The unused low-resolution pooled/dense depth loss is not part of this active objective.

Training uses AdamW(lr0.001,weight_decay1e-4), batch8, EMA0.999, one180-epoch cosine period, gradient clipping1 and patience30. Sampling covers training examples without replacement. Validation RMSE selects the depth checkpoint. The original full v1 selected epoch118 and stopped148. All relevant losses/metrics remained finite; invalid numerical conditions are checked explicitly.

In the later447-point ensemble, beta is0, NLL mean/features are detached and variance gradients are clipped separately; full depth/physics reconstruction remains active. These changes and the training/calibration allocation happen together, so this is not an isolated uncertainty-head ablation.

The OOF study trains three fresh full fold models on188/236/262 points, generates exactly447 held-region errors, then trains the small correction head. It selects head epoch4 by validation NLL, stopping34. Fold feature statistics exclude the held regions and all calibration/validation/test footprints. The final depth ensemble stays frozen. Raw and additionally scaled correction variants were predefined; their results must both be reported.

## 7. Evaluation metrics: definitions before interpretation
''')
    definitions=[('RMSE','sqrt(mean((prediction-reference)^2)), metres; emphasizes larger errors.'),('MAE','mean absolute error, metres; less sensitive to large errors than RMSE.'),('R²','1−SSE/SST, with SST around the reference mean within the reported subset. Negative means worse than that subset-mean constant predictor.'),
        ('Within1m / within2m','100*count(abs(error)<=1 or2m)/n; tolerance-based regression accuracy, not classification accuracy.'),('Bias','mean(prediction-reference), metres. Negative means underprediction.'),('Error SD','Population SD of signed residuals, ddof0. This is measured error spread, not a predicted confidence estimate.'),
        ('Reference/prediction SD','Population spread of depths/predictions; explains range-dependent R². Recorded in CSV/JSON.'),('Mean predicted sigma','Average per-location predicted SD, metres. Distinct from error SD, SD of sigma, and across-seed variability.'),('90% coverage / width','Fraction of reference depths in mean±1.64485*sigma; full width=2*1.64485*sigma. These are Gaussian/moment intervals, not guaranteed bounds.'),
        ('NLL / CRPS / interval score','Proper distribution/interval scores; lower is better. Ensemble NLL/CRPS here consistently use its Gaussian moment approximation, not the separate exact mixture likelihood reported elsewhere.'),('Spatial RMSE intervals','2000 bootstrap resamples of approximately2km groups; conditional on models/reference and not independent-survey or new-seed uncertainty.')]
    lines+=table(['Metric','Meaning'],definitions)
    add('''```text
RMSE² = bias² + population_error_SD²
aleatoric_variance(x) = mean_m(member_head_sigma(x)²)
epistemic_variance(x) = mean_m((member_mean(x)-ensemble_mean(x))²)
total_variance(x) = aleatoric_variance(x) + epistemic_variance(x)
```

Variances combine before taking the square root. Means of per-location SDs do not necessarily combine in quadrature; RMS SDs do. Ensemble disagreement uses1/M population variance, not an extra division by M for the standard error of the mean. A single-model ablation has no measured epistemic value. Uncertainty-off placeholder sigmas are invalid and reported N/A.

## 8. Recomputed depth accuracy tables

Matched component ablations use the same637 training examples, seed42, labels and geographic evaluation positions. The following are separately computed validation/test metrics; no new fitting or training was performed for this report.
''')
    matched=['Full v1','No physics','No uncertainty','Neither']
    for domain in ('0-10','0-20'):
        for split in ('validation','test'):
            add(f'### {split.capitalize()}, reference depth{domain}m')
            lines+=table(['Model','n','RMSE(m)','MAE(m)','R²','Within1m(%)','Within2m(%)','Bias(m)','Error SD(m)'],[
                [name,r(name,split,domain)['n'],*[f(r(name,split,domain).get(k)) for k in ('rmse_m','mae_m','r2','accuracy_within_1m_pct','accuracy_within_2m_pct','bias_m','error_std_population_m')]] for name in matched])
    add('### Full v1 test depth bins: error distribution and its contribution')
    lines+=table(['Bin(m)','n','RMSE','MAE','R²','Within1m(%)','Within2m(%)','Bias','Error SD','Primary SSE(%)'],[
        [b,r('Full v1',domain=b)['n'],*[f(r('Full v1',domain=b).get(k)) for k in ('rmse_m','mae_m','r2','accuracy_within_1m_pct','accuracy_within_2m_pct','bias_m','error_std_population_m','share_of_primary_squared_error_pct')]] for b in ('0-5','5-10','10-15','15-20')])
    add('![Error by depth](figures/depth_error_analysis.png)\n\n![Prediction scatter](figures/prediction_scatter.png)')
    add('### Additional matched loss/head investigations,0–20m')
    extra=['Full v1','No extinction','Squared depth loss','Integrated head candidate']
    lines+=table(['Run','Val RMSE','Test RMSE','Test MAE','Test R²','Test within1m(%)','Test within2m(%)'],[
        [name,f(r(name,'validation')['rmse_m']),*[f(r(name).get(k)) for k in ('rmse_m','mae_m','r2','accuracy_within_1m_pct','accuracy_within_2m_pct')]] for name in extra if name in METRICS])
    add('### Separate447-training-point uncertainty research: depth metrics')
    for split in ('validation','test'):
        lines+=table([split.capitalize()+' model/domain','n','RMSE','MAE','R²','Within1m(%)','Within2m(%)','Bias','Error SD'],[
            [name+'/'+domain,r(name,split,domain)['n'],*[f(r(name,split,domain).get(k)) for k in ('rmse_m','mae_m','r2','accuracy_within_1m_pct','accuracy_within_2m_pct','bias_m','error_std_population_m')]]
            for name in ('Ensemble raw','OOF raw') for domain in ('0-10','0-20')])
    add('OOF corrections and global calibration leave depth means unchanged, so their depth metrics are exactly identical to the source ensemble. That equality is a verification result, not four independent improvements.')
    add('## 9. Uncertainty results, separately from depth accuracy')
    for split in ('validation','test'):
        add(f'### {split.capitalize()} uncertainty,0–10m and0–20m')
        names=['Full v1','No physics','No uncertainty','Neither','Ensemble raw','Ensemble scaled','OOF raw','OOF scaled']
        lines+=table(['Model/domain','Mean sigma(m)','Error SD(m)','90% coverage(%)','90% width(m)','NLL','CRPS(m)','Interval score90'],[
            [name+'/'+domain,*[f(r(name,split,domain).get(k)) for k in ('mean_predicted_sigma_m','error_std_population_m','coverage90_pct','width90_m','gaussian_moment_nll','gaussian_moment_crps_m','interval_score90_m')]] for domain in ('0-10','0-20') for name in names])
    add('### Raw three-member ensemble decomposition')
    lines+=table(['Split/domain','Mean aleatoric SD','Mean epistemic SD','Mean total SD'],[
        [split+'/'+domain,*[f(r('Ensemble raw',split,domain).get(k)) for k in ('mean_aleatoric_sigma_m','mean_epistemic_sigma_m','mean_predicted_sigma_m')]] for split in ('validation','test') for domain in ('0-10','0-20')])
    add('''The learned OOF head corrects total predictive risk. It does not identify a newly measured pure aleatoric component, and its multiplier must not be arbitrarily split between aleatoric and epistemic sources. Raw component values are preserved.

The predefined raw OOF option reduces overall test width9.33→7.73m, with coverage96.5→89.0% for nominal90%. Validation width8.71→7.14m, coverage93.8→90.4%. It is promising, not uniformly reliable: raw OOF test coverage by bin is94.1/94.8/87.5/85.4%; the two deeper bins still under-cover. Extra global scaling makes test width9.41m, so that variant did not achieve narrowing. Narrower intervals alone cannot establish improvement if coverage or proper scores degrade.

## 10. Why are the results lower or higher?

### A. Why is0–10m RMSE1.264m, while0–20m is2.561m?
''')
    shallow=r('Full v1',domain='0-10');full=r('Full v1');deep_count=full['n']-shallow['n'];share=shallow['share_of_primary_squared_error_pct']
    add(f'''**Verified mathematical explanation:** the shallow75 points are{100*75/228:.1f}% of the primary test samples but only{share:.2f}% of its squared error. The deeper{deep_count} points contribute{100-share:.2f}%. RMSE is not the unweighted average of bin RMSEs:

```text
RMSE_0–20 = sqrt(sum_bin(n_bin * RMSE_bin²) / sum_bin(n_bin))
```

The10–15m and15–20m bins have RMSE3.255/2.800m and contribute45.33/46.66% of total squared error. Restricting the reporting domain removes most high-error cases; it is **not** evidence that the model was retrained to halve error on the same full dataset. Always label the depth domain and count next to RMSE.

**Plausible physical explanation:** optical bottom contrast generally becomes more difficult to distinguish from water-column/background effects with depth. Turbidity, bottom type, glint and adjacency may matter. These are scientific hypotheses consistent with SDB optics, not local measured causal attributions. Our uncalibrated K/z_ext cannot establish the exact invisible-depth boundary. [IHO guidance](https://iho.int/uploads/user/pubs/bathy/B_13_Ed100_032024.pdf).

### B. Why is10–15m particularly difficult?

**Verified residual pattern:** bias is−1.863m and error SD2.669m. Both systematic underprediction and variable residuals contribute: `3.255² ≈ (−1.863)² + 2.669²`. Fifteen-to-twenty metres also underpredicts (bias−1.613m), so the problem is not an isolated single-bin number.

**Verified data associations:** the mid-bin audit has median6 clear observations and2 cloud/shadow observations versus8 and0 in the other0–20m test locations. The four highest-error approximately2km groups contribute{ev['mid_bin_top4_spatial_group_sse_share_pct']:.1f}% of mid-bin squared error. Existing inspection also found some high-error groups with clear observations on all12 dates; cloud alone is not a sufficient explanation.

**Not established:** the proportions caused by actual optical invisibility, a wrong GEBCO reference, benthic variation, land contamination, the weak extinction prior or model bias cannot be separated from these saved predictions. Coarse-target smoothness and mixed shoreline pixels are plausible reference limitations. The study lacks independent dense local survey depths and measured water optical properties needed for attribution.

Shallow positive bias and deeper negative bias are consistent with compression toward the middle of the depth distribution. Across0–20m, prediction SD4.467m is smaller than reference SD4.746m. This supports investigating smoothing/regression-to-the-mean and loss bias, but does not identify a unique mechanism. Removing extinction alone changes validation RMSE1.963→2.095m and test2.561→2.550m; it does not resolve the issue or prove that the extinction term caused it. The squared-depth objective gives2.100/2.590m validation/test, so a more explicitly RMSE-aligned term did not improve this joint optimization run.

### C. Why can R² be negative despite roughly1m RMSE?

R² depends on the spread of reference depths in the evaluated subset. In0–5m, reference SD is only0.534m, while RMSE is1.094m: `1−1.094²/0.534² ≈ −3.197`. This is a real negative bin-specific R²: the bin-mean constant predictor has lower squared error there. It must not be hidden.

Combined0–10m reference SD is2.068m and R²0.626. Combined0–20m reference SD is4.746m and R²0.709 despite larger RMSE. A larger R² across a wider depth range therefore does not mean its absolute errors are smaller. The ensemble's0–10m R² is−0.096, documenting a shallow-performance limitation.

### D. Does the ablation prove physics and uncertainty help?

On the same0–10m test labels, full/no-physics/no-uncertainty/neither RMSE is1.264/1.373/1.665/1.887m. That supports a **numerical shallow-domain advantage** of the full configuration in this seed. Physics-off removes seven input features, FiLM conditioning and physics losses together; it does not isolate one physical law. Uncertainty-off changes NLL optimization, not merely the display of confidence intervals.

On0–20m the ablations have slightly smaller test RMSE than full, while full has the best validation score. The previously computed paired spatial difference intervals span zero. Do not claim a universal causal benefit or remove difficult bins to make that claim. Repeated matched seeds or another independent geographic/reference evaluation are needed.

No-extinction and squared-loss runs worsen validation RMSE relative to the selected configuration; the integrated-head candidate's tiny validation RMSE gain comes with worse uncertainty diagnostics. This explains why changing a loss or adding a head was not automatically promoted as a better complete model.

### E. Why does the ensemble improve overall test RMSE but worsen shallow RMSE?

**Verified confounding changes:** shallow training support falls from187 to80 in the half-open0–10m bins (inclusive≤10 counts189→82). Calibration/buffering removes training examples and feature references are refitted; the variance head/gradient handling and ensemble averaging also change. The later447-point training set is deeper on average and calibration points are disproportionately shallow. These are not matched-only head changes.

Ensemble averaging may reduce model-specific error, but all members can share bias. The loss of shallow supervision and preprocessing shift are plausible contributors to worse0–10m RMSE2.165m versus1.264m for v1. They have not been isolated experimentally. Overall test RMSE2.372m versus2.561m is an exploratory numerical difference with a paired spatial95% interval including zero. V1 stays the validation-selected depth reference.

### F. Why are uncertainty values different from RMSE, and why do widths change?

RMSE is an observed average error against labels. Predicted sigma is a model estimate before reference depth is known; it is not required to equal RMSE at each point. For v1 in0–10m, mean sigma1.802m and90% coverage98.7% indicate conservative intervals. Across0–20m, mean sigma1.606m but coverage71.1% indicates undercoverage. Global averaging can hide strong conditional miscalibration.

The uncertainty calibration set has only15 geographic groups and some high-error locations, so global scaling must widen every interval to cover them. Its factor1.759 gives96.5% overall test coverage, above the nominal90%. The OOF head learns location-dependent risk from held-region errors and has slightly better error ranking (.162→.187), enabling narrower aggregate intervals. The result is empirical, not proof that all errors are anticipated. Depth RMSE stays2.372m because no depth mean is changed.

### G. Is the dataset responsible? Is GEBCO better than ICESat-2?

The reference, sample distribution and image/reference pairing affect the measured result. A coarse smooth grid can be easier to reproduce than sparse tracks; that does not make it more accurate seabed truth. Our GEBCO test and ATL24 experiments use different labels/locations/depth distributions, so their absolute RMSEs cannot rank sensor quality or isolate a model improvement.

The old public ATL24 run has test RMSE4.741m and uses2018–2024 photons with a2024 image composite. Audits found large date gaps, many single-photon labels and about7m horizontal uncertainty. The officialV002 recovery uses a different reference set: composite control6.512m versus matched/tide6.093m, with171 test points;113 are15–20m, compared with only15/233 in the old public run. The new set is much deeper, and only29 test points have usable historical clear images. On those29 themselves, error worsens5.207→5.869m; the lower total matched score is not proof temporal repair solved the problem.

Product-version/flag interpretation, profile sparsity, horizontal footprint, tide residuals, true local morphology and independent reference quality remain limitations. The EOT20 water-level check supports some relative variation (coastal held-date surface RMSE0.269m), not a complete datum/surge correction. A better reference can improve label validity but cannot manufacture a visible bottom in an optical image.

### H. Can the number be compared directly with published papers?

No direct claim of superiority follows from RMSE alone. The original HybridBathNet paper uses a different reference and study design, including NOAA airborne lidar, rather than our coarse GEBCO labels. Match depth range, reference resolution, geography, independent splitting and sample support before comparing errors. Our implementation combines related ideas but is not asserted to be an exact reproduction of that published architecture. [Qian et al.](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1636124/full).

## 11. Historical development: comparisons that must stay separate
''')
    lines+=table(['Stage','Validation/test RMSE(m)','What changed / comparison limit'],[
        ['Earlier expanded-smooth study','2.594 /3.554','2176 train,65 validation/65 test, broader historical depth domain.'],
        ['Nearshore selected full v1','1.963 /2.561','637/208/343; primary0–20m n228. New shelf eligibility and denser distinct nearshore sampling.'],
        ['Full v1 shallow reporting','1.778 /1.264','No new training: same predictions, restricted0–10m n72/75.'],
        ['Three-member uncertainty ensemble','2.219 /2.372','447 train+123 calibration, changed feature fitting/head/optimization; not an isolated architecture effect.'],
        ['OOF uncertainty option','2.219 /2.372','Point predictions unchanged; test interval width7.73m at89.0% nominal90% coverage.']])
    add('''The repairs already made include native GEBCO reprojection, radiometric offset handling, training-only feature fitting, finite-value guards, mask-aware physics, EMA buffer/best-checkpoint correctness, metre-consistent variance arithmetic, no-replacement sampling and the active joint reconstruction objective. Their contribution was not measured as one controlled factorial experiment, so the entire historical gain cannot be apportioned among them.

## 12. What we can defend, and what remains open
''')
    lines+=table(['Statement','Status'],[
        ['The selected full model achieves0–10m test RMSE1.264m against GEBCO on75 points.','Verified from saved predictions.'],
        ['54.7%/93.3% of shallow test predictions are within1/2m.','Verified tolerance counts41/75 and70/75.'],
        ['Above10m contributes about92% of primary squared error.','Verified algebra, not a physical-cause diagnosis.'],
        ['The OOF head reduces overall interval width17.2% with89.0% test coverage.','Verified exploratory tradeoff; deep-bin undercoverage remains.'],
        ['Physics gives a reliable measured extinction depth or accurate physical Kd.','Not established.'],
        ['GEBCO metrics prove sub-metre surveyed accuracy or superiority to ICESat-2.','Not established.'],
        ['The full model universally beats every ablation.','Not supported across all domains/metrics/seeds.'],
        ['Three ensemble members capture all epistemic uncertainty.','Not established; shared bias and geographic shift can remain.']])
    add('''Recommended next work is to retain shallow supervision while reserving independent calibration groups; validate reference quality and shoreline mixing; expand independent spatial/optical evidence; and assess deeper-bin conditional coverage without tuning to these test labels. Keep geography fixed as requested. No new training was started to prepare this report, and no model/default was silently promoted.

## Appendix A. Full numerical exports and reproducibility

* [evaluation_metrics.csv](evaluation_metrics.csv): all models, validation/test and0–5/5–10/10–15/15–20/20–25/0–10/0–15/0–20/0–25 domains, including reference/prediction/error SD, median/p90 errors, tolerance counts, sigma, coverage and proper scores.
* [evaluation_metrics.json](evaluation_metrics.json): includes2000-resample spatial RMSE intervals for key matched/ensemble/OOF summaries and error-contribution values.
* [dataset_counts.json](dataset_counts.json): recalculated counts and target distributions; boundary conventions are explicit above.
* [architecture_audit.json](architecture_audit.json) and[layer_inventory.csv](layer_inventory.csv): strict reload, actual shapes and every registered leaf, including unexecuted modules.
* [prediction_provenance.json](prediction_provenance.json): hashes of prediction inputs used for recomputation.
* [analysis_evidence.json](analysis_evidence.json): source quality summaries and independent spatial-error concentration calculation.
* [Source project handoff](../../PROJECT_CONTEXT.md),[matched ablation report](../../runs/2026-11-06/gebco_component_ablation_v1/REPORT.md),[OOF study](../../runs/2026-11-06/gebco_oof_uncertainty_v1/REPORT.md),[calibration literature note](../../docs/GEBCO_UNCERTAINTY_CALIBRATION_2026-11-06.md).

The historical root `results/hybridbathnet_results.json` is not the authority for these current experiments. Reporting code reads preserved prediction/checkpoint artifacts; no historical result, dataset or model is overwritten. Original source-ensemble checkpoint hashes and the risk-head reload were already verified in the OOF study.

## Appendix B. Interpretation checklist for your presentation

When stating any result, name the reference dataset, depth range, sample count, spatial split and whether it is a single model or ensemble. Say “agreement with GEBCO,” not independently surveyed seabed accuracy. Present validation and test separately. Explain R² with its reference variance. Report uncertainty coverage together with width. Clearly label causal hypotheses and unavailable metrics. Do not compare the shallow1.264m number directly with the older3.554m full-depth result as a same-test improvement.
''')
    # Compact across-seed stability appendix, not confused with epistemic sigma.
    add('## Appendix C. Seed variability of the447-training-point members')
    rows=[]
    for split in ('validation','test'):
        for domain in ('0-10','0-20'):
            values=[r(f'Ensemble member seed{s}',split,domain)['rmse_m'] for s in (42,1,7)]
            rows.append([split+'/'+domain,*[f(v) for v in values],f(np.mean(values)),f(np.std(values,ddof=1))])
    lines+=table(['Split/domain','Seed42 RMSE','Seed1 RMSE','Seed7 RMSE','Mean RMSE','Sample SD across seeds'],rows)
    add('This sample SD uses only three runs and measures variation in a dataset-level score. It is not the per-location epistemic SD, not residual SD, and not an ensemble RMSE; ensemble RMSE is calculated after averaging predictions.')
    appendix=ROOT/'atl24_appendix_metrics.json'
    if appendix.exists():
        data=json.loads(appendix.read_text())
        add('## Appendix D. Separate ATL24 diagnostic evaluations')
        add('These are not matched to the GEBCO benchmark. Public-v1 and officialV002 also have different labels/locations/depth distributions; compare the two official arms only within their common reference. The exact source version of the public service subset remains unverified. Official scores use acquisition-relative depths after inverse tide conversion. Small bins have a small-sample flag in the export; they are descriptive, not reliable performance estimates.')
        for domain in ('0-10','0-20'):
            lines+=table(['ATL24 model/split/domain','n','RMSE','MAE','R²','Within1m(%)','Within2m(%)','Bias','Error SD','Mean sigma','90% coverage'],[
                [name+'/'+split+'/'+domain,a['n'],*[f(a.get(k)) for k in ('rmse_m','mae_m','r2','accuracy_within_1m_pct','accuracy_within_2m_pct','bias_m','error_std_population_m','mean_predicted_sigma_m','coverage90_pct')]]
                for name,parts in data.items() for split,summary in parts.items() for a in [summary[domain]]])
        add('[Complete ATL24 appendix metrics](atl24_appendix_metrics.csv) | [ATL24 prediction provenance](atl24_prediction_provenance.json). No ATL24 training was performed for this report.')
    return '\n'.join(lines)+'\n'


def inline(text):
    text=html.escape(text)
    text=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2">',text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    return text


def polish(markdown):
    """Improve prose unit spacing without touching URLs or fenced/code text."""
    result=[];code=False
    for line in markdown.splitlines():
        if line.startswith('```'):code=not code;result.append(line);continue
        if code:result.append(line);continue
        chunks=re.split(r'(`[^`]+`|!?\[[^\]]*\]\([^)]+\))',line)
        for i in range(0,len(chunks),2):
            versions={}
            def protect(match):
                key='VERSIONTOKEN'+chr(65+len(versions));versions[key]=match[0];return key
            value=re.sub(r'\bv\d+\b',protect,chunks[i],flags=re.IGNORECASE)
            value=re.sub(r'(?<=[a-z])(?=\d)',' ',value)
            value=re.sub(r'(?<=\d)(?=(?:km|m|arcsec|pixels?|points?|epochs?|layers?|channels?)\b)',' ',value)
            for key,original in versions.items():value=value.replace(key,original)
            chunks[i]=value
        result.append(''.join(chunks))
    return '\n'.join(result)+'\n'


def html_document(markdown):
    out=[];code=False;table_open=False;list_open=False
    for line in markdown.splitlines():
        if line.startswith('```'):
            if table_open:out.append('</tbody></table>');table_open=False
            out.append('</code></pre>' if code else '<pre><code>');code=not code;continue
        if code:out.append(html.escape(line)+'\n');continue
        if line.startswith('|'):
            cells=[c.strip() for c in line.strip().strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?',c or ' ') for c in cells):continue
            if not table_open:
                out.append('<table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in cells)+'</tr></thead><tbody>');table_open=True
            else:out.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in cells)+'</tr>')
            continue
        if table_open:out.append('</tbody></table>');table_open=False
        if line.startswith('* '):
            if not list_open:out.append('<ul>');list_open=True
            out.append('<li>'+inline(line[2:])+'</li>');continue
        if list_open:out.append('</ul>');list_open=False
        match=re.match(r'^(#{1,6}) (.*)',line)
        if match:
            level=len(match[1]);out.append(f'<h{level}>'+inline(match[2])+f'</h{level}>')
        elif line.strip():out.append('<p>'+inline(line)+'</p>')
    if table_open:out.append('</tbody></table>')
    if list_open:out.append('</ul>')
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HybridBathNet study report</title><style>'+'''
body{font:15px/1.65 system-ui,sans-serif;color:#203047;max-width:1250px;margin:40px auto;padding:0 26px}h1,h2,h3{line-height:1.3;color:#173754}h2{border-bottom:2px solid #d9e6ef;padding-bottom:9px;margin-top:42px}table{border-collapse:collapse;width:100%;font-size:12px;margin:20px 0}th,td{border:1px solid #d0dae4;padding:8px;text-align:left;vertical-align:top}th{background:#eaf1f7}tr:nth-child(even){background:#f7f9fc}img{max-width:100%;height:auto}pre{background:#f1f5f8;padding:15px;white-space:pre-wrap}code{font-size:.9em}a{color:#1262a1}p{margin:13px 0}@media print{body{font-size:10pt;max-width:none;margin:0;padding:0}h2{break-before:page}table{font-size:8pt}tr,img{break-inside:avoid}a{color:#203047;text-decoration:none}}@page{size:A4 landscape;margin:15mm}
'''+ '</style></head><body>'+''.join(out)+'</body></html>'


def main():
    figures();text=polish(report())
    reference=['# HybridBathNet layer-by-layer reference','',
        'Generated from strict checkpoint reload and one evaluation forward pass. Input example batch size is1; model weights are not modified. Registered layers can exist without executing in a particular configuration.','']
    for name,a in ARCH.items():
        reference += ['## '+name,'']
        reference += table(['Module','Type/specification','Input','Output','Parameters','Executed'],[
            [row['module'],row['specification'],str(row['input']),str(row['output']),row['parameters'],row['executed']] for row in a['layers']])
    reference='\n'.join(reference)
    (ROOT/'LAYER_REFERENCE.md').write_text(reference,encoding='utf-8')
    (ROOT/'LAYER_REFERENCE.html').write_text(html_document(reference),encoding='utf-8')
    text+='\n[Readable layer-by-layer reference](LAYER_REFERENCE.md) | [Printable layer reference](LAYER_REFERENCE.html).\n'
    (ROOT/'HYBRIDBATHNET_STUDY_REPORT.md').write_text(text,encoding='utf-8')
    (ROOT/'HYBRIDBATHNET_STUDY_REPORT.html').write_text(html_document(text),encoding='utf-8')
    print('Study report created:',len(text.split()),'words; models:',len(METRICS))


if __name__=='__main__':main()
