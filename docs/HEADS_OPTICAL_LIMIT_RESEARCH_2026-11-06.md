# Physics, uncertainty and ICESat-2: evidence and implementation

## Completed recovery experiments

The [recovery report](../runs/2026-11-06/recovery_v2/REPORT.md) now contains the actual five full-model runs, official ATL24 pairing/tide audit, published visibility inference, independent calibration results and checkpoint verification. GEBCO ensemble validation/test RMSE is 2.219/2.372 m; its shallow bins and validation worsen versus the preserved full v1, so it is not promoted. Official ATL24 matched/tide test RMSE is 6.093 m versus 6.512 m for its matched-reference control, but validation worsens and the spatial difference interval includes zero. Only 29 test locations have usable historical clear imagery, and that stratum itself worsens. These outcomes do not establish causal improvement from temporal pairing or a validated physics limit.

The published visibility adapter matches inspected source features exactly and dense-network inference numerically, but yields zero supported held-out locations at the recommended threshold on the evaluated scene. Keep automatic masking disabled. Local optical truth, measured attenuation/background/contrast/noise, and uncertainty under geographic shift remain unresolved. All proposed/current distinctions in this review must be read with those completed experimental outcomes. All 31 tests pass; no recovery job remains active.

Date: 6 November 2026. This is a targeted primary-source review, not an exhaustive review of every SDB paper. Confirmed project findings are distinguished from proposed improvements. No sub-metre accuracy or calibrated optical mask is established.

## What the current experiments establish

The full GEBCO model has validation/test RMSE1.963/2.561m on its 0–20m nearshore benchmark. Removing physics, uncertainty or both slightly lowers test RMSE but worsens validation; all paired spatial intervals for the test differences span zero. These single-seed ablations do not establish that either branch helps test RMSE. Retaining the proposed novelty requires demonstrating predictive, physical and uncertainty value separately.

ATL24 validation/test RMSE3.443/4.741m is a different reference and sample benchmark. Its score cannot be subtracted from GEBCO's score to rank reference quality. A model can fit a smooth, coarse interpolated surface more readily while being less accurate against real seabed observations.

The existing code reconstructs three reflectance bands from depth, independently learned per-pixel effective attenuation and albedo, and a training-percentile background. Accurate reconstruction alone cannot identify physical depth: different combinations of attenuation, albedo and depth can explain the same radiance. The heuristic blue/green-plus-red K and the learned reconstruction K are different quantities; neither currently has local optical validation. L2A surface reflectance is not automatically water-leaving remote-sensing reflectance in sr^-1.

The legacy expression `.0166 + .156*(B/G)^(-1.14) + .430*R/G` has no verified sensor-specific calibration in this repository. We did not find primary evidence validating that exact combined expression for shallow Sentinel-2 water. [NASA's Kd product description](https://modis.gsfc.nasa.gov/data/dataprod/kd_490.php) specifies an empirical relationship using water remote-sensing reflectance for supported ocean-colour sensors; it does not validate adding this red ratio or applying it to bottom-contaminated Sentinel L2A reflectance. The code documentation now explicitly calls the result a heuristic prior. It is retained only as a matched legacy conditioning/weak-penalty term in the current candidate, not an operational optical limit.

## Relevant literature and what we can transfer

| Primary source | Finding relevant here | Application and limit |
|---|---|---|
| [Lee et al., 1998, shallow-water semianalytical model](https://opg.optica.org/ao/abstract.cfm?uri=ao-37-27-6329) | Water-column and bottom contributions must both be modeled. | Use wavelength-dependent water and bottom terms; our exponential approximation is not a complete Lee/HOPE inversion. |
| [Blake, multispatial/multitemporal inversion, 2020](https://arxiv.org/abs/2002.02298) | Shared spatial and temporal structure helps constrain bathymetry, turbidity and bottom composition. | Share water optics within plausible water zones and scenes, allow bottom differences; do not force constant water properties across the whole AOI. |
| [Liu et al., 2025, BITE](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL118498) | Effective target visibility changes sharply with attenuation and bottom brightness; examples range from a few metres to about20m. | No universal Sentinel-2 depth cutoff. Their target-detection limit is not a bathymetric RMSE guarantee or locally calibrated cutoff for Kerala. |
| [Global optically shallow/deep Sentinel-2 delineation, 2024](https://doi.org/10.1016/j.rse.2024.114302) | Bottom detectability can be formulated as shallow/deep discrimination; the abstract discusses a spectral difference from modeled deep water. | Add a separately validated bottom-visibility detector. Full text was inaccessible during this review; do not treat its abstract as an implementation recipe. |
| [Atmospheric correction assessment for Sentinel-2 SDB, 2022](https://www.mdpi.com/2306-5338/9/12/215) | In that study, aquatic correction processors improved bathymetric retrieval relative to the tested Sen2Cor product. | Compare ACOLITE DSF or another aquatic correction on local scenes; improvement is site-dependent, not guaranteed. |
| [Qian et al., HybridBathNet, 2025](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1636124/full) | Uses NOAA airborne bathymetric lidar and a WDI branch with multiscale feature fusion. The reference starts at1m and is resampled to10m. | It is not an ATL24 experiment or a replication of our reconstruction/variance model. Its large dense reference dataset differs substantially from our sparse track labels. |
| [Seitzer et al., 2022, beta-NLL](https://arxiv.org/abs/2203.09168) | Heteroscedastic NLL optimization can settle at poor mean estimates; beta-NLL mitigates this problem. | We already used beta=.5; its presence does not prove calibration or remove all shared-gradient interference. |
| [Stirn et al., 2023, faithful heteroscedastic regression](https://arxiv.org/abs/2212.09184) | Separate mean and variance gradient effects to preserve mean learning. | We implement detached variance features and detached mean in NLL. Our Huber-plus-physics objective differs from their squared-error theorem; do not claim its theorem directly. |
| [Lakshminarayanan et al., 2017, deep ensembles](https://arxiv.org/abs/1612.01474) | Independently trained predictors quantify model disagreement as well as observation noise. | After validation chooses a configuration, evaluate3 independent seeds; ensemble uncertainty is not a guarantee under geographic shift. |
| [Angelopoulos and Bates, conformal prediction](https://arxiv.org/abs/2107.07511) | Calibration can turn residual scores into prediction intervals under its sampling assumptions. | Reserve independent calibration groups. Reusing validation for selection and calibration or assuming adjacent track pixels are exchangeable weakens nominal-coverage claims. Scaling intervals cannot change depth RMSE. |
| [Parrish et al., ATL24 assessment, 2025](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025EA004391) | High-confidence ATL24 points still include some false bottom detections. Confidence is a classification score, not a depth standard deviation. | Inspect profile continuity, reference consistency and product flags, not only confidence>=.6. Published direct lidar/reference accuracy is not the same quantity as Sentinel-to-depth model error. |
| [NSIDC data dictionary](https://nsidc.org/sites/default/files/documents/technical-reference/icesat2_atl24_data_dict_v001.pdf), [SlideRule ATL24 documentation](https://docs.slideruleearth.io/user-guide/icesat2) | Supplies height, uncertainty, classification and feasibility fields. The sensor-depth flag concerns ATLAS detection. | Verify version-specific semantics and retrieve missing fields. The accessible NSIDC dictionary is V001; our official data are V002 and public subset is V003-advertised. Do not transfer flags blindly across versions. |
| [GEBCO grid development](https://www.gebco.net/data-products/gridded-bathymetry-data/grid-development), [IHO SDB guide](https://iho.int/uploads/user/pubs/bathy/B_13_Ed100_032024.pdf) | Grid sampling/reference provenance and optical/measurement uncertainty matter. | GEBCO's15arcsec spacing is about450m here; resampling to10m does not create10m measured bathymetry. Report agreement with GEBCO separately from surveyed accuracy. |

## A defensible optical limit

There is no generally accurate one-step conversion of Sentinel-2 blue/green ratios into a local maximum reliable depth. The operational quantity should be a **scene-, location- and bottom-dependent probability of usable optical support**, accompanied by coverage and depth error on independent calibration locations.

For an explicitly simplified reflectance model:

`R_lambda(z) = R_infinity,lambda + C_lambda exp[-kappa_lambda z]`

where `C=R_bottom_at_surface-R_infinity` and `kappa=Kd+Ku_bottom` (m^-1), detection above `tau` times total reflectance error gives:

`z_limit,lambda = max(0, log(|C_lambda|/(tau sigma_R,lambda))/kappa_lambda)`.

The log formula is an algebraic diagnostic under these assumptions. The full shallow-water signal also includes attenuation of water-column backscatter and angular dependence; `2K` is an approximation, not separately retrieved downward/upward attenuation. Positive conservative bounds on kappa/noise and a lower contrast bound give a lower support estimate, provided those bounds have demonstrated joint coverage. A dark bottom may be undetectable even in clear shallow water. A universal1%-light rule ignores both contrast and error.

Required work before a mask is reliable:

1. Process individual scenes with aquatic atmospheric/glint/adjacency controls. Compare locally validated corrected reflectance products. Do not feed L2A reflectance into an Rrs formula by changing its name or dividing by pi without checking the product definition and corrections.
2. Estimate local deep-water background for each scene and water type from independent verified deep controls. A low percentile of shallow reflectance is not necessarily deep-water background; land adjacency/turbidity can dominate it.
3. Constrain spectral attenuation and a small bottom-spectrum mixture jointly across multiple observed depths/scenes. Use training-only reference constraints and optical priors. Do not force a universal blue/green ordering in CDOM-rich water. Additional Sentinel bands can assist aerosol/water-quality checks but cannot create depth truth.
4. Estimate total correction/background/radiometric error, including band covariance. Across-date standard deviation mixes environmental changes and is not sensor noise. A known-parameter local information bound `sigma_z >= 1/sqrt(sum[(dR_lambda/dz / sigma_R,lambda)^2])` is optimistic when nuisance parameters are unknown; it is not a calibrated model interval.
5. Validate actual bottom visibility and depth risk on separate geographic calibration groups. Use both bottom/deep discrimination and accuracy/coverage gates. Evaluate uncertainty about the limit itself. Report all eligible test samples first; any selective score must include retained fraction and groups. A mask cannot turn a difficult full-set RMSE into an unqualified improved score.

[src/optical_limit.py](../src/optical_limit.py) now adds a multiband conservative diagnostic requiring a declared number of supported bands, plus conditional local depth precision. They default to uncalibrated. They cannot manufacture missing optical measurements or certify our learned K/background. The old operational calibration guard remains active.

The Richardson et al. paper has a [published implementation and pretrained models](https://github.com/yulunwu8/Optically-Shallow-Deep), making an independent visibility baseline a practical next step. Its documented input is Sentinel L1C SAFE (optionally with ACOLITE L2R), not our four-band L2A composite. It outputs shallow-water probability rather than a limit in metres. Its documented Python3.10/TensorFlow2.13 stack differs from this project's Python3.14/PyTorch venv; it was not installed or applied here. Obtain compatible scene inputs and an approved compatible runtime or verified weight conversion, then assess geographic calibration on this AOI before treating its probability as an operational mask. Availability of a pretrained classifier does not establish local reliability.

## What the frozen ATL24 audit found

[Full audit JSON](../runs/2026-11-06/physics_uncertainty_research_v1/atl24_failure_audit.json) and [per-centre CSV](../runs/2026-11-06/physics_uncertainty_research_v1/atl24_sample_audit.csv) were reconstructed without changing labels/splits/predictions.

[Historical-scene query manifest](../runs/2026-11-06/physics_uncertainty_research_v1/atl24_scene_pairing_manifest.csv) covers75 distinct selected track/pass/date combinations with the original AOI and proposed ±10day query windows. These are acquisition requests to investigate, not downloaded/quality-selected scenes. A pass can span more than one spatial split; scene reuse does not replace the fixed geographic separation.

| Evidence | Training | Validation | Test |
|---|---:|---:|---:|
| Patch centres | 570 | 112 | 233 |
| Single-photon labels | 400 | 73 | 156 |
| No photon within30days of any Jan–Mar2024 scene | 559 | 104 | 231 |
| Median nearest scene gap, days | 755 | 392 | 1065 |
| 15–20m label count | 39 | 6 | 15 |

Depth subtraction exactly matches `surface_h-ortho_h`. Median reported horizontal uncertainty is about7.6m in each split, while reported vertical uncertainty is around0.15m. Those product uncertainties omit some collocation, temporal and model errors and do not certify a bottom classification. No selected10m pixel has repeated acquisition dates, so robust cross-date median labels were not actually available. All15 deep test labels use single photons; their mean depth bias is−10.321m and nominal90% coverage is only26.7%. The apparently reasonable overall84.98% coverage hides that deep failure. The shallow test bin has+2.929m bias. These patterns motivate reference and optical investigation, not deletion of difficult test points.

Confirmed differences explain why the GEBCO and ATL24 tasks differ; they do not by themselves prove the cause of every ATL24 error. Temporal gaps matter especially where the bed changes or water levels differ, but stable-bed multi-year pairing can sometimes work, as other studies demonstrate. Our AOI's turbidity has not been independently quantified. Do not attribute all error to the AOI or declare ATL24 inferior.

## Next ATL24 dataset and training design

1. Obtain one consistent official product version and confirm complete AOI returns and quality-field definitions. Inspect along-track photon profiles, including sea-surface photons, for coherent bottoms and surface confusion. Use product flags and signal/profile evidence with predeclared criteria, not GEBCO agreement or model residuals to select labels.
2. Pair each retained track/pass with the nearest usable Sentinel scene, initially checking a ±10day window. If too sparse, retrieve historical scenes matching the existing2018–2024 points, rather than pairing all old tracks to one2024 composite or duplicating photons. Choose windows using training/physical stability evidence. A new matched-scene benchmark must be documented alongside the frozen original.
3. Reconcile instantaneous water-column depth and a common vertical datum. `surface_h-ortho_h` is instantaneous depth; a fixed bed elevation needs the sea level at the Sentinel acquisition. ATL24 is already refraction corrected: applying another generic1.33 correction would be wrong. Check geoid/product/scene datum and tidal uncertainty.
4. Aggregate supported photons locally within a pass, retaining count, robust spread, confidence, horizontal/vertical uncertainty and dates. Use footprint-aware optical pooling or training-only positional augmentation consistent with the actual uncertainty definition. Do not claim uncertainty shrinks as1/sqrt(N) for correlated photons. Neighbourhood support must not bridge shorelines or steep discontinuities.
5. Preserve geographic train/test identities and footprint buffers. Group repeated passes/pixels to prevent leakage; also report transfer to unseen passes where support allows. Improve training depth coverage using new independent track support and balanced unique sampling; never duplicate evaluation pixels.
6. Treat label uncertainty explicitly: a reference-aware likelihood can include measurement and collocation variance in addition to model aleatoric variance. For positional error, local bed gradient and position covariance determine an additional vertical component; confirm the scalar sigma_thu interpretation before using it as a covariance. Classification confidence is not inverse depth variance.
7. Train the integrated model and compare against the frozen ATL24 experiment on declared scopes. Report full and per-bin RMSE, geographic intervals, interval calibration/width, uncertainty-error ranking, and support coverage. Keep GEBCO as a separate auxiliary/pretraining experiment if used; do not silently mix its labels into an ATL24-only claim.

## Implemented candidate and decision rule

New optional model switches detach variance features and share reconstruction attenuation over valid patch water. The objective can detach its NLL mean. Historical defaults and state-dict parameter shapes are retained. In the candidate, ordinary Gaussian NLL trains the variance head; Huber depth and existing optical losses train the backbone and physics. It retains the prior weights for a controlled first comparison. This is inspired by gradient separation, not the exact faithful squared-error theorem. The patch-homogeneous water assumption may fail across plume boundaries; learned per-band K is still an effective coefficient, not a measured Kd product.

The current trainer still clips a single global gradient norm. Although the variance loss has no direct backbone/depth gradient, its norm can change the common clipping factor; this experiment therefore does not guarantee identical optimizer steps to a mean-only model. A future exact separation experiment should clip the variance and depth/physics parameter groups separately. The conformal helper now uses the finite-sample order statistic and returns an unbounded interval if too few independent calibration examples support the requested coverage; geographic exchangeability is still a separate requirement. No conformal rescaling is used to claim an RMSE gain in the current experiment.

The predefined seed42 full integrated GEBCO experiment at [integrated_head_research_v1](../runs/2026-11-06/integrated_head_research_v1/) is **complete**. It used the same frozen637/208/343 patches, schedule and EMA validation selection as v1, stopping at166 with best epoch136. All eligible test centres were scored after selection. [Detailed results and reload audit](../runs/2026-11-06/integrated_head_research_v1/HEAD_COMPARISON.md).

| Measure, 0–20m | Existing full model | Integrated candidate |
|---|---:|---:|
| Validation RMSE | 1.963m | 1.961m |
| Test RMSE | 2.561m | 2.527m |
| Validation Gaussian NLL | 2.058 | 2.146 |
| Test Gaussian NLL | 2.596 | 2.634 |
| Test nominal90% coverage | 71.05% | 68.42% |
| Test mean90% interval width | 5.285m | 5.154m |
| Test sigma/absolute-error Spearman | 0.232 | −0.002 |

Paired spatial95% interval for candidate minus full test RMSE is[−0.217,+0.153]m, so a reliable depth improvement is not established. The candidate has a tiny numerical validation-RMSE advantage but worsened validation NLL and held-out uncertainty. It is not promoted as an improved joint model. Reconstruction RMSE0.001609 and exact prediction-equivalent reloads verify execution, not physical attenuation calibration.26 tests pass, including gradient separation and finite-sample calibration rank. All models/checkpoints are preserved; no training is active.

Next: prioritize independent optical visibility/background calibration and temporally/datum-matched ATL24 pairs; then compare a richer variance branch and independently trained ensembles with dedicated calibration geography. The weak residual ranking of this one-layer variance head shows that gradient separation alone did not provide useful test uncertainty here. Further runs should distinguish this uncertainty limitation from water-optics sharing, using validation and multiple seeds. No operational optical mask or reliable RMSE gain is claimed, and no further test-driven search is running.
