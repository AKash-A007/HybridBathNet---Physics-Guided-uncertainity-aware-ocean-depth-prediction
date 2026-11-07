# Literature verification and research direction

Updated 2026-11-05 from the user's three-panel literature-review image. The image is a research lead, not a verified bibliography. Publisher/author sources below establish paper identity and the stated points; items marked limited-access are not full-paper reviews. Do not repeat every limitation in the slide as an independently verified finding.

## References checked

| Slide entry | Primary source | Verified scope / implication |
|---|---|---|
| Qian et al., Physics-guided deep neural networks for bathymetric mapping using Sentinel-2 multi-spectral imagery | [Frontiers, 2025](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1636124/full) | Already introduces **HybridBathNet**, with a physical bathymetry branch and feature fusion. Cite it as prior work; combining physics with deep learning alone is not our new contribution. |
| Xie et al., Nearshore Bathymetry from ICESat-2 LiDAR and Sentinel-2 Imagery Datasets Using Physics-Informed CNN | [Remote Sensing 16, 511, 2024](https://www.mdpi.com/2072-4292/16/3/511) | Uses lidar-derived depth references and radiative-transfer features. Its workflow corrects reference effects including tides/refraction. Sentinel imagery is not the depth ground truth. |
| Wang et al., Physics-Informed Neural Network for Bathymetry Inversion Coupling Seafloor Slope Effects and Radiative Transfer Constraints Using ICESat-2 and Sentinel-2 Data | [Remote Sensing 18, 1291, 2026](https://www.mdpi.com/2072-4292/18/9/1291) | Publisher-indexed abstract describes slope-aware physics and ICESat-2/Sentinel-2. Full-page retrieval rate-limited; exact slide thresholds and datasets not independently verified here. |
| Christofilakos et al., A Cloud-Based Framework for the Quantification of the Uncertainty of a Machine Learning Produced Satellite-Derived Bathymetry | [Author institution full text, 2025](https://elib.dlr.de/216314/1/remotesensing-17-03060.pdf) | Distinguishes ICESat-2 training labels from in-situ validation. The proposed uncertainty index's relation to actual errors is weak; uncertainty must be tested, not merely emitted by a network. |
| Liu, Shi and Fang, A Review on Bathymetric Inversion Research Based on Deep Learning Models and Remote Sensing Images | [Remote Sensing 18, 720, 2026](https://www.mdpi.com/2072-4292/18/5/720) | Publisher result located; full-page retrieval rate-limited. The slide's exact study count and deterministic-model prevalence were not audited. Treat those as pending, not established project facts. |
| Lv et al., BathyFormer | [Remote Sensing 17, 1195, 2025](https://www.mdpi.com/2072-4292/17/7/1195) | Uses high-resolution multispectral imagery, CUDEM references, and independent hydrographic validation. Reported 0.55–0.73 m RMSE concerns depths 2–5 m, not our much broader GEBCO depth range. |
| Wu et al., Satellite-Derived Bathymetry Using a Fast Feature Cascade Learning Model in Turbid Coastal Waters | [Journal of Remote Sensing](https://spj.science.org/doi/10.34133/remotesensing.0272) | Publisher abstract emphasizes turbid waters and limited field samples. Use as motivation for limited-label comparisons, not proof of transfer to this AOI. |
| Ferchichi et al., Deep learning-based uncertainty quantification for spatio-temporal environmental remote sensing: a systematic literature review | [DOI](https://doi.org/10.1016/j.neucom.2025.130242) | Citation identity located in bibliographic indexing; primary full text not inspected. No quantitative conclusions from the slide are adopted here. |
| Liu et al., Generalized satellite-derived bathymetry across spatial and temporal domains | [ISPRS publisher page](https://www.sciencedirect.com/science/article/pii/S092427162500379X) | Publisher abstract describes domain adaptation and spatial/temporal experiments. Supports explicitly testing transfer rather than random patch-only validation. |
| Lee et al., Satellite-Derived Bathymetry Using Sentinel-2 and Airborne Hyperspectral Data: A Deep Learning Approach with Adaptive Interpolation | [Remote Sensing 17, 2594, 2025](https://www.mdpi.com/2072-4292/17/15/2594) | Spatial harmonization/interpolation and higher-resolution references are central. Resampling is an evaluation-design issue as well as a technical alignment operation. |
| Zhang and Al Shehhi, Bathymetry estimation for coastal regions using self-attention | [Scientific Reports](https://www.nature.com/articles/s41598-024-83705-9) | TransBathy uses self-attention and examines unseen areas; depth-dependent errors matter. This does not establish that replacing our CNN with attention will fix coarse labels. |
| Chybicki et al., Study of various machine learning approaches for Sentinel-2 derived bathymetry | [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0291595) | Highlights how acquisition, reference quality, preprocessing and hyperparameters affect comparison. Motivates matched datasets for the six-channel versus sixteen-channel study. |
| Xie et al., Satellite-derived bathymetry combined with Sentinel-2 and ICESat-2 datasets using machine learning | [Frontiers author article PDF, 2023](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1111817/pdf?isPublishedV2=false) | "Only satellite data" means active ICESat-2 plus passive Sentinel-2; it does not mean Sentinel-2 reflectance supplies independent depth labels. |
| Xu et al., Machine Learning Based Estimation of Coastal Bathymetry From ICESat-2 and Sentinel-2 Data | [IEEE Xplore](https://ieeexplore.ieee.org/document/10306283/) | Multitemporal Sentinel imagery is composited and paired with ICESat-2 topographic information. More dates can improve optical quality without creating new independent seabed measurements. |
| Nguyen, Banno and Ha, Satellite derived bathymetry using empirical and machine learning approaches: a case study in the highly dynamic coastal water | [Publisher article PDF](https://www.tandfonline.com/doi/pdf/10.1080/21664250.2024.2445418) | Article identity verified (online 2024, journal volume 2025). Slide-specific performance/limitations require a closer methods review before reuse. |

## What the GEBCO audit changes

[Official GEBCO 2020 documentation](https://www.gebco.net/data-products/gridded-bathymetry-data/gebco-2020) describes 15-arc-second, pixel-centre-registered elevations in metres. The local subset has 72 x 61 native cells. Its metadata describes sea-level/bathymetric height but also contains an inconsistent ellipsoid standard_name: do not infer a precise vertical-datum conversion from that single field.

Our code had a coarse nearest-neighbour reprojection to UTM followed by bilinear interpolation to the image grid. The corrected path warps from native geographic cell centres directly to the target grid. This removes avoidable interpolation distortion, but it does not turn approximately 460 m source spacing into measured 10 m bathymetry. Correct coordinate handling cannot resolve missing underwater reference detail or tidal/scene-age discrepancies.

## Preserve the intended contribution: deep learning + physics + uncertainty

The user's explicit requirement is **one integrated model where the physics module, depth predictor and uncertainty module operate together**. Preserve shared learning and an explicit joint loss; this is not a plan to replace the hybrid with three unrelated models or tabular regressors. Current training already combines physics-conditioned depth estimation and heteroscedastic uncertainty. The optical reconstruction branch is not supervised by the active scalar loss, so completing that integration needs a tested model/loss change.

However, **novelty is a hypothesis to demonstrate**, not a consequence of putting these words in a title. Qian et al. already use the HybridBathNet name and physics-guided architecture. Cite and distinguish our extension from that baseline. Reference labels used for supervision/evaluation are a separate issue from whether the model components work jointly.

Proposed contribution to evaluate: **reference-quality-aware, physics-guided bathymetry with calibrated predictive uncertainty under spatial and temporal shifts**. This is a proposed direction, not a first-in-literature claim.

Required evidence:

1. A matched deterministic DL baseline, physics-guided DL, uncertainty-aware DL, and combined model. Hold labels, split, budget and seeds fixed. Preserve the user's exclusion of tabular training.
2. Physically meaningful and mask-aware optical features/constraints; report ablations and violations rather than calling heuristic coefficients calibrated physics.
3. Separate aleatoric prediction variance from epistemic/model variability. Current results use a heteroscedastic Gaussian head; neither MC dropout nor a calibrated deep ensemble is active.
4. Assess NLL, interval coverage and width, calibration by depth, and failure detection in new regions/dates alongside RMSE. The selected original-label model's test 68% coverage is only 55.38%, while nominal 90% coverage is 87.69%; uncertainty remains imperfect.
5. Use a separate calibration partition/geographic groups if adding interval calibration. Do not fit calibration on test labels or promise independent-pixel conformal guarantees for correlated tiles.
6. Distinguish error against resampled GEBCO from actual bathymetric accuracy against independent depths. Reference uncertainty can be modelled separately only with evidence or explicit assumptions, not chosen to make intervals look calibrated.

## Can we train, validate and test using Sentinel-2 only?

**As input imagery: yes. As independent depth ground truth: no.** Three spatially separate subsets of Sentinel-2 can train and test an optical reconstruction/self-supervised task. Without external depth reference they cannot establish RMSE in metres for true seabed depth.

A simplified optical model illustrates the ambiguity: R = R_inf + A exp(-2 K z). Changing K and z while keeping their product fixed can leave reflectance unchanged; unknown bottom reflectance A adds ambiguity. Low reconstruction error is not proof of accurate absolute depth. Pseudo-labels generated by the same physics model can teach consistency, but evaluating against them measures agreement with that model, not independent bathymetry.

Viable later directions:

- **Satellite-only but not Sentinel-only:** Sentinel-2 inputs with quality-controlled ICESat-2 bathymetric references, including refraction/tide/datum handling, geographic separation and independent validation where possible. Check actual AOI track coverage/usable bottom returns before promising a dataset.
- **Self-supervised pretraining:** use per-scene Sentinel imagery, optical reconstruction and temporal consistency, then anchor/calibrate absolute depths with trusted labels. Evaluate reconstruction and depth separately.
- **Weakly supervised GEBCO:** use native-cell-aware targets or aggregation and honest reference-scale uncertainty. Do not call interpolated subpixels independent surveyed depths.

Do not automatically switch to Sentinel-only depth scoring if a short pilot fails. A failure to improve RMSE is not evidence that the corrected coordinates are wrong, nor a justification to remove the reference against which accuracy is defined.
