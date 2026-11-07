"""Bottom-detection limit under an explicit two-way attenuation/noise model.

This is not a Sentinel-2 Kd retrieval. Kd, Ku, bottom contrast and noise require
independent estimation/calibration before using the result as an accuracy mask.
"""
import math
import torch


def bottom_detection_limit(k_down, k_up, bottom_contrast, noise_std,
                           detection_snr=3., valid_mask=None):
    """S_bottom(z)=contrast*exp(-(Kd+Ku)*z); detect at S >= snr*noise.

    All attenuations are m^-1; contrast/noise share linear-reflectance units.
    Return zero if the bottom contrast is below threshold even at zero depth.
    Invalid inputs return NaN and valid=False, not a plausible-looking cutoff.
    The SNR threshold is a declared operational assumption, not a fitted fact.
    """
    if not math.isfinite(detection_snr) or detection_snr<=0:
        raise ValueError('Detection SNR must be finite and positive')
    kd,ku,contrast,noise=torch.broadcast_tensors(k_down.float(),k_up.float(),
                                               bottom_contrast.float(),noise_std.float())
    valid=torch.isfinite(kd)&torch.isfinite(ku)&torch.isfinite(contrast)&torch.isfinite(noise)
    valid=valid&(kd>0)&(ku>0)&(contrast>=0)&(noise>0)
    if valid_mask is not None: valid=valid&valid_mask.bool()
    ratio=contrast/(detection_snr*noise.clamp_min(torch.finfo(noise.dtype).tiny))
    safe_attenuation=(kd+ku).clamp_min(torch.finfo(kd.dtype).tiny)
    depth=torch.log(ratio.clamp_min(1.))/safe_attenuation
    depth=torch.where(valid,depth,torch.full_like(depth,float('nan')))
    return {'z_limit_m':depth,'valid':valid,'detectable_at_surface':valid&(ratio>=1),
            'detection_snr':detection_snr,'operationally_calibrated':False}


def conservative_bottom_detection_limit(k_down_upper, k_up_upper,
                                        bottom_contrast_lower, noise_upper,
                                        detection_snr=3., valid_mask=None):
    """A lower bound on optical support from matched parameter uncertainty bounds.

    Bounds must come from independent, locally relevant measurements in matching
    reflectance units and bands. Their *joint* coverage must be assessed; putting
    unrelated marginal quantiles here is not an automatic confidence guarantee.
    This function alone does not certify an operational mask.
    """
    result=bottom_detection_limit(k_down_upper,k_up_upper,
                                  bottom_contrast_lower,noise_upper,
                                  detection_snr=detection_snr,valid_mask=valid_mask)
    result['interpretation']='conservative_parameter_bound; requires independent calibration'
    result['operationally_calibrated']=False
    return result


def multiband_bottom_detection_limit(k_down_upper, k_up_upper,
                                     bottom_contrast_lower, noise_upper,
                                     detection_snr=3., required_bands=2,
                                     band_dim=1, valid_mask=None):
    """Conservative per-band detection model, then require k supported bands.

    Inputs are matching per-band parameter bounds in common reflectance units.
    Contrast is |R_bottom_at_surface - R_deep|, not bottom albedo alone.
    Noise includes radiometric/background/correction error; temporal scene std
    cannot be substituted without validation. The kth deepest band limit is
    returned. This is a declared detection rule, not a false-alarm guarantee:
    correlated bands, nuisance parameters and threshold choice need calibration.
    Detection of a bottom contribution does not guarantee accurate depth inversion.
    """
    result = conservative_bottom_detection_limit(
        k_down_upper, k_up_upper, bottom_contrast_lower, noise_upper,
        detection_snr=detection_snr, valid_mask=valid_mask)
    bands = result['z_limit_m'].shape[band_dim]
    if not isinstance(required_bands, int) or not 1 <= required_bands <= bands:
        raise ValueError('required_bands must be an integer between 1 and band count')
    valid = result['valid'].sum(dim=band_dim) >= required_bands
    limits = result['z_limit_m'].masked_fill(~result['valid'], -float('inf'))
    limit = limits.sort(dim=band_dim, descending=True).values.select(band_dim, required_bands-1)
    limit = torch.where(valid, limit, torch.full_like(limit, float('nan')))
    detectable = result['detectable_at_surface'].sum(dim=band_dim) >= required_bands
    return {'z_limit_m': limit, 'valid': valid, 'per_band_z_limit_m': result['z_limit_m'],
            'detectable_at_surface': detectable, 'required_bands': required_bands,
            'operationally_calibrated': False}


def conditional_depth_precision(depth, attenuation_total, bottom_contrast,
                                noise_std, band_dim=1):
    """Local Gaussian precision bound with known optical nuisance parameters.

    Model: R(z)=R_deep+contrast*exp(-attenuation_total*z). Independent band
    errors are assumed. Unknown attenuation/albedo/background, spatial error,
    tide and model discrepancy are omitted: this is an optimistic diagnostic,
    never a calibrated uncertainty head or operational confidence interval.
    """
    z, attenuation, contrast, noise = torch.broadcast_tensors(
        depth.float(), attenuation_total.float(), bottom_contrast.float(), noise_std.float())
    good = (torch.isfinite(z) & (z >= 0) & torch.isfinite(attenuation) & (attenuation > 0)
            & torch.isfinite(contrast) & (contrast >= 0) & torch.isfinite(noise) & (noise > 0))
    safe_z = torch.where(good, z, torch.zeros_like(z))
    safe_k = torch.where(good, attenuation, torch.ones_like(attenuation))
    safe_c = torch.where(good, contrast, torch.zeros_like(contrast))
    safe_noise = torch.where(good, noise, torch.ones_like(noise))
    derivative = safe_k * safe_c * torch.exp(-safe_k * safe_z)
    information = ((derivative / safe_noise) ** 2).sum(dim=band_dim)
    valid = good.all(dim=band_dim) & (information > 0)
    sigma = torch.where(valid, information.clamp_min(torch.finfo(information.dtype).tiny).rsqrt(),
                        torch.full_like(information, float('nan')))
    return {'conditional_sigma_lower_bound_m': sigma, 'valid': valid,
            'operationally_calibrated': False, 'assumption': 'known nuisance parameters; independent Gaussian band errors'}


def optical_support_flags(predicted_depth, optical_limit, calibrated=False,
                          predicted_sigma=None, uncertainty_multiplier=1.645):
    """Flag support without clipping predictions. Requires validated input claim.

    If supplied, sigma adds a one-sided upper-depth margin. This is not a joint
    probabilistic guarantee when either depth or optical-limit uncertainty is uncalibrated.
    """
    if not calibrated or not optical_limit.get('operationally_calibrated',False):
        raise ValueError('An independently calibrated and validated optical limit is required')
    if not math.isfinite(uncertainty_multiplier) or uncertainty_multiplier<0:
        raise ValueError('Uncertainty multiplier must be finite and nonnegative')
    depth=predicted_depth.float()
    valid_depth=torch.isfinite(depth)&(depth>=0)
    if predicted_sigma is not None:
        valid_depth=valid_depth&torch.isfinite(predicted_sigma)&(predicted_sigma>=0)
        depth=depth+uncertainty_multiplier*predicted_sigma.float()
    supported=(optical_limit['valid']&valid_depth&torch.isfinite(depth)&(depth>=0)
               &(depth<=optical_limit['z_limit_m']))
    return {'supported':supported,'out_of_optical_support':~supported,
            'coverage_fraction':supported.float().mean()}
