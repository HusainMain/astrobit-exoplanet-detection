"""Feature engineering for AstroBit classifier — v2.

Major additions:
- TLS features alongside BLS
- BLS/TLS agreement features
- Systematic period distance (targets 372.5d artifact)
- Multi-detrend stability features
- Multi-peak BLS features
- Transit event consistency features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import correlation_lags
from astropy.timeseries import LombScargle

from .config import (
    FEATURE_COLUMNS, PERIOD_MIN_DAYS, PERIOD_MAX_DAYS,
    KEPLER_SYSTEMATIC_PERIODS,
)
from .period_search import (
    clean_for_bls, search, detrend_multi_window, _bls_search,
)


# ── BLS features ───────────────────────────────────────────────────────

def _bls_features(time: np.ndarray, flux: np.ndarray):
    """Run BLS and extract detection features + all peaks."""
    result = search(time, flux)
    best_bls = result["best_bls"]
    bls_peaks = result["bls_peaks"]
    tls = result["tls"]

    period = best_bls["period"]
    depth_ppm = best_bls["depth_ppm"]
    duration_hours = best_bls["duration_hours"]
    sde = best_bls["sde"]
    t0 = best_bls["transit_time"]

    # Count transits
    if not np.isnan(period) and period > 0:
        baseline = time.max() - time.min()
        n_transits = int(baseline / period)
    else:
        n_transits = 0

    # SNR approximation
    valid = ~np.isnan(flux)
    scatter = np.std(flux[valid]) * 1e6 if valid.sum() > 0 else 1.0
    snr = depth_ppm / (scatter / np.sqrt(n_transits)) if n_transits > 0 and scatter > 0 else 0.0

    bls_dict = {
        "bls_sde": sde,
        "bls_period": period,
        "bls_depth_ppm": depth_ppm,
        "bls_duration_hours": duration_hours,
        "bls_snr": snr,
        "bls_n_transits": n_transits,
        "bls_t0": t0,
    }

    return bls_dict, bls_peaks, tls


# ── TLS features ───────────────────────────────────────────────────────

def _tls_features(tls_result: dict, flux: np.ndarray) -> dict:
    """Extract TLS detection features."""
    period = tls_result.get("period", np.nan)
    sde = tls_result.get("sde", 0.0)
    depth_ppm = tls_result.get("depth_ppm", 0.0)
    duration_hours = tls_result.get("duration_hours", 0.0)

    if not np.isnan(period) and period > 0:
        baseline_len = 1460  # approximate 4-year baseline
        n_transits = int(baseline_len / period)
    else:
        n_transits = 0

    valid = ~np.isnan(flux)
    scatter = np.std(flux[valid]) * 1e6 if valid.sum() > 0 else 1.0
    snr = depth_ppm / (scatter / np.sqrt(n_transits)) if n_transits > 0 and scatter > 0 else 0.0

    return {
        "tls_sde": sde,
        "tls_period": period,
        "tls_depth_ppm": depth_ppm,
        "tls_duration_hours": duration_hours,
        "tls_snr": snr,
        "tls_n_transits": n_transits,
    }


# ── BLS/TLS agreement features ────────────────────────────────────────

def _agreement_features(bls: dict, tls: dict) -> dict:
    """Compute agreement between BLS and TLS period searches."""
    bls_period = bls["bls_period"]
    tls_period = tls.get("tls_period", np.nan)

    if np.isnan(bls_period) or np.isnan(tls_period) or tls_period <= 0:
        return {
            "period_agreement": 0.0,
            "period_ratio": 0.0,
            "sde_ratio": 0.0,
            "depth_agreement": 0.0,
            "duration_agreement": 0.0,
        }

    # Period agreement: how close are they (0 = different, 1 = identical)
    ratio = bls_period / tls_period
    # Check for alias: ratio near 1, 2, 1/2, 3, 1/3
    alias_ratios = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]
    min_alias_dist = min([abs(ratio - ar) for ar in alias_ratios])
    agreement = 1.0 - min(min_alias_dist, 1.0)

    # SDE ratio
    bls_sde = bls["bls_sde"]
    tls_sde = tls.get("tls_sde", 0.0)
    sde_ratio = tls_sde / bls_sde if bls_sde > 0 else 0.0

    # Depth agreement
    bls_depth = bls["bls_depth_ppm"]
    tls_depth = tls.get("tls_depth_ppm", 0.0)
    if bls_depth > 0 and tls_depth > 0:
        depth_ratio = min(bls_depth, tls_depth) / max(bls_depth, tls_depth)
    else:
        depth_ratio = 0.0

    # Duration agreement
    bls_dur = bls["bls_duration_hours"]
    tls_dur = tls.get("tls_duration_hours", 0.0)
    if bls_dur > 0 and tls_dur > 0:
        dur_ratio = min(bls_dur, tls_dur) / max(bls_dur, tls_dur)
    else:
        dur_ratio = 0.0

    return {
        "period_agreement": agreement,
        "period_ratio": ratio,
        "sde_ratio": sde_ratio,
        "depth_agreement": depth_ratio,
        "duration_agreement": dur_ratio,
    }


# ── Systematic period distance (targets 372.5d artifact) ──────────────

def _systematic_distance_features(bls_period: float) -> dict:
    """Distance to known Kepler systematic periods.

    Key periods: 372.5d (spacecraft roll), 186.25d (half), 93.125d (quarter),
    32d (data downlink), 3d (reaction wheel desaturation).
    """
    if np.isnan(bls_period) or bls_period <= 0:
        return {"dist_to_systematic": 1.0, "min_systematic_ratio": 0.0}

    # Distance as fraction of systematic period
    distances = [abs(bls_period - sp) / sp for sp in KEPLER_SYSTEMATIC_PERIODS]
    min_dist = min(distances)

    # Also check harmonics: P/2, P/3, 2P, 3P
    all_systematics = []
    for sp in KEPLER_SYSTEMATIC_PERIODS:
        all_systematics.extend([sp, sp/2, sp/3, sp*2, sp*3])

    harmonic_distances = [abs(bls_period - sp) / sp for sp in all_systematics if sp > 0]
    min_harmonic_dist = min(harmonic_distances) if harmonic_distances else 1.0

    # Ratio to nearest systematic
    nearest_sys = min(KEPLER_SYSTEMATIC_PERIODS, key=lambda sp: abs(bls_period - sp))
    ratio = bls_period / nearest_sys if nearest_sys > 0 else 0.0

    return {
        "dist_to_systematic": min_harmonic_dist,
        "min_systematic_ratio": ratio,
    }


# ── Multi-peak BLS features ───────────────────────────────────────────

def _multi_peak_features(bls_peaks: list[dict]) -> dict:
    """Features from top 3 BLS peaks."""
    while len(bls_peaks) < 3:
        bls_peaks.append({"sde": 0.0, "period": np.nan})

    peak2 = bls_peaks[1] if len(bls_peaks) > 1 else {"sde": 0.0, "period": np.nan}
    peak3 = bls_peaks[2] if len(bls_peaks) > 2 else {"sde": 0.0, "period": np.nan}

    # Peak separation: how different are peak 1 and peak 2?
    p1 = bls_peaks[0]["period"]
    p2 = peak2["period"]
    if not np.isnan(p1) and not np.isnan(p2) and p1 > 0:
        sep = abs(p1 - p2) / p1
    else:
        sep = 0.0

    return {
        "bls_peak2_sde": peak2["sde"],
        "bls_peak2_period": peak2["period"],
        "bls_peak3_sde": peak3["sde"],
        "bls_peak3_period": peak3["period"],
        "bls_peak_separation": sep,
    }


# ── Light curve statistics ─────────────────────────────────────────────

def _lc_statistics(flux: np.ndarray) -> dict:
    """Compute basic statistics on cleaned flux."""
    valid = ~np.isnan(flux)
    f = flux[valid]
    if len(f) < 100:
        return {k: 0.0 for k in ["lc_rms", "lc_skew", "lc_kurtosis",
                                   "lc_entropy", "lc_autocorr_lag1", "lc_autocorr_lag5"]}

    f_centered = f - np.mean(f)
    rms = float(np.std(f_centered))
    sk = float(skew(f_centered))
    ku = float(kurtosis(f_centered))

    # Entropy of binned flux distribution
    hist, _ = np.histogram(f_centered, bins=50, density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

    # Autocorrelation
    f_norm = (f_centered / rms) if rms > 0 else f_centered
    n = len(f_norm)
    ac_lag1 = float(np.mean(f_norm[:-1] * f_norm[1:])) if n > 1 else 0.0
    ac_lag5 = float(np.mean(f_norm[:-5] * f_norm[5:])) if n > 5 else 0.0

    return {
        "lc_rms": rms,
        "lc_skew": sk,
        "lc_kurtosis": ku,
        "lc_entropy": entropy,
        "lc_autocorr_lag1": ac_lag1,
        "lc_autocorr_lag5": ac_lag5,
    }


# ── Lomb-Scargle features ─────────────────────────────────────────────

def _ls_features(time: np.ndarray, flux: np.ndarray) -> dict:
    """Lomb-Scargle periodogram features."""
    valid = ~np.isnan(flux)
    if valid.sum() < 100:
        return {"ls_peak_power": 0.0, "ls_second_peak_ratio": 0.0, "ls_peak_period": 0.0}

    t = time[valid]
    f = flux[valid]

    try:
        freq, power = LombScargle(t, f).autopower(
            minimum_frequency=1.0 / PERIOD_MAX_DAYS,
            maximum_frequency=1.0 / PERIOD_MIN_DAYS,
            samples_per_peak=2,
        )
    except Exception:
        return {"ls_peak_power": 0.0, "ls_second_peak_ratio": 0.0, "ls_peak_period": 0.0}

    period = 1.0 / freq

    # Top 2 peaks (well-separated)
    order = np.argsort(power)[::-1]
    peak1_idx = order[0]
    peak1_power = float(power[peak1_idx])
    peak1_period = float(period[peak1_idx])

    # Find second peak at least 10% away from first
    peak2_power = 0.0
    for i in order[1:]:
        if abs(period[i] - peak1_period) > 0.1 * peak1_period:
            peak2_power = float(power[i])
            break

    ratio = peak2_power / peak1_power if peak1_power > 0 else 0.0

    return {
        "ls_peak_power": peak1_power,
        "ls_second_peak_ratio": ratio,
        "ls_peak_period": peak1_period,
    }


# ── Vetting features ──────────────────────────────────────────────────

def _vetting_features(
    time: np.ndarray, flux: np.ndarray,
    period: float, t0: float, duration_hours: float,
) -> dict:
    """Transit vetting: odd-even depth, secondary eclipse, quarter consistency."""
    if np.isnan(period) or period <= 0:
        return {"odd_even_depth_ratio": 1.0, "secondary_eclipse_depth": 0.0,
                "quarter_signal_fraction": 0.0}

    duration_days = duration_hours / 24.0
    valid = ~np.isnan(flux)
    t = time[valid]
    f = flux[valid]

    # Phase fold
    phase = ((t - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    in_transit = np.abs(phase) < (duration_days / (2 * period))
    out_transit = np.abs(phase) > (duration_days / (2 * period) + 0.1)

    # Odd-even depth ratio
    transit_times = t[in_transit]
    if len(transit_times) > 4:
        transit_numbers = np.floor((transit_times - t0) / period).astype(int)
        odd_mask = transit_numbers % 2 == 1
        even_mask = transit_numbers % 2 == 0

        if odd_mask.sum() > 2 and even_mask.sum() > 2:
            odd_depth = np.median(f[in_transit][~odd_mask]) - np.median(f[in_transit][odd_mask])
            even_depth = np.median(f[in_transit][~even_mask]) - np.median(f[in_transit][even_mask])
            oe_ratio = abs(odd_depth / even_depth) if abs(even_depth) > 1e-10 else 1.0
        else:
            oe_ratio = 1.0
    else:
        oe_ratio = 1.0

    # Secondary eclipse: check at phase 0.5
    sec_phase = np.abs(phase - 0.5)
    sec_in = sec_phase < (duration_days / (2 * period))
    if sec_in.sum() > 2 and out_transit.sum() > 50:
        sec_depth = np.median(f[out_transit]) - np.median(f[sec_in])
        sec_depth_ppm = sec_depth * 1e6
    else:
        sec_depth_ppm = 0.0

    # Quarter signal fraction
    quarters = np.unique(np.floor(t / 90).astype(int))
    n_with_signal = 0
    for q in quarters:
        q_mask = (np.floor(t / 90).astype(int) == q) & in_transit
        if q_mask.sum() > 1:
            n_with_signal += 1
    qfrac = n_with_signal / max(1, len(quarters))

    return {
        "odd_even_depth_ratio": oe_ratio,
        "secondary_eclipse_depth": sec_depth_ppm,
        "quarter_signal_fraction": qfrac,
    }


# ── Transit shape features ────────────────────────────────────────────

def _transit_shape_features(
    time: np.ndarray, flux: np.ndarray,
    period: float, t0: float, duration_hours: float,
) -> dict:
    """Measure transit shape: ingress sharpness, boxiness."""
    if np.isnan(period) or period <= 0:
        return {"transit_ingress_sharpness": 0.0, "transit_boxiness": 0.0}

    duration_days = duration_hours / 24.0
    valid = ~np.isnan(flux)
    t = time[valid]
    f = flux[valid]

    phase = ((t - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    # Bin the transit region
    half_dur = duration_days / (2 * period)
    in_range = np.abs(phase) < half_dur * 3
    if in_range.sum() < 20:
        return {"transit_ingress_sharpness": 0.0, "transit_boxiness": 0.0}

    bins = np.linspace(-half_dur * 3, half_dur * 3, 30)
    bin_idx = np.digitize(phase[in_range], bins) - 1
    bin_flux = np.array([np.median(f[in_range][bin_idx == i]) if (bin_idx == i).sum() > 0
                         else np.nan for i in range(len(bins) - 1)])
    bin_centers = (bins[:-1] + bins[1:]) / 2

    valid_bins = ~np.isnan(bin_flux)
    if valid_bins.sum() < 10:
        return {"transit_ingress_sharpness": 0.0, "transit_boxiness": 0.0}

    # Ingress sharpness
    transit_bins = np.abs(bin_centers) < half_dur
    out_bins = np.abs(bin_centers) > half_dur * 1.5
    if transit_bins.sum() > 0 and out_bins.sum() > 0:
        transit_med = np.median(bin_flux[transit_bins & valid_bins]) if (transit_bins & valid_bins).sum() > 0 else 1.0
        out_med = np.median(bin_flux[out_bins & valid_bins]) if (out_bins & valid_bins).sum() > 0 else 1.0
        depth = out_med - transit_med
        ingress_bins = (bin_centers > half_dur * 0.5) & (bin_centers < half_dur * 1.5)
        if ingress_bins.sum() > 1 and valid_bins[ingress_bins].sum() > 1:
            ingress_flux = bin_flux[ingress_bins & valid_bins]
            ingress_slope = np.max(np.abs(np.diff(ingress_flux))) / depth if depth > 1e-10 else 0.0
        else:
            ingress_slope = 0.0
    else:
        ingress_slope = 0.0
        depth = 0.0

    # Boxiness
    if transit_bins.sum() > 3 and valid_bins[transit_bins].sum() > 3:
        transit_flux = bin_flux[transit_bins & valid_bins]
        boxiness = np.std(transit_flux) / depth if depth > 1e-10 else 0.0
    else:
        boxiness = 0.0

    return {
        "transit_ingress_sharpness": float(ingress_slope),
        "transit_boxiness": float(boxiness),
    }


# ── Transit event consistency features ─────────────────────────────────

def _event_consistency_features(
    time: np.ndarray, flux: np.ndarray,
    period: float, t0: float, duration_hours: float,
) -> dict:
    """Individual transit depth/timing consistency."""
    if np.isnan(period) or period <= 0:
        return {"depth_cv": 0.0, "depth_mad_ratio": 0.0,
                "timing_rms": 0.0, "n_expected": 0.0, "transit_coverage": 0.0}

    duration_days = duration_hours / 24.0
    valid = ~np.isnan(flux)
    t = time[valid]
    f = flux[valid]

    baseline = t.max() - t.min()
    n_expected = baseline / period

    # Find individual transit events
    transit_numbers = np.floor((t - t0) / period + 0.5).astype(int)
    unique_transits = np.unique(transit_numbers)

    depths = []
    timings = []
    for tn in unique_transits:
        t_mask = transit_numbers == tn
        if t_mask.sum() < 3:
            continue
        transit_t = t[t_mask]
        transit_f = f[t_mask]

        # Depth: out-of-transit median minus in-transit median
        phase_local = ((transit_t - t0) % period) / period
        phase_local[phase_local > 0.5] -= 1.0
        in_t = np.abs(phase_local) < (duration_days / (2 * period))
        out_t = np.abs(phase_local) > (duration_days / (2 * period) + 0.05)

        if in_t.sum() > 0 and out_t.sum() > 0:
            depth = np.median(transit_f[out_t]) - np.median(transit_f[in_t])
            depths.append(depth * 1e6)  # ppm

            # Timing: midpoint of transit
            transit_mid = np.median(transit_t[in_t])
            expected_time = t0 + tn * period
            timings.append(transit_mid - expected_time)

    n_observed = len(depths)
    coverage = n_observed / n_expected if n_expected > 0 else 0.0

    if n_observed < 2:
        return {"depth_cv": 0.0, "depth_mad_ratio": 0.0,
                "timing_rms": 0.0, "n_expected": n_expected, "transit_coverage": coverage}

    depths = np.array(depths)
    timings = np.array(timings)

    # Depth coefficient of variation
    depth_mean = np.mean(depths)
    depth_std = np.std(depths)
    depth_cv = depth_std / abs(depth_mean) if abs(depth_mean) > 1e-10 else 0.0

    # Depth MAD / median ratio
    depth_median = np.median(depths)
    depth_mad = np.median(np.abs(depths - depth_median))
    depth_mad_ratio = depth_mad / abs(depth_median) if abs(depth_median) > 1e-10 else 0.0

    # Timing RMS
    timing_rms = np.sqrt(np.mean(timings**2))

    return {
        "depth_cv": float(depth_cv),
        "depth_mad_ratio": float(depth_mad_ratio),
        "timing_rms": float(timing_rms),
        "n_expected": float(n_expected),
        "transit_coverage": float(coverage),
    }


# ── Multi-detrend stability features ──────────────────────────────────

def _detrend_stability_features(
    df: pd.DataFrame, bls_period: float,
) -> dict:
    """Does the BLS period survive different detrend window sizes?"""
    if np.isnan(bls_period) or bls_period <= 0:
        return {"detrend_stability_score": 0.0, "detrend_period_std": 0.0}

    from .config import DETREND_WINDOWS_DAYS

    periods = []
    for w in DETREND_WINDOWS_DAYS:
        t, f = detrend_multi_window(df, window_days=w)
        if len(t) < 200:
            continue
        bls_results = _bls_search(t, f)
        if bls_results:
            periods.append(bls_results[0]["period"])

    if len(periods) < 2:
        return {"detrend_stability_score": 0.0, "detrend_period_std": 0.0}

    periods = np.array(periods)

    # Stability: fraction of windows recovering a similar period
    matches = sum(1 for p in periods if abs(p - bls_period) / bls_period < 0.05)
    stability = matches / len(periods)

    # Period spread
    period_std = np.std(periods) / bls_period if bls_period > 0 else 0.0

    return {
        "detrend_stability_score": float(stability),
        "detrend_period_std": float(period_std),
    }


# ── Main feature extraction ───────────────────────────────────────────

def extract_features_star(
    lc_raw: pd.DataFrame,
    kepid: int,
    labels_row: dict | None = None,
    skip_detrend_stability: bool = True,
) -> dict:
    """Extract all features for a single star.

    Parameters
    ----------
    lc_raw : raw light curve DataFrame
    kepid : Kepler ID
    labels_row : dict with kepmag, teff, logg, radius (optional)
    skip_detrend_stability : if True, skip multi-detrend (slow)
    """
    # Clean for BLS (uses wotan now)
    t, f = clean_for_bls(lc_raw, window_days=0.5)
    if len(t) == 0:
        return {col: 0.0 for col in FEATURE_COLUMNS}

    # BLS + TLS features
    bls, bls_peaks, tls = _bls_features(t, f)

    # TLS features
    tls_dict = _tls_features(tls, f)

    # BLS/TLS agreement
    agreement = _agreement_features(bls, tls_dict)

    # Systematic distance
    systematic = _systematic_distance_features(bls["bls_period"])

    # Multi-peak BLS
    multi_peak = _multi_peak_features(bls_peaks)

    # Event consistency
    event_consistency = _event_consistency_features(
        t, f, bls["bls_period"], bls["bls_t0"], bls["bls_duration_hours"]
    )

    # LC statistics
    lc_stats = _lc_statistics(f)

    # LS features
    ls = _ls_features(t, f)

    # Vetting
    vetting = _vetting_features(
        t, f, bls["bls_period"], bls["bls_t0"], bls["bls_duration_hours"]
    )

    # Transit shape
    shape = _transit_shape_features(
        t, f, bls["bls_period"], bls["bls_t0"], bls["bls_duration_hours"]
    )

    # Stellar properties
    stellar = {}
    if labels_row is not None:
        for key in ["kepmag", "teff", "logg", "radius"]:
            stellar[key] = float(labels_row.get(key, 0.0) or 0.0)
    else:
        stellar = {"kepmag": 0.0, "teff": 0.0, "logg": 0.0, "radius": 0.0}

    # Detrend stability (slow — skip for initial feature build)
    if skip_detrend_stability:
        detrend_stab = {"detrend_stability_score": 0.0, "detrend_period_std": 0.0}
    else:
        detrend_stab = _detrend_stability_features(lc_raw, bls["bls_period"])

    # Combine
    features = {}
    features.update(bls)
    features.update(tls_dict)
    features.update(agreement)
    features.update(systematic)
    features.update(multi_peak)
    features.update(event_consistency)
    features.update(lc_stats)
    features.update(ls)
    features.update(vetting)
    features.update(shape)
    features.update(stellar)
    features.update(detrend_stab)

    # Remove non-feature keys
    for k in ["bls_t0"]:
        features.pop(k, None)

    return features


def build_feature_matrix(
    light_curves: dict[int, pd.DataFrame],
    labels_df: pd.DataFrame,
    kepids: list[int],
    verbose: bool = False,
    skip_detrend_stability: bool = True,
) -> pd.DataFrame:
    """Build feature matrix for a set of stars."""
    labels_dict = {}
    if labels_df is not None and not labels_df.empty:
        for _, row in labels_df.iterrows():
            labels_dict[int(row["kepid"])] = row.to_dict()

    rows = []
    for i, kepid in enumerate(kepids):
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(kepids)}] extracting features...")

        lc = light_curves.get(kepid)
        if lc is None:
            rows.append({col: 0.0 for col in FEATURE_COLUMNS})
            continue

        label_row = labels_dict.get(kepid)
        feat = extract_features_star(lc, kepid, label_row, skip_detrend_stability)
        rows.append(feat)

    df = pd.DataFrame(rows, index=kepids)

    # Ensure all columns exist (fill missing with 0)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    return df[FEATURE_COLUMNS]
