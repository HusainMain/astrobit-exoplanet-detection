"""Light curve preprocessing for AstroBit.

Transforms raw SAP flux into clean, normalized light curves ready for
period search. The pipeline per quarter:
  1. Mask bad cadences (quality != 0)
  2. Sigma-clip outliers
  3. Detrend by dividing by a smooth trend (preserves transit shape)
  4. Stitch quarters together
  5. Flux is ~1.0 baseline with transit dips going below
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


# ── Quality filtering ──────────────────────────────────────────────────

def mask_bad_cadences(lc: pd.DataFrame) -> pd.DataFrame:
    """Return copy with bad cadences masked (flux set to NaN)."""
    df = lc.copy()
    bad = df["quality"] != 0
    df.loc[bad, "flux"] = np.nan
    df.loc[bad, "flux_err"] = np.nan
    return df


# ── Sigma clipping ─────────────────────────────────────────────────────

def sigma_clip_flux(
    flux: np.ndarray,
    threshold: float = 5.0,
    window: int = 200,
) -> np.ndarray:
    """Sigma-clip outliers relative to a rolling median.

    Points deviating by more than `threshold` sigma from the local median
    are set to NaN.
    """
    flux = flux.copy()
    valid = ~np.isnan(flux)
    if valid.sum() < window:
        return flux

    s = pd.Series(np.where(valid, flux, np.nan))
    rolling_median = s.rolling(window, center=True, min_periods=window // 2).median()
    rolling_mad = (s - rolling_median).abs().rolling(window, center=True, min_periods=window // 2).median()
    rolling_sigma = rolling_mad * 1.4826

    deviation = np.abs(flux - rolling_median.values)
    bad = deviation > (threshold * rolling_sigma.values)
    flux[bad] = np.nan
    return flux


# ── Detrending (divide-by-trend, not subtract) ─────────────────────────

def detrend_spline(
    time: np.ndarray,
    flux: np.ndarray,
    smoothing_factor: float = 1e7,
) -> np.ndarray:
    """Detrend by dividing by a smoothing spline fit.

    Returns flux / trend, so baseline is ~1.0 and transits dip below 1.
    The smoothing factor should be large (1e6-1e8) so the spline captures
    only long-term stellar variability, not short-duration transits.
    """
    valid = ~np.isnan(flux)
    if valid.sum() < 10:
        med = np.nanmedian(flux[valid])
        return flux / med if abs(med) > 1e-10 else flux

    try:
        spline = UnivariateSpline(
            time[valid], flux[valid],
            s=smoothing_factor,
            k=3,
        )
        trend = spline(time)
        median_trend = np.nanmedian(trend)
        if abs(median_trend) < 1e-10:
            return flux / np.nanmedian(flux[valid])
        return flux / trend
    except Exception:
        median_flux = np.nanmedian(flux[valid])
        return flux / median_flux if abs(median_flux) > 1e-10 else flux


def detrend_poly(
    time: np.ndarray,
    flux: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """Detrend by dividing by a polynomial fit."""
    valid = ~np.isnan(flux)
    if valid.sum() < order + 1:
        median_flux = np.nanmedian(flux[valid])
        return flux / median_flux if abs(median_flux) > 1e-10 else flux

    coeffs = np.polyfit(time[valid], flux[valid], order)
    trend = np.polyval(coeffs, time)
    median_trend = np.nanmedian(trend)
    if abs(median_trend) < 1e-10:
        return flux / np.nanmedian(flux[valid])
    return flux / trend


def detrend_rolling_median(
    flux: np.ndarray,
    window: int = 2000,
) -> np.ndarray:
    """Detrend by dividing by a large-window rolling median."""
    s = pd.Series(np.where(~np.isnan(flux), flux, np.nan))
    trend = s.rolling(window, center=True, min_periods=window // 4).median()
    trend = trend.bfill().ffill()
    trend_vals = trend.values
    median_trend = np.nanmedian(trend_vals)
    if abs(median_trend) < 1e-10:
        return flux / np.nanmedian(flux) if np.nanmedian(flux) != 0 else flux
    return flux / trend_vals


# ── Quarter processing ─────────────────────────────────────────────────

def process_quarter(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    method: str = "rolling",
    sigma_threshold: float = 5.0,
    clip_window: int = 200,
    spline_smoothing: float = 1e8,
    poly_order: int = 3,
    rolling_window: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a single quarter: clip → detrend (divide by trend).

    Returns (time, flux_processed, flux_err_processed).
    """
    # 1. Sigma clip outliers
    flux_clipped = sigma_clip_flux(flux, threshold=sigma_threshold, window=clip_window)

    # 2. Detrend by dividing by trend
    if method == "spline":
        flux_proc = detrend_spline(time, flux_clipped, smoothing_factor=spline_smoothing)
    elif method == "poly":
        flux_proc = detrend_poly(time, flux_clipped, order=poly_order)
    elif method == "rolling":
        flux_proc = detrend_rolling_median(flux_clipped, window=rolling_window)
    else:
        raise ValueError(f"Unknown detrend method: {method}")

    return time, flux_proc, flux_err


# ── Full light curve preprocessing ─────────────────────────────────────

def preprocess_light_curve(
    lc: pd.DataFrame,
    method: str = "rolling",
    sigma_threshold: float = 5.0,
    clip_window: int = 200,
    spline_smoothing: float = 1e8,
    poly_order: int = 3,
    rolling_window: int = 2000,
) -> pd.DataFrame:
    """Full preprocessing pipeline for a single star.

    Steps:
      1. Mask bad cadences (quality != 0)
      2. Per-quarter: sigma clip → divide-by-trend
      3. Stitch quarters back together

    Returns DataFrame with same columns, flux is ~1.0 baseline.
    """
    df = mask_bad_cadences(lc)
    result = df.copy()

    for q in sorted(df["quarter"].unique()):
        mask = df["quarter"] == q
        q_time = df.loc[mask, "time"].values
        q_flux = df.loc[mask, "flux"].values
        q_err = df.loc[mask, "flux_err"].values

        _, q_flux_proc, q_err_proc = process_quarter(
            q_time, q_flux, q_err,
            method=method,
            sigma_threshold=sigma_threshold,
            clip_window=clip_window,
            spline_smoothing=spline_smoothing,
            poly_order=poly_order,
            rolling_window=rolling_window,
        )

        result.loc[mask, "flux"] = q_flux_proc
        result.loc[mask, "flux_err"] = q_err_proc

    return result


# ── Batch preprocessing ────────────────────────────────────────────────

def preprocess_batch(
    light_curves: dict[int, pd.DataFrame],
    **kwargs,
) -> dict[int, pd.DataFrame]:
    """Preprocess all light curves in a batch."""
    processed = {}
    for kepid, lc in light_curves.items():
        processed[kepid] = preprocess_light_curve(lc, **kwargs)
    return processed
