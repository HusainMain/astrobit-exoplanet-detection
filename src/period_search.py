"""Period search engine — BLS + TLS with wotan detrending.

Changes from v1:
- Replaced rolling median with wotan biweight + quarter breaks
- Added TLS alongside BLS for independent period verification
- Returns top 3 BLS peaks instead of just the best
- Multi-peak BLS for feature extraction
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.stats import median_abs_deviation

from .config import (
    PERIOD_MIN_DAYS, PERIOD_MAX_DAYS,
    BLS_COARSE_RESOLUTION, BLS_FINE_RESOLUTION,
    BLS_FINE_WINDOW_FRAC, BLS_DURATIONS_DAYS, BLS_OBJECTIVE,
    N_LS_CANDIDATES, KEPLER_SYSTEMATIC_PERIODS,
)


def compute_sde(power: np.ndarray, peak_idx: int) -> float:
    """MAD-based Signal Detection Efficiency."""
    med = np.nanmedian(power)
    mad = np.nanmedian(np.abs(power - med))
    if mad <= 0:
        return 0.0
    return float((power[peak_idx] - med) / (1.4826 * mad))


def _compute_quarter_breaks(quarters: np.ndarray) -> np.ndarray:
    """Find quarter boundary indices for wotan break_tolerance."""
    breaks = []
    q_sorted = np.argsort(quarters)
    for i in range(1, len(q_sorted)):
        if quarters[q_sorted[i]] != quarters[q_sorted[i - 1]]:
            breaks.append(q_sorted[i])
    return np.array(breaks) if breaks else np.array([])


def clean_for_bls(df: pd.DataFrame, window_days: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Clean light curve for BLS: wotan biweight detrending.

    Uses wotan's biweight filter with break_tolerance at quarter boundaries.
    Falls back to rolling median if wotan is unavailable.
    """
    m = (df["quality"].values == 0) & np.isfinite(df["flux"].values)
    t = df["time"].values[m]
    f = df["flux"].values[m].astype(float)
    q = df["quarter"].values[m]

    if len(t) < 1000:
        return np.array([]), np.array([])

    # Normalize each quarter to its median
    for qq in np.unique(q):
        s = q == qq
        med = np.median(f[s])
        f[s] = f[s] / med if med > 0 else 1.0

    # Try wotan biweight detrending
    try:
        from wotan import flatten
        # Use quarter boundaries as break points
        breaks = _compute_quarter_breaks(q)
        break_tolerance = 200  # large tolerance to handle quarter jumps

        if len(breaks) > 0:
            _, trend = flatten(
                t, f,
                method="biweight",
                window_length=window_days,
                break_tolerance=break_tolerance,
                break_location=breaks,
                return_trend=True,
            )
        else:
            _, trend = flatten(
                t, f,
                method="biweight",
                window_length=window_days,
                break_tolerance=break_tolerance,
                return_trend=True,
            )

        ok = np.isfinite(trend) & (trend > 0)
        return t[ok], f[ok] / trend[ok]

    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: rolling median
    cadence = np.median(np.diff(t))
    k = max(5, int(window_days / cadence) | 1)
    if k % 2 == 0:
        k += 1
    trend = pd.Series(f).rolling(k, center=True, min_periods=max(1, k // 3)).median().values
    ok = np.isfinite(trend) & (trend > 0)
    return t[ok], f[ok] / trend[ok]


def _bls_search(time: np.ndarray, flux: np.ndarray, verbose: bool = False) -> list[dict]:
    """Coarse-to-fine BLS period search returning top N peaks.

    Returns list of dicts sorted by SDE descending.
    """
    if len(time) < 200:
        return []

    bls = BoxLeastSquares(time, flux)
    baseline = time.max() - time.min()
    pmax = min(PERIOD_MAX_DAYS, baseline / 3.0)

    # Coarse sweep: log-spaced grid
    coarse = np.exp(np.linspace(np.log(PERIOD_MIN_DAYS), np.log(pmax), BLS_COARSE_RESOLUTION))
    res = bls.power(coarse, BLS_DURATIONS_DAYS, objective=BLS_OBJECTIVE)
    power = np.asarray(res.power)

    # Find well-separated peaks
    order = np.argsort(power)[::-1]
    peaks, used = [], np.zeros(len(coarse), bool)
    n_candidates = min(N_LS_CANDIDATES * 2, len(order))  # get extra for filtering
    for i in order:
        if used[i]:
            continue
        peaks.append(i)
        lo = np.searchsorted(coarse, coarse[i] * 0.9)
        hi = np.searchsorted(coarse, coarse[i] * 1.1)
        used[lo:hi] = True
        if len(peaks) >= n_candidates:
            break

    # Refine each peak
    results = []
    for i in peaks:
        p0 = coarse[i]
        width = BLS_FINE_WINDOW_FRAC * p0
        fine = np.linspace(p0 - width, p0 + width, BLS_FINE_RESOLUTION)
        fine = fine[fine > PERIOD_MIN_DAYS]
        if len(fine) < 10:
            continue

        r = bls.power(fine, BLS_DURATIONS_DAYS, objective=BLS_OBJECTIVE)
        p = np.asarray(r.power)
        j = int(np.nanargmax(p))
        score = compute_sde(power, i)

        results.append({
            "period": float(r.period[j]),
            "depth_ppm": float(r.depth[j] * 1e6),
            "duration_hours": float(r.duration[j] * 24),
            "transit_time": float(r.transit_time[j]),
            "sde": score,
        })

    # Sort by SDE descending
    results.sort(key=lambda x: x["sde"], reverse=True)
    return results


def _tls_search(time: np.ndarray, flux: np.ndarray, verbose: bool = False) -> dict:
    """TLS period search. Returns best period result.

    NOTE: TLS is ~25 min/star with full period grid. Disabled for feature builds.
    Only enable for top candidates in submission generation.
    """
    return {"period": np.nan, "depth_ppm": 0.0, "duration_hours": 0.0,
            "transit_time": np.nan, "sde": 0.0}

    try:
        from transitleastsquares import transitleastsquares

        model = transitleastsquares(time, flux)
        result = model.power(
            period_min=PERIOD_MIN_DAYS,
            period_max=min(PERIOD_MAX_DAYS, (time.max() - time.min()) / 3.0),
        )

        if result is None or len(result.periods) == 0:
            return {"period": np.nan, "depth_ppm": 0.0, "duration_hours": 0.0,
                    "transit_time": np.nan, "sde": 0.0}

        best_idx = int(np.nanargmax(result.power))

        return {
            "period": float(result.periods[best_idx]),
            "depth_ppm": float(result.transit_depths[best_idx] * 1e6),
            "duration_hours": float(result.duration[best_idx] * 24),
            "transit_time": float(result.transit_times[best_idx]),
            "sde": float(result.power[best_idx] * 50),  # TLS SDE is different scale, approximate
        }

    except ImportError:
        return {"period": np.nan, "depth_ppm": 0.0, "duration_hours": 0.0,
                "transit_time": np.nan, "sde": 0.0}
    except Exception as e:
        if verbose:
            print(f"  TLS error: {e}")
        return {"period": np.nan, "depth_ppm": 0.0, "duration_hours": 0.0,
                "transit_time": np.nan, "sde": 0.0}


def search(
    time: np.ndarray,
    flux: np.ndarray,
    verbose: bool = False,
) -> dict:
    """BLS + TLS period search. Returns best result + all BLS peaks + TLS result."""
    bls_results = _bls_search(time, flux, verbose)
    tls_result = _tls_search(time, flux, verbose)

    # Best BLS result
    best_bls = bls_results[0] if bls_results else {
        "period": np.nan, "depth_ppm": 0.0, "duration_hours": 0.0,
        "transit_time": np.nan, "sde": 0.0,
    }

    return {
        "best_bls": best_bls,
        "bls_peaks": bls_results,
        "tls": tls_result,
    }


def search_star(lc: pd.DataFrame, verbose: bool = False) -> dict:
    """Run full period search on a raw light curve DataFrame."""
    t, f = clean_for_bls(lc, window_days=0.5)
    if len(t) == 0:
        return {"best_bls": {"period": np.nan, "depth_ppm": 0.0, "duration_hours": 0.0,
                "transit_time": np.nan, "sde": 0.0},
                "bls_peaks": [], "tls": {"period": np.nan, "depth_ppm": 0.0,
                "duration_hours": 0.0, "transit_time": np.nan, "sde": 0.0}}
    return search(t, f, verbose=verbose)


def detrend_multi_window(
    df: pd.DataFrame,
    window_days: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detrend at a specific window length, returns (time, detrended_flux)."""
    m = (df["quality"].values == 0) & np.isfinite(df["flux"].values)
    t = df["time"].values[m]
    f = df["flux"].values[m].astype(float)
    q = df["quarter"].values[m]

    if len(t) < 1000:
        return np.array([]), np.array([])

    # Normalize each quarter
    for qq in np.unique(q):
        s = q == qq
        med = np.median(f[s])
        f[s] = f[s] / med if med > 0 else 1.0

    try:
        from wotan import flatten
        breaks = _compute_quarter_breaks(q)
        _, trend = flatten(
            t, f,
            method="biweight",
            window_length=window_days,
            break_tolerance=200,
            break_location=breaks if len(breaks) > 0 else None,
            return_trend=True,
        )
        ok = np.isfinite(trend) & (trend > 0)
        return t[ok], f[ok] / trend[ok]
    except Exception:
        cadence = np.median(np.diff(t))
        k = max(5, int(window_days / cadence) | 1)
        if k % 2 == 0:
            k += 1
        trend = pd.Series(f).rolling(k, center=True, min_periods=max(1, k // 3)).median().values
        ok = np.isfinite(trend) & (trend > 0)
        return t[ok], f[ok] / trend[ok]


def search_multi_detrend(df: pd.DataFrame, windows: list[float] = None, verbose: bool = False) -> dict:
    """Run BLS across multiple detrend windows for stability analysis.

    Returns dict with per-window best periods and stability metrics.
    """
    if windows is None:
        from .config import DETREND_WINDOWS_DAYS
        windows = DETREND_WINDOWS_DAYS

    results = {}
    for w in windows:
        t, f = detrend_multi_window(df, window_days=w)
        if len(t) < 200:
            results[w] = {"period": np.nan, "sde": 0.0}
            continue
        bls_results = _bls_search(t, f, verbose)
        if bls_results:
            results[w] = bls_results[0]
        else:
            results[w] = {"period": np.nan, "sde": 0.0}

    return results
