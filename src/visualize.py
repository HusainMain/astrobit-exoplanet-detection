"""Visualization tools for AstroBit light curves and analysis."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.config import OUTPUTS_DIR


def plot_light_curve(
    lc: pd.DataFrame,
    kepid: int,
    label: Optional[int] = None,
    truth: Optional[dict] = None,
    save_path: Optional[Path] = None,
    title_suffix: str = "",
) -> None:
    """Plot a single light curve with optional truth overlay.

    Parameters
    ----------
    lc : DataFrame with time, flux, flux_err, quality, quarter
    kepid : Kepler ID
    label : 0 or 1 (optional)
    truth : dict with period_days, depth_ppm, etc. (optional)
    save_path : where to save the figure
    title_suffix : extra text for the title
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), height_ratios=[3, 1, 1])
    fig.suptitle(f"KIC {kepid}  |  label={label}  {title_suffix}", fontsize=14)

    time = lc["time"].values
    flux = lc["flux"].values
    quality = lc["quality"].values
    quarter = lc["quarter"].values

    # Mask NaN
    valid = ~np.isnan(flux)

    # ── Panel 1: Full light curve ──────────────────────────────────────
    ax = axes[0]
    ax.scatter(time[valid], flux[valid], s=0.3, alpha=0.5, c="steelblue", rasterized=True)
    ax.set_ylabel("Flux (e-/s)")
    ax.set_xlabel("Time (BKJD days)")

    # Mark transit epochs if truth provided
    if truth is not None:
        period = truth["period_days"]
        epoch = truth["epoch_t0"]
        depth = truth["depth_ppm"]
        duration = truth["duration_hours"] / 24.0
        # Find all transit epochs within the time range
        t_min, t_max = time[valid].min(), time[valid].max()
        n_periods = int((t_max - epoch) / period) + 1
        transit_times = epoch + np.arange(-int((epoch - t_min) / period), n_periods) * period
        transit_times = transit_times[(transit_times >= t_min) & (transit_times <= t_max)]
        for tt in transit_times:
            ax.axvline(tt, color="red", alpha=0.3, lw=0.5)
        ax.set_title(
            f"Period={period:.2f}d  Depth={depth:.0f}ppm  "
            f"Duration={truth['duration_hours']:.1f}h  "
            f"Bin={truth.get('bin', 'N/A')}",
            fontsize=10,
        )
    else:
        ax.set_title("Raw light curve (no truth overlay)")

    # ── Panel 2: Per-cadence SNR ───────────────────────────────────────
    ax = axes[1]
    flux_err = lc["flux_err"].values
    median_flux = np.nanmedian(flux[valid])
    snr = (flux - median_flux) / flux_err
    snr[~valid] = 0
    ax.scatter(time[valid], snr[valid], s=0.2, alpha=0.4, c="gray", rasterized=True)
    ax.set_ylabel("SNR")
    ax.set_xlabel("Time (BKJD days)")
    ax.set_ylim(-10, 10)

    # ── Panel 3: Quarter markers ───────────────────────────────────────
    ax = axes[2]
    q_colors = plt.cm.tab20(np.linspace(0, 1, 18))
    for q in sorted(np.unique(quarter[valid])):
        mask = (quarter == q) & valid
        ax.scatter(time[mask], np.ones(mask.sum()), s=0.5, c=[q_colors[q]], label=f"Q{q}")
    ax.set_ylabel("Quarter")
    ax.set_xlabel("Time (BKJD days)")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_folded_transit(
    lc: pd.DataFrame,
    period: float,
    epoch: float,
    kepid: int,
    depth_ppm: Optional[float] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot phase-folded light curve at a given period and epoch."""
    time = lc["time"].values
    flux = lc["flux"].values
    valid = ~np.isnan(flux)

    # Normalize flux
    flux_norm = flux / np.nanmedian(flux[valid])

    # Phase fold
    phase = ((time - epoch) % period) / period
    phase[phase > 0.5] -= 1.0  # center transit at phase 0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(phase[valid], flux_norm[valid], s=0.5, alpha=0.3, c="steelblue", rasterized=True)

    if depth_ppm is not None:
        depth_frac = depth_ppm / 1e6
        ax.axhline(1.0 - depth_frac, color="red", ls="--", alpha=0.5, label=f"Depth={depth_ppm:.0f}ppm")
        ax.legend()

    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"KIC {kepid} — Folded at P={period:.2f}d")
    ax.set_xlim(-0.1, 0.1)  # zoom on transit

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_detection_summary(
    results: list[dict],
    save_path: Optional[Path] = None,
) -> None:
    """Plot summary of detections across all stars.

    results: list of dicts with keys kepid, prediction, confidence, period, etc.
    """
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Confidence distribution
    ax = axes[0]
    ax.hist(df["confidence"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")

    # Predictions by confidence
    ax = axes[1]
    detected = df[df["prediction"] == 1]
    not_detected = df[df["prediction"] == 0]
    ax.hist(detected["confidence"], bins=20, alpha=0.7, label="Detected", color="green")
    ax.hist(not_detected["confidence"], bins=20, alpha=0.7, label="Not detected", color="red")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Confidence by Prediction")

    # Period distribution of detected
    ax = axes[2]
    if len(detected) > 0 and detected["period"].notna().any():
        ax.hist(detected["period"].dropna(), bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Count")
        ax.set_title("Periods of Detected Signals")
    else:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Periods of Detected Signals")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
