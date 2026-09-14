"""Unified data loader for AstroBit.

Provides functions to load parquet light curves, labels, and ground truth
for any split (train, dev, test).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    TRAIN_PARQUETS, DEV_PARQUETS, TEST_PARQUETS,
    TRAIN_LABELS, TRAIN_TRUTH, DEV_LABELS, DEV_TRUTH,
)


# ── Parquet loading ────────────────────────────────────────────────────

def _kepid_from_filename(name: str) -> int:
    """Extract integer KIC ID from 'KIC_1234567.parquet'."""
    return int(name.replace("KIC_", "").replace(".parquet", ""))


def load_light_curve(kepid: int, split: str = "train") -> pd.DataFrame:
    """Load raw SAP flux light curve for a single star.

    Parameters
    ----------
    kepid : int
        Kepler ID (e.g. 8165946).
    split : str
        'train', 'dev', or 'test'.

    Returns
    -------
    pd.DataFrame with columns: time, flux, flux_err, quality, quarter.
    """
    dirs = {"train": TRAIN_PARQUETS, "dev": DEV_PARQUETS, "test": TEST_PARQUETS}
    parquet_dir = dirs[split]
    path = parquet_dir / f"KIC_{kepid}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No parquet file for KIC {kepid} in {split}: {path}")
    return pd.read_parquet(path)


def load_all_light_curves(split: str = "train") -> dict[int, pd.DataFrame]:
    """Load all light curves for a split. Returns {kepid: DataFrame}."""
    dirs = {"train": TRAIN_PARQUETS, "dev": DEV_PARQUETS, "test": TEST_PARQUETS}
    parquet_dir = dirs[split]
    result = {}
    for f in sorted(parquet_dir.glob("KIC_*.parquet")):
        kepid = _kepid_from_filename(f.name)
        result[kepid] = pd.read_parquet(f)
    return result


# ── Label loading ──────────────────────────────────────────────────────

def load_labels(split: str = "train") -> pd.DataFrame:
    """Load stellar labels (kepid, label, koi_disposition, stellar props).

    For 'test' split, returns empty DataFrame (no labels available).
    """
    paths = {"train": TRAIN_LABELS, "dev": DEV_LABELS}
    if split == "test":
        return pd.DataFrame()
    return pd.read_csv(paths[split])


def load_truth(split: str = "train") -> pd.DataFrame:
    """Load ground-truth injected signal parameters.

    Returns DataFrame with columns:
        kepid, injected, bin, period_days, epoch_t0, depth_ppm,
        duration_hours, rp_rs, n_transits
    """
    paths = {"train": TRAIN_TRUTH, "dev": DEV_TRUTH}
    if split == "test":
        return pd.DataFrame()
    return pd.read_csv(paths[split])


# ── Combined loaders ──────────────────────────────────────────────────

def load_split(split: str = "train") -> dict:
    """Load everything for a split: light curves, labels, truth.

    Returns dict with keys:
        'light_curves': {kepid: DataFrame}
        'labels': DataFrame (or empty)
        'truth': DataFrame (or empty)
        'kepids': list[int]
    """
    lc = load_all_light_curves(split)
    labels = load_labels(split)
    truth = load_truth(split)
    kepids = sorted(lc.keys())
    return {
        "light_curves": lc,
        "labels": labels,
        "truth": truth,
        "kepids": kepids,
    }


# ── Helper: get label for a star ───────────────────────────────────────

def get_label(kepid: int, labels_df: pd.DataFrame) -> Optional[int]:
    """Return binary label (0 or 1) for a kepid, or None if not found."""
    row = labels_df[labels_df["kepid"] == kepid]
    if row.empty:
        return None
    return int(row.iloc[0]["label"])


def get_truth_for_star(kepid: int, truth_df: pd.DataFrame) -> Optional[dict]:
    """Return truth dict for a kepid, or None if not in truth table."""
    row = truth_df[truth_df["kepid"] == kepid]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# ── List parquet files in a directory ──────────────────────────────────

def list_parquet_kepids(split: str = "train") -> list[int]:
    """Return sorted list of KIC IDs that have parquet files."""
    dirs = {"train": TRAIN_PARQUETS, "dev": DEV_PARQUETS, "test": TEST_PARQUETS}
    parquet_dir = dirs[split]
    if not parquet_dir.exists():
        return []
    return []


# ── Canonical target definition ────────────────────────────────────────

def make_target(labels_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    """Build the canonical competition target: label==1 OR injected==1.

    This is the single source of truth for what counts as a positive.
    Starter notebook defines: has_planet = (label==1) | (injected==1).

    Parameters
    ----------
    labels_df : DataFrame with columns [kepid, label, ...]
    truth_df  : DataFrame with columns [kepid, injected, ...]

    Returns
    -------
    dict with:
        'target': dict[int, int]  -- kepid -> 0 or 1
        'n_positives': int
        'n_negatives': int
        'label1_only': set[int]   -- known KOI, no injected signal
        'injected_only': set[int] -- injected signal, label=0
        'both': set[int]          -- both label=1 and injected=1
        'injected_kepids': set[int] -- all truth.csv kepids (for period recovery)
    """
    label1 = set()
    if labels_df is not None and not labels_df.empty:
        label1 = set(labels_df[labels_df["label"] == 1]["kepid"].values)

    injected = set()
    if truth_df is not None and not truth_df.empty:
        injected = set(truth_df[truth_df["injected"] == 1]["kepid"].values)

    both = label1 & injected
    label1_only = label1 - injected
    injected_only = injected - label1

    all_kepids = label1 | injected
    target = {k: 1 for k in all_kepids}

    return {
        "target": target,
        "n_positives": len(all_kepids),
        "n_negatives": 0,  # caller must supply total count
        "label1_only": label1_only,
        "injected_only": injected_only,
        "both": both,
        "injected_kepids": injected,
    }


def get_star_id_from_path(path: str) -> str:
    """Extract star_id from parquet filename stem.

    For train/dev: 'KIC_10064054.parquet' -> 'KIC_10064054'
    For private test: 'STAR_0000.parquet' -> 'STAR_0000'
    """
    from pathlib import Path
    return Path(path).stem
    return sorted(_kepid_from_filename(f.name) for f in parquet_dir.glob("KIC_*.parquet"))
