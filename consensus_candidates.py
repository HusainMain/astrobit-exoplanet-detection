"""
Multi-window consensus candidate clustering for period selection.

For each star:
  1. Generate top-20 BLS candidates from 3 detrend windows
  2. Alias-cluster periods across windows
  3. Compute cluster features
  4. Score clusters with z-normalized weighted formula
  5. Select best cluster as final period

Target: period recovery >= 20/30 on dev (up from 9/30 rank-0 baseline).
"""
import sys, time, io, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.stats import zscore, median_abs_deviation
from scipy.signal import savgol_filter

from src.config import (
    PERIOD_MIN_DAYS, PERIOD_MAX_DAYS,
    BLS_COARSE_RESOLUTION, BLS_FINE_RESOLUTION,
    BLS_FINE_WINDOW_FRAC, BLS_DURATIONS_DAYS, BLS_OBJECTIVE,
    N_CANDIDATE_PEAKS, KEPLER_SYSTEMATIC_PERIODS,
)
from src.period_search import _compute_quarter_breaks, compute_sde
from src.data_loader import load_all_light_curves, load_labels, load_truth, make_target
from build_candidates import extract_candidate_features


# ── Alias constants ──────────────────────────────────────────────────

ALIAS_RATIOS = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]
ALIAS_TOLERANCE = 0.02  # 2%


# ── Detrending functions ─────────────────────────────────────────────

def detrend_wotan_biweight(time, flux, quarter, window_days):
    """Wotan biweight detrending with quarter breaks."""
    try:
        from wotan import flatten
        breaks = _compute_quarter_breaks(quarter)
        _, trend = flatten(
            time, flux, method="biweight",
            window_length=window_days, break_tolerance=200,
            break_location=breaks if len(breaks) > 0 else None,
            return_trend=True,
        )
        ok = np.isfinite(trend) & (trend > 0)
        return time[ok], flux[ok] / trend[ok]
    except Exception:
        cadence = np.median(np.diff(time))
        k = max(5, int(window_days / cadence) | 1)
        if k % 2 == 0:
            k += 1
        trend = pd.Series(flux).rolling(k, center=True, min_periods=max(1, k // 3)).median().values
        ok = np.isfinite(trend) & (trend > 0)
        return time[ok], flux[ok] / trend[ok]


def detrend_savgol_per_quarter(time, flux, quarter, window_days=2.0, polyorder=3):
    """Savitzky-Golay detrending per quarter."""
    t_out, f_out = [], []
    for qq in np.unique(quarter):
        mask = quarter == qq
        t_q = time[mask]
        f_q = flux[mask]
        if len(f_q) < 50:
            t_out.append(t_q)
            f_out.append(f_q)
            continue
        cadence = np.median(np.diff(t_q))
        n_points = int((t_q[-1] - t_q[0]) / cadence) + 1
        t_uniform = np.linspace(t_q[0], t_q[-1], n_points)
        f_interp = np.interp(t_uniform, t_q, f_q)
        win_pts = max(int(window_days / cadence), 5)
        if win_pts % 2 == 0:
            win_pts += 1
        if win_pts > len(f_interp):
            win_pts = len(f_interp) if len(f_interp) % 2 == 1 else len(f_interp) - 1
        if win_pts < polyorder + 2:
            win_pts = polyorder + 2
            if win_pts % 2 == 0:
                win_pts += 1
        trend = savgol_filter(f_interp, win_pts, polyorder)
        trend_orig = np.interp(t_q, t_uniform, trend)
        ok = np.isfinite(trend_orig) & (trend_orig > 0)
        t_out.append(t_q[ok])
        f_out.append(f_q[ok] / trend_orig[ok])
    return np.concatenate(t_out), np.concatenate(f_out)


# ── BLS search (top-N peaks) ────────────────────────────────────────

def bls_search_multi(time, flux, n_candidates=20):
    """Coarse-to-fine BLS returning top N peaks with full features."""
    if len(time) < 200:
        return []

    bls = BoxLeastSquares(time, flux)
    baseline = time.max() - time.min()
    pmax = min(PERIOD_MAX_DAYS, baseline / 3.0)

    coarse = np.exp(np.linspace(np.log(PERIOD_MIN_DAYS), np.log(pmax), BLS_COARSE_RESOLUTION))
    res = bls.power(coarse, BLS_DURATIONS_DAYS, objective=BLS_OBJECTIVE)
    power = np.asarray(res.power)

    order = np.argsort(power)[::-1]
    peaks, used = [], np.zeros(len(coarse), bool)
    for i in order:
        if used[i]:
            continue
        peaks.append(i)
        lo = np.searchsorted(coarse, coarse[i] * 0.9)
        hi = np.searchsorted(coarse, coarse[i] * 1.1)
        used[lo:hi] = True
        if len(peaks) >= n_candidates:
            break

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
        sde = compute_sde(power, i)

        results.append({
            "period": float(r.period[j]),
            "t0": float(r.transit_time[j]),
            "depth": float(r.depth[j]),
            "duration": float(r.duration[j]),
            "sde": sde,
        })

    results.sort(key=lambda x: x["sde"], reverse=True)
    return results


# ── Alias clustering ─────────────────────────────────────────────────

def are_aliases(p1, p2):
    """Check if two periods are aliases of each other."""
    if p1 <= 0 or p2 <= 0:
        return False
    ratio = p1 / p2
    return min([abs(ratio - ar) for ar in ALIAS_RATIOS]) < ALIAS_TOLERANCE


def cluster_periods(candidates):
    """Group candidates into alias clusters across all windows.

    candidates: list of dicts with 'period', 'window', 'rank', etc.
    Returns list of clusters (each cluster is a list of candidate indices).
    """
    n = len(candidates)
    assigned = [-1] * n
    clusters = []

    for i in range(n):
        if assigned[i] >= 0:
            continue
        cluster_id = len(clusters)
        cluster = [i]
        assigned[i] = cluster_id
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            for k in cluster:
                if are_aliases(candidates[k]["period"], candidates[j]["period"]):
                    cluster.append(j)
                    assigned[j] = cluster_id
                    break
        clusters.append(cluster)
    return clusters


# ── Cluster feature computation ──────────────────────────────────────

def score_cluster(cluster_indices, candidates):
    """Compute cluster features and weighted score.

    Returns dict with all cluster features plus consensus_score.
    """
    cands = [candidates[i] for i in cluster_indices]

    # Basic stats
    sdes = np.array([c.get("bls_sde", c.get("sde", 0.0)) for c in cands])
    periods = np.array([c.get("period", c.get("bls_period", 0.0)) for c in cands])
    windows = [c["window"] for c in cands]
    ranks = np.array([c["rank"] for c in cands])
    reranker_scores = np.array([c.get("reranker_score", 0.0) for c in cands])
    transit_coverages = np.array([c.get("transit_coverage", 0.0) for c in cands])
    depth_cvs = np.array([c.get("depth_cv", 0.0) for c in cands])
    box_fit_snrs = np.array([c.get("box_fit_snr", 0.0) for c in cands])
    sec_eclipse_depths = np.array([c.get("secondary_eclipse_depth", 0.0) for c in cands])

    # Window support
    unique_windows = set(windows)
    n_windows = len(unique_windows)
    window_support = n_windows / 3.0

    # Window-specific flags
    has_baseline = int("biweight_0.5d" in unique_windows)
    has_2d = int("biweight_2d" in unique_windows)
    has_savgol = int("savgol_2d" in unique_windows)

    # Cross-window agreement
    cross_window_agreement = int(has_baseline and (has_2d or has_savgol))
    deep_agreement = int(has_2d and has_savgol)

    # Rank stats
    best_rank = int(ranks.min())
    mean_rank = float(ranks.mean())

    # SDE stats
    max_sde = float(sdes.max())
    mean_sde = float(sdes.mean())
    sum_sde = float(sdes.sum())

    # Reranker stats
    max_reranker = float(reranker_scores.max())
    mean_reranker = float(reranker_scores.mean())

    # Transit coverage stats
    best_coverage = float(transit_coverages.max())
    mean_coverage = float(transit_coverages.mean())

    # Depth CV (lower is more consistent = better)
    best_depth_cv = float(depth_cvs.min()) if len(depth_cvs) > 0 else 0.0

    # Box-fit SNR (higher is better)
    best_box_snr = float(box_fit_snrs.max())

    # Secondary eclipse (lower is better for transit vs EB)
    best_sec_eclipse = float(sec_eclipse_depths.min())

    # Systematic distance
    dist_to_sys = min([abs(periods[np.argmax(sdes)] - sp) / sp
                       for sp in KEPLER_SYSTEMATIC_PERIODS])
    near_systematic = 1 if abs(periods[np.argmax(sdes)] - 372.0) < 20 else 0

    # Period spread (alias-normalized)
    # Normalize periods to a canonical form (divide by closest alias ratio)
    canonical = []
    ref = periods[np.argmax(sdes)]
    for p in periods:
        ratios = [p / ref if ref > 0 else 1.0]
        canonical.append(p / min(ALIAS_RATIOS, key=lambda ar: abs(p / ref - ar) if ref > 0 else 1.0))
    period_spread = np.std(canonical) / ref if ref > 0 else 0.0

    # Median period (from canonical)
    median_period = float(np.median(periods))

    # Score gap: difference between best and second-best cluster
    # (computed externally, set to 0 here as placeholder)
    score_gap = 0.0

    # Only baseline window flag
    only_baseline = int(has_baseline and not has_2d and not has_savgol)

    # Best candidate (highest SDE)
    best_idx = int(np.argmax(sdes))

    # ── Consensus score (z-normalized weighted) ──────────────────────
    # Will be z-normalized across all clusters of a star after computation

    # Systematic penalty: strong suppression for periods near 372d
    near_372 = 1.0 if abs(periods[best_idx] - 372.5) < 25 else 0.0
    near_372_harmonic = 1.0 if any(abs(periods[best_idx] - sp) / sp < 0.05
                                    for sp in [372.5, 186.25, 93.125]) else 0.0

    # Bonus: period NOT near systematic AND has reranker support
    non_sys_bonus = (1.0 - near_372) * max_reranker

    consensus_score = (
        2.00 * max_reranker                          # reranker is primary signal
        + 0.40 * min(max_sde / 30.0, 1.0)           # SDE as secondary
        + 0.30 * window_support                      # weak window bonus
        + 0.20 * best_coverage                       # transit coverage
        + 0.15 * score_gap                           # separation from next cluster
        + 1.00 * non_sys_bonus                       # strong bonus: non-systematic + reranker
        - 3.00 * near_372                            # heavy penalty for 372d
        - 1.50 * near_372_harmonic                   # penalty for harmonics
        - 0.10 * (best_rank / 10.0)                 # weak rank preference
    )

    return {
        "best_period": float(periods[best_idx]),
        "best_sde": float(sdes[best_idx]),
        "best_depth_ppm": float(cands[best_idx].get("bls_depth_ppm", cands[best_idx].get("depth_ppm", 0.0))),
        "best_duration_hours": float(cands[best_idx].get("bls_duration_hours", cands[best_idx].get("duration_hours", 0.0))),
        "best_t0": float(cands[best_idx].get("t0", 0.0)),
        "max_sde": max_sde,
        "mean_sde": mean_sde,
        "sum_sde": sum_sde,
        "n_candidates": len(cands),
        "n_windows": n_windows,
        "window_support": window_support,
        "has_baseline": has_baseline,
        "has_2d": has_2d,
        "has_savgol": has_savgol,
        "cross_window_agreement": cross_window_agreement,
        "deep_agreement": deep_agreement,
        "best_rank": best_rank,
        "mean_rank": mean_rank,
        "max_reranker_score": max_reranker,
        "mean_reranker_score": mean_reranker,
        "best_transit_coverage": best_coverage,
        "mean_transit_coverage": mean_coverage,
        "best_depth_cv": best_depth_cv,
        "best_box_fit_snr": best_box_snr,
        "best_secondary_eclipse": best_sec_eclipse,
        "dist_to_systematic": dist_to_sys,
        "near_systematic": near_systematic,
        "median_period": median_period,
        "period_spread": period_spread,
        "only_baseline_window": only_baseline,
        "score_gap_to_next": score_gap,  # placeholder, set after scoring
        "consensus_score_raw": consensus_score,
    }


# ── Main pipeline ────────────────────────────────────────────────────

def build_consensus(split_name, output_csv, verbose=True):
    """Build multi-window consensus candidate table."""
    print(f"\n{'='*70}")
    print(f"CONSENSUS CANDIDATE TABLE — {split_name.upper()}")
    print(f"{'='*70}")

    # Load data
    print(f"\nLoading {split_name} light curves...")
    lc = load_all_light_curves(split_name)
    labels = load_labels(split_name)
    truth = load_truth(split_name)
    target_info = make_target(labels, truth)
    has_planet_map = target_info["target"]

    truth_dict = {}
    if truth is not None and not truth.empty:
        for _, row in truth.iterrows():
            truth_dict[int(row["kepid"])] = row.to_dict()

    labels_dict = {}
    if labels is not None and not labels.empty:
        for _, row in labels.iterrows():
            labels_dict[int(row["kepid"])] = row.to_dict()

    print(f"  {len(lc)} stars, {target_info['n_positives']} positive, "
          f"{len(truth_dict)} with truth")

    # Load existing reranker scores if available
    reranker_map = {}
    try:
        rerank_df = pd.read_csv(f"outputs/{split_name}_reranked_v4.csv")
        for _, row in rerank_df.iterrows():
            key = (int(row["kepid"]), row["bls_period"])
            reranker_map[key] = float(row.get("proba_ensemble", 0.0))
        print(f"  Loaded reranker scores for {len(reranker_map)} candidate-key pairs")
    except FileNotFoundError:
        print(f"  No reranker file found, using SDE as proxy")
    except Exception as e:
        print(f"  Warning loading reranker: {e}")

    # Define windows
    windows = [
        ("biweight_0.5d", lambda t, f, q: detrend_wotan_biweight(t, f, q, 0.5)),
        ("biweight_2d", lambda t, f, q: detrend_wotan_biweight(t, f, q, 2.0)),
        ("savgol_2d", lambda t, f, q: detrend_savgol_per_quarter(t, f, q, 2.0)),
    ]

    # ── Step 1: Generate multi-window candidates ─────────────────────
    print(f"\n{'='*60}")
    print("STEP 1: Generate multi-window candidates")
    print(f"{'='*60}")

    t0 = time.time()
    all_candidates = []

    for i, (kepid, lc_data) in enumerate(lc.items()):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(lc) - i - 1)
            print(f"  [{i+1}/{len(lc)}] {elapsed:.0f}s, ETA {eta:.0f}s, "
                  f"{len(all_candidates)} candidates", flush=True)

        m = (lc_data["quality"].values == 0) & np.isfinite(lc_data["flux"].values)
        t = lc_data["time"].values[m]
        f = lc_data["flux"].values[m].astype(float)
        q = lc_data["quarter"].values[m]

        if len(t) < 1000:
            continue

        # Normalize each quarter
        for qq in np.unique(q):
            s = q == qq
            med = np.median(f[s])
            f[s] = f[s] / med if med > 0 else 1.0

        hp = has_planet_map.get(kepid, 0)
        true_period = truth_dict.get(kepid, {}).get("period_days", None)

        for win_name, detrend_fn in windows:
            t_d, f_d = detrend_fn(t, f, q)
            if len(t_d) < 200:
                continue

            peaks = bls_search_multi(t_d, f_d, n_candidates=N_CANDIDATE_PEAKS)

            for rank, peak in enumerate(peaks):
                # Extract full candidate features
                cand = extract_candidate_features(
                    t_d, f_d, peak["period"], peak["t0"], peak["depth"], peak["duration"]
                )
                # Map keys for consistency with clustering code
                cand["period"] = cand["bls_period"]
                cand["depth_ppm"] = cand["bls_depth_ppm"]
                cand["duration_hours"] = cand["bls_duration_hours"]
                cand["kepid"] = kepid
                cand["window"] = win_name
                cand["rank"] = rank
                cand["bls_sde"] = peak["sde"]
                cand["has_planet"] = hp
                cand["t0"] = peak["t0"]

                # 372d vetting
                cand["dist_to_systematic"] = abs(peak["period"] - 372.0)
                cand["near_systematic"] = 1 if abs(peak["period"] - 372.0) < 20 else 0

                # Reranker score
                rerank_key = (kepid, peak["period"])
                cand["reranker_score"] = reranker_map.get(rerank_key, 0.0)

                # Label: is this candidate correct?
                if true_period is not None and true_period > 0:
                    ratio = peak["period"] / true_period
                    min_alias_dist = min([abs(ratio - ar) for ar in ALIAS_RATIOS])
                    cand["is_correct"] = 1 if min_alias_dist < 0.02 else 0
                else:
                    cand["is_correct"] = np.nan

                # Stellar properties
                if kepid in labels_dict:
                    for key in ["kepmag", "teff", "logg", "radius"]:
                        cand[key] = float(labels_dict[kepid].get(key, 0) or 0)
                else:
                    for key in ["kepmag", "teff", "logg", "radius"]:
                        cand[key] = 0.0

                all_candidates.append(cand)

    elapsed = time.time() - t0
    total_cands = len(all_candidates)
    print(f"  Done in {elapsed:.0f}s, {total_cands} total candidates")

    # ── Step 2: Cluster and score per star ───────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2: Alias-cluster and score")
    print(f"{'='*60}")

    star_results = []
    all_cluster_features = []

    # Pre-group candidates by kepid for fast lookup
    cands_by_kepid = {}
    for c in all_candidates:
        k = c["kepid"]
        if k not in cands_by_kepid:
            cands_by_kepid[k] = []
        cands_by_kepid[k].append(c)

    t1 = time.time()
    n_stars = len(cands_by_kepid)
    for si, kepid in enumerate(cands_by_kepid):
        if (si + 1) % 20 == 0:
            elapsed2 = time.time() - t1
            eta2 = elapsed2 / (si + 1) * (n_stars - si - 1)
            print(f"  [{si+1}/{n_stars}] {elapsed2:.0f}s, ETA {eta2:.0f}s", flush=True)

        star_cands = cands_by_kepid[kepid]
        if not star_cands:
            continue

        # Cluster periods
        clusters = cluster_periods(star_cands)

        # Score each cluster
        scored = []
        for cluster_indices in clusters:
            sc = score_cluster(cluster_indices, star_cands)
            scored.append(sc)

        # Compute score gaps (difference between best and second-best)
        if len(scored) >= 2:
            scored.sort(key=lambda x: x["consensus_score_raw"], reverse=True)
            gap = scored[0]["consensus_score_raw"] - scored[1]["consensus_score_raw"]
            scored[0]["score_gap_to_next"] = gap

        # Z-normalize scores across clusters of this star
        if len(scored) >= 2:
            raw_scores = np.array([s["consensus_score_raw"] for s in scored])
            # Z-score, but handle edge cases
            if np.std(raw_scores) > 1e-10:
                z_scores = zscore(raw_scores)
                for idx, s in enumerate(scored):
                    s["consensus_score"] = float(z_scores[idx])
            else:
                for s in scored:
                    s["consensus_score"] = 0.0
        elif len(scored) == 1:
            scored[0]["consensus_score"] = 1.0  # single cluster gets max score

        # Sort by consensus score
        scored.sort(key=lambda x: x["consensus_score"], reverse=True)

        # Get truth info
        hp = has_planet_map.get(kepid, 0)
        true_period = truth_dict.get(kepid, {}).get("period_days", None)
        true_depth = truth_dict.get(kepid, {}).get("depth_ppm", None)
        true_duration = truth_dict.get(kepid, {}).get("duration_hours", None)
        true_bin = truth_dict.get(kepid, {}).get("bin", None)

        best = scored[0] if scored else None

        # Check if best period is correct
        is_correct = np.nan
        if best and true_period and true_period > 0:
            ratio = best["best_period"] / true_period
            min_alias_dist = min([abs(ratio - ar) for ar in ALIAS_RATIOS])
            is_correct = 1 if min_alias_dist < 0.02 else 0

        # Also check rank-0 (highest SDE) for comparison
        rank0_period = None
        rank0_correct = np.nan
        if star_cands:
            rank0_cand = max(star_cands, key=lambda c: c["bls_sde"])
            rank0_period = rank0_cand["period"]
            if true_period and true_period > 0:
                ratio0 = rank0_period / true_period
                min_alias0 = min([abs(ratio0 - ar) for ar in ALIAS_RATIOS])
                rank0_correct = 1 if min_alias0 < 0.02 else 0

        star_results.append({
            "kepid": kepid,
            "has_planet": hp,
            "is_correct": is_correct,
            "rank0_correct": rank0_correct,
            "true_period": true_period,
            "true_depth_ppm": true_depth,
            "true_duration_hours": true_duration,
            "true_bin": true_bin,
            "n_clusters": len(scored),
            **(best if best else {}),
        })

        # Store all cluster features for analysis
        for rank_i, sc in enumerate(scored):
            all_cluster_features.append({
                "kepid": kepid,
                "cluster_rank": rank_i,
                "has_planet": hp,
                "is_correct_cluster": is_correct if rank_i == 0 else np.nan,
                **sc,
            })

    results_df = pd.DataFrame(star_results)
    clusters_df = pd.DataFrame(all_cluster_features)

    # ── Step 3: Evaluate ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: Evaluate consensus period selection")
    print(f"{'='*60}")

    # Period recovery
    truth_stars = results_df[results_df["is_correct"].notna()]
    n_truth = len(truth_stars)

    if n_truth > 0:
        consensus_correct = int((truth_stars["is_correct"] == 1).sum())
        consensus_recovery = consensus_correct / n_truth

        rank0_correct = int((truth_stars["rank0_correct"] == 1).sum())
        rank0_recovery = rank0_correct / n_truth

        # Oracle: any candidate correct
        star_cands_map = {}
        for c in all_candidates:
            if c["kepid"] not in star_cands_map:
                star_cands_map[c["kepid"]] = []
            star_cands_map[c["kepid"]].append(c)

        oracle_correct = 0
        for kepid in truth_dict:
            if kepid not in star_cands_map:
                continue
            true_p = truth_dict[kepid].get("period_days", 0)
            if true_p <= 0:
                continue
            for c in star_cands_map[kepid]:
                ratio = c["period"] / true_p
                min_alias = min([abs(ratio - ar) for ar in ALIAS_RATIOS])
                if min_alias < 0.02:
                    oracle_correct += 1
                    break

        print(f"\n  Period Recovery (truth stars: {n_truth}):")
        print(f"    Rank-0 (highest SDE):  {rank0_correct}/{n_truth} = {rank0_recovery:.3f}")
        print(f"    Consensus selected:    {consensus_correct}/{n_truth} = {consensus_recovery:.3f}")
        print(f"    Top-20 oracle:         {oracle_correct}/{n_truth} = {oracle_correct/n_truth:.3f}")
        print(f"    Improvement over rank-0: {consensus_recovery - rank0_recovery:+.3f}")

        # Per-bin breakdown
        if "true_bin" in truth_stars.columns:
            print(f"\n  Per-bin breakdown:")
            for bin_name in ["earth_analog", "shallow", "mid", "deep"]:
                bin_stars = truth_stars[truth_stars["true_bin"] == bin_name]
                if len(bin_stars) > 0:
                    bin_correct = int((bin_stars["is_correct"] == 1).sum())
                    bin_r0 = int((bin_stars["rank0_correct"] == 1).sum())
                    print(f"    {bin_name:>14s}: consensus={bin_correct}/{len(bin_stars)}, "
                          f"rank0={bin_r0}/{len(bin_stars)}")
    else:
        print("\n  No truth stars to evaluate!")
        consensus_recovery = 0

    # Detection quality: does cluster quality separate TP vs FP?
    print(f"\n{'='*60}")
    print("STEP 4: Detection quality analysis")
    print(f"{'='*60}")

    # For stars with has_planet=1 vs has_planet=0, compare cluster features
    hp_stars = results_df[results_df["has_planet"] == 1]
    np_stars = results_df[results_df["has_planet"] == 0]

    if len(hp_stars) > 0 and len(np_stars) > 0:
        print(f"\n  Planet stars: {len(hp_stars)}, No-planet stars: {len(np_stars)}")

        # Compare key cluster features
        for feat in ["best_sde", "window_support", "n_windows", "max_reranker_score",
                      "best_transit_coverage", "dist_to_systematic", "consensus_score",
                      "cross_window_agreement", "deep_agreement"]:
            if feat in results_df.columns:
                hp_val = hp_stars[feat].median()
                np_val = np_stars[feat].median()
                ratio = hp_val / np_val if abs(np_val) > 1e-10 else float("inf")
                print(f"    {feat:<30s} planet={hp_val:10.4f}  nopl={np_val:10.4f}  ratio={ratio:.2f}")

        # AP using consensus_score as confidence
        from sklearn.metrics import average_precision_score, roc_auc_score
        y_true = results_df["has_planet"].values
        y_score = results_df["consensus_score"].values

        # Handle NaN scores
        valid = ~np.isnan(y_score)
        if valid.sum() > 10:
            ap = average_precision_score(y_true[valid], y_score[valid])
            try:
                auc = roc_auc_score(y_true[valid], y_score[valid])
            except:
                auc = 0.5
            print(f"\n  Detection AP (consensus_score): {ap:.4f}")
            print(f"  Detection AUC (consensus_score): {auc:.4f}")
        else:
            print(f"\n  Not enough valid scores for AP computation")

    # ── Per-star detail for truth stars ──────────────────────────────
    print(f"\n--- Per-star detail (truth stars) ---")
    for _, row in truth_stars.sort_values("consensus_score", ascending=False).iterrows():
        correct_str = "YES" if row["is_correct"] == 1 else "NO"
        r0_str = "YES" if row["rank0_correct"] == 1 else "NO"
        print(f"  KepID {int(row['kepid'])}: "
              f"P={row.get('best_period', 0):.2f}d "
              f"(true={row.get('true_period', 0):.2f}d) "
              f"[{correct_str}] "
              f"SDE={row.get('best_sde', 0):.1f} "
              f"score={row.get('consensus_score', 0):.3f} "
              f"win={row.get('n_windows', 0)}/3 "
              f"rank0={r0_str} "
              f"bin={row.get('true_bin', '?')}")

    # ── Save ─────────────────────────────────────────────────────────
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved {output_csv}")

    clusters_out = output_csv.replace(".csv", "_clusters.csv")
    clusters_df.to_csv(clusters_out, index=False)
    print(f"Saved {clusters_out}")

    return results_df, clusters_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    args = parser.parse_args()

    output = f"outputs/{args.split}_consensus_v1.csv"
    results, clusters = build_consensus(args.split, output)
