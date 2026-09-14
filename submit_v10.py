"""
v10: Multi-window consensus for period selection.
1. Generate top-20 candidates for 3 detrending windows
2. Alias-cluster periods across windows
3. Score each cluster with consensus features
4. Select period from best cluster
5. Keep v7 detection: SDE>8 + detector confidence
"""
import sys, time, io, os
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter

from src.config import (
    PERIOD_MIN_DAYS, PERIOD_MAX_DAYS,
    BLS_COARSE_RESOLUTION, BLS_FINE_RESOLUTION,
    BLS_FINE_WINDOW_FRAC, BLS_DURATIONS_DAYS, BLS_OBJECTIVE,
    N_CANDIDATE_PEAKS, KEPLER_SYSTEMATIC_PERIODS,
)
from src.period_search import _compute_quarter_breaks, compute_sde
from src.data_loader import load_all_light_curves, load_labels, load_truth, make_target
from sklearn.metrics import average_precision_score


# ── Detrending (from bake-off) ──────────────────────────────────────

def detrend_wotan_biweight(time, flux, quarter, window_days):
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


# ── BLS search ──────────────────────────────────────────────────────

def bls_search_multi(time, flux, n_candidates=20):
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
            "depth_ppm": float(r.depth[j] * 1e6),
            "duration_hours": float(r.duration[j] * 24),
            "t0": float(r.transit_time[j]),
            "depth": float(r.depth[j]),
            "duration": float(r.duration[j]),
            "sde": sde,
        })
    results.sort(key=lambda x: x["sde"], reverse=True)
    return results


# ── Alias clustering ─────────────────────────────────────────────────

ALIAS_RATIOS = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]
ALIAS_TOLERANCE = 0.02  # 2%


def are_aliases(p1, p2):
    """Check if two periods are aliases of each other."""
    if p1 <= 0 or p2 <= 0:
        return False
    ratio = p1 / p2
    return min([abs(ratio - ar) for ar in ALIAS_RATIOS]) < ALIAS_TOLERANCE


def cluster_periods(candidates):
    """Group candidates into alias clusters across all windows.

    candidates: list of dicts with 'period', 'sde', 'window', 'rank'
    Returns list of clusters, each cluster is a list of candidate indices.
    """
    n = len(candidates)
    assigned = [-1] * n
    clusters = []

    for i in range(n):
        if assigned[i] >= 0:
            continue
        # Start new cluster
        cluster_id = len(clusters)
        cluster = [i]
        assigned[i] = cluster_id
        # Find all candidates that alias with this one
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            # Check if candidate j aliases with any candidate already in cluster
            for k in cluster:
                if are_aliases(candidates[k]["period"], candidates[j]["period"]):
                    cluster.append(j)
                    assigned[j] = cluster_id
                    break
        clusters.append(cluster)
    return clusters


def score_cluster(cluster_indices, candidates):
    """Score a cluster of alias-aligned candidates.

    Returns consensus features for the cluster.
    """
    cands = [candidates[i] for i in cluster_indices]

    # Basic stats
    sdes = [c["sde"] for c in cands]
    periods = [c["period"] for c in cands]
    windows = [c["window"] for c in cands]
    ranks = [c["rank"] for c in cands]

    max_sde = max(sdes)
    mean_sde = np.mean(sdes)
    sum_sde = np.sum(sdes)

    # Window support
    unique_windows = set(windows)
    n_windows = len(unique_windows)
    window_support = n_windows / 3.0  # fraction of 3 windows

    # Window-specific flags
    has_05d = "biweight_0.5d" in unique_windows
    has_2d = "biweight_2d" in unique_windows
    has_savgol = "savgol_2d" in unique_windows

    # Consensus agreement
    # Does 0.5d see it AND (2d OR savgol see it)?
    cross_window_agreement = int(has_05d and (has_2d or has_savgol))
    # Do 2d and savgol agree (both see an alias)?
    deep_agreement = int(has_2d and has_savgol)

    # Rank consistency
    min_rank = min(ranks)
    mean_rank = np.mean(ranks)
    rank_std = np.std(ranks) if len(ranks) > 1 else 0

    # Best period (from highest SDE candidate)
    best_idx = np.argmax(sdes)
    best_period = periods[best_idx]
    best_sde = sdes[best_idx]
    best_depth = cands[best_idx]["depth_ppm"]
    best_duration = cands[best_idx]["duration_hours"]
    best_t0 = cands[best_idx]["t0"]

    # Systematic distance
    dist_to_sys = min([abs(best_period - sp) / sp for sp in KEPLER_SYSTEMATIC_PERIODS])
    near_systematic = 1 if abs(best_period - 372.0) < 20 else 0

    # Only-seen-in-0.5d flag (likely artifact if true)
    only_05d = int(has_05d and not has_2d and not has_savgol)

    # Consensus score (composite)
    consensus_score = (
        min(max_sde / 20, 1.0) * 0.25          # Strong signal
        + window_support * 0.25                    # Recurs across windows
        + cross_window_agreement * 0.20            # 0.5d + deep agree
        + deep_agreement * 0.15                    # 2d + savgol agree
        + (1.0 - min_rank / 10) * 0.10            # Low rank = good
        + (1.0 - near_systematic) * 0.05           # Not near 372d
    )

    # Penalize only-0.5d candidates
    if only_05d:
        consensus_score *= 0.5

    return {
        "best_period": best_period,
        "best_sde": best_sde,
        "best_depth_ppm": best_depth,
        "best_duration_hours": best_duration,
        "best_t0": best_t0,
        "max_sde": max_sde,
        "mean_sde": mean_sde,
        "sum_sde": sum_sde,
        "n_candidates": len(cands),
        "n_windows": n_windows,
        "window_support": window_support,
        "has_05d": int(has_05d),
        "has_2d": int(has_2d),
        "has_savgol": int(has_savgol),
        "cross_window_agreement": cross_window_agreement,
        "deep_agreement": deep_agreement,
        "min_rank": min_rank,
        "mean_rank": mean_rank,
        "rank_std": rank_std,
        "dist_to_systematic": dist_to_sys,
        "near_systematic": near_systematic,
        "only_05d": only_05d,
        "consensus_score": consensus_score,
    }


# ── Main pipeline ────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("v10: MULTI-WINDOW CONSENSUS PERIOD SELECTION")
    print("=" * 70)

    # Load data
    print("\nLoading dev light curves...")
    lc = load_all_light_curves("dev")
    labels = load_labels("dev")
    truth = load_truth("dev")
    target_info = make_target(labels, truth)
    has_planet_map = target_info["target"]

    truth_dict = {}
    if truth is not None:
        for _, row in truth.iterrows():
            truth_dict[int(row["kepid"])] = row.to_dict()

    print(f"  {len(lc)} stars, {target_info['n_positives']} positive")

    # Load existing detector and reranker
    det = pd.read_csv("outputs/dev_detected_v7.csv")
    det_map = dict(zip(det["kepid"], det["proba"]))
    rerank = pd.read_csv("outputs/dev_reranked_v4.csv")

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
    all_star_candidates = {}  # kepid -> list of candidate dicts

    for i, (kepid, lc_data) in enumerate(lc.items()):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(lc) - i - 1)
            print(f"  [{i+1}/{len(lc)}] {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)

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

        star_cands = []
        for win_name, detrend_fn in windows:
            t_d, f_d = detrend_fn(t, f, q)
            if len(t_d) < 200:
                continue
            peaks = bls_search_multi(t_d, f_d, n_candidates=20)
            for rank, peak in enumerate(peaks):
                star_cands.append({
                    "kepid": kepid,
                    "window": win_name,
                    "rank": rank,
                    "period": peak["period"],
                    "sde": peak["sde"],
                    "depth_ppm": peak["depth_ppm"],
                    "duration_hours": peak["duration_hours"],
                    "t0": peak["t0"],
                    "depth": peak["depth"],
                    "duration": peak["duration"],
                })

        all_star_candidates[kepid] = star_cands

    elapsed = time.time() - t0
    total_cands = sum(len(v) for v in all_star_candidates.values())
    print(f"  Done in {elapsed:.0f}s, {total_cands} total candidates")

    # ── Step 2: Cluster and score per star ───────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2: Alias-cluster and score")
    print(f"{'='*60}")

    star_results = []
    for kepid, cands in all_star_candidates.items():
        if not cands:
            continue

        # Cluster periods
        clusters = cluster_periods(cands)

        # Score each cluster
        scored = []
        for cluster_indices in clusters:
            sc = score_cluster(cluster_indices, cands)
            scored.append(sc)

        # Sort by consensus score
        scored.sort(key=lambda x: x["consensus_score"], reverse=True)

        # Best cluster = selected period
        best = scored[0]
        hp = has_planet_map.get(kepid, 0)
        true_period = truth_dict.get(kepid, {}).get("period_days", None)

        # Check if best period is correct
        if true_period and true_period > 0:
            ratio = best["best_period"] / true_period
            alias_ratios = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]
            min_alias_dist = min([abs(ratio - ar) for ar in alias_ratios])
            is_correct = 1 if min_alias_dist < 0.02 else 0
        else:
            is_correct = np.nan

        star_results.append({
            "kepid": kepid,
            "has_planet": hp,
            "is_correct": is_correct,
            "true_period": true_period,
            **best,
            "n_clusters": len(scored),
        })

    results_df = pd.DataFrame(star_results)

    # ── Step 3: Evaluate ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: Evaluate v10 period selection")
    print(f"{'='*60}")

    # SDE>8 detection
    results_df["prediction"] = (results_df["best_sde"] > 8).astype(int)
    tp = ((results_df["prediction"] == 1) & (results_df["has_planet"] == 1)).sum()
    fp = ((results_df["prediction"] == 1) & (results_df["has_planet"] == 0)).sum()
    fn = ((results_df["prediction"] == 0) & (results_df["has_planet"] == 1)).sum()
    tn = ((results_df["prediction"] == 0) & (results_df["has_planet"] == 0)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print(f"\n  Detection (SDE>8):")
    print(f"    Predicted: {tp+fp}, TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"    Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

    # Period recovery
    truth_stars = results_df[results_df["is_correct"].notna()]
    if len(truth_stars) > 0:
        rank0_correct = (truth_stars["is_correct"] == 1).sum()
        rank0_recovery = rank0_correct / len(truth_stars)
        print(f"\n  Rank-0 period recovery: {rank0_correct}/{len(truth_stars)} = {rank0_recovery:.3f}")
    else:
        rank0_recovery = 0
        print(f"\n  Rank-0 period recovery: 0/0")

    # Top-20 oracle (check if any cluster has the correct period)
    oracle_correct = 0
    for kepid, cands in all_star_candidates.items():
        true_period = truth_dict.get(kepid, {}).get("period_days", None)
        if true_period is None or true_period <= 0:
            continue
        # Check if any candidate period is an alias of true_period
        found = False
        for c in cands:
            ratio = c["period"] / true_period
            alias_ratios = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]
            min_alias_dist = min([abs(ratio - ar) for ar in alias_ratios])
            if min_alias_dist < 0.02:
                found = True
                break
        if found:
            oracle_correct += 1
    n_truth = len([k for k, v in truth_dict.items() if v.get("period_days", 0) > 0])
    oracle_recovery = oracle_correct / max(n_truth, 1)
    print(f"  Top-20 oracle recovery: {oracle_correct}/{n_truth} = {oracle_recovery:.3f}")

    # Near-372d fraction
    n_near = (results_df["near_systematic"] == 1).sum()
    n_total = len(results_df)
    print(f"  Near-372d fraction: {n_near}/{n_total} = {n_near/n_total:.3f}")

    # AP using consensus_score as confidence
    y_true = results_df["has_planet"].values
    y_score = results_df["consensus_score"].values
    ap = average_precision_score(y_true, y_score)
    print(f"  AP (consensus score): {ap:.3f}")

    # ── Compare with v7 rank-0 ──────────────────────────────────────
    print(f"\n--- v10 vs v7 (rank-0 period recovery) ---")
    # v7 rank-0 recovery (from candidates_v4)
    v7_cands = pd.read_csv("outputs/dev_candidates_v4.csv")
    v7_rank0 = v7_cands[v7_cands["rank"] == 0]
    v7_rank0_eval = v7_rank0[v7_rank0["is_correct"].notna()]
    v7_rank0_correct = (v7_rank0_eval["is_correct"] == 1).sum()
    v7_rank0_recovery = v7_rank0_correct / len(v7_rank0_eval) if len(v7_rank0_eval) > 0 else 0
    print(f"  v7 rank-0 recovery: {v7_rank0_correct}/{len(v7_rank0_eval)} = {v7_rank0_recovery:.3f}")
    print(f"  v10 rank-0 recovery: {rank0_correct}/{len(truth_stars)} = {rank0_recovery:.3f}")
    print(f"  Improvement: {rank0_recovery - v7_rank0_recovery:+.3f}")

    # ── Per-star detail ──────────────────────────────────────────────
    print(f"\n--- Per-star period selection (truth stars) ---")
    for _, row in truth_stars.sort_values("consensus_score", ascending=False).iterrows():
        correct_str = "YES" if row["is_correct"] == 1 else "NO"
        windows_str = f"0.5d={'Y' if row['has_05d'] else 'N'} 2d={'Y' if row['has_2d'] else 'N'} sg={'Y' if row['has_savgol'] else 'N'}"
        print(f"  KepID {int(row['kepid'])}: P={row['best_period']:.2f}d "
              f"(true={row['true_period']:.2f}d) [{correct_str}] "
              f"SDE={row['best_sde']:.1f} score={row['consensus_score']:.3f} "
              f"win={row['n_windows']}/3 {windows_str}")

    # ── Save results ─────────────────────────────────────────────────
    results_df.to_csv("outputs/dev_v10_consensus.csv", index=False)
    print(f"\nSaved outputs/dev_v10_consensus.csv")

    return results_df


if __name__ == "__main__":
    results = main()
