"""Build candidate-level features for reranking.

For each star, extracts features at each of the top-6 BLS peaks independently.
Labels each candidate as positive/negative using truth.csv.
"""
import sys, time, io
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.stats import skew, kurtosis
from astropy.timeseries import LombScargle

from src.config import (
    PERIOD_MIN_DAYS, PERIOD_MAX_DAYS,
    BLS_COARSE_RESOLUTION, BLS_FINE_RESOLUTION,
    BLS_FINE_WINDOW_FRAC, BLS_DURATIONS_DAYS, BLS_OBJECTIVE,
    N_CANDIDATE_PEAKS, KEPLER_SYSTEMATIC_PERIODS,
)
from src.period_search import clean_for_bls, _compute_quarter_breaks


def extract_candidate_features(time, flux, period, t0, depth, duration):
    """Extract features for a single candidate period."""
    features = {}

    # Basic BLS features
    features["bls_period"] = period
    features["bls_depth_ppm"] = depth * 1e6
    features["bls_duration_hours"] = duration * 24

    # Count transits
    baseline = time.max() - time.min()
    n_transits = baseline / period if period > 0 else 0
    features["bls_n_transits"] = n_transits

    # SNR
    valid = ~np.isnan(flux)
    scatter = np.std(flux[valid]) * 1e6 if valid.sum() > 0 else 1.0
    snr = (depth * 1e6) / (scatter / np.sqrt(n_transits)) if n_transits > 0 and scatter > 0 else 0
    features["bls_snr"] = snr

    # Systematic distance
    distances = [abs(period - sp) / sp for sp in KEPLER_SYSTEMATIC_PERIODS]
    features["dist_to_systematic"] = min(distances)

    # Transit event features
    duration_days = duration
    valid = ~np.isnan(flux)
    t = time[valid]
    f = flux[valid]

    # Phase fold
    phase = ((t - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    in_transit = np.abs(phase) < (duration_days / (2 * period))

    # Find individual events
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
        phase_local = ((transit_t - t0) % period) / period
        phase_local[phase_local > 0.5] -= 1.0
        in_t = np.abs(phase_local) < (duration_days / (2 * period))
        out_t = np.abs(phase_local) > (duration_days / (2 * period) + 0.05)

        if in_t.sum() > 0 and out_t.sum() > 0:
            d = np.median(transit_f[out_t]) - np.median(transit_f[in_t])
            depths.append(d * 1e6)
            transit_mid = np.median(transit_t[in_t])
            expected = t0 + tn * period
            timings.append(transit_mid - expected)

    n_observed = len(depths)
    n_expected = baseline / period if period > 0 else 0
    features["transit_coverage"] = n_observed / n_expected if n_expected > 0 else 0
    features["n_expected"] = n_expected

    if n_observed >= 2:
        depths = np.array(depths)
        timings = np.array(timings)
        features["depth_cv"] = np.std(depths) / abs(np.mean(depths)) if abs(np.mean(depths)) > 1e-10 else 0
        features["depth_mad_ratio"] = np.median(np.abs(depths - np.median(depths))) / abs(np.median(depths)) if abs(np.median(depths)) > 1e-10 else 0
        features["timing_rms"] = np.sqrt(np.mean(timings**2))
    else:
        features["depth_cv"] = 0
        features["depth_mad_ratio"] = 0
        features["timing_rms"] = 0

    # Odd-even depth ratio — compare odd vs even transit events
    transit_times = t[in_transit]
    if len(transit_times) > 4:
        transit_nums_at = np.floor((transit_times - t0) / period).astype(int)
        odd_t = transit_nums_at % 2 == 1
        even_t = transit_nums_at % 2 == 0

        odd_transits = np.unique(transit_nums_at[odd_t])
        even_transits = np.unique(transit_nums_at[even_t])

        odd_depths = []
        for tn in odd_transits:
            t_mask = transit_numbers == tn
            if t_mask.sum() < 3:
                continue
            transit_f_local = f[t_mask]
            transit_t_local = t[t_mask]
            phase_local = ((transit_t_local - t0) % period) / period
            phase_local[phase_local > 0.5] -= 1.0
            in_t = np.abs(phase_local) < (duration_days / (2 * period))
            out_t = np.abs(phase_local) > (duration_days / (2 * period) + 0.05)
            if in_t.sum() > 0 and out_t.sum() > 0:
                odd_depths.append(np.median(transit_f_local[out_t]) - np.median(transit_f_local[in_t]))

        even_depths = []
        for tn in even_transits:
            t_mask = transit_numbers == tn
            if t_mask.sum() < 3:
                continue
            transit_f_local = f[t_mask]
            transit_t_local = t[t_mask]
            phase_local = ((transit_t_local - t0) % period) / period
            phase_local[phase_local > 0.5] -= 1.0
            in_t = np.abs(phase_local) < (duration_days / (2 * period))
            out_t = np.abs(phase_local) > (duration_days / (2 * period) + 0.05)
            if in_t.sum() > 0 and out_t.sum() > 0:
                even_depths.append(np.median(transit_f_local[out_t]) - np.median(transit_f_local[in_t]))

        if odd_depths and even_depths:
            odd_med = np.median(odd_depths)
            even_med = np.median(even_depths)
            features["odd_even_depth_ratio"] = abs(odd_med / even_med) if abs(even_med) > 1e-10 else 1.0
        else:
            features["odd_even_depth_ratio"] = 1.0
    else:
        features["odd_even_depth_ratio"] = 1.0

    # Secondary eclipse at phase 0.5
    sec_phase = np.abs(phase - 0.5)
    sec_in = sec_phase < (duration_days / (2 * period))
    out_transit = np.abs(phase) > (duration_days / (2 * period) + 0.1)
    if sec_in.sum() > 2 and out_transit.sum() > 50:
        features["secondary_eclipse_depth"] = (np.median(f[out_transit]) - np.median(f[sec_in])) * 1e6
    else:
        features["secondary_eclipse_depth"] = 0

    # Quarter signal fraction
    quarters = np.unique(np.floor(t / 90).astype(int))
    n_with_signal = 0
    for q in quarters:
        q_mask = (np.floor(t / 90).astype(int) == q) & in_transit
        if q_mask.sum() > 1:
            n_with_signal += 1
    features["quarter_signal_fraction"] = n_with_signal / max(1, len(quarters))

    # === NEW: Folded-curve shape features ===

    # 1. In-transit vs out-of-transit scatter ratio
    in_flux = f[in_transit]
    out_flux = f[~in_transit & (np.abs(phase) < 0.4)]
    if len(in_flux) > 5 and len(out_flux) > 50:
        in_scatter = np.std(in_flux)
        out_scatter = np.std(out_flux)
        features["in_out_scatter_ratio"] = in_scatter / out_scatter if out_scatter > 0 else 1.0
    else:
        features["in_out_scatter_ratio"] = 1.0

    # 2. Folded curve box-fit quality (how box-shaped is the dip)
    # Bin the folded curve and measure how well a box fits
    n_bins = 50
    phase_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_idx = np.digitize(phase, phase_edges) - 1
    bin_medians = np.array([np.median(f[bin_idx == i]) if (bin_idx == i).sum() > 0 else np.nan
                           for i in range(n_bins)])
    bin_phases = (phase_edges[:-1] + phase_edges[1:]) / 2

    # Fit a box: transit bins vs out-of-transit bins
    transit_bins = np.abs(bin_phases) < (duration_days / (2 * period))
    out_bins = np.abs(bin_phases) > (duration_days / (2 * period) + 0.05)
    valid_bins = ~np.isnan(bin_medians)

    if transit_bins.sum() > 0 and out_bins.sum() > 0 and valid_bins.sum() > 5:
        transit_level = np.nanmedian(bin_medians[transit_bins & valid_bins])
        out_level = np.nanmedian(bin_medians[out_bins & valid_bins])
        box_depth = out_level - transit_level

        # Residuals from box model
        model = np.where(transit_bins, transit_level, out_level)
        residuals = bin_medians[valid_bins] - model[valid_bins]
        noise = np.nanstd(bin_medians[out_bins & valid_bins]) if out_bins.sum() > 2 else 1.0
        features["box_fit_snr"] = box_depth / noise if noise > 0 else 0
        features["box_fit_residual_rms"] = np.nanstd(residuals) / noise if noise > 0 else 0
    else:
        features["box_fit_snr"] = 0
        features["box_fit_residual_rms"] = 1.0

    # 3. Event depth trend (is depth consistent over time?)
    if n_observed >= 3:
        event_times = []
        for tn in unique_transits:
            t_mask = transit_numbers == tn
            if t_mask.sum() >= 3:
                event_times.append(np.median(t[t_mask]))
        if len(depths) >= 3 and len(event_times) >= 3:
            # Slope of depth vs time
            event_times = np.array(event_times[:len(depths)])
            depth_arr = np.array(depths[:len(event_times)])
            if np.std(event_times) > 0:
                slope = np.polyfit(event_times, depth_arr, 1)[0]
                features["depth_trend_slope"] = slope
            else:
                features["depth_trend_slope"] = 0
        else:
            features["depth_trend_slope"] = 0
    else:
        features["depth_trend_slope"] = 0

    # 4. Duration consistency (individual transit durations)
    durations = []
    for tn in unique_transits[:20]:
        t_mask = transit_numbers == tn
        if t_mask.sum() >= 3:
            t_event = t[t_mask]
            f_event = f[t_mask]
            ph = ((t_event - t0) % period) / period
            ph[ph > 0.5] -= 1.0
            dur = (ph.max() - ph.min()) * period
            durations.append(dur)
    if len(durations) >= 2:
        durations = np.array(durations)
        features["duration_consistency"] = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
    else:
        features["duration_consistency"] = 0

    # 5. Baseline RMS (scatter far from transit)
    far_out = np.abs(phase) > 0.3
    if far_out.sum() > 50:
        features["baseline_rms"] = np.std(f[far_out]) * 1e6
    else:
        features["baseline_rms"] = 0

    return features


def bls_search_multi(time, flux):
    """BLS search returning top N peaks."""
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
        if len(peaks) >= N_CANDIDATE_PEAKS:
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

        # Compute SDE
        med = np.nanmedian(power)
        mad = np.nanmedian(np.abs(power - med))
        sde = (power[i] - med) / (1.4826 * mad) if mad > 0 else 0

        results.append({
            "period": float(r.period[j]),
            "t0": float(r.transit_time[j]),
            "depth": float(r.depth[j]),
            "duration": float(r.duration[j]),
            "sde": sde,
        })

    results.sort(key=lambda x: x["sde"], reverse=True)
    return results


def build_candidates(split_name, output_csv):
    """Build candidate-level feature matrix."""
    from src.data_loader import load_all_light_curves, load_labels, load_truth

    print(f"\n{'='*60}", flush=True)
    print(f"Building candidates for {split_name}...", flush=True)
    print(f"{'='*60}", flush=True)

    lc = load_all_light_curves(split_name)
    labels = load_labels(split_name)
    truth = load_truth(split_name)

    labels_dict = {}
    if labels is not None:
        for _, row in labels.iterrows():
            labels_dict[int(row["kepid"])] = row.to_dict()

    truth_dict = {}
    if truth is not None and not truth.empty:
        for _, row in truth.iterrows():
            truth_dict[int(row["kepid"])] = row.to_dict()

    from src.data_loader import make_target
    target_info = make_target(labels, truth)
    has_planet_map = target_info["target"]  # kepid -> 0 or 1

    kepids = sorted(lc.keys())
    print(f"  {len(kepids)} stars, {len(truth_dict)} with truth, {target_info['n_positives']} has_planet (correct target)", flush=True)

    t0 = time.time()
    all_candidates = []

    for i, kepid in enumerate(kepids):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(kepids) - i - 1)
            print(f"  [{i+1}/{len(kepids)}] {elapsed:.0f}s, ETA {eta:.0f}s, {len(all_candidates)} candidates", flush=True)

        t, f = clean_for_bls(lc[kepid], window_days=0.5)
        if len(t) < 200:
            continue

        # Get top 20 BLS peaks (expanded from 6)
        peaks = bls_search_multi(t, f)

        # Get true period if available
        true_period = truth_dict.get(kepid, {}).get("period_days", None)

        # Correct competition target
        hp = has_planet_map.get(kepid, 0)

        # 372d vetting: fraction of top-20 peaks in systematic band (330-400d)
        n_sys_peaks = sum(1 for p in peaks if 330 < p["period"] < 400)
        systematic_fraction = n_sys_peaks / len(peaks) if peaks else 0

        for rank, peak in enumerate(peaks):
            # Extract features at this candidate period
            cand = extract_candidate_features(
                t, f, peak["period"], peak["t0"], peak["depth"], peak["duration"]
            )
            cand["kepid"] = kepid
            cand["rank"] = rank
            cand["bls_sde"] = peak["sde"]
            cand["has_planet"] = hp  # correct competition target

            # 372d vetting features (per-candidate)
            cand["systematic_fraction"] = systematic_fraction
            cand["dist_to_systematic"] = abs(peak["period"] - 372.0)
            cand["near_systematic"] = 1 if abs(peak["period"] - 372.0) < 20 else 0

            # Label: is this candidate close to the true period?
            # Only label for stars WITH truth (injected signals)
            if true_period is not None and true_period > 0:
                ratio = peak["period"] / true_period
                alias_ratios = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]
                min_alias_dist = min([abs(ratio - ar) for ar in alias_ratios])
                cand["is_correct"] = 1 if min_alias_dist < 0.02 else 0
            else:
                cand["is_correct"] = np.nan  # unknown — do NOT use as negative in training

            # Star-level features
            if kepid in labels_dict:
                for key in ["kepmag", "teff", "logg", "radius"]:
                    cand[key] = float(labels_dict[kepid].get(key, 0) or 0)
            else:
                for key in ["kepmag", "teff", "logg", "radius"]:
                    cand[key] = 0

            all_candidates.append(cand)

    total = time.time() - t0
    print(f"  Done: {total:.0f}s, {len(all_candidates)} candidates", flush=True)

    df = pd.DataFrame(all_candidates)

    # Report coverage with correct target
    n_with_truth = df["is_correct"].notna().sum()
    n_correct = (df["is_correct"] == 1).sum()
    n_injected = len(truth_dict)
    print(f"  {n_with_truth} candidates from {n_injected} truth stars, {n_correct} correct", flush=True)

    # Coverage by rank (for truth stars only)
    truth_df = df[df["is_correct"].notna()]
    if len(truth_df) > 0:
        print(f"  Coverage by rank (truth stars only):", flush=True)
        for rank in range(min(20, N_CANDIDATE_PEAKS)):
            rc = (truth_df[(truth_df["rank"] == rank)]["is_correct"] == 1).sum()
            total_at_rank = len(truth_df[truth_df["rank"] == rank])
            if total_at_rank == 0:
                break
            print(f"    Rank {rank:>2}: {rc}/{total_at_rank} correct", flush=True)

        # Oracle: how many stars have at least one correct candidate?
        oracle = truth_df.groupby("kepid")["is_correct"].max()
        n_oracle = (oracle == 1).sum()
        print(f"  Oracle (any rank correct): {n_oracle}/{len(truth_dict)} = {n_oracle/len(truth_dict)*100:.1f}%", flush=True)

    # Has_planet distribution
    hp_counts = df["has_planet"].value_counts().to_dict()
    print(f"  has_planet: {hp_counts}", flush=True)

    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}", flush=True)
    return df


if __name__ == "__main__":
    train_df = build_candidates("train", r"C:\Users\husai\OneDrive\Pictures\Documents\AstroBit\outputs\train_candidates.csv")
    dev_df = build_candidates("dev", r"C:\Users\husai\OneDrive\Pictures\Documents\AstroBit\outputs\dev_candidates.csv")
