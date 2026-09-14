"""
Targeted Injection Augmentation for Ranker — v2.

Failure mode: 8/10 missed stars have selected period within 20d of 372d/186d/93d.
The ranker picks systematic aliases even when true period has higher SDE.

Strategy: inject shallow signals at 50-200d into quiet stars. The existing
Kepler 372d systematic creates competing peaks. Ranker must learn to prefer
the non-systematic period.

~120 targeted injections, ~10 minutes.
"""
import sys, warnings, io, time
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRanker

from src.data_loader import load_all_light_curves, load_labels, load_truth, make_target
from src.period_search import clean_for_bls
from src.config import (
    PERIOD_MIN_DAYS, PERIOD_MAX_DAYS,
    BLS_COARSE_RESOLUTION, BLS_FINE_RESOLUTION,
    BLS_FINE_WINDOW_FRAC, BLS_DURATIONS_DAYS, BLS_OBJECTIVE,
    N_CANDIDATE_PEAKS, KEPLER_SYSTEMATIC_PERIODS,
    OUTPUTS_DIR, MODELS_DIR,
)
from build_candidates import extract_candidate_features, bls_search_multi
from train_reranker_v3 import CANDIDATE_FEATURES

ALIAS_RATIOS = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]


def is_alias(p1, p2):
    if p1 <= 0 or p2 <= 0:
        return False
    return min([abs(p1 / p2 - ar) for ar in ALIAS_RATIOS]) < 0.02


def inject_transit(time_arr, flux_arr, period_days, depth_ppm, duration_hours, t0):
    depth_frac = depth_ppm * 1e-6
    dur_days = duration_hours / 24.0
    t_min, t_max = time_arr.min(), time_arr.max()
    n_orbits = int(np.ceil((t_max - t0) / period_days)) + 1
    centers = t0 + np.arange(-n_orbits, n_orbits + 1) * period_days
    centers = centers[(centers >= t_min - dur_days) & (centers <= t_max + dur_days)]
    flux = flux_arr.copy()
    for tc in centers:
        in_t = np.abs(time_arr - tc) < (dur_days / 2.0)
        flux[in_t] *= (1.0 - depth_frac)
    return flux


def process_one_host(lc_df, period, depth_ppm, duration_hours, t0, kepid_syn):
    m = (lc_df["quality"].values == 0) & np.isfinite(lc_df["flux"].values)
    t = lc_df["time"].values[m].copy()
    f_raw = lc_df["flux"].values[m].astype(float).copy()
    q = lc_df["quarter"].values[m].copy()

    if len(t) < 1000:
        return []

    f_injected = inject_transit(t, f_raw, period, depth_ppm, duration_hours, t0)

    for qq in np.unique(q):
        s = q == qq
        med = np.median(f_injected[s])
        if med > 0:
            f_injected[s] /= med

    try:
        from wotan import flatten
        from src.period_search import _compute_quarter_breaks
        breaks = _compute_quarter_breaks(q)
        _, trend = flatten(t, f_injected, method="biweight", window_length=0.5,
                          break_tolerance=200,
                          break_location=breaks if len(breaks) > 0 else None,
                          return_trend=True)
        ok = np.isfinite(trend) & (trend > 0)
        t, f = t[ok], f_injected[ok] / trend[ok]
    except Exception:
        cadence = np.median(np.diff(t))
        k = max(5, int(0.5 / cadence) | 1)
        if k % 2 == 0:
            k += 1
        trend = pd.Series(f_injected).rolling(k, center=True, min_periods=max(1, k // 3)).median().values
        ok = np.isfinite(trend) & (trend > 0)
        t, f = t[ok], f_injected[ok] / trend[ok]

    if len(t) < 200:
        return []

    peaks = bls_search_multi(t, f)
    if not peaks:
        return []

    n_sys = sum(1 for p in peaks if 330 < p["period"] < 400)
    syst_frac = n_sys / len(peaks)

    candidates = []
    for rank_i, peak in enumerate(peaks):
        cand = extract_candidate_features(t, f, peak["period"], peak["t0"], peak["depth"], peak["duration"])
        cand["kepid"] = kepid_syn
        cand["rank"] = rank_i
        cand["bls_sde"] = peak["sde"]
        cand["has_planet"] = 1
        cand["systematic_fraction"] = syst_frac
        cand["dist_to_systematic"] = abs(peak["period"] - 372.0)
        cand["near_systematic"] = 1 if abs(peak["period"] - 372.0) < 20 else 0

        ratio = peak["period"] / period
        cand["is_correct"] = 1 if min([abs(ratio - ar) for ar in ALIAS_RATIOS]) < 0.02 else 0

        for key in ["kepmag", "teff", "logg", "radius"]:
            cand[key] = 0
        candidates.append(cand)

    return candidates


def main():
    print("=" * 70)
    print("TARGETED INJECTION AUGMENTATION v2")
    print("=" * 70)

    # Load data
    print("Loading light curves...")
    all_lcs = load_all_light_curves("train")
    truth = load_truth("train")
    labels = load_labels("train")
    target_info = make_target(labels, truth)
    has_planet_map = target_info["target"]

    truth_dict = {}
    for _, row in truth.iterrows():
        truth_dict[int(row["kepid"])] = row.to_dict()

    # Select quiet host stars (no known planet)
    truth_kepids = set(truth["kepid"].values) if truth is not None else set()
    label0 = labels[labels["label"] == 0]
    quiet_hosts = label0[~label0["kepid"].isin(truth_kepids)]["kepid"].values
    no_planet_stars = [int(k) for k in quiet_hosts]
    print(f"  Quiet no-planet stars: {len(no_planet_stars)}")

    # ── Targeted injection parameters ──────────────────────────────────
    # Match failure mode: signals that BLS CAN find but ranker gets confused
    # BLS needs depth >= 1000ppm at 50-150d to beat the 372d systematic
    rng = np.random.RandomState(42)

    # Failure mode 1: periods 50-150d, depths 1000-2500ppm
    # BLS finds them at rank 2-7, but ranker picks 372d alias instead
    fm1_specs = []
    for _ in range(60):
        p = rng.uniform(50, 150)
        d = rng.uniform(1000, 2500)
        dur = rng.uniform(2.0, 6.0)
        fm1_specs.append(("fm1", p, d, dur))

    # Failure mode 2: periods 150-300d, depths 500-1500ppm
    # BLS finds them at rank 5-15, weak signal, ranker confused
    fm2_specs = []
    for _ in range(40):
        p = rng.uniform(150, 300)
        d = rng.uniform(500, 1500)
        dur = rng.uniform(3.0, 8.0)
        fm2_specs.append(("fm2", p, d, dur))

    # Control: deep signals that should always work (no regression)
    ctrl_specs = []
    for _ in range(20):
        p = rng.uniform(10, 50)
        d = rng.uniform(2000, 5000)
        dur = rng.uniform(2.0, 6.0)
        ctrl_specs.append(("ctrl", p, d, dur))

    all_specs = fm1_specs + fm2_specs + ctrl_specs
    rng.shuffle(all_specs)

    # Assign host stars (one injection per star, cycle through quiet stars)
    host_stars = rng.choice(no_planet_stars, size=min(len(all_specs), len(no_planet_stars)), replace=False)

    print(f"  Targeted injections: {len(all_specs)}")
    print(f"    FM1 (50-150d, shallow): {len(fm1_specs)}")
    print(f"    FM2 (150-250d, very weak): {len(fm2_specs)}")
    print(f"    Control (deep): {len(ctrl_specs)}")

    # ── Run injections ─────────────────────────────────────────────────
    all_candidates = []
    n_success = 0
    t0 = time.time()

    for i, (spec, kepid_syn) in enumerate(zip(all_specs, host_stars)):
        if i % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(all_specs)}] {elapsed:.0f}s elapsed, {n_success} successful...")

        fm_tag, period, depth, duration = spec
        lc_df = all_lcs[kepid_syn]

        # Random t0
        t_vals = lc_df["time"].values
        t0_inject = rng.uniform(t_vals.min() + 10, t_vals.max() - 10)

        cands = process_one_host(lc_df, period, depth, duration, t0_inject, kepid_syn)

        if cands:
            # Find the injected period candidate
            inj_cands = [c for c in cands if is_alias(c["bls_period"], period)]
            if inj_cands:
                n_success += 1
                for c in cands:
                    c["injected_bin"] = fm_tag
                all_candidates.extend(cands)

    elapsed = time.time() - t0
    print(f"\n  Done: {n_success}/{len(all_specs)} successful injections in {elapsed:.0f}s")

    # Save augmented candidates
    inj_df = pd.DataFrame(all_candidates)
    # Merge with existing augmented data
    existing_aug = pd.read_csv(OUTPUTS_DIR / "train_candidates_augmented.csv")
    combined_aug = pd.concat([existing_aug, inj_df], ignore_index=True)
    combined_aug.to_csv(OUTPUTS_DIR / "train_candidates_augmented_v2.csv", index=False)
    print(f"  Saved: {OUTPUTS_DIR / 'train_candidates_augmented_v2.csv'} ({len(combined_aug)} rows)")

    # ── Train ranker with targeted + real data ─────────────────────────
    print(f"\n{'='*70}")
    print("TRAINING RANKER v2")
    print("=" * 70)

    real_train_df = pd.read_csv(OUTPUTS_DIR / "train_candidates_v4.csv")
    aug_v2 = combined_aug.drop(columns=["injected_bin"], errors="ignore")
    train_all = pd.concat([real_train_df, aug_v2], ignore_index=True)

    available_feats = [f for f in CANDIDATE_FEATURES if f in train_all.columns]
    train_supervised = train_all[train_all["is_correct"].notna()].copy()
    train_sorted = train_supervised.sort_values("kepid").reset_index(drop=True)
    gs = train_sorted.groupby("kepid").size().values
    X = train_sorted[available_feats].values
    y = train_sorted["is_correct"].astype(int).values

    print(f"  Training on {len(y)} candidates from {len(gs)} stars")
    print(f"  Positive: {y.sum()}, Negative: {len(y) - y.sum()}")

    ranker_v2 = LGBMRanker(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        metric="ndcg",
        eval_at=[1, 3, 5],
        random_state=42,
        verbose=-1,
    )
    ranker_v2.fit(X, y, group=gs)

    # ── Evaluate on dev ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DEV EVALUATION")
    print("=" * 70)

    dev_candidates = pd.read_csv(OUTPUTS_DIR / "dev_candidates_v4.csv")
    dev_scored = dev_candidates.copy()
    dev_scored["ranker_v2_score"] = ranker_v2.predict(dev_scored[available_feats].fillna(0).values)

    truth_dev = load_truth("dev")
    truth_dev_dict = {}
    for _, row in truth_dev.iterrows():
        truth_dev_dict[int(row["kepid"])] = row.to_dict()

    # Compare ranker v1 vs v2
    print("\n  Period recovery:")
    for method_name, score_col in [
        ("Ranker v1 (augmented 132)", "ranker_score"),
        ("Ranker v2 (targeted +120)", "ranker_v2_score"),
    ]:
        if score_col == "ranker_score":
            with open(MODELS_DIR / "augmented_lgbm_ranker.pkl", "rb") as f:
                old_artifact = pickle.load(f)
            old_ranker = old_artifact["model"]
            dev_scored["ranker_score"] = old_ranker.predict(dev_scored[old_artifact["feature_columns"]].fillna(0).values)

        correct = 0
        n = 0
        for kepid_val, grp in dev_scored.groupby("kepid"):
            kepid = int(kepid_val)
            true_p = truth_dev_dict.get(kepid, {}).get("period_days", None)
            if true_p is None or true_p <= 0:
                continue
            n += 1
            p = grp.sort_values(score_col, ascending=False).iloc[0]["bls_period"]
            if is_alias(p, true_p):
                correct += 1
        print(f"    {method_name}: {correct}/{n} = {correct/n:.3f}")

    # Per-bin
    print("\n  Per-bin (ranker v2):")
    for bin_name in ["earth_analog", "shallow", "mid", "deep"]:
        correct = 0
        n = 0
        for kepid_val, grp in dev_scored.groupby("kepid"):
            kepid = int(kepid_val)
            true_info = truth_dev_dict.get(kepid, {})
            true_p = true_info.get("period_days", None)
            true_bin = true_info.get("bin", None)
            if true_p is None or true_p <= 0 or true_bin != bin_name:
                continue
            n += 1
            p = grp.sort_values("ranker_v2_score", ascending=False).iloc[0]["bls_period"]
            if is_alias(p, true_p):
                correct += 1
        if n > 0:
            print(f"    {bin_name:>14s}: {correct}/{n}")

    # Check specifically the 10 previously-missed-but-present stars
    print("\n  Previously missed-but-present stars:")
    for kepid_val, grp in dev_scored.groupby("kepid"):
        kepid = int(kepid_val)
        true_info = truth_dev_dict.get(kepid, {})
        true_p = true_info.get("period_days", None)
        if true_p is None or true_p <= 0:
            continue

        # Check if v2 now gets it right
        p_v2 = grp.sort_values("ranker_v2_score", ascending=False).iloc[0]["bls_period"]
        ok_v2 = is_alias(p_v2, true_p)

        # Check v1
        p_v1 = grp.sort_values("ranker_score", ascending=False).iloc[0]["bls_period"]
        ok_v1 = is_alias(p_v1, true_p)

        if not ok_v1:  # Was missed by v1
            status = "FIXED" if ok_v2 else "still missed"
            print(f"    KIC_{kepid:08d}: true={true_p:.2f}d, v1={p_v1:.2f}d, v2={p_v2:.2f}d -> {status}")

    # Save ranker v2
    artifact_v2 = {
        "model": ranker_v2,
        "feature_columns": available_feats,
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "training_samples": len(y),
        "training_groups": len(gs),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "augmented_injections_v1": 132,
        "augmented_injections_v2": n_success,
        "augmentation_type": "targeted (shallow 50-250d, deep control)",
        "description": "LGBMRanker v2 with targeted augmentation targeting systematic alias failure mode.",
    }

    out_path = MODELS_DIR / "augmented_lgbm_ranker_v2.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact_v2, f)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    import pickle
    main()
