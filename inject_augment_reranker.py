"""
Synthetic Injection Augmentation for Reranker — Optimized.
One injection per quiet host star. ~112 injections, ~8 minutes.
"""
import sys, warnings, io, time
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_loader import load_all_light_curves, load_labels, load_truth, make_target
from src.period_search import clean_for_bls
from src.config import (
    PERIOD_MIN_DAYS, PERIOD_MAX_DAYS,
    BLS_COARSE_RESOLUTION, BLS_FINE_RESOLUTION,
    BLS_FINE_WINDOW_FRAC, BLS_DURATIONS_DAYS, BLS_OBJECTIVE,
    N_CANDIDATE_PEAKS, KEPLER_SYSTEMATIC_PERIODS,
)
from build_candidates import extract_candidate_features, bls_search_multi

ALIAS_RATIOS = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]
CANDIDATE_FEATURES = [
    "bls_period", "bls_depth_ppm", "bls_duration_hours",
    "bls_n_transits", "bls_snr", "bls_sde",
    "dist_to_systematic", "systematic_fraction", "near_systematic",
    "transit_coverage", "n_expected",
    "depth_cv", "depth_mad_ratio", "timing_rms",
    "odd_even_depth_ratio", "secondary_eclipse_depth", "quarter_signal_fraction",
    "in_out_scatter_ratio", "box_fit_snr", "box_fit_residual_rms",
    "depth_trend_slope", "duration_consistency", "baseline_rms",
    "kepmag", "teff", "logg", "radius",
]


def inject_transit(time_arr, flux_arr, period_days, depth_ppm, duration_hours, t0):
    """Inject box transit into flux array. Returns new flux."""
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
    """Inject -> detrend -> BLS -> features. Returns candidate list."""
    # Get raw quality-masked arrays
    m = (lc_df["quality"].values == 0) & np.isfinite(lc_df["flux"].values)
    t = lc_df["time"].values[m].copy()
    f_raw = lc_df["flux"].values[m].astype(float).copy()
    q = lc_df["quarter"].values[m].copy()
    
    if len(t) < 1000:
        return []
    
    # Inject into raw flux
    f_injected = inject_transit(t, f_raw, period, depth_ppm, duration_hours, t0)
    
    # Per-quarter median normalize
    for qq in np.unique(q):
        s = q == qq
        med = np.median(f_injected[s])
        if med > 0:
            f_injected[s] /= med
    
    # Wotan biweight detrend
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
        if k % 2 == 0: k += 1
        trend = pd.Series(f_injected).rolling(k, center=True, min_periods=max(1, k//3)).median().values
        ok = np.isfinite(trend) & (trend > 0)
        t, f = t[ok], f_injected[ok] / trend[ok]
    
    if len(t) < 200:
        return []
    
    # BLS search
    peaks = bls_search_multi(t, f)
    if not peaks:
        return []
    
    n_sys = sum(1 for p in peaks if 330 < p["period"] < 400)
    syst_frac = n_sys / len(peaks)
    
    candidates = []
    for rank, peak in enumerate(peaks):
        cand = extract_candidate_features(t, f, peak["period"], peak["t0"], peak["depth"], peak["duration"])
        cand["kepid"] = kepid_syn
        cand["rank"] = rank
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
    print("SYNTHETIC INJECTION AUGMENTATION (optimized)")
    print("=" * 70)
    
    rng = np.random.RandomState(42)
    
    # Load data
    print("\n[1] Loading data...")
    t0 = time.time()
    train_lc = load_all_light_curves("train")
    train_labels = load_labels("train")
    train_truth = load_truth("train")
    print(f"  Loaded in {time.time()-t0:.0f}s")
    
    # Find quiet hosts
    truth_kepids = set(train_truth["kepid"].values) if train_truth is not None else set()
    label0 = train_labels[train_labels["label"] == 0]
    quiet_hosts = label0[~label0["kepid"].isin(truth_kepids)]["kepid"].values
    print(f"  Quiet hosts: {len(quiet_hosts)}")
    
    # Sample parameters: one per host, from truth distribution
    print("\n[2] Sampling injection parameters (1 per host)...")
    injections = []
    for i, host_kepid in enumerate(quiet_hosts):
        # Sample bin with oversampling of hard cases
        bin_weights = {"earth_analog": 0.30, "shallow": 0.25, "mid": 0.25, "deep": 0.20}
        bin_name = rng.choice(list(bin_weights.keys()), p=list(bin_weights.values()))
        
        bin_truth = train_truth[train_truth["bin"] == bin_name]
        row = bin_truth.sample(1, random_state=rng).iloc[0]
        
        period = row["period_days"] * rng.uniform(0.8, 1.2)
        depth = row["depth_ppm"] * rng.uniform(0.8, 1.2)
        duration = row["duration_hours"] * rng.uniform(0.8, 1.2)
        
        period = np.clip(period, PERIOD_MIN_DAYS, PERIOD_MAX_DAYS)
        depth = np.clip(depth, 50, 5000)
        duration = np.clip(duration, 1.0, 15.0)
        
        baseline = 1400
        t0_epoch = rng.uniform(0, period) if period > 0 else 0
        
        injections.append({
            "host_kepid": host_kepid,
            "bin": bin_name,
            "period_days": period,
            "depth_ppm": depth,
            "duration_hours": duration,
            "epoch_t0": t0_epoch,
        })
    
    # Add near-372d alias injections
    for i in range(30):
        host_kepid = quiet_hosts[i % len(quiet_hosts)]
        base = rng.choice([372.5, 186.25, 93.125, 372.5*2])
        period = base * rng.uniform(0.97, 1.03)
        period = np.clip(period, PERIOD_MIN_DAYS, PERIOD_MAX_DAYS)
        depth = rng.uniform(80, 2500)
        duration = rng.uniform(2, 14)
        injections.append({
            "host_kepid": host_kepid,
            "bin": "near_systematic",
            "period_days": period,
            "depth_ppm": depth,
            "duration_hours": duration,
            "epoch_t0": rng.uniform(0, period),
        })
    
    print(f"  {len(injections)} injections planned")
    
    # Process all injections
    print(f"\n[3] Injecting and running BLS...")
    all_candidates = []
    n_success = 0
    
    t_start = time.time()
    for i, inj in enumerate(injections):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(injections) - i - 1)
            print(f"  [{i+1}/{len(injections)}] {elapsed:.0f}s, ETA {eta:.0f}s, "
                  f"{n_success} success, {len(all_candidates)} cands")
        
        host_kepid = inj["host_kepid"]
        if host_kepid not in train_lc:
            continue
        
        kepid_syn = 9000000 + i
        cands = process_one_host(
            train_lc[host_kepid], inj["period_days"], inj["depth_ppm"],
            inj["duration_hours"], inj["epoch_t0"], kepid_syn
        )
        if cands:
            n_success += 1
            for c in cands:
                c["injected_bin"] = inj["bin"]
            all_candidates.extend(cands)
    
    elapsed = time.time() - t_start
    print(f"\n  Done: {elapsed:.0f}s, {n_success} success, {len(all_candidates)} candidates")
    
    inj_df = pd.DataFrame(all_candidates)
    n_correct = (inj_df["is_correct"] == 1).sum()
    oracle = inj_df.groupby("kepid")["is_correct"].max()
    n_oracle = (oracle == 1).sum()
    print(f"  Correct candidates: {n_correct}")
    print(f"  Oracle: {n_oracle}/{n_success} = {n_oracle/n_success*100:.1f}%")
    
    # Coverage by rank
    print(f"\n  Coverage by rank:")
    for rank in range(20):
        rc = (inj_df[inj_df["rank"] == rank]["is_correct"] == 1).sum()
        total = len(inj_df[inj_df["rank"] == rank])
        if total == 0: break
        print(f"    Rank {rank:>2}: {rc}/{total}")
    
    # Save augmented candidates
    inj_df.to_csv("outputs/train_candidates_augmented.csv", index=False)
    
    # ── Train reranker ─────────────────────────────────────────────────
    print(f"\n[4] Training reranker...")
    
    real_train = pd.read_csv("outputs/train_candidates_v4.csv")
    real_dev = pd.read_csv("outputs/dev_candidates_v4.csv")
    
    inj_train = inj_df.drop(columns=["injected_bin"], errors="ignore")
    combined_train = pd.concat([real_train, inj_train], ignore_index=True)
    
    available_feats = [f for f in CANDIDATE_FEATURES if f in combined_train.columns]
    
    train_mask = combined_train["is_correct"].notna()
    X_train = combined_train.loc[train_mask, available_feats].values
    y_train = combined_train.loc[train_mask, "is_correct"].astype(int).values
    
    dev_mask = real_dev["is_correct"].notna()
    X_dev = real_dev.loc[dev_mask, available_feats].values
    y_dev = real_dev.loc[dev_mask, "is_correct"].astype(int).values
    
    print(f"  Train: {len(y_train)} cands ({y_train.sum()} pos, {len(y_train)-y_train.sum()} neg)")
    print(f"  Dev:   {len(y_dev)} cands ({y_dev.sum()} pos, {len(y_dev)-y_dev.sum()} neg)")
    
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=(len(y_train)-y_train.sum())/max(y_train.sum(),1),
        eval_metric="aucpr", early_stopping_rounds=50,
        random_state=42, verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)
    px = xgb.predict_proba(X_dev)[:, 1]
    print(f"  XGB:  AUC={roc_auc_score(y_dev, px):.3f}, AP={average_precision_score(y_dev, px):.3f}")
    
    # LightGBM
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=(len(y_train)-y_train.sum())/max(y_train.sum(),1),
        metric="average_precision", early_stopping_rounds=50,
        random_state=42, verbose=-1,
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_dev, y_dev)])
    pl = lgbm.predict_proba(X_dev)[:, 1]
    print(f"  LGBM: AUC={roc_auc_score(y_dev, pl):.3f}, AP={average_precision_score(y_dev, pl):.3f}")
    
    # Ensemble
    pe = 0.5 * px + 0.5 * pl
    print(f"  Ens:  AUC={roc_auc_score(y_dev, pe):.3f}, AP={average_precision_score(y_dev, pe):.3f}")
    
    # ── Period recovery on dev ─────────────────────────────────────────
    print(f"\n[5] Period recovery on dev...")
    
    # Score ALL dev candidates with augmented model
    dev_all_feats = real_dev[available_feats].fillna(0).values
    dev_proba_aug = 0.5 * xgb.predict_proba(dev_all_feats)[:, 1] + 0.5 * lgbm.predict_proba(dev_all_feats)[:, 1]
    real_dev = real_dev.copy()
    real_dev["proba_aug"] = dev_proba_aug
    
    # Load original reranker
    orig_rerank = pd.read_csv("outputs/dev_reranked_v4.csv")
    
    truth_dev = load_truth("dev")
    truth_dev_dict = {}
    for _, row in truth_dev.iterrows():
        truth_dev_dict[int(row["kepid"])] = row.to_dict()
    
    def is_alias(p1, p2):
        if p1 <= 0 or p2 <= 0: return False
        ratio = p1 / p2
        return min([abs(ratio - ar) for ar in ALIAS_RATIOS]) < 0.02
    
    results = []
    for kepid in sorted(real_dev["kepid"].unique()):
        true_p = truth_dev_dict.get(kepid, {}).get("period_days", None)
        if true_p is None or true_p <= 0: continue
        
        # Original reranker top-1
        orig = orig_rerank[orig_rerank["kepid"] == kepid].sort_values("proba_ensemble", ascending=False)
        orig_p = orig.iloc[0]["bls_period"] if len(orig) > 0 else np.nan
        orig_ok = is_alias(orig_p, true_p) if not np.isnan(orig_p) else False
        
        # Augmented reranker top-1
        aug = real_dev[real_dev["kepid"] == kepid].sort_values("proba_aug", ascending=False)
        aug_p = aug.iloc[0]["bls_period"] if len(aug) > 0 else np.nan
        aug_ok = is_alias(aug_p, true_p) if not np.isnan(aug_p) else False
        
        true_bin = truth_dev_dict.get(kepid, {}).get("bin", "?")
        results.append({
            "kepid": kepid, "true_period": true_p, "true_bin": true_bin,
            "orig_p": orig_p, "orig_ok": orig_ok,
            "aug_p": aug_p, "aug_ok": aug_ok,
        })
    
    rdf = pd.DataFrame(results)
    n = len(rdf)
    orig_n = rdf["orig_ok"].sum()
    aug_n = rdf["aug_ok"].sum()
    
    print(f"\n  Period recovery ({n} truth stars):")
    print(f"    Original:  {orig_n}/{n} = {orig_n/n:.3f}")
    print(f"    Augmented: {aug_n}/{n} = {aug_n/n:.3f}")
    
    print(f"\n  Per-bin:")
    for b in ["earth_analog", "shallow", "mid", "deep"]:
        sub = rdf[rdf["true_bin"] == b]
        if len(sub) > 0:
            o = sub["orig_ok"].sum()
            a = sub["aug_ok"].sum()
            print(f"    {b:>14s}: orig={o}/{len(sub)}, aug={a}/{len(sub)}")
    
    fixed = rdf[(rdf["orig_ok"] == False) & (rdf["aug_ok"] == True)]
    broken = rdf[(rdf["orig_ok"] == True) & (rdf["aug_ok"] == False)]
    
    if len(fixed) > 0:
        print(f"\n  Fixed ({len(fixed)}):")
        for _, r in fixed.iterrows():
            print(f"    KepID {int(r['kepid'])}: {r['orig_p']:.2f} -> {r['aug_p']:.2f} (true={r['true_period']:.2f}) [{r['true_bin']}]")
    
    if len(broken) > 0:
        print(f"\n  Broken ({len(broken)}):")
        for _, r in broken.iterrows():
            print(f"    KepID {int(r['kepid'])}: {r['orig_p']:.2f} -> {r['aug_p']:.2f} (true={r['true_period']:.2f}) [{r['true_bin']}]")
    
    # Feature importance
    print(f"\n--- Top-10 features ---")
    imp = pd.DataFrame({"f": available_feats, "imp": xgb.feature_importances_}).sort_values("imp", ascending=False)
    for _, r in imp.head(10).iterrows():
        print(f"  {r['f']:30s}: {r['imp']:.4f}")
    
    rdf.to_csv("outputs/dev_augmented_recovery.csv", index=False)
    print(f"\nSaved outputs/dev_augmented_recovery.csv")


if __name__ == "__main__":
    main()
