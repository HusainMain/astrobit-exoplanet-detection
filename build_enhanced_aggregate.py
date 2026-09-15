"""
Enhanced aggregate features: add period-quality / candidate-quality features.
Key additions:
  - Reranker probability for rank-0 and best candidate
  - Reranker score gap (top-1 minus top-2)
  - Non-systematic candidate count and best non-sys reranker score
  - Rank-0 event-quality features (depth_cv, timing_rms, quarter_signal, box_fit_snr)
  - Period-quality composite score
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from src.data_loader import load_labels, load_truth, make_target


def build_enhanced_aggregate(split="dev"):
    """Build enhanced aggregate features with period-quality additions."""
    print(f"{'='*70}")
    print(f"ENHANCED AGGREGATE FEATURES for {split}")
    print(f"{'='*70}")

    df = pd.read_csv(f"outputs/{split}_candidates.csv")
    labels = load_labels(split)
    truth = load_truth(split)
    target_info = make_target(labels, truth)
    has_planet_map = target_info["target"]

    # Merge reranker probabilities if available
    rerank_path = f"outputs/{split}_reranked_v4.csv"
    import os
    has_reranker = os.path.exists(rerank_path)
    if has_reranker:
        rerank = pd.read_csv(rerank_path)
        rerank_cols = ["kepid", "rank", "proba_ensemble", "proba_xgb", "proba_lgbm"]
        df = pd.merge(df, rerank[rerank_cols], on=["kepid", "rank"], how="left")
        print(f"  Reranker loaded: {rerank_path}")
    else:
        print(f"  No reranker file — skipping reranker features")

    feat_cols = [
        "bls_sde", "bls_period", "bls_depth_ppm", "bls_duration_hours",
        "bls_n_transits", "bls_snr",
        "dist_to_systematic", "systematic_fraction", "near_systematic",
        "transit_coverage", "n_expected", "depth_cv", "depth_mad_ratio",
        "timing_rms", "odd_even_depth_ratio", "secondary_eclipse_depth",
        "quarter_signal_fraction", "in_out_scatter_ratio",
        "box_fit_snr", "box_fit_residual_rms",
        "depth_trend_slope", "duration_consistency", "baseline_rms",
    ]

    star_features = []
    for kepid, group in df.groupby("kepid"):
        feats = {"kepid": kepid}
        feats["label"] = has_planet_map.get(kepid, 0)
        n = len(group)

        # Basic stats
        feats["n_candidates"] = n

        # ── Per-feature aggregates across top-20 ──
        for col in feat_cols:
            if col not in group.columns:
                continue
            vals = group[col].values
            feats[f"{col}_max"] = np.nanmax(vals)
            feats[f"{col}_min"] = np.nanmin(vals)
            feats[f"{col}_mean"] = np.nanmean(vals)
            feats[f"{col}_std"] = np.nanstd(vals)

        # ── Rank-0 features (the "best" candidate) ──
        r0 = group[group["rank"] == 0]
        if len(r0) > 0:
            for col in feat_cols:
                if col in r0.columns:
                    feats[f"r0_{col}"] = r0[col].values[0]

        # ── SDE gap between rank-0 and rank-1 ──
        r0_sde = group[group["rank"] == 0]["bls_sde"].values
        r1_sde = group[group["rank"] == 1]["bls_sde"].values
        if len(r0_sde) > 0 and len(r1_sde) > 0:
            feats["sde_gap_0_1"] = r0_sde[0] - r1_sde[0]
        else:
            feats["sde_gap_0_1"] = 0

        # ── Number of candidates with SDE > threshold ──
        for thresh in [5, 8, 10, 15, 20]:
            feats[f"n_sde_gt_{thresh}"] = (group["bls_sde"] > thresh).sum()

        # ── Best non-systematic candidate ──
        non_sys = group[(group["dist_to_systematic"] > 20) | (group["bls_period"] < 50)]
        if len(non_sys) > 0:
            feats["best_non_sys_sde"] = non_sys["bls_sde"].max()
            feats["best_non_sys_period"] = non_sys.loc[non_sys["bls_sde"].idxmax(), "bls_period"]
            feats["best_non_sys_depth"] = non_sys.loc[non_sys["bls_sde"].idxmax(), "bls_depth_ppm"]
        else:
            feats["best_non_sys_sde"] = 0
            feats["best_non_sys_period"] = 0
            feats["best_non_sys_depth"] = 0

        # ── Fraction of candidates near systematic ──
        feats["frac_near_systematic"] = group["near_systematic"].mean()

        # ── Best transit coverage among high-SDE candidates ──
        high_sde = group[group["bls_sde"] > 8]
        if len(high_sde) > 0:
            feats["best_coverage_high_sde"] = high_sde["transit_coverage"].max()
            feats["best_ntransits_high_sde"] = high_sde["bls_n_transits"].max()
        else:
            feats["best_coverage_high_sde"] = 0
            feats["best_ntransits_high_sde"] = 0

        # ── Period consistency ──
        periods = group["bls_period"].values
        if len(periods) >= 2:
            max_agree = 0
            for p in periods:
                agree = np.sum(np.abs(periods - p) / p < 0.05)
                max_agree = max(max_agree, agree)
            feats["max_period_agreement"] = max_agree
        else:
            feats["max_period_agreement"] = 1

        # ══════════════════════════════════════════════════════════════
        # NEW: PERIOD-QUALITY / CANDIDATE-QUALITY FEATURES
        # ══════════════════════════════════════════════════════════════

        # ── Rank-0 row (used throughout) ──
        r0_row = group[group["rank"] == 0]

        # ── Reranker features (only if reranker file exists) ──
        if has_reranker:
            if len(r0_row) > 0 and "proba_ensemble" in r0_row.columns:
                feats["r0_reranker_proba"] = r0_row["proba_ensemble"].values[0]
            else:
                feats["r0_reranker_proba"] = 0

            if "proba_ensemble" in group.columns:
                valid = group["proba_ensemble"].dropna()
                feats["best_reranker_proba"] = valid.max() if len(valid) > 0 else 0
            else:
                feats["best_reranker_proba"] = 0

            r0_proba = group[group["rank"] == 0]["proba_ensemble"].values
            r1_proba = group[group["rank"] == 1]["proba_ensemble"].values
            if len(r0_proba) > 0 and len(r1_proba) > 0 and not np.isnan(r0_proba[0]) and not np.isnan(r1_proba[0]):
                feats["reranker_proba_gap"] = r0_proba[0] - r1_proba[0]
            else:
                feats["reranker_proba_gap"] = 0

            if len(non_sys) > 0 and "proba_ensemble" in non_sys.columns:
                valid_ns = non_sys["proba_ensemble"].dropna()
                feats["best_non_sys_reranker_proba"] = valid_ns.max() if len(valid_ns) > 0 else 0
            else:
                feats["best_non_sys_reranker_proba"] = 0
        else:
            feats["r0_reranker_proba"] = 0
            feats["best_reranker_proba"] = 0
            feats["reranker_proba_gap"] = 0
            feats["best_non_sys_reranker_proba"] = 0

        # ── Non-systematic candidate count ──
        feats["n_non_sys_candidates"] = len(non_sys)

        # ── Rank-0 event-quality features ──
        if len(r0_row) > 0:
            feats["r0_depth_cv"] = r0_row["depth_cv"].values[0] if "depth_cv" in r0_row.columns else 0
            feats["r0_timing_rms"] = r0_row["timing_rms"].values[0] if "timing_rms" in r0_row.columns else 0
            feats["r0_quarter_signal_fraction"] = r0_row["quarter_signal_fraction"].values[0] if "quarter_signal_fraction" in r0_row.columns else 0
            feats["r0_box_fit_snr"] = r0_row["box_fit_snr"].values[0] if "box_fit_snr" in r0_row.columns else 0
            feats["r0_in_out_scatter_ratio"] = r0_row["in_out_scatter_ratio"].values[0] if "in_out_scatter_ratio" in r0_row.columns else 0
            feats["r0_odd_even_depth_ratio"] = r0_row["odd_even_depth_ratio"].values[0] if "odd_even_depth_ratio" in r0_row.columns else 1.0
        else:
            feats["r0_depth_cv"] = 0
            feats["r0_timing_rms"] = 0
            feats["r0_quarter_signal_fraction"] = 0
            feats["r0_box_fit_snr"] = 0
            feats["r0_in_out_scatter_ratio"] = 0
            feats["r0_odd_even_depth_ratio"] = 1.0

        # ── Period-is-systematic flag ──
        feats["r0_is_systematic"] = 1 if (len(r0_row) > 0 and r0_row["near_systematic"].values[0] == 1) else 0

        # ── Has good non-systematic alternative ──
        feats["has_good_non_sys"] = 1 if (len(non_sys) > 0 and non_sys["bls_sde"].max() > 8) else 0

        # ── SDE ratio: rank-0 SDE vs best non-systematic SDE ──
        if len(r0_row) > 0 and feats["best_non_sys_sde"] > 0:
            feats["sde_ratio_r0_vs_nonsys"] = r0_row["bls_sde"].values[0] / feats["best_non_sys_sde"]
        else:
            feats["sde_ratio_r0_vs_nonsys"] = 1.0

        # ── Rank-1 vs rank-2 SDE gap (alternatives competing?) ──
        r1_sde_val = group[group["rank"] == 1]["bls_sde"].values
        r2_sde_val = group[group["rank"] == 2]["bls_sde"].values
        if len(r1_sde_val) > 0 and len(r2_sde_val) > 0:
            feats["sde_gap_1_2"] = r1_sde_val[0] - r2_sde_val[0]
        else:
            feats["sde_gap_1_2"] = 0

        # ── Count of candidates with depth consistency < 0.3 (good transit) ──
        if "depth_cv" in group.columns:
            feats["n_good_depth_cv"] = (group["depth_cv"] < 0.3).sum()
        else:
            feats["n_good_depth_cv"] = 0

        # ── Period quality composite score (no reranker dependency) ──
        r0_dist = feats.get("r0_dist_to_systematic", 20)
        r0_tc = feats.get("r0_transit_coverage", 0)
        r0_bfs = feats.get("r0_box_fit_snr", 0)
        r0_dcv = feats.get("r0_depth_cv", 1.0)
        feats["period_quality_score"] = (
            min(r0_dist / 50, 1.0) * 0.25
            + min(r0_tc, 1.0) * 0.25
            + min(r0_bfs / 200, 1.0) * 0.25
            + max(1.0 - r0_dcv, 0.0) * 0.25
        )

        # ── Star-level labels ──
        if kepid in target_info["target"]:
            feats["has_planet"] = target_info["target"][kepid]
        else:
            feats["has_planet"] = 0

        # ── Stellar properties ──
        labels_dict = {}
        if labels is not None:
            for _, row in labels.iterrows():
                labels_dict[int(row["kepid"])] = row.to_dict()
        if kepid in labels_dict:
            for key in ["kepmag", "teff", "logg", "radius"]:
                feats[key] = float(labels_dict[kepid].get(key, 0) or 0)

        star_features.append(feats)

    agg_df = pd.DataFrame(star_features)
    print(f"  {len(agg_df)} stars, {len(agg_df.columns)} features")

    # Show new features
    new_feats = [c for c in agg_df.columns if c.startswith("r0_reranker") or
                 c.startswith("best_reranker") or c.startswith("reranker_") or
                 c.startswith("n_non_sys") or c.startswith("r0_depth_cv") or
                 c.startswith("r0_timing_rms") or c.startswith("r0_quarter") or
                 c.startswith("r0_box_fit") or c.startswith("r0_in_out") or
                 c.startswith("r0_is_sys") or c.startswith("has_good_non") or
                 c.startswith("sde_ratio") or c.startswith("period_quality")]
    print(f"  New period-quality features: {len(new_feats)}")
    for f in new_feats:
        print(f"    {f}")

    return agg_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "private"], default=None,
                        help="Which split to build (default: train + dev)")
    args = parser.parse_args()

    if args.split:
        splits = [args.split]
    else:
        splits = ["train", "dev"]

    for s in splits:
        agg = build_enhanced_aggregate(s)
        out = f"outputs/{s}_aggregate_features.csv"
        agg.to_csv(out, index=False)
        print(f"\nSaved {out} ({agg.shape[1]} cols)")
