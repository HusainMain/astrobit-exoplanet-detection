"""
Clean dev comparison: v7 baseline, original reranker, augmented binary, augmented LGBMRanker.

Separates detection metrics from period recovery metrics.
"""
import sys, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from lightgbm import LGBMRanker, LGBMClassifier
from src.data_loader import load_labels, load_truth, make_target
from src.config import OUTPUTS_DIR, MODELS_DIR
from train_reranker_v3 import CANDIDATE_FEATURES

ALIAS_RATIOS = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]

def is_alias(p1, p2):
    if p1 <= 0 or p2 <= 0:
        return False
    return min([abs(p1 / p2 - ar) for ar in ALIAS_RATIOS]) < 0.02


def load_ranker():
    with open(MODELS_DIR / "augmented_lgbm_ranker.pkl", "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["feature_columns"]


def main():
    print("=" * 70)
    print("CLEAN DEV COMPARISON")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    candidates = pd.read_csv(OUTPUTS_DIR / "dev_candidates_v4.csv")
    det = pd.read_csv(OUTPUTS_DIR / "dev_detected_v7.csv")
    orig_rerank = pd.read_csv(OUTPUTS_DIR / "dev_reranked_v4.csv")

    labels = load_labels("dev")
    truth = load_truth("dev")
    target_info = make_target(labels, truth)
    has_planet_map = target_info["target"]

    truth_dict = {}
    for _, row in truth.iterrows():
        truth_dict[int(row["kepid"])] = row.to_dict()

    ranker, feature_cols = load_ranker()

    # ── Score candidates with ranker ───────────────────────────────────
    candidates = candidates.copy()
    X = candidates[feature_cols].fillna(0).values
    candidates["ranker_score"] = ranker.predict(X)

    # ── DETECTION COMPARISON ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DETECTION METRICS (same detection model, different thresholds)")
    print("=" * 70)

    # All methods use the same detector — just different threshold strategies
    merged_det = det[["kepid", "proba"]].copy()
    merged_det["label"] = merged_det["kepid"].map(has_planet_map).fillna(0).astype(int)

    y_true = merged_det["label"].values
    y_score = merged_det["proba"].values

    ap = average_precision_score(y_true, y_score)

    # Various thresholds
    thresholds = [0.15, 0.20, 0.23, 0.25, 0.30, 0.50]
    print(f"\n  AP (all detectors): {ap:.4f}")
    print(f"\n  {'Thresh':>6s} {'Pred+':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s}")

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  {t:>6.2f} {tp+fp:>6d} {tp:>4d} {fp:>4d} {fn:>4d} "
              f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

    # ── PERIOD RECOVERY COMPARISON ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("PERIOD RECOVERY (ranked candidates, top-1 per star)")
    print("=" * 70)

    methods = [
        ("v7 rank-0 (highest SDE)", candidates, "bls_sde"),
        ("Original reranker", orig_rerank, "proba_ensemble"),
        ("Augmented LGBMRanker", candidates, "ranker_score"),
    ]

    results = {}
    for name, df, score_col in methods:
        correct = 0
        n = 0
        detail = []
        for kepid_val, grp in df.groupby("kepid"):
            kepid = int(kepid_val)
            true_p = truth_dict.get(kepid, {}).get("period_days", None)
            if true_p is None or true_p <= 0:
                continue
            n += 1
            p = grp.sort_values(score_col, ascending=False).iloc[0]["bls_period"]
            ok = is_alias(p, true_p)
            if ok:
                correct += 1
            detail.append((kepid, p, true_p, ok))
        results[name] = (correct, n, detail)
        print(f"\n  {name}: {correct}/{n} = {correct/n:.3f}")

    # ── Per-bin breakdown for augmented ranker ──────────────────────────
    print(f"\n  Per-bin (augmented LGBMRanker):")
    for bin_name in ["earth_analog", "shallow", "mid", "deep"]:
        correct = 0
        n = 0
        for kepid_val, grp in candidates.groupby("kepid"):
            kepid = int(kepid_val)
            true_info = truth_dict.get(kepid, {})
            true_p = true_info.get("period_days", None)
            true_bin = true_info.get("bin", None)
            if true_p is None or true_p <= 0 or true_bin != bin_name:
                continue
            n += 1
            p = grp.sort_values("ranker_score", ascending=False).iloc[0]["bls_period"]
            if is_alias(p, true_p):
                correct += 1
        if n > 0:
            print(f"    {bin_name:>14s}: {correct}/{n}")

    # ── Comparison table ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  {'Method':<30s} {'Period':>10s} {'AP':>8s}")
    print(f"  {'-'*48}")
    for name, _, _ in methods:
        c, n, _ = results[name]
        print(f"  {name:<30s} {c:>2d}/{n:<2d} = {c/n:.3f}  {'(same)':>8s}")

    print(f"\n  Detection AP: {ap:.4f} (same detector for all methods)")
    print(f"  Detection threshold used in v8+: 0.23")

    # ── Stars where methods disagree ───────────────────────────────────
    print(f"\n{'='*70}")
    print("AGREEMENT / DISAGREEMENT ANALYSIS")
    print("=" * 70)

    rank0_detail = results["v7 rank-0 (highest SDE)"][2]
    rerank_detail = results["Original reranker"][2]
    lgbm_detail = results["Augmented LGBMRanker"][2]

    rank0_map = {k: (p, t, ok) for k, p, t, ok in rank0_detail}
    rerank_map = {k: (p, t, ok) for k, p, t, ok in rerank_detail}
    lgbm_map = {k: (p, t, ok) for k, p, t, ok in lgbm_detail}

    all_kepids = sorted(set(list(rank0_map.keys()) + list(rerank_map.keys()) + list(lgbm_map.keys())))

    agree_all_correct = 0
    agree_all_wrong = 0
    lgbm_unique_correct = 0
    lgbm_unique_wrong = 0
    rerank_unique_correct = 0
    disagree_lgbm_correct = []

    for k in all_kepids:
        r0 = rank0_map.get(k, (0, 0, False))[2]
        rr = rerank_map.get(k, (0, 0, False))[2]
        lg = lgbm_map.get(k, (0, 0, False))[2]

        if r0 and rr and lg:
            agree_all_correct += 1
        elif not r0 and not rr and not lg:
            agree_all_wrong += 1
        elif lg and not r0 and not rr:
            lgbm_unique_correct += 1
            disagree_lgbm_correct.append(k)
        elif not lg:
            lgbm_unique_wrong += 1

    print(f"\n  All 3 correct:   {agree_all_correct}")
    print(f"  All 3 wrong:     {agree_all_wrong}")
    print(f"  LGBM only right: {lgbm_unique_correct}")
    print(f"  LGBM misses (where at least 1 other gets it): {lgbm_unique_wrong}")
    if disagree_lgbm_correct:
        print(f"  LGBM-unique correct stars: {disagree_lgbm_correct}")

    # ── Stars the augmented ranker gets right but rank-0 misses ────────
    print(f"\n  Stars where augmented ranker helps over rank-0:")
    for k in all_kepids:
        r0 = rank0_map.get(k, (0, 0, False))
        lg = lgbm_map.get(k, (0, 0, False))
        if not r0[2] and lg[2]:
            print(f"    KIC_{k:08d}: true_p={r0[1]:.2f}d, rank0_p={r0[0]:.2f}d, lgbm_p={lg[0]:.2f}d")

    # ── Stars where augmented ranker loses to rank-0 ───────────────────
    print(f"\n  Stars where augmented ranker loses to rank-0:")
    for k in all_kepids:
        r0 = rank0_map.get(k, (0, 0, False))
        lg = lgbm_map.get(k, (0, 0, False))
        if r0[2] and not lg[2]:
            print(f"    KIC_{k:08d}: true_p={r0[1]:.2f}d, rank0_p={r0[0]:.2f}d, lgbm_p={lg[0]:.2f}d")


if __name__ == "__main__":
    main()
