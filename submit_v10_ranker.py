"""
Submission v10-ranker: Augmented LGBMRanker for period selection.

Architecture:
  BLS top-20 candidates -> candidate features -> augmented LGBMRanker score
  -> select best-ranked candidate per star
  -> detection confidence from v7 detector (star-level)
  -> period/depth/duration from ranker-selected candidate
"""
import sys, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from lightgbm import LGBMRanker
from src.data_loader import load_labels, load_truth, make_target
from src.config import OUTPUTS_DIR, MODELS_DIR

ALIAS_RATIOS = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]


def is_alias(p1, p2):
    if p1 <= 0 or p2 <= 0:
        return False
    return min([abs(p1 / p2 - ar) for ar in ALIAS_RATIOS]) < 0.02


def load_ranker():
    """Load the trained augmented ranker from disk."""
    model_path = MODELS_DIR / "augmented_lgbm_ranker.pkl"
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["feature_columns"]


def main():
    print("=" * 70)
    print("SUBMISSION v10-ranker: Augmented LGBMRanker period selection")
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

    # ── Load ranker ────────────────────────────────────────────────────
    ranker, feature_cols = load_ranker()
    print(f"  Loaded ranker: {len(feature_cols)} features")

    # ── Score all candidates with ranker ───────────────────────────────
    candidates = candidates.copy()
    X = candidates[feature_cols].fillna(0).values
    candidates["ranker_score"] = ranker.predict(X)

    # ── Star-level: pick best candidate per star ───────────────────────
    best_per_star = candidates.loc[
        candidates.groupby("kepid")["ranker_score"].idxmax()
    ].copy()

    # Merge detection scores
    best_per_star = best_per_star.merge(
        det[["kepid", "proba"]].rename(columns={"proba": "det_proba"}),
        on="kepid", how="left"
    )
    best_per_star["det_proba"] = best_per_star["det_proba"].fillna(0)

    # ── Period recovery evaluation ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("DEV PERIOD RECOVERY")
    print("=" * 70)

    for method_name, score_df, proba_col in [
        ("Rank-0 (highest SDE)", candidates, "bls_sde"),
        ("Original reranker", orig_rerank, "proba_ensemble"),
        ("Augmented ranker", candidates, "ranker_score"),
    ]:
        correct = 0
        n = 0
        for kepid_val, grp in score_df.groupby("kepid"):
            kepid = int(kepid_val)
            true_p = truth_dict.get(kepid, {}).get("period_days", None)
            if true_p is None or true_p <= 0:
                continue
            n += 1
            p = grp.sort_values(proba_col, ascending=False).iloc[0]["bls_period"]
            if is_alias(p, true_p):
                correct += 1
        print(f"  {method_name:30s}: {correct}/{n} = {correct/n:.3f}")

    # ── Per-bin breakdown ──────────────────────────────────────────────
    print(f"\n  Per-bin (augmented ranker):")
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

    # ── Detection evaluation ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DETECTION METRICS")
    print("=" * 70)

    y_true = best_per_star["kepid"].map(has_planet_map).fillna(0).astype(int).values
    y_score = best_per_star["det_proba"].values

    ap = average_precision_score(y_true, y_score)
    f1, prec, rec, _ = precision_recall_fscore_support(
        y_true, (y_score >= 0.23).astype(int), average="binary", zero_division=0
    )
    print(f"  AP: {ap:.4f}")
    print(f"  At threshold 0.23: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    # ── Generate submission ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("GENERATING SUBMISSION")
    print("=" * 70)

    submission = best_per_star[["kepid"]].copy()
    submission["star_id"] = submission["kepid"].apply(lambda x: f"KIC_{int(x):08d}")
    submission["prediction"] = (best_per_star["det_proba"] >= 0.23).astype(int)
    submission["confidence"] = np.clip(best_per_star["det_proba"].values, 0.01, 0.99)
    submission["period"] = np.where(
        submission["prediction"] == 1,
        best_per_star["bls_period"].values,
        0.0
    )
    submission["depth_ppm"] = np.where(
        submission["prediction"] == 1,
        best_per_star["bls_depth_ppm"].values,
        0.0
    )
    submission["duration_hours"] = np.where(
        submission["prediction"] == 1,
        best_per_star["bls_duration_hours"].values,
        0.0
    )

    submission = submission[["star_id", "prediction", "confidence", "period", "depth_ppm", "duration_hours"]]

    n_pos = submission["prediction"].sum()
    print(f"  Stars: {len(submission)}")
    print(f"  Predicted positive: {n_pos}")

    # Show top positives
    pos = submission[submission["prediction"] == 1].sort_values("confidence", ascending=False)
    print(f"\n  Top positive predictions:")
    for _, row in pos.head(10).iterrows():
        print(f"    {row['star_id']}: P={row['period']:.2f}d, "
              f"depth={row['depth_ppm']:.0f}ppm, conf={row['confidence']:.3f}")
    if len(pos) > 10:
        print(f"    ... and {len(pos) - 10} more")

    out_path = OUTPUTS_DIR / "submission_v10_ranker.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return submission


if __name__ == "__main__":
    main()
