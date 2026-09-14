"""
Submission v7: SDE>8 for hard predictions + detector probability for confidence.
Separates detection decision from confidence ranking.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from src.data_loader import load_labels, load_truth, make_target


def main():
    print("=" * 70)
    print("SUBMISSION v7: SDE>8 + detector confidence")
    print("=" * 70)

    # Load data
    det = pd.read_csv("outputs/dev_detected_v7.csv")  # star-level detector output
    rank0 = pd.read_csv("outputs/dev_candidates_v4.csv")
    rank0 = rank0[rank0["rank"] == 0].copy()
    rerank = pd.read_csv("outputs/dev_reranked_v4.csv")

    labels = load_labels("dev")
    truth = load_truth("dev")
    target_info = make_target(labels, truth)
    has_planet_map = target_info["target"]

    # Merge detector probabilities with rank-0 candidates
    merged = pd.merge(
        rank0[["kepid", "bls_sde", "bls_period", "bls_depth_ppm", "bls_duration_hours",
               "bls_n_transits", "bls_snr", "dist_to_systematic", "systematic_fraction",
               "near_systematic", "transit_coverage", "box_fit_snr"]],
        det[["kepid", "proba"]],
        on="kepid", how="left"
    )
    merged["label"] = merged["kepid"].map(has_planet_map).fillna(0).astype(int)

    # Also get reranker probability for comparison
    rerank_star = rerank.groupby("kepid").agg(
        reranker_proba=("proba_ensemble", "max"),
        reranker_period=("bls_period", lambda x: x.iloc[rerank.loc[x.index, "proba_ensemble"].argmax()]),
    ).reset_index()
    merged = pd.merge(merged, rerank_star, on="kepid", how="left")

    print(f"  {len(merged)} stars, {merged['label'].sum()} positive")

    # ── Strategy 1: SDE>8 baseline (reference) ──────────────────────
    print("\n--- Strategy 1: SDE>8 baseline ---")
    y = merged["label"].values
    pred_sde = (merged["bls_sde"] > 8).astype(int).values
    f1, prec, rec, _ = precision_recall_fscore_support(y, pred_sde, average="binary", zero_division=0)
    print(f"  Predicted: {pred_sde.sum()}, TP: {(pred_sde & y).sum()}, FP: {pred_sde.sum() - (pred_sde & y).sum()}")
    print(f"  F1={f1:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")

    # ── Strategy 2: Detector probability as confidence ───────────────
    print("\n--- Strategy 2: Detector probability (confidence) ---")
    proba_det = merged["proba"].values
    ap_det = average_precision_score(y, proba_det)
    auc_det = roc_auc_score(y, proba_det)
    print(f"  AP: {ap_det:.3f}, AUC: {auc_det:.3f}")

    # ── Strategy 3: Reranker probability as confidence ───────────────
    print("\n--- Strategy 3: Reranker probability (confidence) ---")
    proba_rerank = merged["reranker_proba"].fillna(0).values
    ap_rerank = average_precision_score(y, proba_rerank)
    auc_rerank = roc_auc_score(y, proba_rerank)
    print(f"  AP: {ap_rerank:.3f}, AUC: {auc_rerank:.3f}")

    # ── Strategy 4: Combined confidence (detector + reranker) ────────
    print("\n--- Strategy 4: Combined confidence ---")
    # Normalize both to [0,1]
    det_norm = (proba_det - proba_det.min()) / (proba_det.max() - proba_det.min() + 1e-9)
    rerank_norm = (proba_rerank - proba_rerank.min()) / (proba_rerank.max() - proba_rerank.min() + 1e-9)
    proba_combined = 0.6 * det_norm + 0.4 * rerank_norm
    ap_combined = average_precision_score(y, proba_combined)
    auc_combined = roc_auc_score(y, proba_combined)
    print(f"  AP: {ap_combined:.3f}, AUC: {auc_combined:.3f}")

    # ── Strategy 5: SDE-based confidence (simple baseline) ───────────
    print("\n--- Strategy 5: SDE-based confidence ---")
    sde_vals = merged["bls_sde"].values
    proba_sde = np.clip((sde_vals - 5) / 40, 0.01, 0.99)
    ap_sde = average_precision_score(y, proba_sde)
    auc_sde = roc_auc_score(y, proba_sde)
    print(f"  AP: {ap_sde:.3f}, AUC: {auc_sde:.3f}")

    # ── Find best confidence source ──────────────────────────────────
    print(f"\n--- Confidence ranking comparison ---")
    scores = {
        "SDE-based": (ap_sde, auc_sde),
        "Reranker": (ap_rerank, auc_rerank),
        "Detector": (ap_det, auc_det),
        "Combined": (ap_combined, auc_combined),
    }
    best_name = max(scores, key=lambda k: scores[k][0])
    for name, (ap, auc) in sorted(scores.items(), key=lambda x: x[1][0], reverse=True):
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:<12s}: AP={ap:.3f}, AUC={auc:.3f}{marker}")

    # ── Use best confidence for submission ───────────────────────────
    best_ap, best_auc = scores[best_name]
    if best_name == "Detector":
        confidences = proba_det
    elif best_name == "Combined":
        confidences = proba_combined
    elif best_name == "Reranker":
        confidences = proba_rerank
    else:
        confidences = proba_sde

    # ── Period selection: reranker if confident, else rank-0 ─────────
    # For each star, pick the best period based on confidence
    best_periods = []
    best_depths = []
    best_durations = []
    for _, row in merged.iterrows():
        # Use reranker period if reranker is confident
        if row.get("reranker_proba", 0) >= 0.5:
            best_periods.append(row.get("reranker_period", row["bls_period"]))
            # Get depth/duration for this period from reranked data
            star_rerank = rerank[rerank["kepid"] == row["kepid"]]
            if len(star_rerank) > 0:
                best_r = star_rerank.iloc[star_rerank["proba_ensemble"].argmax()]
                best_depths.append(best_r["bls_depth_ppm"])
                best_durations.append(best_r["bls_duration_hours"])
            else:
                best_depths.append(row["bls_depth_ppm"])
                best_durations.append(row["bls_duration_hours"])
        else:
            best_periods.append(row["bls_period"])
            best_depths.append(row["bls_depth_ppm"])
            best_durations.append(row["bls_duration_hours"])

    merged["best_period"] = best_periods
    merged["best_depth"] = best_depths
    merged["best_duration"] = best_durations

    # ── Hard prediction: SDE>8 (guaranteed recall) ───────────────────
    merged["prediction"] = (merged["bls_sde"] > 8).astype(int)
    merged["confidence"] = confidences

    # ── Verify improvement over v6 ───────────────────────────────────
    print(f"\n--- Final v7 submission ---")
    y_pred = merged["prediction"].values
    y_conf = merged["confidence"].values
    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    f1_final = 2 * tp / max(2 * tp + fp + fn, 1)
    ap_final = average_precision_score(y, y_conf)
    print(f"  Predicted positive: {y_pred.sum()}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  F1: {f1_final:.3f}")
    print(f"  AP (confidence ranking): {ap_final:.3f}")

    # ── Generate submission CSV ──────────────────────────────────────
    submission = pd.DataFrame({
        "star_id": merged["kepid"].apply(lambda x: f"KIC_{int(x):08d}"),
        "prediction": merged["prediction"],
        "confidence": np.clip(merged["confidence"], 0.01, 0.99),
        "period": np.where(merged["prediction"] == 1, merged["best_period"], 0.0),
        "depth_ppm": np.where(merged["prediction"] == 1, merged["best_depth"], 0.0),
        "duration_hours": np.where(merged["prediction"] == 1, merged["best_duration"], 0.0),
    })

    # Ensure confidence spread
    conf = submission["confidence"].values
    print(f"\n  Confidence: min={conf.min():.3f}, max={conf.max():.3f}, "
          f"unique={submission['confidence'].nunique()}")

    # Show positive predictions
    pos = submission[submission["prediction"] == 1].sort_values("confidence", ascending=False)
    print(f"  Positive predictions: {len(pos)}")
    for _, row in pos.head(10).iterrows():
        print(f"    {row['star_id']}: P={row['period']:.2f}d, "
              f"depth={row['depth_ppm']:.0f}ppm, conf={row['confidence']:.3f}")
    if len(pos) > 10:
        print(f"    ... and {len(pos) - 10} more")

    submission.to_csv("outputs/submission_v7.csv", index=False)
    print(f"\n  Saved outputs/submission_v7.csv ({len(submission)} rows)")

    return submission


if __name__ == "__main__":
    main()
