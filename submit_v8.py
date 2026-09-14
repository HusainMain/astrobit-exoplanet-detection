"""
Submission v8: Detector probability for both prediction AND confidence.
No SDE>8 — let the detector decide. Sweep thresholds with recall constraints.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score,
    roc_auc_score, precision_recall_curve
)
from src.data_loader import load_labels, load_truth, make_target


def main():
    print("=" * 70)
    print("SUBMISSION v8: Detector-threshold (no SDE>8)")
    print("=" * 70)

    # Load data
    det = pd.read_csv("outputs/dev_detected_v7.csv")
    rank0 = pd.read_csv("outputs/dev_candidates_v4.csv")
    rank0 = rank0[rank0["rank"] == 0].copy()
    rerank = pd.read_csv("outputs/dev_reranked_v4.csv")

    labels = load_labels("dev")
    truth = load_truth("dev")
    target_info = make_target(labels, truth)
    has_planet_map = target_info["target"]

    # Merge
    merged = pd.merge(
        rank0[["kepid", "bls_sde", "bls_period", "bls_depth_ppm", "bls_duration_hours",
               "bls_n_transits", "bls_snr", "dist_to_systematic", "systematic_fraction",
               "near_systematic", "transit_coverage", "box_fit_snr"]],
        det[["kepid", "proba"]],
        on="kepid", how="left"
    )
    merged["label"] = merged["kepid"].map(has_planet_map).fillna(0).astype(int)

    # Reranker for period selection
    rerank_star = rerank.groupby("kepid").agg(
        reranker_proba=("proba_ensemble", "max"),
        reranker_period=("bls_period", lambda x: x.iloc[rerank.loc[x.index, "proba_ensemble"].argmax()]),
        reranker_depth=("bls_depth_ppm", lambda x: x.iloc[rerank.loc[x.index, "proba_ensemble"].argmax()]),
        reranker_duration=("bls_duration_hours", lambda x: x.iloc[rerank.loc[x.index, "proba_ensemble"].argmax()]),
    ).reset_index()
    merged = pd.merge(merged, rerank_star, on="kepid", how="left")

    y = merged["label"].values
    proba = merged["proba"].values

    print(f"  {len(merged)} stars, {y.sum()} positive, {len(y) - y.sum()} negative")
    print(f"  Detector proba: min={proba.min():.3f}, max={proba.max():.3f}")

    # ── THRESHOLD SWEEP ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("THRESHOLD SWEEP (detector probability)")
    print(f"{'='*70}")
    print(f"{'Thresh':>6s} {'Pred+':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'F1*Rec':>7s}")

    results = []
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred = (proba >= thresh).astype(int)
        tp = int(((y_pred == 1) & (y == 1)).sum())
        fp = int(((y_pred == 1) & (y == 0)).sum())
        fn = int(((y_pred == 0) & (y == 1)).sum())
        tn = int(((y_pred == 0) & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1_weighted = f1 * rec  # Penalize low recall
        results.append({
            "thresh": thresh, "pred_pos": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "prec": prec, "rec": rec, "f1": f1, "f1_rec": f1_weighted,
        })

    rdf = pd.DataFrame(results)

    # Show key thresholds
    print(f"\n--- Key thresholds ---")
    key_threshs = [0.10, 0.15, 0.20, 0.23, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    for t in key_threshs:
        idx = (rdf["thresh"] - t).abs().idxmin()
        r = rdf.loc[idx]
        print(f"  {t:.2f}: pred={int(r['pred_pos']):>3d}, TP={int(r['tp']):>2d}, FP={int(r['fp']):>2d}, "
              f"FN={int(r['fn']):>2d}, Prec={r['prec']:.3f}, Rec={r['rec']:.3f}, F1={r['f1']:.3f}")

    # ── BEST F1 (unconstrained) ─────────────────────────────────────
    best_f1_row = rdf.loc[rdf["f1"].idxmax()]
    print(f"\n--- Best F1 (unconstrained) ---")
    print(f"  Threshold: {best_f1_row['thresh']:.2f}")
    print(f"  Predicted: {int(best_f1_row['pred_pos'])}, TP={int(best_f1_row['tp'])}, "
          f"FP={int(best_f1_row['fp'])}, FN={int(best_f1_row['fn'])}")
    print(f"  Precision: {best_f1_row['prec']:.3f}, Recall: {best_f1_row['rec']:.3f}, F1: {best_f1_row['f1']:.3f}")

    # ── RECALL-CONSTRAINED THRESHOLDS ────────────────────────────────
    for min_recall in [1.00, 0.98, 0.95, 0.90]:
        eligible = rdf[rdf["rec"] >= min_recall]
        if len(eligible) == 0:
            print(f"\n--- Recall >= {min_recall:.0%}: NO THRESHOLD MEETS CONSTRAINT ---")
            continue
        best = eligible.loc[eligible["f1"].idxmax()]
        print(f"\n--- Best F1 where Recall >= {min_recall:.0%} ---")
        print(f"  Threshold: {best['thresh']:.2f}")
        print(f"  Predicted: {int(best['pred_pos'])}, TP={int(best['tp'])}, "
              f"FP={int(best['fp'])}, FN={int(best['fn'])}")
        print(f"  Precision: {best['prec']:.3f}, Recall: {best['rec']:.3f}, F1: {best['f1']:.3f}")

    # ── FP VETO LAYER ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FP VETO LAYER")
    print(f"{'='*70}")

    # Apply veto: flag stars as "vetoed" if they match FP signature
    # FP signature: near_systematic=1 AND transit_coverage<0.4 AND low non-systematic SDE
    merged["veto_score"] = 0
    # Near systematic
    merged.loc[merged["near_systematic"] == 1, "veto_score"] += 1
    # Low coverage
    merged.loc[merged["transit_coverage"] < 0.3, "veto_score"] += 1
    # Low box_fit_snr
    merged.loc[merged["box_fit_snr"] < 50, "veto_score"] += 1
    # High systematic fraction
    merged.loc[merged["systematic_fraction"] > 0.08, "veto_score"] += 1

    # Veto stars with score >= 3 (very likely FP)
    print(f"\n  Veto score distribution: {dict(merged['veto_score'].value_counts().sort_index())}")

    # Test different veto thresholds
    for veto_thresh in [2, 3]:
        for det_thresh in [0.20, 0.23, 0.25, 0.30]:
            y_pred_raw = (proba >= det_thresh).astype(int)
            vetoed = merged["veto_score"] >= veto_thresh
            y_pred_vetoed = y_pred_raw.copy()
            y_pred_vetoed[vetoed] = 0  # Remove vetoed predictions

            tp = int(((y_pred_vetoed == 1) & (y == 1)).sum())
            fp = int(((y_pred_vetoed == 1) & (y == 0)).sum())
            fn = int(((y_pred_vetoed == 0) & (y == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            n_vetoed = vetoed.sum()
            n_vetoed_tp = int(((y_pred_raw == 1) & vetoed & (y == 1)).sum())
            n_vetoed_fp = int(((y_pred_raw == 1) & vetoed & (y == 0)).sum())
            print(f"  Veto>={veto_thresh}, Det>={det_thresh:.2f}: "
                  f"pred={int(y_pred_vetoed.sum())}, TP={tp}, FP={fp}, "
                  f"F1={f1:.3f}, Rec={rec:.3f} "
                  f"[vetoed {n_vetoed}: {n_vetoed_tp} TP, {n_vetoed_fp} FP]")

    # ── BEST COMBINED STRATEGY ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("BEST STRATEGIES")
    print(f"{'='*70}")

    strategies = []

    # Strategy A: Best F1 (unconstrained)
    t = best_f1_row["thresh"]
    y_pred = (proba >= t).astype(int)
    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    strategies.append(("A: Best F1", t, tp, fp, fn, prec, rec, f1))

    # Strategy B: Recall >= 98%
    eligible = rdf[rdf["rec"] >= 0.98]
    if len(eligible) > 0:
        best = eligible.loc[eligible["f1"].idxmax()]
        t = best["thresh"]
        y_pred = (proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y == 1)).sum())
        fp = int(((y_pred == 1) & (y == 0)).sum())
        fn = int(((y_pred == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        strategies.append(("B: Rec>=98%", t, tp, fp, fn, prec, rec, f1))

    # Strategy C: Recall >= 95%
    eligible = rdf[rdf["rec"] >= 0.95]
    if len(eligible) > 0:
        best = eligible.loc[eligible["f1"].idxmax()]
        t = best["thresh"]
        y_pred = (proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y == 1)).sum())
        fp = int(((y_pred == 1) & (y == 0)).sum())
        fn = int(((y_pred == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        strategies.append(("C: Rec>=95%", t, tp, fp, fn, prec, rec, f1))

    # Strategy D: Best F1 with veto>=3
    best_v1_f1, best_v1_t = 0, 0.2
    for det_t in np.arange(0.10, 0.80, 0.01):
        y_pred = (proba >= det_t).astype(int)
        vetoed = merged["veto_score"] >= 3
        y_pred[vetoed] = 0
        tp = int(((y_pred == 1) & (y == 1)).sum())
        fp = int(((y_pred == 1) & (y == 0)).sum())
        fn = int(((y_pred == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_v1_f1:
            best_v1_f1 = f1
            best_v1_t = det_t
    y_pred = (proba >= best_v1_t).astype(int)
    vetoed = merged["veto_score"] >= 3
    y_pred[vetoed] = 0
    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    strategies.append(("D: Veto>=3", best_v1_t, tp, fp, fn, prec, rec, f1))

    print(f"{'Strategy':<16s} {'Thresh':>6s} {'Pred+':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    for name, t, tp, fp, fn, prec, rec, f1 in strategies:
        pred = tp + fp
        print(f"  {name:<14s} {t:.2f}  {pred:>6d} {tp:>4d} {fp:>4d} {fn:>4d} "
              f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

    # ── PICK BEST AND GENERATE SUBMISSION ────────────────────────────
    # Choose Strategy B (Rec>=98%) as default — balances F1 and recall
    best_strat = None
    for s in strategies:
        if "98%" in s[0]:
            best_strat = s
            break
    if best_strat is None:
        best_strat = strategies[0]  # Fallback to best F1

    strat_name, best_t, best_tp, best_fp, best_fn, _, _, best_f1 = best_strat
    print(f"\n--- Selected: {strat_name} (threshold={best_t:.2f}, F1={best_f1:.3f}) ---")

    # Generate predictions
    merged["prediction"] = (proba >= best_t).astype(int)

    # Apply veto if using strategy D
    if "Veto" in strat_name:
        vetoed = merged["veto_score"] >= 3
        merged.loc[vetoed, "prediction"] = 0

    # Period selection: reranker if confident, else rank-0
    merged["best_period"] = merged["bls_period"]
    merged["best_depth"] = merged["bls_depth_ppm"]
    merged["best_duration"] = merged["bls_duration_hours"]

    # Use reranker period where reranker is confident
    rerank_confident = merged["reranker_proba"].fillna(0) >= 0.5
    merged.loc[rerank_confident, "best_period"] = merged.loc[rerank_confident, "reranker_period"]
    merged.loc[rerank_confident, "best_depth"] = merged.loc[rerank_confident, "reranker_depth"]
    merged.loc[rerank_confident, "best_duration"] = merged.loc[rerank_confident, "reranker_duration"]

    # Confidence = detector probability (already calibrated)
    merged["confidence"] = np.clip(proba, 0.01, 0.99)

    # Final metrics
    y_pred = merged["prediction"].values
    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    f1_final = 2 * tp / max(2 * tp + fp + fn, 1)
    ap_final = average_precision_score(y, merged["confidence"].values)

    print(f"\n  FINAL: predicted={y_pred.sum()}, TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  F1: {f1_final:.3f}, AP: {ap_final:.3f}")

    # Build submission
    submission = pd.DataFrame({
        "star_id": merged["kepid"].apply(lambda x: f"KIC_{int(x):08d}"),
        "prediction": merged["prediction"],
        "confidence": merged["confidence"],
        "period": np.where(merged["prediction"] == 1, merged["best_period"], 0.0),
        "depth_ppm": np.where(merged["prediction"] == 1, merged["best_depth"], 0.0),
        "duration_hours": np.where(merged["prediction"] == 1, merged["best_duration"], 0.0),
    })

    pos = submission[submission["prediction"] == 1].sort_values("confidence", ascending=False)
    print(f"\n  Positive predictions: {len(pos)}")
    for _, row in pos.head(10).iterrows():
        print(f"    {row['star_id']}: P={row['period']:.2f}d, "
              f"depth={row['depth_ppm']:.0f}ppm, conf={row['confidence']:.3f}")
    if len(pos) > 10:
        print(f"    ... and {len(pos) - 10} more")

    submission.to_csv("outputs/submission_v8.csv", index=False)
    print(f"\n  Saved outputs/submission_v8.csv ({len(submission)} rows)")

    return submission


if __name__ == "__main__":
    main()
