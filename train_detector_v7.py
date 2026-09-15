"""
Star-level detector using top-20 candidate aggregate features.
Train on train_aggregate, evaluate on dev or predict on private.
"""
import sys, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, precision_recall_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-private", action="store_true",
                        help="Train on train, predict on private (no evaluation)")
    args = parser.parse_args()

    print("=" * 70)
    print("STAR-LEVEL DETECTOR (aggregate features)")
    print("=" * 70)

    train = pd.read_csv("outputs/train_aggregate_features.csv")

    if args.predict_private:
        pred_target = "private"
        pred_df = pd.read_csv("outputs/private_aggregate_features.csv")
        print(f"Mode: train on train, predict on private")
    else:
        pred_target = "dev"
        pred_df = pd.read_csv("outputs/dev_aggregate_features.csv")
        print(f"Mode: train on train, evaluate on dev")

    print(f"Train: {len(train)} stars, {len(train.columns)} features")
    print(f"Predict target ({pred_target}): {len(pred_df)} stars")
    print(f"Train has_planet: {dict(train['has_planet'].value_counts())}")

    # Features: exclude label columns and kepid
    exclude = {"kepid", "label", "has_planet"}
    feat_cols = [c for c in train.columns if c not in exclude and train[c].dtype in ["float64", "int64", "float32", "int32"]]
    print(f"\nFeatures: {len(feat_cols)}")

    X_train = train[feat_cols].values
    y_train = train["has_planet"].values.astype(int)

    n_pos_train = y_train.sum()
    n_neg_train = len(y_train) - n_pos_train
    print(f"Train: {n_pos_train} positive, {n_neg_train} negative")

    # ── XGBoost ─────────────────────────────────────────────────────
    print("\n--- XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=n_neg_train / max(n_pos_train, 1),
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    # ── LightGBM ────────────────────────────────────────────────────
    print("--- LightGBM ---")
    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=n_neg_train / max(n_pos_train, 1),
        metric="average_precision",
        early_stopping_rounds=50,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train)])

    # ── Predict on target ──────────────────────────────────────────
    X_pred = pred_df[feat_cols].values
    proba_xgb = xgb.predict_proba(X_pred)[:, 1]
    proba_lgbm = lgbm.predict_proba(X_pred)[:, 1]
    proba = 0.5 * proba_xgb + 0.5 * proba_lgbm

    if args.predict_private:
        # Private: no ground truth, just save predictions
        pred_df["proba"] = proba
        pred_out = "outputs/private_detected_v7.csv"
        pred_df[["kepid", "proba"]].to_csv(pred_out, index=False)
        print(f"\nSaved {pred_out}")

        # Show confidence distribution
        print(f"\nConfidence distribution:")
        print(f"  min={proba.min():.3f}, median={np.median(proba):.3f}, max={proba.max():.3f}")
        print(f"  >0.5: {(proba > 0.5).sum()}, >0.23: {(proba > 0.23).sum()}")
        return pred_df, 0.23, feat_cols, xgb, lgbm

    # ── Dev evaluation ──────────────────────────────────────────────
    y_dev = pred_df["has_planet"].values.astype(int)
    print(f"\nDev: {y_dev.sum()} positive, {len(y_dev) - y_dev.sum()} negative")

    # ── Metrics at various thresholds ────────────────────────────────
    print(f"\n--- Threshold sweep ---")
    print(f"{'Thresh':>6s} {'Pred+':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")

    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.20, 0.90, 0.01):
        y_pred = (proba >= thresh).astype(int)
        tp = int(((y_pred == 1) & (y_dev == 1)).sum())
        fp = int(((y_pred == 1) & (y_dev == 0)).sum())
        fn = int(((y_pred == 0) & (y_dev == 1)).sum())
        tn = int(((y_pred == 0) & (y_dev == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
        if thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80] or abs(thresh - best_thresh) < 0.005:
            marker = " <-- best" if abs(thresh - best_thresh) < 0.005 else ""
            print(f"  {thresh:.2f}  {tp+fp:6d} {tp:4d} {fp:4d} {fn:4d} {tn:4d} {prec:6.3f} {rec:6.3f} {f1:6.3f}{marker}")

    print(f"\nBest threshold: {best_thresh:.2f}, F1: {best_f1:.3f}")

    # ── Final metrics at best threshold ──────────────────────────────
    y_pred_best = (proba >= best_thresh).astype(int)
    tp = int(((y_pred_best == 1) & (y_dev == 1)).sum())
    fp = int(((y_pred_best == 1) & (y_dev == 0)).sum())
    fn = int(((y_pred_best == 0) & (y_dev == 1)).sum())
    tn = int(((y_pred_best == 0) & (y_dev == 0)).sum())
    print(f"\n--- Best threshold results ---")
    print(f"  Predicted positive: {tp + fp}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {tp/max(tp+fp,1):.3f}")
    print(f"  Recall: {tp/max(tp+fn,1):.3f}")
    print(f"  F1: {best_f1:.3f}")
    print(f"  Accuracy: {(tp+tn)/(tp+fp+fn+tn):.3f}")

    # ── AUC and AP ───────────────────────────────────────────────────
    auc = roc_auc_score(y_dev, proba)
    ap = average_precision_score(y_dev, proba)
    print(f"\n  AUC: {auc:.3f}")
    print(f"  AP (PR-AUC): {ap:.3f}")

    # ── Feature importance ───────────────────────────────────────────
    print(f"\n--- XGBoost top-15 features ---")
    importances = pd.DataFrame({
        "feature": feat_cols,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    for _, row in importances.head(15).iterrows():
        print(f"  {row['feature']:<40s}: {row['importance']:.4f}")

    # ── Per-star analysis ────────────────────────────────────────────
    print(f"\n--- Per-star detection (best threshold={best_thresh:.2f}) ---")
    pred_df["proba"] = proba
    pred_df["pred"] = y_pred_best

    # Detected positives
    detected_pos = pred_df[(pred_df["pred"] == 1) & (pred_df["has_planet"] == 1)]
    false_pos = pred_df[(pred_df["pred"] == 1) & (pred_df["has_planet"] == 0)]
    missed = pred_df[(pred_df["pred"] == 0) & (pred_df["has_planet"] == 1)]

    print(f"\n  Detected {len(detected_pos)}/{int(y_dev.sum())} positive stars")
    print(f"  False positives: {len(false_pos)}")
    print(f"  Missed: {len(missed)}")
    if len(missed) > 0:
        for _, row in missed.sort_values("proba", ascending=False).iterrows():
            print(f"    KepID {int(row['kepid'])}: proba={row['proba']:.3f}, "
                  f"r0_sde={row.get('r0_bls_sde', 0):.1f}, "
                  f"dist_372={row.get('r0_dist_to_systematic', 0):.1f}")

    # ── Confidence calibration check ─────────────────────────────────
    print(f"\n--- Confidence distribution by class ---")
    for label, name in [(1, "POS"), (0, "NEG")]:
        vals = proba[y_dev == label]
        print(f"  {name}: min={vals.min():.3f}, p25={np.percentile(vals, 25):.3f}, "
              f"median={np.median(vals):.3f}, p75={np.percentile(vals, 75):.3f}, max={vals.max():.3f}")

    # Save
    pred_df.to_csv(f"outputs/{pred_target}_detected_v7.csv", index=False)
    print(f"\nSaved outputs/{pred_target}_detected_v7.csv")

    # Return results for submission generation
    return pred_df, best_thresh, feat_cols, xgb, lgbm


if __name__ == "__main__":
    main()
