"""
Star-level detector using top-20 candidate aggregate features.
Train on train_aggregate, evaluate on dev_aggregate.
Separate detection from period selection.
"""
import sys, warnings
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
    print("=" * 70)
    print("STAR-LEVEL DETECTOR (aggregate features)")
    print("=" * 70)

    train = pd.read_csv("outputs/train_aggregate_features.csv")
    dev = pd.read_csv("outputs/dev_aggregate_features.csv")

    print(f"Train: {len(train)} stars, {len(train.columns)} features")
    print(f"Dev: {len(dev)} stars")
    print(f"Train has_planet: {dict(train['has_planet'].value_counts())}")
    print(f"Dev has_planet: {dict(dev['has_planet'].value_counts())}")

    # Features: exclude label columns and kepid
    exclude = {"kepid", "label", "has_planet"}
    feat_cols = [c for c in train.columns if c not in exclude and train[c].dtype in ["float64", "int64", "float32", "int32"]]
    print(f"\nFeatures: {len(feat_cols)}")

    X_train = train[feat_cols].values
    y_train = train["has_planet"].values.astype(int)
    X_dev = dev[feat_cols].values
    y_dev = dev["has_planet"].values.astype(int)

    n_pos_train = y_train.sum()
    n_neg_train = len(y_train) - n_pos_train
    print(f"Train: {n_pos_train} positive, {n_neg_train} negative")
    print(f"Dev: {y_dev.sum()} positive, {len(y_dev) - y_dev.sum()} negative")

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

    dev_proba_xgb = xgb.predict_proba(X_dev)[:, 1]

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

    dev_proba_lgbm = lgbm.predict_proba(X_dev)[:, 1]

    # ── Ensemble ─────────────────────────────────────────────────────
    dev_proba = 0.5 * dev_proba_xgb + 0.5 * dev_proba_lgbm

    # ── Metrics at various thresholds ────────────────────────────────
    print(f"\n--- Threshold sweep ---")
    print(f"{'Thresh':>6s} {'Pred+':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")

    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.20, 0.90, 0.01):
        y_pred = (dev_proba >= thresh).astype(int)
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
    y_pred_best = (dev_proba >= best_thresh).astype(int)
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
    auc = roc_auc_score(y_dev, dev_proba)
    ap = average_precision_score(y_dev, dev_proba)
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
    dev["proba"] = dev_proba
    dev["pred"] = y_pred_best

    # Detected positives
    detected_pos = dev[(dev["pred"] == 1) & (dev["has_planet"] == 1)]
    false_pos = dev[(dev["pred"] == 1) & (dev["has_planet"] == 0)]
    missed = dev[(dev["pred"] == 0) & (dev["has_planet"] == 1)]

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
        vals = dev_proba[y_dev == label]
        print(f"  {name}: min={vals.min():.3f}, p25={np.percentile(vals, 25):.3f}, "
              f"median={np.median(vals):.3f}, p75={np.percentile(vals, 75):.3f}, max={vals.max():.3f}")

    # Save
    dev.to_csv("outputs/dev_detected_v7.csv", index=False)
    print(f"\nSaved outputs/dev_detected_v7.csv")

    # Return results for submission generation
    return dev, best_thresh, feat_cols, xgb, lgbm


if __name__ == "__main__":
    dev, best_thresh, feat_cols, xgb_model, lgbm_model = main()
