"""
Star-level detector v9: enhanced features with period-quality additions.
Train on train_aggregate_features_v2, evaluate on dev_aggregate_features_v2.
Compare against v7 baseline (136 features) vs v9 (147 features).
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
    print("STAR-LEVEL DETECTOR v9 (enhanced period-quality features)")
    print("=" * 70)

    train = pd.read_csv("outputs/train_aggregate_features_v2.csv")
    dev = pd.read_csv("outputs/dev_aggregate_features_v2.csv")

    print(f"Train: {len(train)} stars, {len(train.columns)} features")
    print(f"Dev: {len(dev)} stars")
    print(f"Train has_planet: {dict(train['has_planet'].value_counts())}")
    print(f"Dev has_planet: {dict(dev['has_planet'].value_counts())}")

    # Features: exclude label columns and kepid
    exclude = {"kepid", "label", "has_planet"}
    feat_cols = [c for c in train.columns if c not in exclude
                 and train[c].dtype in ["float64", "int64", "float32", "int32"]]
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
    print(f"{'Thresh':>6s} {'Pred+':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s}")

    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.10, 0.90, 0.01):
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

    # Print key thresholds
    for t in [0.15, 0.20, 0.23, 0.30, 0.40, 0.50]:
        y_pred = (dev_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_dev == 1)).sum())
        fp = int(((y_pred == 1) & (y_dev == 0)).sum())
        fn = int(((y_pred == 0) & (y_dev == 1)).sum())
        tn = int(((y_pred == 0) & (y_dev == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        marker = " <-- BEST" if abs(t - best_thresh) < 0.005 else ""
        print(f"  {t:.2f}  {tp+fp:6d} {tp:4d} {fp:4d} {fn:4d} {tn:4d} "
              f"{prec:6.3f} {rec:6.3f} {f1:6.3f}{marker}")

    # Best threshold detail
    y_pred_best = (dev_proba >= best_thresh).astype(int)
    tp = int(((y_pred_best == 1) & (y_dev == 1)).sum())
    fp = int(((y_pred_best == 1) & (y_dev == 0)).sum())
    fn = int(((y_pred_best == 0) & (y_dev == 1)).sum())
    tn = int(((y_pred_best == 0) & (y_dev == 0)).sum())
    print(f"\nBest threshold: {best_thresh:.2f}, F1: {best_f1:.3f}")
    print(f"  Predicted: {tp+fp}, TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {tp/max(tp+fp,1):.3f}, Recall: {tp/max(tp+fn,1):.3f}")

    # AUC and AP
    auc = roc_auc_score(y_dev, dev_proba)
    ap = average_precision_score(y_dev, dev_proba)
    print(f"\n  AUC: {auc:.3f}")
    print(f"  AP (PR-AUC): {ap:.3f}")

    # ── Feature importance ───────────────────────────────────────────
    print(f"\n--- XGBoost top-20 features ---")
    importances = pd.DataFrame({
        "feature": feat_cols,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    for _, row in importances.head(20).iterrows():
        print(f"  {row['feature']:<40s}: {row['importance']:.4f}")

    # ── New feature importances ──────────────────────────────────────
    print(f"\n--- New period-quality feature importances ---")
    new_feats = [f for f in feat_cols if any(f.startswith(p) for p in [
        "r0_reranker", "best_reranker", "reranker_proba_gap",
        "r0_depth_cv", "r0_timing_rms", "r0_quarter", "r0_box_fit",
        "r0_in_out", "r0_odd_even", "n_non_sys", "r0_is_sys",
        "has_good_non", "sde_ratio", "sde_gap_1_2", "n_good_depth",
        "period_quality",
    ])]
    new_imp = importances[importances["feature"].isin(new_feats)]
    for _, row in new_imp.iterrows():
        print(f"  {row['feature']:<40s}: {row['importance']:.4f}")

    # ── Confidence calibration check ─────────────────────────────────
    print(f"\n--- Confidence distribution by class ---")
    for label, name in [(1, "POS"), (0, "NEG")]:
        vals = dev_proba[y_dev == label]
        print(f"  {name}: min={vals.min():.3f}, p25={np.percentile(vals, 25):.3f}, "
              f"median={np.median(vals):.3f}, p75={np.percentile(vals, 75):.3f}, max={vals.max():.3f}")

    # Save
    dev["proba"] = dev_proba
    dev["proba_xgb"] = dev_proba_xgb
    dev["proba_lgbm"] = dev_proba_lgbm
    dev.to_csv("outputs/dev_detected_v9.csv", index=False)
    print(f"\nSaved outputs/dev_detected_v9.csv")

    return dev, best_thresh, feat_cols, xgb, lgbm


if __name__ == "__main__":
    dev, best_thresh, feat_cols, xgb_model, lgbm_model = main()
