"""
Train a reranker model to pick the best candidate per star from top-20 peaks.

Uses proper train→dev split: train on train_candidates_v3, evaluate on dev_candidates_v3.
Uses correct target: has_planet = (label==1) | (injected==1).
Only stars with truth (truth.csv) are used for supervision — unknown stars are NaN.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r'C:\Users\husai\OneDrive\Pictures\Documents\AstroBit')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_fscore_support
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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

TARGET = "has_planet"


def load_and_prepare(split):
    df = pd.read_csv(f"outputs/{split}_candidates_v4.csv")

    # Only supervise on truth stars (is_correct is not NaN)
    df["label"] = df["has_planet"].astype(int)

    return df


def star_level_metrics(df, proba_col="proba"):
    """Compute star-level metrics using best candidate per star."""
    stars = df.groupby("kepid").agg(
        best_proba=(proba_col, "max"),
        label=("label", "first"),
    ).reset_index()

    y_true = stars["label"].values
    y_score = stars["best_proba"].values

    # Precision at top-k
    k = int(y_true.sum())
    top_k_idx = np.argsort(y_score)[::-1][:k]
    p_at_k = y_true[top_k_idx].sum() / k

    # AUC
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.5
    ap = average_precision_score(y_true, y_score)

    # Hard predictions (threshold at 0.5)
    y_pred = (y_score >= 0.5).astype(int)
    f1, prec, rec, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # Count detected (any candidate above 0.5)
    detected = stars[stars["best_proba"] >= 0.5]["label"].sum()
    total_pos = int(y_true.sum())

    return {
        "n_stars": len(stars),
        "n_pos": total_pos,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "detected": detected,
        "p_at_k": p_at_k,
        "auc": auc,
        "ap": ap,
    }


def main():
    print("=" * 60)
    print("Training reranker v4 (top-20 + 372d vetting)")
    print("=" * 60)

    train_df = load_and_prepare("train")
    dev_df = load_and_prepare("dev")

    print(f"\nTrain: {len(train_df)} candidates, {train_df['kepid'].nunique()} stars")
    print(f"Dev: {len(dev_df)} candidates, {dev_df['kepid'].nunique()} stars")

    train_truth = train_df[train_df["is_correct"].notna()]
    dev_truth = dev_df[dev_df["is_correct"].notna()]

    print(f"Train truth stars: {train_truth['kepid'].nunique()}, "
          f"correct candidates: {int(train_truth['is_correct'].sum())}")
    print(f"Dev truth stars: {dev_truth['kepid'].nunique()}, "
          f"correct candidates: {int(dev_truth['is_correct'].sum())}")

    # Filter features to those available
    available_feats = [f for f in CANDIDATE_FEATURES if f in train_df.columns]
    print(f"Features ({len(available_feats)}): {available_feats}")

    X_train = train_df[available_feats].values
    X_dev = dev_df[available_feats].values

    # Target: per-candidate binary label (is this the correct period?)
    # For truth stars: is_correct == 1
    # For non-truth stars: NaN (don't train on these)
    train_mask = train_df["is_correct"].notna()
    y_train = train_df.loc[train_mask, "is_correct"].astype(int).values
    X_train_supervised = X_train[train_mask.values]

    print(f"\nSupervised training: {len(y_train)} candidates from truth stars")
    print(f"  Positive: {y_train.sum()}, Negative: {len(y_train) - y_train.sum()}")

    # ── Train XGBoost ────────────────────────────────────────────────
    print("\n--- XGBoost Reranker ---")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train_supervised, y_train, eval_set=[(X_train_supervised, y_train)], verbose=False)

    dev_proba_xgb = xgb.predict_proba(X_dev)[:, 1]
    dev_df["proba_xgb"] = dev_proba_xgb

    # Per-candidate correctness
    dev_mask = dev_df["is_correct"].notna()
    y_dev = dev_df.loc[dev_mask, "is_correct"].astype(int).values
    cand_auc_xgb = roc_auc_score(y_dev, dev_proba_xgb[dev_mask.values])
    cand_ap_xgb = average_precision_score(y_dev, dev_proba_xgb[dev_mask.values])
    print(f"  Per-candidate AUC: {cand_auc_xgb:.3f}, AP: {cand_ap_xgb:.3f}")

    # Star-level metrics
    metrics_xgb = star_level_metrics(dev_df, "proba_xgb")
    print(f"  Star-level: F1={metrics_xgb['f1']:.3f}, AP={metrics_xgb['ap']:.3f}, "
          f"AUC={metrics_xgb['auc']:.3f}, P@K={metrics_xgb['p_at_k']:.3f}")
    print(f"  Detected: {metrics_xgb['detected']}/{metrics_xgb['n_pos']} positive stars")

    # ── Train LightGBM ──────────────────────────────────────────────
    print("\n--- LightGBM Reranker ---")
    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
        metric="average_precision",
        early_stopping_rounds=50,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train_supervised, y_train, eval_set=[(X_train_supervised, y_train)])

    dev_proba_lgbm = lgbm.predict_proba(X_dev)[:, 1]
    dev_df["proba_lgbm"] = dev_proba_lgbm

    cand_auc_lgbm = roc_auc_score(y_dev, dev_proba_lgbm[dev_mask.values])
    cand_ap_lgbm = average_precision_score(y_dev, dev_proba_lgbm[dev_mask.values])
    print(f"  Per-candidate AUC: {cand_auc_lgbm:.3f}, AP: {cand_ap_lgbm:.3f}")

    metrics_lgbm = star_level_metrics(dev_df, "proba_lgbm")
    print(f"  Star-level: F1={metrics_lgbm['f1']:.3f}, AP={metrics_lgbm['ap']:.3f}, "
          f"AUC={metrics_lgbm['auc']:.3f}, P@K={metrics_lgbm['p_at_k']:.3f}")
    print(f"  Detected: {metrics_lgbm['detected']}/{metrics_lgbm['n_pos']} positive stars")

    # ── Ensemble ─────────────────────────────────────────────────────
    print("\n--- Ensemble (0.5*XGB + 0.5*LGBM) ---")
    dev_df["proba_ensemble"] = 0.5 * dev_proba_xgb + 0.5 * dev_proba_lgbm

    cand_auc_ens = roc_auc_score(y_dev, dev_df.loc[dev_mask.values, "proba_ensemble"].values)
    cand_ap_ens = average_precision_score(y_dev, dev_df.loc[dev_mask.values, "proba_ensemble"].values)
    print(f"  Per-candidate AUC: {cand_auc_ens:.3f}, AP: {cand_ap_ens:.3f}")

    metrics_ens = star_level_metrics(dev_df, "proba_ensemble")
    print(f"  Star-level: F1={metrics_ens['f1']:.3f}, AP={metrics_ens['ap']:.3f}, "
          f"AUC={metrics_ens['auc']:.3f}, P@K={metrics_ens['p_at_k']:.3f}")
    print(f"  Detected: {metrics_ens['detected']}/{metrics_ens['n_pos']} positive stars")

    # ── Show per-star detection for ensemble ─────────────────────────
    print("\n--- Per-star detection (ensemble) ---")
    dev_star = dev_df.groupby("kepid").agg(
        best_proba=("proba_ensemble", "max"),
        best_period=("bls_period", lambda x: x.iloc[dev_df.loc[x.index, "proba_ensemble"].argmax()]),
        true_correct=("is_correct", "max"),
        label=("label", "first"),
        n_correct=("is_correct", lambda x: (x == 1).sum()),
    ).reset_index()

    # Stars with truth
    dev_truth_stars = dev_star[dev_star["true_correct"].notna()]
    detected = dev_truth_stars[dev_truth_stars["best_proba"] >= 0.5]
    missed = dev_truth_stars[(dev_truth_stars["true_correct"] == 1) & (dev_truth_stars["best_proba"] < 0.5)]

    print(f"  Detected: {len(detected)}/{len(dev_truth_stars)} truth stars")
    for _, row in detected.iterrows():
        print(f"    KepID {int(row['kepid'])}: P={row['best_period']:.3f}d, proba={row['best_proba']:.3f}, "
              f"correct={'YES' if row['true_correct']==1 else 'NO'}")

    if len(missed) > 0:
        print(f"  Missed (has_planet but proba < 0.5):")
        for _, row in missed.iterrows():
            print(f"    KepID {int(row['kepid'])}: P={row['best_period']:.3f}d, proba={row['best_proba']:.3f}, "
                  f"correct={'YES' if row['true_correct']==1 else 'NO'}")

    # ── Non-truth stars: show distribution ───────────────────────────
    non_truth = dev_star[dev_star["true_correct"].isna()]
    hp_non_truth = non_truth[non_truth["label"] == 1]
    print(f"\n  Non-truth stars with has_planet=1: {len(hp_non_truth)}")
    if len(hp_non_truth) > 0:
        print(f"  Confidence distribution:")
        for _, row in hp_non_truth.sort_values("best_proba", ascending=False).iterrows():
            print(f"    KepID {int(row['kepid'])}: P={row['best_period']:.3f}d, proba={row['best_proba']:.3f}")

    # ── Feature importance ───────────────────────────────────────────
    print("\n--- XGBoost feature importance (top-10) ---")
    importances = pd.DataFrame({
        "feature": available_feats,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    for _, row in importances.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # Save
    dev_df.to_csv("outputs/dev_reranked_v4.csv", index=False)
    print(f"\nSaved outputs/dev_reranked_v4.csv")

    return dev_df, metrics_ens


if __name__ == "__main__":
    main()
