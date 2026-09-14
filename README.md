# AstroBit — Kepler Exoplanet Detection Pipeline

Detecting transiting exoplanets from Kepler light curves and characterizing their orbital periods, depths, and durations.

## Competition Overview

The AstroBit ML competition asks us to:

1. **Detect** which of 446 Kepler stars host transiting exoplanets
2. **Characterize** each detection's orbital period, transit depth, and transit duration
3. Score on **Detection F1 + PR-AUC** combined with **characterization accuracy** (period/depth/duration within tolerances)

**Data:**
- 269 train stars (30 with known planets)
- 89 dev stars (30 with known planets)
- 87 private test stars (hidden)
- Light curves: preprocessed parquet files with time, flux, quarter, and quality columns

**Difficulty bins** (from `train_truth.csv`):
- `earth_analog` — shallow, long-period signals (hardest)
- `shallow` — small depth
- `mid` — moderate depth
- `deep` — large depth (easiest)

---

## Approaches Tried (Chronological)

### 1. Starter Notebook Baseline

Started from the competition's starter notebook. Basic BLS period search with `wotan` detrending and `astropy.timeseries.BoxLeastSquares`. Found the right period for deep signals but completely failed on mid/shallow/earth_analog bins.

**Problem discovered:** BLS period search was dominated by a ~372-day Kepler systematic artifact. True periods would rank 3000th+ out of 20,000 candidates.

### 2. Detrend Window Fix

**Critical finding:** The detrend window controls everything. A 1-day rolling median window (default) was too long — it smoothed out the transits themselves. Reducing to 0.5 days made BLS actually find real periods.

- 1-day window: true period ranks 3452/20000
- 0.5-day window: true period ranks 1/20000 (SDE=56 for deep signal)

This single change was the biggest improvement in the entire project.

### 3. Feature Engineering + Star-Level Classifier

Built ~55 star-level features (BLS SDE, period, depth, duration, SNR, transit consistency metrics, stellar properties, light curve statistics) and trained XGBoost + LightGBM classifiers to predict which stars host planets.

**Result:** Detection AP ~0.80. Good but not enough for characterization.

### 4. Candidate-Level Reranker

Realized that star-level detection wasn't enough — we needed to pick the *right period* from BLS's top-20 candidates per star.

Built a **reranker** (XGB + LGBM ensemble) that scores each of the 20 candidates per star and picks the best one. Used 27 candidate-level features:
- BLS metrics (period, depth, duration, SNR, SDE)
- Systematic distance (how close to the 372d artifact)
- Transit shape features (odd-even ratio, secondary eclipse, quarter consistency)
- Timing/depth consistency across transits

**Result:** 13/30 correct periods on dev = 43.3%

### 5. Multi-Window Consensus Clustering

Tried running BLS across 5 different detrend windows (0.25d, 0.5d, 1.0d, 2.0d, 3.0d) and clustering peaks that appeared across windows. Idea: if a period shows up in multiple detrend windows, it's more likely real.

**Result:** Hurt both recovery AND detection. The aggressive filtering removed real signals. Abandoned.

### 6. Synthetic Injection Augmentation

The fundamental problem was data: only 30 positive train stars with truth. Built a synthetic injection pipeline:
1. Pick quiet host stars (no known planets)
2. Inject synthetic box transits with parameters sampled from the truth distribution
3. Re-run BLS to get candidate lists with known ground truth
4. Train the reranker on real + synthetic data

**132 injections** (102 from truth distribution + 30 near-systematic aliases + some extras). Each injection: ~4 seconds (raw flux → inject → detrend → BLS → features).

**Result:** 16/30 correct periods on dev = 53.3% (+3 over original reranker)

Key augmentation details:
- Injected periods sampled from truth distribution with ±20% noise
- 30 near-systematic injections (periods near 372d, 186d, 93d) to teach the model these are bad
- Balanced: earth_analog 30%, shallow 25%, mid 25%, deep 20%
- Final: 5,280 augmented candidates added to 5,380 real candidates = 10,660 training rows

### 7. Targeted Injection Augmentation (v2)

Tried to be smarter: only inject periods that would appear near the 372d/186d/93d systematic peaks, to teach the model to disambiguate real from aliased signals.

111 targeted injections matching the exact failure mode of the v1 ranker.

**Result:** No improvement. Still 16/30. The targeted injections didn't add new information the model didn't already have.

### 8. LGBMRanker Approach

Switched from binary classification (XGB + LGBM ensemble) to **LGBMRanker** — a learning-to-rank model that directly optimizes NDCG on the candidate list per star.

- Uses the same 27 candidate features
- Trained with group-aware labels (is_correct = 1 for the true period candidate)
- NDCG@5 optimization

**Result:** 16/30 — same as the augmented binary classifier, but simpler architecture (single model, no ensemble)

---

## Final Pipeline

The final submission (`submit_v10_ranker.py`) combines:

### Detection (Star-Level)
- **v7 detector** (`train_detector_v7.py`): LightGBM classifier on star-level features
- Predicts probability of planet presence per star
- Threshold 0.23: Precision=0.647, Recall=0.863, F1=0.739
- Detection AP: 0.7996

### Period Selection (Candidate-Level)
- **Augmented LGBMRanker**: scores each of BLS's top-20 candidates per star
- Trained on real data + 132 synthetic injections
- Picks the highest-ranked candidate as the final period/depth/duration

### Feature Set (27 features)
```
bls_period, bls_depth_ppm, bls_duration_hours, bls_n_transits, bls_snr, bls_sde,
dist_to_systematic, systematic_fraction, near_systematic, transit_coverage, n_expected,
depth_cv, depth_mad_ratio, timing_rms, odd_even_depth_ratio, secondary_eclipse_depth,
quarter_signal_fraction, in_out_scatter_ratio, box_fit_snr, box_fit_residual_rms,
depth_trend_slope, duration_consistency, baseline_rms, kepmag, teff, logg, radius
```

### Period Recovery
- **Rank-0 (highest SDE):** 9/30 (30%) — just pick the strongest BLS peak
- **Original reranker:** 13/30 (43.3%) — XGB+LGBM ensemble without augmentation
- **Augmented LGBMRanker:** 16/30 (53.3%) — final best

---

## Results

### Dev Set (30 stars with truth)

| Metric | Rank-0 | Orig Reranker | Augmented Ranker |
|---|---|---|---|
| Period recovery | 9/30 (30%) | 13/30 (43%) | **16/30 (53%)** |

### Per-Bin Breakdown (Augmented Ranker)
- **earth_analog:** 3/7
- **shallow:** 4/8
- **mid:** 3/8
- **deep:** 6/7

### Detection
- AP: 0.7996
- Precision@0.23: 0.647
- Recall@0.23: 0.863
- F1@0.23: 0.739

### Missed Stars Analysis

Of the 14 missed dev stars:
- **4 stars:** Correct period not in BLS top-20 at all (fundamental BLS limitation)
- **10 stars:** Correct period present in top-20 but misranked (systematic alias confusion — 8/10 selected period within 20d of 372d/186d/93d)

---

## Key Insights

1. **Detrend window is everything.** The 0.5-day rolling median window was the single biggest factor in making BLS work. Too long smooths transits; too short leaves noise.

2. **The 372-day Kepler systematic is the #1 enemy.** It dominates BLS periodograms, creates aliases at 186d and 93d, and confuses any model that doesn't explicitly account for it.

3. **More data beats more features.** The injection augmentation (+3 correct periods) was worth more than all the feature engineering experiments combined.

4. **The oracle is 30/30.** Every correct period exists somewhere in the top-20 candidates. The problem is ranking, not detection. This means the ceiling for period recovery is 100% — we just need a better ranker.

5. **Ensembles didn't help.** XGB + LGBM ensemble gave the same result as LGBM alone. The models are too correlated.

6. **Multi-window consensus hurt.** Filtering candidates by cross-window agreement removed real signals. The noise from aggressive filtering outweighed the benefit.

7. **Targeted augmentation didn't help either.** Teaching the model about systematic aliases directly didn't add information beyond what random augmentation already provided.

---

## Project Structure

```
AstroBit/
├── src/
│   ├── config.py              # Paths, constants, hyperparameters
│   ├── data_loader.py         # Load parquets, labels, truth CSVs
│   ├── features.py            # Star-level feature extraction (~55 features)
│   ├── period_search.py       # BLS search, wotan detrending, clean_for_bls()
│   ├── preprocess.py          # Quality masking, sigma clipping, detrending
│   └── visualize.py           # Light curve visualization
├── build_candidates.py        # extract_candidate_features(), bls_search_multi()
├── build_enhanced_aggregate.py # Aggregate features for star-level detector
├── consensus_candidates.py    # Multi-window consensus (experimental, abandoned)
├── inject_augment_reranker.py # Synthetic injection augmentation (132 injections)
├── inject_targeted_augment.py # Targeted augmentation (111 injections, no improvement)
├── train_reranker_v3.py       # Original reranker training (XGB+LGBM ensemble)
├── train_detector_v7.py       # Star-level detector training
├── train_detector_v9.py       # Updated detector (experimental)
├── submit_v7.py               # Detector-only submission
├── submit_v8.py               # Detector + original reranker
├── submit_v10.py              # Detector + augmented reranker
├── submit_v10_ranker.py       # Final submission (LGBMRanker + v7 detector)
├── eval_clean_comparison.py   # Clean dev comparison (all methods)
├── starter_notebook.ipynb     # Competition starter notebook
├── models/
│   └── augmented_lgbm_ranker.pkl  # Trained ranker (model + metadata)
├── outputs/                   # Generated CSVs and visualizations
├── train_pack/                # Training data (269 stars)
├── dev_pack/                  # Dev data (89 stars)
└── private_pack/              # Private test data (87 stars)
```

## Running the Pipeline

```bash
# 1. Extract candidates from BLS
python build_candidates.py

# 2. Train the star-level detector
python train_detector_v7.py

# 3. Train the augmented reranker
python inject_augment_reranker.py

# 4. Generate submission
python submit_v10_ranker.py
```

## Dependencies

- Python 3.8+
- numpy, pandas, scipy
- astropy (BLS periodogram)
- wotan (light curve detrending)
- lightgbm, xgboost (models)
- scikit-learn (metrics, preprocessing)
- matplotlib, seaborn (visualization)

## Submission Format

```csv
star_id,prediction,confidence,period,depth_ppm,duration_hours
KIC_0012345678,1,0.85,12.34,150.0,4.2
KIC_0012345679,0,0.15,0.0,0.0,0.0
```

---

## What Didn't Work (Lessons Learned)

| Approach | Result | Why It Failed |
|---|---|---|
| Multi-window consensus | Hurt recovery & detection | Aggressive filtering removed real signals |
| Anchor period feature | No improvement | Not discriminative enough |
| Extra features (55 total) | Marginal | Diminishing returns beyond ~20 features |
| XGB + LGBM ensemble | Same as LGBM alone | Too correlated |
| Test-time augmentation | No improvement | Noise outweighed signal |
| Targeted injection augmentation | No improvement | Didn't add new information |
| TLS (Transit Least Squares) | Not used | Couldn't install in time |

## What Worked

| Approach | Improvement | Why It Helped |
|---|---|---|
| 0.5-day detrend window | Massive | Unmasked real transits from BLS |
| Candidate-level reranker | +4/30 | Picked best from top-20 instead of rank-0 |
| Synthetic injection augmentation | +3/30 | More training data for the ranker |
| Systematic distance feature | Implicit | Taught model to avoid 372d artifact |
| 27 candidate features | Implicit | Transit shape, timing, depth consistency |
