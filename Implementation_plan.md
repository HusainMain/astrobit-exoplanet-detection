# AstroBit — Exoplanet Detection Pipeline

## Implementation Plan & Status Tracker

**Overall Progress: 28%**

---

## Phase 1: Data Loading & Exploration [100%] ✅

- [x] 1a. Create project structure (`src/`, `models/`, `outputs/`, `configs/`)
- [x] 1b. Create `config.py` with paths, constants, hyperparameters
- [x] 1c. Build `data_loader.py` — unified loader for parquets, labels, truth
- [x] 1d. Build `visualize.py` — plot raw light curves across all difficulty bins
- [x] 1e. Sanity checks: verify all kepid ↔ parquet filename mappings match

> All checks passed. 269 train + 89 dev parquets load correctly.
> Generated 9 sample visualizations in `outputs/phase1_viz/`.

---

## Phase 2: Light Curve Preprocessing [100%] ✅

- [x] 2a. Quality bitmask filtering — mask cadences where `quality != 0`
- [x] 2b. Per-quarter sigma-clipping — remove >5σ outliers from rolling median
- [x] 2c. Per-quarter detrending — rolling median (robust) or spline (smooth), divide by trend
- [x] 2d. Quarter stitching — quarters processed independently, stitched together
- [x] 2e. Flux normalization — divide-by-trend gives baseline ~1.0 with transit dips below
- [x] 2f. Validation — verified injected transits survive preprocessing (depth within 7-50% across all bins)

> Rolling median (window=2000 cadences ~40 days) is default — robust and fast.
> Spline (smoothing=1e8) available as alternative for smooth light curves.
> Key insight: divide-by-trend, not subtract, preserves transit depth correctly.

---

## Phase 3: Period Search Engine [50%] 🔄

> **Critical discovery**: BLS periodogram is dominated by long-period stellar
> variability. The detrend window MUST be ≤0.5 days for BLS to find transits.
> Starter notebook's 1-day window is too long — true period ranks 3452/20000.
> With 0.5-day window: true period ranks 1/20000 (SDE=56) for DEEP signal.

- [x] 3a. BLS with 0.5-day cleaning — coarse-to-fine, log-spaced grid, `objective="likelihood"`
- [x] 3b. Coarse sweep (20k periods) → fine refinement (±2%, 600 trials) around top 8 peaks
- [x] 3c. MAD-based SDE computation
- [x] 3d. Validate on training set — **1/4 exact** (DEEP: 0.00% error)
- [ ] 3e. Improve long-period detection (MID/SHALLOW/EARTH_ANALOG still fail)
- [ ] 3f. Benchmark: 22.3s/star, est 32 min for 87 stars — within budget

> Results: DEEP=PASS (SDE=56), MID=FAIL (2× alias, 4.34%), SHALLOW=FAIL, EARTH_ANALOG=FAIL
> Remaining: long-period signals need additional detectors or classifier rescue

---

## Phase 4: Feature Engineering [0%]

> Vetting features inspired by starter notebook: odd-even depth, secondary eclipse,
> quarter consistency — each false positive removed = precision gained.

- [ ] 4a. BLS features: MAD-based SDE, period, depth, duration, SNR, n_transits
- [ ] 4b. Light curve statistics: RMS, skewness, kurtosis, entropy, autocorrelation
- [ ] 4c. Periodicity features: LS peak power, second peak ratio, autocorrelation peaks
- [ ] 4d. Stellar properties: kepmag, teff, logg, radius (normalized)
- [ ] 4e. Vetting: odd-even depth ratio, secondary eclipse presence, quarter-by-quarter signal consistency
- [ ] 4f. Transit shape: ingress/egress sharpness, boxiness score
- [ ] 4g. Build feature matrix: one row per star, ~30 features total

---

## Phase 5: Classifier Training [0%]

- [ ] 5a. Define train/dev split strategy (all 269 train → classify, validate on 89 dev)
- [ ] 5b. Handle class imbalance: `scale_pos_weight` = 202/67 ≈ 3
- [ ] 5c. Train XGBoost classifier on combined feature set
- [ ] 5d. Train LightGBM classifier as second model
- [ ] 5e. Probability calibration — Platt scaling / isotonic regression on dev set
- [ ] 5f. Threshold optimization — maximize F1 on dev set
- [ ] 5g. Feature importance analysis and selection
- [ ] 5h. Save trained models

---

## Phase 6: Ensemble & Confidence Ranking [0%]

- [ ] 6a. Combine BLS detection + classifier confidence into final prediction logic
- [ ] 6b. Implement confidence ranking for PR-AUC optimization
- [ ] 6c. Define fallback logic for stars with weak/no BLS signal
- [ ] 6d. Ablation study — measure contribution of each component on dev set

---

## Phase 7: Dev Set Evaluation & Tuning [0%]

- [ ] 7a. Full evaluation: precision, recall, F1, PR-AUC, average precision
- [ ] 7b. Difficulty breakdown: per-bin scores (earth_analog, shallow, mid, deep)
- [ ] 7c. Characterization accuracy: period error (within 2%), depth error
- [ ] 7d. Error analysis: missed detections, false positives, systematic failures
- [ ] 7e. Hyperparameter tuning based on dev performance
- [ ] 7f. Confidence distribution review: ensure meaningful spread (not flat)

---

## Phase 8: Test Set Prediction & Submission [0%]

> Submission validation from starter notebook: 87 rows, correct columns,
> star_id format, prediction 0/1, confidence 0-1, all detections have params.

- [ ] 8a. Wait for `private_pack.zip` release (hour 40)
- [ ] 8b. Run full pipeline on 87 private stars
- [ ] 8c. Generate `submission_<teamname>.csv` in required format
- [ ] 8d. Validate: 87 rows, STAR_XXXX format, no missing detection params
- [ ] 8e. Confidence distribution review: meaningful spread (not flat)
- [ ] 8f. Final sanity checks: period/depth/duration reasonable ranges

---

## Phase 9: Documentation & Reproducibility [0%]

- [ ] 9a. Write `README.md` with methodology description
- [ ] 9b. Create `requirements.txt` with pinned dependencies
- [ ] 9c. Document entry point and run instructions
- [ ] 9d. Ensure `submission_<teamname>.csv` is reproducible from scratch

---

## Key Metrics to Beat (Dev Set)

| Metric | Baseline | Target |
|---|---|---|
| F1 (detection) | — | >0.85 |
| PR-AUC | — | >0.90 |
| Period accuracy | — | >90% within 2% |
| Confidence spread | — | min <0.1, max >0.9, unique >20 |

---

## File Structure (Target)

```
AstroBit/
├── Implementation_plan.md
├── requirements.txt
├── README.md
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── period_search.py
│   ├── features.py
│   ├── classifier.py
│   ├── ensemble.py
│   ├── evaluate.py
│   └── submit.py
├── models/
│   └── (saved models)
├── outputs/
│   └── submission_<teamname>.csv
├── train_pack/
│   ├── train_labels.csv
│   ├── train_truth.csv
│   └── train/*.parquet
├── dev_pack/
│   ├── dev_labels.csv
│   ├── dev_truth.csv
│   └── dev/*.parquet
└── private_pack/  (when released)
```
