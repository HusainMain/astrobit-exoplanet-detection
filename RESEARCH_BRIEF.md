# AstroBit Exoplanet Detection — Research Brief

## Competition
ML competition to detect transiting exoplanets from raw Kepler light curves.
- 269 training stars, 89 dev stars, 87 private test stars
- Each star has ~65K cadences over ~4 years (29.4-min cadence)
- Training labels: binary (planet/no-planet) + injected synthetic signals on label=0 stars
- Scoring: Detection F1 + PR-AUC (confidence ranking) + Characterization (period/depth/duration accuracy)
- Submission: 87 rows with star_id, prediction, confidence, period, depth_ppm, duration_hours

## Current Pipeline
1. Quality mask (quality == 0) → per-quarter median normalize → 0.5-day rolling median detrend
2. BLS period search: 1000 log-spaced coarse periods (3-400d), top 3 peaks refined with 100 fine trials
3. Extract 24 features: BLS (SDE, period, depth, duration, SNR, n_transits), LC stats (RMS, skew, kurtosis, entropy, autocorr), LS periodicity, stellar props, vetting (odd-even, secondary eclipse, quarter signal fraction), transit shape
4. XGBoost classifier: OOF AP=0.770, F1=0.703

## The Problem
BLS with 0.5-day detrend creates a **spurious signal at ~350 days** in almost every star:
- 105/112 non-planet training stars have BLS SDE > 10
- Features derived from the BLS period are noise (wrong period → wrong fold → wrong vetting)
- Classifier can't separate real planets from false detections
- Characterization: only 34.5% correct periods on dev

## What Works
- Stars with high BLS SDE (>50) DO recover the correct period
- ML classifier confidence ranking is decent (top 10 predictions mostly correct)
- Deep signals (depth > 800 ppm) are detectable; shallow signals (100-250 ppm) are lost
- The 0.5-day detrend IS critical: with 1-day window, DEEP signal drops from rank 1/20000 to rank 3452/20000

## What Doesn't Work
- BLS periods near ~350 days are almost always spurious
- Odd-even depth ratio is all zeros (BLS finds wrong period → can't compute correctly)
- Folded SNR is near zero for all stars (folding at wrong period)
- RMS improvement is zero (transit model at wrong period doesn't help)

## Key Data Insight
All 88 injected signals in train are on label=0 stars (hidden synthetic transits). The 67 label=1 stars are real Kepler planets NOT in the injected truth table. So we have two types of positives:
1. Real Kepler planets (label=1): known to the classifier via labels
2. Injected synthetic signals (in truth.csv): synthetic transits on otherwise quiet stars

## What We Need
1. **Why does BLS create spurious ~350d signals?** Is it a detrend artifact, a BLS edge effect, or stellar variability?
2. **How do other Kepler pipelines handle this?** What detrend + BLS configurations are standard?
3. **What features actually separate real transits from spurious BLS peaks?** Literature on transit detection validation.
4. **Is there a better period search method than BLS?** Transit least squares, box-fitting, convolution approaches?
5. **How do competition winners typically approach this?** Feature engineering strategies for transit detection.
