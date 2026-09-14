"""AstroBit configuration — paths, constants, hyperparameters."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "train_pack"
DEV_DIR = ROOT / "dev_pack"
TEST_DIR = ROOT / "private_pack"

TRAIN_PARQUETS = TRAIN_DIR / "train"
DEV_PARQUETS = DEV_DIR / "dev"
TEST_PARQUETS = TEST_DIR  # private stars live directly here

TRAIN_LABELS = TRAIN_DIR / "train_labels.csv"
TRAIN_TRUTH = TRAIN_DIR / "train_truth.csv"
DEV_LABELS = DEV_DIR / "dev_labels.csv"
DEV_TRUTH = DEV_DIR / "dev_truth.csv"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ── Light curve constants ──────────────────────────────────────────────
CADENCE_MINUTES = 29.4          # Kepler short-cadence equivalent
CADENCE_DAYS = CADENCE_MINUTES / (24 * 60)
KEPLER_QUARTERS = list(range(0, 18))  # Q0–Q17

# Quality bitmask: 0 = good cadence
# Common bad flags: 1 (attitude tweak), 2 (safe mode), 4 (coarse point),
#   8 (deflectors), 16 (isolated transient), 8192 (stray light)
# We mask anything with quality != 0
QUALITY_GOOD = 0

# ── Period search range ────────────────────────────────────────────────
PERIOD_MIN_DAYS = 3.0           # starter: 3.0 (need ≥3 transits at max period)
PERIOD_MAX_DAYS = 400.0

# Coarse-to-fine parameters (adopted from starter notebook)
N_LS_CANDIDATES = 6             # top coarse peaks to refine (for main period search)
N_CANDIDATE_PEAKS = 20          # expanded peaks for candidate reranking
BLS_COARSE_RESOLUTION = 1000    # sufficient resolution for period recovery
BLS_FINE_RESOLUTION = 100       # trials per refinement window (fast for features)
BLS_FINE_WINDOW_FRAC = 0.02     # ±2% around coarse peak for fine sweep
BLS_DURATIONS_DAYS = [0.05, 0.1, 0.2, 0.4, 0.8]  # trial durations in days
BLS_OBJECTIVE = "likelihood"    # more robust than default "transit"

# ── Preprocessing hyperparameters ──────────────────────────────────────
SIGMA_CLIP_THRESHOLD = 5.0
DETREND_METHOD = "rolling"       # "rolling" (robust) or "spline" (smooth)
DETREND_POLY_ORDER = 3
DETREND_SPLINE_SMOOTHING = 1e8  # smoothing factor for spline detrending
ROLLING_WINDOW_CADENCES = 2000  # ~40 days at 29.4-min cadence

# ── Classifier ─────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 3.0,    # ≈202/67 for class imbalance
    "random_state": 42,
    "n_jobs": -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 3.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ── TLS parameters ──────────────────────────────────────────────────────
TLS_ENABLED = True
TLS_TRANSIT_DEPTH_MIN = 1e-6     # 1 ppm
TLS_TRANSIT_DEPTH_MAX = 0.1      # 10%
TLS_RADIUS_MIN = 0.5             # minimum stellar radius for TLS
TLS_MASS_MIN = 0.5               # minimum stellar mass for TLS

# ── Multi-detrend windows ──────────────────────────────────────────────
DETREND_WINDOWS_DAYS = [0.25, 0.5, 1.0, 2.0, 3.0]

# ── Known Kepler systematic periods ────────────────────────────────────
KEPLER_SYSTEMATIC_PERIODS = [372.5, 186.25, 93.125, 32.0, 3.0]

# ── Feature list ───────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    # BLS features
    "bls_sde", "bls_period", "bls_depth_ppm", "bls_duration_hours",
    "bls_snr", "bls_n_transits",
    # TLS features (NEW)
    "tls_sde", "tls_period", "tls_depth_ppm", "tls_duration_hours",
    "tls_snr", "tls_n_transits",
    # BLS/TLS agreement (NEW)
    "period_agreement", "period_ratio", "sde_ratio",
    "depth_agreement", "duration_agreement",
    # Light curve statistics
    "lc_rms", "lc_skew", "lc_kurtosis", "lc_entropy",
    "lc_autocorr_lag1", "lc_autocorr_lag5",
    # Periodicity
    "ls_peak_power", "ls_second_peak_ratio",
    "ls_peak_period",
    # Stellar properties
    "kepmag", "teff", "logg", "radius",
    # Vetting (from starter notebook suggestions)
    "odd_even_depth_ratio", "secondary_eclipse_depth",
    "quarter_signal_fraction",
    # Transit shape
    "transit_ingress_sharpness", "transit_boxiness",
    # Systematic period distance (NEW) — targets 372.5d artifact
    "dist_to_systematic", "min_systematic_ratio",
    # Multi-detrend stability (NEW) — does period survive different detrenders?
    "detrend_stability_score", "detrend_period_std",
    # Multi-peak BLS features (NEW) — carry top 3 peaks
    "bls_peak2_sde", "bls_peak2_period",
    "bls_peak3_sde", "bls_peak3_period",
    "bls_peak_separation",
    # Transit event consistency (NEW)
    "depth_cv", "depth_mad_ratio", "timing_rms",
    "n_expected", "transit_coverage",
]

# ── Submission ─────────────────────────────────────────────────────────
SUBMISSION_COLUMNS = [
    "star_id", "prediction", "confidence",
    "period", "depth_ppm", "duration_hours",
]
