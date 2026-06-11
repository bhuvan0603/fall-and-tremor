"""
Tremor Detection — Inference Module
====================================
Loads the trained XGBoost pipeline (tremor_model.joblib), accepts raw
velocity signals (from CSV upload), windows + extracts features, and
returns per-window predictions with an overall verdict.

Used by:
  - app/app.py   (Flask route /api/tremor-analyze)
  - webcam_fall_detector.py  (optional real-time integration)
"""

import numpy as np
import joblib
from pathlib import Path

from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew, entropy as sp_entropy

# ── Paths & constants ─────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "models" / "tremor_model.joblib"
SAMPLE_RATE  = 100   # Hz
WINDOW_SIZE  = 200   # samples (2 s)
STEP_SIZE    = 100   # samples (1 s, 50 % overlap)

# ── Singleton model cache ─────────────────────────────────────────────────────
_pipeline = None
_threshold = 0.5   # per-window probability threshold (loaded from model file)


def load_model(path=None):
    """Load the trained sklearn/XGBoost pipeline. Returns True on success."""
    global _pipeline, _threshold
    p = Path(path) if path else MODEL_PATH
    if not p.exists():
        print(f"[TREMOR] Model not found: {p}")
        return False
    obj = joblib.load(str(p))
    # Support both old (bare pipeline) and new (dict with threshold) formats
    if isinstance(obj, dict) and "pipeline" in obj:
        _pipeline = obj["pipeline"]
        _threshold = obj.get("threshold", 0.5)
    else:
        _pipeline = obj
        _threshold = 0.5
    print(f"[TREMOR] Model loaded from {p}  (threshold={_threshold})")
    return True


def is_loaded() -> bool:
    return _pipeline is not None


# =============================================================================
#  Feature extraction  (mirrors tremor_train_rf.py exactly)
# =============================================================================

def _bandpass(signal, low, high, fs=SAMPLE_RATE, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    if len(signal) <= 3 * max(len(a), len(b)):
        return np.zeros_like(signal)
    return filtfilt(b, a, signal)


def _zcr(x):
    return float(np.mean(np.abs(np.diff(np.sign(x))) > 0))


def _energy(x):
    return float(np.mean(x ** 2))


def _extract_features(window):
    """Extract 37 features from a single 1-D window (must match training)."""
    x = window.flatten().astype(np.float64)
    n = len(x)

    f = {}

    # -- Time-domain --
    f["mean"]          = np.mean(x)
    f["std"]           = np.std(x)
    f["var"]           = np.var(x)
    f["rms"]           = np.sqrt(np.mean(x ** 2))
    f["max"]           = np.max(x)
    f["min"]           = np.min(x)
    f["peak2peak"]     = np.ptp(x)
    f["skewness"]      = float(skew(x))
    f["kurtosis"]      = float(kurtosis(x))
    f["zcr"]           = _zcr(x)
    f["energy"]        = _energy(x)
    f["abs_mean"]      = np.mean(np.abs(x))
    f["q25"]           = np.percentile(x, 25)
    f["q75"]           = np.percentile(x, 75)
    f["iqr"]           = f["q75"] - f["q25"]
    f["median"]        = np.median(x)

    diff1 = np.diff(x)
    f["mean_abs_diff"] = np.mean(np.abs(diff1))
    f["std_diff"]      = np.std(diff1)

    # -- Frequency-domain (Welch PSD) --
    freqs, psd = welch(x, fs=SAMPLE_RATE, nperseg=min(128, n),
                       noverlap=min(64, n // 2))
    total_power = np.sum(psd) + 1e-12

    f["dom_freq"]          = freqs[np.argmax(psd)]
    f["dom_freq_power"]    = np.max(psd)
    f["spectral_centroid"] = np.sum(freqs * psd) / total_power

    psd_norm = psd / total_power
    psd_norm = psd_norm[psd_norm > 0]
    f["spectral_entropy"]  = float(sp_entropy(psd_norm))

    def _bp(lo, hi):
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        return float(np.sum(psd[idx])) if len(idx) else 0.0

    f["power_1_3hz"]   = _bp(1, 3)
    f["power_3_6hz"]   = _bp(3, 6)
    f["power_4_6hz"]   = _bp(4, 6)
    f["power_6_12hz"]  = _bp(6, 12)
    f["power_12_25hz"] = _bp(12, 25)
    f["power_0_50hz"]  = total_power

    f["ratio_tremor_total"] = f["power_3_6hz"]  / total_power
    f["ratio_4_6_total"]    = f["power_4_6hz"]  / total_power
    f["ratio_6_12_total"]   = f["power_6_12hz"] / total_power
    f["ratio_tremor_high"]  = f["power_3_6hz"]  / (f["power_12_25hz"] + 1e-12)

    bp46  = _bandpass(x, 4.0, 6.0)
    bp612 = _bandpass(x, 6.0, 12.0)
    f["bp_4_6_rms"]            = np.sqrt(np.mean(bp46 ** 2))
    f["bp_4_6_energy"]         = _energy(bp46)
    f["bp_6_12_rms"]           = np.sqrt(np.mean(bp612 ** 2))
    f["bp_6_12_energy"]        = _energy(bp612)
    f["bp_ratio_4_6_vs_6_12"] = f["bp_4_6_energy"] / (f["bp_6_12_energy"] + 1e-12)

    return f


# =============================================================================
#  Public API
# =============================================================================

def analyze_signal(velocity: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """
    Analyze a raw velocity signal for tremor.

    Parameters
    ----------
    velocity : 1-D numpy array of velocity values (any length).
    sample_rate : int, Hz (default 100).

    Returns
    -------
    dict with keys:
        tremor_detected : bool   — overall verdict
        confidence      : float  — mean tremor probability across windows
        num_windows     : int
        window_results  : list of dicts per window
            { window_idx, start_s, end_s, probability, label }
        signal_length   : int    — total samples
        duration_s      : float  — total seconds
    """
    if _pipeline is None:
        raise RuntimeError("Tremor model not loaded. Call load_model() first.")

    vel = np.asarray(velocity, dtype=np.float64).flatten()
    n = len(vel)

    if n < WINDOW_SIZE:
        # Pad short signals
        vel = np.pad(vel, (0, WINDOW_SIZE - n), mode="constant")
        n = len(vel)

    # -- Sliding windows + z-score per window --
    starts = list(range(0, n - WINDOW_SIZE + 1, STEP_SIZE))
    if not starts:
        starts = [0]

    windows = []
    for s in starts:
        w = vel[s : s + WINDOW_SIZE].copy()
        std = w.std()
        if std > 1e-8:
            w = (w - w.mean()) / std
        else:
            w = w - w.mean()
        windows.append(w)

    # -- Feature extraction --
    feat_dicts = [_extract_features(w) for w in windows]
    feat_names = list(feat_dicts[0].keys())
    X = np.array([[d[k] for k in feat_names] for d in feat_dicts], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    # -- Predict --
    probs = _pipeline.predict_proba(X)[:, 1]   # P(tremor)

    # -- Per-window results --
    window_results = []
    for i, (s, p) in enumerate(zip(starts, probs)):
        window_results.append({
            "window_idx":  i,
            "start_s":     round(s / sample_rate, 2),
            "end_s":       round((s + WINDOW_SIZE) / sample_rate, 2),
            "probability": round(float(p), 4),
            "label":       "Tremor" if p >= _threshold else "No Tremor",
        })

    mean_prob  = float(np.mean(probs))
    tremor_pct = float(np.mean(probs >= _threshold))

    return {
        "tremor_detected": bool(tremor_pct >= 0.5),
        "confidence":      round(mean_prob, 4),
        "tremor_window_pct": round(tremor_pct * 100, 1),
        "num_windows":     len(window_results),
        "window_results":  window_results,
        "signal_length":   int(len(velocity)),
        "duration_s":      round(len(velocity) / sample_rate, 2),
    }
