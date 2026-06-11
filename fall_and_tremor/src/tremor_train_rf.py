"""
Phase 2b - Tremor Detection: Feature-Engineered ML Pipeline
============================================================
Replaces CNN-LSTM with hand-crafted features + Random Forest / XGBoost.

Loads windowed data from Phase 1, extracts ~25 statistical + frequency
features per window, trains both RF and XGBoost with 5-fold stratified
cross-validation, picks the best model, and saves it.

Input  : data/tremor/X_train.npy   shape (N, 200, 1)  z-scored velocity
         data/tremor/y_train.npy   shape (N,)
Output : models/tremor_model.joblib
         data/tremor/training_history.json   (CV results)
         data/tremor/model_metadata.json
"""

import json
import sys
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew, entropy as sp_entropy

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ── Try to import XGBoost (optional) ─────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] xgboost not installed - will skip XGBoost model")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_DIR    = BASE_DIR / "models"
DATA_DIR     = BASE_DIR / "data" / "tremor"
MODEL_PATH   = MODEL_DIR / "tremor_model.joblib"
HISTORY_OUT  = DATA_DIR / "training_history.json"
METADATA_OUT = DATA_DIR / "model_metadata.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 200
SAMPLE_RATE  = 100   # Hz
RANDOM_SEED  = 42
N_FOLDS      = 5     # stratified CV folds


# =============================================================================
#  FEATURE EXTRACTION
# =============================================================================

def bandpass_filter(signal, low, high, fs=SAMPLE_RATE, order=4):
    """Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    if len(signal) <= 3 * max(len(a), len(b)):
        return np.zeros_like(signal)
    return filtfilt(b, a, signal)


def zero_crossing_rate(x):
    """Fraction of consecutive pairs with a sign change."""
    return np.mean(np.abs(np.diff(np.sign(x))) > 0)


def signal_energy(x):
    """Mean squared amplitude."""
    return np.mean(x ** 2)


def extract_features_single(window):
    """
    Extract hand-crafted features from a single 1-D velocity window.
    Returns a 1-D feature vector (dict -> values).
    """
    x = window.flatten().astype(np.float64)
    n = len(x)
    dt = 1.0 / SAMPLE_RATE

    feats = {}

    # ── 1. Time-domain statistics ──────────────────────────────────────────
    feats["mean"]       = np.mean(x)
    feats["std"]        = np.std(x)
    feats["var"]        = np.var(x)
    feats["rms"]        = np.sqrt(np.mean(x ** 2))
    feats["max"]        = np.max(x)
    feats["min"]        = np.min(x)
    feats["peak2peak"]  = np.ptp(x)
    feats["skewness"]   = float(skew(x))
    feats["kurtosis"]   = float(kurtosis(x))
    feats["zcr"]        = zero_crossing_rate(x)
    feats["energy"]     = signal_energy(x)
    feats["abs_mean"]   = np.mean(np.abs(x))

    # Percentiles
    feats["q25"]        = np.percentile(x, 25)
    feats["q75"]        = np.percentile(x, 75)
    feats["iqr"]        = feats["q75"] - feats["q25"]
    feats["median"]     = np.median(x)

    # Signal dynamics
    diff1 = np.diff(x)
    feats["mean_abs_diff"]  = np.mean(np.abs(diff1))
    feats["std_diff"]       = np.std(diff1)

    # ── 2. Frequency-domain features (Welch PSD) ──────────────────────────
    freqs, psd = welch(x, fs=SAMPLE_RATE, nperseg=min(128, n),
                       noverlap=min(64, n // 2))
    total_power = np.sum(psd) + 1e-12

    # Dominant frequency
    feats["dom_freq"]       = freqs[np.argmax(psd)]
    feats["dom_freq_power"] = np.max(psd)

    # Spectral centroid
    feats["spectral_centroid"] = np.sum(freqs * psd) / total_power

    # Spectral entropy
    psd_norm = psd / total_power
    psd_norm = psd_norm[psd_norm > 0]
    feats["spectral_entropy"] = float(sp_entropy(psd_norm))

    # Band powers (critical Parkinson's tremor bands)
    def band_power(f_low, f_high):
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        return np.sum(psd[idx]) if len(idx) > 0 else 0.0

    feats["power_1_3hz"]   = band_power(1, 3)    # sub-tremor
    feats["power_3_6hz"]   = band_power(3, 6)    # resting tremor
    feats["power_4_6hz"]   = band_power(4, 6)    # narrower PD tremor
    feats["power_6_12hz"]  = band_power(6, 12)   # essential/action tremor
    feats["power_12_25hz"] = band_power(12, 25)  # higher-freq noise
    feats["power_0_50hz"]  = total_power         # total (already computed)

    # Tremor band ratios (key discriminators)
    feats["ratio_tremor_total"]  = feats["power_3_6hz"]  / total_power
    feats["ratio_4_6_total"]     = feats["power_4_6hz"]  / total_power
    feats["ratio_6_12_total"]    = feats["power_6_12hz"] / total_power
    feats["ratio_tremor_high"]   = feats["power_3_6hz"]  / (feats["power_12_25hz"] + 1e-12)

    # ── 3. Bandpass-filtered signal stats ─────────────────────────────────
    bp_4_6  = bandpass_filter(x, 4.0, 6.0)
    bp_6_12 = bandpass_filter(x, 6.0, 12.0)

    feats["bp_4_6_rms"]    = np.sqrt(np.mean(bp_4_6 ** 2))
    feats["bp_4_6_energy"] = signal_energy(bp_4_6)
    feats["bp_6_12_rms"]   = np.sqrt(np.mean(bp_6_12 ** 2))
    feats["bp_6_12_energy"] = signal_energy(bp_6_12)
    feats["bp_ratio_4_6_vs_6_12"] = (feats["bp_4_6_energy"] /
                                      (feats["bp_6_12_energy"] + 1e-12))

    return feats


def extract_features_all(X):
    """
    Extract features from all windows.
    X: (N, 200, 1) -> returns (N, n_features) numpy array + feature names.
    """
    feature_dicts = []
    N = X.shape[0]
    for i in range(N):
        feats = extract_features_single(X[i])
        feature_dicts.append(feats)
        if (i + 1) % 500 == 0 or i == N - 1:
            print(f"  Extracted features: {i+1}/{N}")

    feature_names = list(feature_dicts[0].keys())
    X_feat = np.array([[d[k] for k in feature_names] for d in feature_dicts],
                      dtype=np.float64)

    # Replace NaN/Inf
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1e10, neginf=-1e10)

    return X_feat, feature_names


# =============================================================================
#  MODEL DEFINITIONS
# =============================================================================

def get_models(y=None):
    """Return dict of {name: sklearn Pipeline} to evaluate."""
    models = {}

    # Compute class weight ratio for boosting models
    if y is not None:
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        spw = n_neg / max(n_pos, 1)  # < 1 when tremor is majority
    else:
        spw = 1.0

    # Random Forest
    models["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    # Gradient Boosting (sklearn)
    models["GradientBoosting"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
        )),
    ])

    # XGBoost (if installed)
    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                random_state=RANDOM_SEED,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1,
            )),
        ])

    return models


# =============================================================================
#  TRAINING
# =============================================================================

def cross_validate_models(X_feat, y, models):
    """
    Run stratified K-fold CV on each model.
    Returns dict of {name: {metric: mean, ...}}.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = {}
    for name, pipeline in models.items():
        print(f"\n[CV] {name} ({N_FOLDS}-fold)...")
        cv_results = cross_validate(
            pipeline, X_feat, y, cv=cv, scoring=scoring,
            return_train_score=True, n_jobs=-1
        )
        res = {}
        for metric in scoring:
            test_key  = f"test_{metric}"
            train_key = f"train_{metric}"
            res[f"val_{metric}_mean"]  = float(np.mean(cv_results[test_key]))
            res[f"val_{metric}_std"]   = float(np.std(cv_results[test_key]))
            res[f"train_{metric}_mean"] = float(np.mean(cv_results[train_key]))

        results[name] = res
        print(f"  val_accuracy : {res['val_accuracy_mean']:.4f} +/- {res['val_accuracy_std']:.4f}")
        print(f"  val_precision: {res['val_precision_mean']:.4f} +/- {res['val_precision_std']:.4f}")
        print(f"  val_recall   : {res['val_recall_mean']:.4f} +/- {res['val_recall_std']:.4f}")
        print(f"  val_f1       : {res['val_f1_mean']:.4f} +/- {res['val_f1_std']:.4f}")
        print(f"  val_roc_auc  : {res['val_roc_auc_mean']:.4f} +/- {res['val_roc_auc_std']:.4f}")

    return results


def find_optimal_threshold(X_feat, y, models, best_name):
    """Use cross-validation to find the per-window probability threshold
    that maximizes the balanced accuracy (avg of TPR and TNR)."""
    from sklearn.metrics import balanced_accuracy_score
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    pipeline = models[best_name]

    oof_probs = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X_feat, y):
        pipeline.fit(X_feat[train_idx], y[train_idx])
        oof_probs[val_idx] = pipeline.predict_proba(X_feat[val_idx])[:, 1]

    best_thresh = 0.5
    best_bal_acc = 0.0
    for t in np.arange(0.30, 0.85, 0.01):
        preds = (oof_probs >= t).astype(int)
        bal_acc = balanced_accuracy_score(y, preds)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = round(float(t), 2)

    print(f"[THRESHOLD] Optimal per-window threshold: {best_thresh}")
    print(f"            Balanced accuracy at threshold: {best_bal_acc:.4f}")
    return best_thresh


def train_final_model(X_feat, y, models, cv_results):
    """Train the best model (by AUC) on the FULL dataset and return it."""
    # Pick best model by val_roc_auc_mean
    best_name = max(cv_results, key=lambda n: cv_results[n]["val_roc_auc_mean"])
    best_auc  = cv_results[best_name]["val_roc_auc_mean"]
    print(f"\n[BEST] {best_name}  (val AUC = {best_auc:.4f})")

    # Find optimal per-window threshold using OOF probabilities
    optimal_threshold = find_optimal_threshold(X_feat, y, models, best_name)

    pipeline = models[best_name]

    # Final fit on ALL data
    print(f"[FIT] Training {best_name} on full dataset ({len(y):,} samples)...")
    pipeline.fit(X_feat, y)

    # Final predictions (on train set - just for sanity check)
    y_prob = pipeline.predict_proba(X_feat)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)

    print(f"\n[SANITY CHECK - Full dataset metrics @ threshold={optimal_threshold}]")
    print(f"  Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"  Precision: {precision_score(y, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y, y_pred):.4f}")
    print(f"  F1       : {f1_score(y, y_pred):.4f}")
    print(f"  AUC      : {roc_auc_score(y, y_prob):.4f}")
    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion matrix:\n{cm}")

    return pipeline, best_name, optimal_threshold


def get_feature_importance(pipeline, feature_names, best_name):
    """Extract and display top feature importances."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\n[FEATURES] Top 15 most important ({best_name}):")
        for rank, idx in enumerate(indices[:15], 1):
            print(f"  {rank:2d}. {feature_names[idx]:30s}  {importances[idx]:.4f}")
        return {feature_names[i]: float(importances[i]) for i in indices}
    return {}


# =============================================================================
#  SAVE ARTIFACTS
# =============================================================================

def save_all(pipeline, best_name, cv_results, feature_names, feat_importance,
             optimal_threshold=0.5):
    """Save model, CV results, and metadata."""
    # Save model + threshold together
    save_obj = {"pipeline": pipeline, "threshold": optimal_threshold}
    joblib.dump(save_obj, str(MODEL_PATH))
    print(f"\n[SAVED] Model -> {MODEL_PATH}  (threshold={optimal_threshold})")

    # Save CV history
    with open(str(HISTORY_OUT), "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"[SAVED] CV Results -> {HISTORY_OUT}")

    # Build metadata
    best_cv = cv_results[best_name]
    metadata = {
        "model_type":       f"tremor_{best_name.lower()}",
        "created_at":       datetime.now().isoformat(),
        "window_size":      WINDOW_SIZE,
        "sample_rate_hz":   SAMPLE_RATE,
        "n_features":       len(feature_names),
        "feature_names":    feature_names,
        "labels":           {"0": "No Tremor", "1": "Tremor Detected"},
        "best_model":       best_name,
        "cv_folds":         N_FOLDS,
        "optimal_threshold": optimal_threshold,
        "final_metrics": {
            "val_auc":       round(best_cv["val_roc_auc_mean"], 4),
            "val_auc_std":   round(best_cv["val_roc_auc_std"], 4),
            "val_accuracy":  round(best_cv["val_accuracy_mean"], 4),
            "val_precision": round(best_cv["val_precision_mean"], 4),
            "val_recall":    round(best_cv["val_recall_mean"], 4),
            "val_f1":        round(best_cv["val_f1_mean"], 4),
        },
        "all_models_auc": {
            name: round(res["val_roc_auc_mean"], 4)
            for name, res in cv_results.items()
        },
        "top_features": dict(list(feat_importance.items())[:15]) if feat_importance else {},
        "hyperparameters": {
            "random_seed": RANDOM_SEED,
            "cv_folds":    N_FOLDS,
        },
    }
    with open(str(METADATA_OUT), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Metadata -> {METADATA_OUT}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  TREMOR DETECTION - PHASE 2b: FEATURE-ENGINEERED ML")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading windowed data...")
    X_raw = np.load(str(DATA_DIR / "X_train.npy"))   # (N, 200, 1)
    y     = np.load(str(DATA_DIR / "y_train.npy"))    # (N,)
    print(f"  X: {X_raw.shape}  y: {y.shape}")
    print(f"  Tremor   : {int(y.sum()):,}  ({100*y.mean():.1f}%)")
    print(f"  No-tremor: {int((y==0).sum()):,}  ({100*(1-y.mean()):.1f}%)")

    # ── 2. Feature extraction ─────────────────────────────────────────────
    print("\n[2/5] Extracting features from all windows...")
    X_feat, feature_names = extract_features_all(X_raw)
    print(f"  Feature matrix: {X_feat.shape}  ({len(feature_names)} features)")

    # ── 3. Cross-validate models ──────────────────────────────────────────
    print("\n[3/5] Cross-validating models...")
    models = get_models(y=y)
    cv_results = cross_validate_models(X_feat, y, models)

    # ── 4. Train best model on full data ──────────────────────────────────
    print("\n[4/5] Training final model on full dataset...")
    pipeline, best_name, optimal_threshold = train_final_model(X_feat, y, models, cv_results)

    # Feature importance
    feat_importance = get_feature_importance(pipeline, feature_names, best_name)

    # ── 5. Save ───────────────────────────────────────────────────────────
    print("\n[5/5] Saving artifacts...")
    save_all(pipeline, best_name, cv_results, feature_names, feat_importance,
             optimal_threshold=optimal_threshold)

    # ── Summary ───────────────────────────────────────────────────────────
    best_cv = cv_results[best_name]
    print("\n" + "=" * 60)
    print("  PHASE 2b COMPLETE - TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Best model    : {best_name}")
    print(f"  Features      : {len(feature_names)}")
    print(f"  CV folds      : {N_FOLDS}")
    print(f"  Val AUC       : {best_cv['val_roc_auc_mean']:.4f} +/- {best_cv['val_roc_auc_std']:.4f}")
    print(f"  Val Accuracy  : {best_cv['val_accuracy_mean']:.4f} +/- {best_cv['val_accuracy_std']:.4f}")
    print(f"  Val Precision : {best_cv['val_precision_mean']:.4f} +/- {best_cv['val_precision_std']:.4f}")
    print(f"  Val Recall    : {best_cv['val_recall_mean']:.4f} +/- {best_cv['val_recall_std']:.4f}")
    print(f"  Val F1        : {best_cv['val_f1_mean']:.4f} +/- {best_cv['val_f1_std']:.4f}")
    print(f"\n  All models AUC:")
    for name, res in cv_results.items():
        flag = " <-- BEST" if name == best_name else ""
        print(f"    {name:20s}  {res['val_roc_auc_mean']:.4f}{flag}")
    print(f"\n  Model saved -> {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
