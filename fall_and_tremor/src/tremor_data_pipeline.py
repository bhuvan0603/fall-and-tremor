"""
Phase 1 — Tremor Detection Data Pipeline
=========================================
Reads PhysioNet DBS Tremor dataset (WFDB format), labels recordings,
applies sliding windows, splits subjects, saves .npy arrays and demo CSVs.

Dataset folder structure:
  re* folders (DBS ON)  → label 0 = NO_TREMOR  (tremor suppressed)
  ro* / r*of* folders   → label 1 = TREMOR      (tremor active)

Subject split:
  Training : g*, v* prefixed subjects
  Demo CSVs: s*  prefixed subjects  (s6,s7,s8,s14,s15,s16)
"""

import os
import re
import struct
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "effect-of-deep-brain-stimulation-on-parkinsonian-tremor-1.0.0"
DATA_OUT    = BASE_DIR / "data" / "tremor"
DEMO_OUT    = BASE_DIR / "test_data" / "tremor_samples"

DATA_OUT.mkdir(parents=True, exist_ok=True)
DEMO_OUT.mkdir(parents=True, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 100          # Hz
WINDOW_SEC    = 2            # seconds per window
STEP_SEC      = 1            # 50% overlap
WINDOW_SIZE   = SAMPLE_RATE * WINDOW_SEC   # 200 samples
STEP_SIZE     = SAMPLE_RATE * STEP_SEC     # 100 samples

# Label mapping
LABEL_TREMOR    = 1
LABEL_NO_TREMOR = 0

# Folder → label mapping
# re* = DBS ON (tremor suppressed) → 0
# ro* or r{15,30,45,60}of* = DBS OFF (tremor active) → 1
def get_folder_label(folder_name: str) -> int:
    """Return 1 for tremor-active folders, 0 for DBS-on (suppressed) folders."""
    fn = folder_name.lower()
    if fn.startswith("re"):
        return LABEL_NO_TREMOR        # DBS ON → tremor suppressed
    elif fn.startswith("ro") or re.match(r"r\d+of", fn):
        return LABEL_TREMOR           # DBS OFF / recovering → tremor active
    return -1  # unknown folder, skip


def get_subject_prefix(filename: str) -> str:
    """Extract subject prefix letter (g, v, s) from a filename like 'g1ren.let'."""
    m = re.match(r"([a-z]+)(\d+)", filename.lower())
    if m:
        return m.group(1)   # 'g', 'v', or 's'
    return ""


def read_wfdb_raw(filepath: Path, n_samples: int = None) -> np.ndarray:
    """
    Read a WFDB format-16 binary file (16-bit signed integers, little-endian).
    Returns a 1-D float64 numpy array (raw ADC units — we normalise later).
    """
    data = np.fromfile(str(filepath), dtype="<i2")   # little-endian int16
    if n_samples is not None:
        data = data[:n_samples]
    return data.astype(np.float64)


def parse_file_description() -> dict:
    """
    Parse file_description.txt to build a dict:
      { filename_without_ext: {'folder': ..., 'rate': 100, 'samples': N} }
    """
    desc_path = DATASET_DIR / "file_description.txt"
    records = {}
    current_folder = None

    folder_re   = re.compile(r"^([A-Z]+\s+[A-Z]+\s+[A-Z]+.+)\(n=\d+", re.IGNORECASE)
    data_re     = re.compile(
        r"^(\w+)\s+([\w.]+)\s+([\d.]+)\s+(\S+)\s+([\d.-]+)\s+(\d+)\s+(\d+)"
    )

    with open(desc_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Detect folder context from the sub-folder listing pattern
            # A simpler approach: look for lines with file data
            m = data_re.match(line)
            if m:
                subj, fname, rng, vel_unit, laser, rate, samples = m.groups()
                stem = Path(fname).stem          # e.g. 'g1ren'
                records[stem] = {
                    "filename": fname,
                    "subject":  subj,
                    "rate":     int(rate),
                    "samples":  int(samples),
                }
    return records


def sliding_windows(signal: np.ndarray, window: int, step: int) -> np.ndarray:
    """
    Split 1-D signal into overlapping windows.
    Returns shape (n_windows, window_size).
    """
    starts = range(0, len(signal) - window + 1, step)
    return np.array([signal[i:i + window] for i in starts])


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Z-score normalize each window independently."""
    mean = windows.mean(axis=1, keepdims=True)
    std  = windows.std(axis=1, keepdims=True) + 1e-8
    return (windows - mean) / std


def build_dataset():
    """
    Walk every sub-folder in the dataset, read all recordings,
    window them, label them, and split into train/demo sets.

    Returns:
        X_train, y_train  : windowed signals for training (numpy arrays)
        demo_signals      : dict { stem: (signal_array, label) } for s* subjects
    """
    file_meta = parse_file_description()

    X_train, y_train = [], []
    demo_signals = {}   # { stem: (raw_signal_1d, label) }

    # Iterate over every sub-folder in DATASET_DIR
    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue

        label = get_folder_label(folder.name)
        if label == -1:
            print(f"  [SKIP] Unknown folder: {folder.name}")
            continue

        # Find all signal files (.let / .rit) in this folder
        signal_files = sorted(
            f for f in folder.iterdir()
            if f.suffix.lower() in (".let", ".rit")
        )

        for sig_file in signal_files:
            stem    = sig_file.stem                      # e.g. 'g1ren'
            subj_px = get_subject_prefix(sig_file.name)  # 'g', 'v', 's'

            # Determine expected sample count from metadata (or read all)
            n_samples = None
            if stem in file_meta:
                n_samples = file_meta[stem]["samples"]

            try:
                signal = read_wfdb_raw(sig_file, n_samples)
            except Exception as e:
                print(f"  [ERROR] Could not read {sig_file}: {e}")
                continue

            if len(signal) < WINDOW_SIZE:
                print(f"  [SHORT] {sig_file.name}: only {len(signal)} samples, skipping")
                continue

            print(f"  [OK] {folder.name}/{sig_file.name}  "
                  f"samples={len(signal)}  label={label}  subject_type={subj_px}")

            # ── Demo subjects (s prefix) → save raw signal for CSV export ──────
            if subj_px == "s":
                if stem not in demo_signals:
                    demo_signals[stem] = (signal, label)
                continue   # don't add to training set

            # ── Training subjects (g, v prefix) → window and add ──────────────
            windows = sliding_windows(signal, WINDOW_SIZE, STEP_SIZE)
            windows = normalize_windows(windows)
            labels  = np.full(len(windows), label, dtype=np.int8)

            X_train.append(windows)
            y_train.append(labels)

    if not X_train:
        raise RuntimeError("No training data collected — check dataset path.")

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    return X_train, y_train, demo_signals


def save_training_data(X: np.ndarray, y: np.ndarray):
    """Reshape to (N, window_size, 1) and save .npy files."""
    X = X.reshape(X.shape[0], X.shape[1], 1)   # add channel dim for CNN input
    np.save(str(DATA_OUT / "X_train.npy"), X)
    np.save(str(DATA_OUT / "y_train.npy"), y)

    n_tremor    = int(y.sum())
    n_no_tremor = int((y == 0).sum())
    print(f"\n[SAVED] X_train: {X.shape}  y_train: {y.shape}")
    print(f"        Tremor windows   : {n_tremor}")
    print(f"        No-tremor windows: {n_no_tremor}")
    print(f"        Class ratio      : {n_tremor / max(n_no_tremor, 1):.2f}")
    print(f"        Saved to         : {DATA_OUT}")


def save_demo_csvs(demo_signals: dict):
    """
    Export each demo subject recording as a plain CSV with columns:
        time_s, velocity
    Also save windowed version with label for upload-time inference preview.
    """
    saved = 0
    for stem, (signal, label) in demo_signals.items():
        label_str = "tremor" if label == LABEL_TREMOR else "no_tremor"

        # Raw time-series CSV (what the user uploads)
        timestamps = np.arange(len(signal)) / SAMPLE_RATE
        df = pd.DataFrame({
            "time_s":   np.round(timestamps, 4),
            "velocity": signal,
        })
        out_path = DEMO_OUT / f"{stem}_{label_str}.csv"
        df.to_csv(str(out_path), index=False)
        saved += 1
        print(f"  [CSV] {out_path.name}  ({len(signal)} samples, label={label_str})")

    print(f"\n[SAVED] {saved} demo CSV files to: {DEMO_OUT}")


def print_summary(X: np.ndarray, y: np.ndarray, demo_signals: dict):
    print("\n" + "=" * 55)
    print("  PHASE 1 COMPLETE — DATA PIPELINE SUMMARY")
    print("=" * 55)
    print(f"  Window size : {WINDOW_SIZE} samples ({WINDOW_SEC}s @ {SAMPLE_RATE}Hz)")
    print(f"  Step size   : {STEP_SIZE} samples (50% overlap)")
    print(f"  Train windows: {len(X):,}")
    print(f"    → Tremor   : {int(y.sum()):,}")
    print(f"    → No-tremor: {int((y==0).sum()):,}")
    print(f"  Demo subjects: {len(demo_signals)}")
    for stem, (sig, lbl) in sorted(demo_signals.items()):
        print(f"    {stem:20s}  {len(sig):6d} samples  label={'TREMOR' if lbl else 'NO_TREMOR'}")
    print(f"\n  Output → {DATA_OUT}")
    print(f"  Demo   → {DEMO_OUT}")
    print("=" * 55)


if __name__ == "__main__":
    print("=" * 55)
    print("  TREMOR DETECTION — PHASE 1: DATA PIPELINE")
    print("=" * 55)
    print(f"\nDataset : {DATASET_DIR}")
    print(f"Output  : {DATA_OUT}\n")

    print("[1/3] Reading and windowing recordings...")
    X_train, y_train, demo_signals = build_dataset()

    print("\n[2/3] Saving training arrays...")
    save_training_data(X_train, y_train)

    print("\n[3/3] Exporting demo CSVs...")
    save_demo_csvs(demo_signals)

    print_summary(X_train, y_train, demo_signals)
