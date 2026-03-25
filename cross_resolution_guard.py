"""
Cross-Resolution Leakage Prevention
=====================================
Ensures that when the same underlying instrument exists at multiple
resolutions (e.g. AAPL at 1H, 4H, 1D, 1W), ALL resolutions are assigned
to the same temporal partition (train / purge / val / test) for any
given walk-forward fold.

THE PROBLEM:
  AAPL_1D  bars from Jan–Dec 2020 are in the training set.
  AAPL_4H  bars from Jan–Dec 2020 are in the validation set.

  The model has now "seen" those exact price movements during training —
  just sampled at a coarser resolution. The 4H bars encode the same
  opens, highs, lows, closes, and volume patterns that the 1D bars
  summarise. Validation metrics will be overoptimistic because the
  model is being tested on data it has already learned from in a
  different form.

  This is NOT the same as cross-instrument generalisation (learning
  patterns from AAPL that transfer to MSFT). This is the same
  instrument's data appearing on both sides of the train/val boundary.

THE FIX:
  1. Group all instrument+resolution combos by their BASE instrument
     (the underlying ticker+exchange, ignoring resolution).

  2. Define walk-forward date boundaries at the BASE instrument level.
     All resolutions of that instrument inherit the same boundaries.

  3. When assembling batches, a given calendar date range is either
     ALL-training or ALL-validation for every resolution of that
     instrument. No date range can be training at one resolution
     and validation at another for the same instrument.

Usage:
  from cross_resolution_guard import (
      build_instrument_groups,
      validate_no_cross_resolution_leakage,
      get_safe_split_boundaries,
  )

  # Build groups from your feature files
  groups = build_instrument_groups(feature_files)

  # When creating splits, get boundaries per base instrument
  boundaries = get_safe_split_boundaries(groups, fold_config)

  # After creating splits, validate there's no leakage
  validate_no_cross_resolution_leakage(splits, groups)
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


# ─────────────────────────────────────────────────────────
# Instrument Grouping
# ─────────────────────────────────────────────────────────

def extract_base_instrument(instrument_resolution_key: str) -> str:
    """
    Extract the base instrument name from an instrument+resolution key.

    Examples:
      "NYSE_AAPL_1D"  → "NYSE_AAPL"
      "NYSE_AAPL_4H"  → "NYSE_AAPL"
      "FOREXCOM_EURUSD_1H" → "FOREXCOM_EURUSD"

    The base instrument is the part before the resolution suffix.
    """
    # Known resolutions
    resolutions = {"1H", "4H", "1D", "1W", "1M"}

    parts = instrument_resolution_key.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in resolutions:
        return parts[0]

    # If no resolution suffix found, return as-is
    return instrument_resolution_key


def build_instrument_groups(feature_files: List[str]) -> Dict[str, List[dict]]:
    """
    Group feature files by their base instrument.

    Args:
      feature_files: List of .npz file paths from Stage 2.
                     Expected naming: {EXCHANGE}_{TICKER}_{RESOLUTION}_features.npz

    Returns:
      Dict mapping base_instrument → list of {
          "file": path,
          "resolution": str,
          "base_instrument": str,
          "full_key": str,
          "timestamps": np.array (min/max dates),
      }

    Example output:
      {
        "NYSE_AAPL": [
          {"file": "..._1D_features.npz", "resolution": "1D", ...},
          {"file": "..._4H_features.npz", "resolution": "4H", ...},
          {"file": "..._1W_features.npz", "resolution": "1W", ...},
        ],
        "FOREXCOM_EURUSD": [
          {"file": "..._4H_features.npz", "resolution": "4H", ...},
          {"file": "..._1D_features.npz", "resolution": "1D", ...},
        ],
      }
    """
    groups = defaultdict(list)

    for fpath in feature_files:
        fpath = Path(fpath)
        stem = fpath.stem  # e.g. "NYSE_AAPL_1D_features"

        # Remove the "_features" suffix
        key = stem.replace("_features", "")  # "NYSE_AAPL_1D"

        base = extract_base_instrument(key)
        resolution = key[len(base) + 1:] if len(key) > len(base) else "unknown"

        # Load timestamps to determine date range
        data = np.load(fpath, allow_pickle=True)
        timestamps = pd.to_datetime(data["timestamps"])

        entry = {
            "file": str(fpath),
            "resolution": resolution,
            "base_instrument": base,
            "full_key": key,
            "date_min": timestamps.min(),
            "date_max": timestamps.max(),
            "n_samples": len(timestamps),
        }

        groups[base].append(entry)

    # Sort each group by resolution for consistency
    res_order = {"1W": 0, "1D": 1, "4H": 2, "1H": 3}
    for base in groups:
        groups[base].sort(key=lambda x: res_order.get(x["resolution"], 99))

    return dict(groups)


def print_instrument_groups(groups: Dict[str, List[dict]]):
    """Pretty-print the instrument groups for inspection."""
    print(f"\n{'='*65}")
    print(f"  Cross-Resolution Instrument Groups")
    print(f"{'='*65}")

    multi_res = {k: v for k, v in groups.items() if len(v) > 1}
    single_res = {k: v for k, v in groups.items() if len(v) == 1}

    if multi_res:
        print(f"\n  MULTI-RESOLUTION instruments ({len(multi_res)}):")
        print(f"  (These require cross-resolution leakage protection)\n")
        for base, entries in sorted(multi_res.items()):
            resolutions = [e["resolution"] for e in entries]
            samples = [e["n_samples"] for e in entries]
            date_range = f"{entries[0]['date_min'].date()} → {entries[0]['date_max'].date()}"
            print(f"    {base}")
            for e in entries:
                print(f"      └─ {e['resolution']:4s}  {e['n_samples']:6d} samples  "
                      f"({e['date_min'].date()} → {e['date_max'].date()})")

    if single_res:
        print(f"\n  SINGLE-RESOLUTION instruments ({len(single_res)}):")
        print(f"  (No cross-resolution risk)\n")
        for base, entries in sorted(single_res.items()):
            e = entries[0]
            print(f"    {base:30s}  {e['resolution']:4s}  {e['n_samples']:6d} samples")

    print(f"\n{'='*65}\n")


# ─────────────────────────────────────────────────────────
# Safe Split Boundaries
# ─────────────────────────────────────────────────────────

def get_safe_split_boundaries(
    groups: Dict[str, List[dict]],
    train_years: int = 6,
    val_years: int = 1,
    test_years: int = 1,
    purge_bars: int = 20,
    slide_years: int = 1,
) -> Dict[str, List[dict]]:
    """
    Compute walk-forward date boundaries at the BASE INSTRUMENT level.

    For multi-resolution instruments, all resolutions will inherit the
    same date boundaries — guaranteeing no resolution of an instrument
    can have training data from a period that another resolution uses
    for validation/test.

    Args:
      groups: Output from build_instrument_groups()
      train_years, val_years, test_years: Window lengths
      purge_bars: Gap between train end and val start (handled in Stage 3
                  at the bar level — here we add a conservative date buffer)
      slide_years: How far to slide the window between folds

    Returns:
      Dict mapping base_instrument → list of fold dicts, each containing:
        {
          "fold": int,
          "train_start": datetime, "train_end": datetime,
          "val_start": datetime,   "val_end": datetime,
          "test_start": datetime,  "test_end": datetime,
          "resolutions": [list of resolutions covered],
        }

    These boundaries are SHARED across all resolutions of the instrument.
    """
    boundaries = {}

    for base, entries in groups.items():
        # Find the COMMON date range across all resolutions
        # Use the most restrictive range (latest start, earliest end)
        common_start = max(e["date_min"] for e in entries)
        common_end = min(e["date_max"] for e in entries)

        total_years = (common_end - common_start).days / 365.25
        min_required = train_years + val_years + test_years

        if total_years < min_required:
            warnings.warn(
                f"[{base}] Common date range is {total_years:.1f} years "
                f"(need {min_required}). Fewer folds will be generated."
            )

        # Generate folds
        folds = []
        fold_num = 1
        current_start = common_start

        while True:
            train_start = current_start
            train_end = train_start + pd.DateOffset(years=train_years)

            # Add purge gap (conservative: 30 calendar days covers 20 bars
            # at any resolution including 1D)
            purge_end = train_end + pd.DateOffset(days=30)

            val_start = purge_end
            val_end = val_start + pd.DateOffset(years=val_years)

            test_start = val_end
            test_end = test_start + pd.DateOffset(years=test_years)

            # Check if test_end exceeds available data
            if test_end > common_end:
                break

            folds.append({
                "fold": fold_num,
                "train_start": train_start,
                "train_end": train_end,
                "purge_end": purge_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
                "resolutions": [e["resolution"] for e in entries],
            })

            fold_num += 1
            current_start += pd.DateOffset(years=slide_years)

        boundaries[base] = folds

    return boundaries


# ─────────────────────────────────────────────────────────
# Validation — Post-Split Leakage Check
# ─────────────────────────────────────────────────────────

def validate_no_cross_resolution_leakage(
    splits: Dict[str, dict],
    groups: Dict[str, List[dict]],
) -> Tuple[bool, List[str]]:
    """
    Validate that no cross-resolution leakage exists in the final splits.

    For every base instrument that has multiple resolutions, verify that:
    1. The date ranges assigned to train, val, and test do NOT overlap
       across resolutions for that instrument.
    2. No date appears in training for resolution A and in val/test
       for resolution B of the same instrument.

    Args:
      splits: Dict mapping full_key (e.g. "NYSE_AAPL_1D") to split info
              containing "train_timestamps", "val_timestamps", "test_timestamps"
      groups: Output from build_instrument_groups()

    Returns:
      (is_safe, list_of_violations)
    """
    violations = []

    for base, entries in groups.items():
        if len(entries) <= 1:
            continue  # Single resolution — no cross-res risk

        resolutions = [e["full_key"] for e in entries]

        # For each pair of resolutions, check for date overlap
        for i in range(len(resolutions)):
            for j in range(i + 1, len(resolutions)):
                key_a = resolutions[i]
                key_b = resolutions[j]

                if key_a not in splits or key_b not in splits:
                    continue

                split_a = splits[key_a]
                split_b = splits[key_b]

                # Check: training dates in A vs val/test dates in B
                train_dates_a = set(pd.to_datetime(split_a.get("train_timestamps", [])).date)
                val_dates_b = set(pd.to_datetime(split_b.get("val_timestamps", [])).date)
                test_dates_b = set(pd.to_datetime(split_b.get("test_timestamps", [])).date)

                train_val_overlap = train_dates_a & val_dates_b
                train_test_overlap = train_dates_a & test_dates_b

                if train_val_overlap:
                    violations.append(
                        f"LEAKAGE: {key_a} training dates overlap with "
                        f"{key_b} validation dates on {len(train_val_overlap)} days. "
                        f"Example: {sorted(train_val_overlap)[:3]}"
                    )

                if train_test_overlap:
                    violations.append(
                        f"LEAKAGE: {key_a} training dates overlap with "
                        f"{key_b} test dates on {len(train_test_overlap)} days. "
                        f"Example: {sorted(train_test_overlap)[:3]}"
                    )

                # Check the reverse direction too
                train_dates_b = set(pd.to_datetime(split_b.get("train_timestamps", [])).date)
                val_dates_a = set(pd.to_datetime(split_a.get("val_timestamps", [])).date)
                test_dates_a = set(pd.to_datetime(split_a.get("test_timestamps", [])).date)

                train_val_overlap_rev = train_dates_b & val_dates_a
                train_test_overlap_rev = train_dates_b & test_dates_a

                if train_val_overlap_rev:
                    violations.append(
                        f"LEAKAGE: {key_b} training dates overlap with "
                        f"{key_a} validation dates on {len(train_val_overlap_rev)} days. "
                        f"Example: {sorted(train_val_overlap_rev)[:3]}"
                    )

                if train_test_overlap_rev:
                    violations.append(
                        f"LEAKAGE: {key_b} training dates overlap with "
                        f"{key_a} test dates on {len(train_test_overlap_rev)} days. "
                        f"Example: {sorted(train_test_overlap_rev)[:3]}"
                    )

    is_safe = len(violations) == 0

    if is_safe:
        print("  ✓ Cross-resolution leakage check PASSED — no violations found.")
    else:
        print(f"  ✗ Cross-resolution leakage check FAILED — {len(violations)} violations:")
        for v in violations:
            print(f"    {v}")

    return is_safe, violations


# ─────────────────────────────────────────────────────────
# Sample-Level Filtering
# ─────────────────────────────────────────────────────────

def filter_samples_by_date_boundaries(
    timestamps: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    regimes: np.ndarray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter training samples to only include those whose final bar timestamp
    falls within [start_date, end_date).

    This is the function Stage 3 will call when assembling train/val/test
    sets for each fold, ensuring date boundaries are respected regardless
    of resolution.

    Returns filtered (features, targets, timestamps, regimes).
    """
    ts = pd.to_datetime(timestamps)
    mask = (ts >= start_date) & (ts < end_date)

    return (
        features[mask],
        targets[mask],
        timestamps[mask],
        regimes[mask],
    )


# ─────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────

def run_self_test(processed_dir: str = "data/processed"):
    """
    Run the cross-resolution guard on existing Stage 2 outputs.
    Shows instrument groups, computes safe boundaries, and reports.
    """
    proc_path = Path(processed_dir)
    npz_files = sorted(proc_path.glob("*_features.npz"))

    if not npz_files:
        print(f"No feature files found in {proc_path}. Run Stage 2 first.")
        return

    print(f"Found {len(npz_files)} feature files.\n")

    # Step 1: Build groups
    groups = build_instrument_groups([str(f) for f in npz_files])
    print_instrument_groups(groups)

    # Step 2: Compute safe boundaries
    print("Computing walk-forward boundaries (shared per base instrument)...\n")
    boundaries = get_safe_split_boundaries(groups)

    for base, folds in sorted(boundaries.items()):
        resolutions = folds[0]["resolutions"] if folds else []
        multi = len(resolutions) > 1
        marker = " ⚡ MULTI-RES" if multi else ""

        print(f"  {base} ({', '.join(resolutions)}){marker}")
        print(f"    {len(folds)} fold(s):")
        for fold in folds:
            print(f"      Fold {fold['fold']}: "
                  f"Train {fold['train_start'].date()}→{fold['train_end'].date()} | "
                  f"Val {fold['val_start'].date()}→{fold['val_end'].date()} | "
                  f"Test {fold['test_start'].date()}→{fold['test_end'].date()}")
        print()

    # Summary
    multi_res_instruments = [k for k, v in groups.items() if len(v) > 1]
    if multi_res_instruments:
        print(f"  ⚡ {len(multi_res_instruments)} instrument(s) have multiple resolutions.")
        print(f"     All resolutions will share identical date boundaries per fold.")
        print(f"     This prevents cross-resolution temporal leakage.\n")
    else:
        print(f"  ✓ No multi-resolution instruments found. No cross-resolution risk.\n")


if __name__ == "__main__":
    run_self_test()