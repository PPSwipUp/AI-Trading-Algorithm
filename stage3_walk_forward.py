"""
Stage 3: Walk-Forward Validation (Disk-Streaming)
===================================================
Time-series-safe train/val/test splitting with cross-resolution
leakage prevention.

MEMORY STRATEGY:
  With 200+ instrument files, even ONE fold's concatenated training array
  can exceed 4 GB. This version never concatenates — it saves each
  instrument's contribution as a separate shard file within each fold.
  Peak RAM = one instrument at a time (~16 MB).

  Output structure:
    data/splits/fold_N/
        train/
            INSTRUMENT_KEY.npz   (features, targets, timestamps, regimes)
            ...
        val/
            INSTRUMENT_KEY.npz
            ...
        test/
            INSTRUMENT_KEY.npz
            ...
        fold_manifest.json       (sample counts, metadata, regime balance)

  Stage 5 (training loop) loads shards on demand per batch.

Usage:
    python stage3_walk_forward.py                # Generate all splits
    python stage3_walk_forward.py --dry-run      # Show plan only
    python stage3_walk_forward.py --max-folds 6  # Limit fold count
"""

import os
import gc
import sys
import json
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from cross_resolution_guard import extract_base_instrument

# ─────────────────────────────────────────────────────────
# Configuration (from blueprint Section 3.2)
# ─────────────────────────────────────────────────────────

TRAIN_YEARS = 8               # 8yr training window (was 6) — more samples per fold
VAL_YEARS = 1
TEST_YEARS = 1
SLIDE_YEARS = 1
PURGE_DAYS = 30
MIN_FOLDS = 4
DEFAULT_MAX_FOLDS = 8
MAX_INSTRUMENT_SHARE = 1.0    # No cap — the 15% cap was throwing away 4H data
MIN_REGIME_SHARE = 0.05


# ─────────────────────────────────────────────────────────
# Pass 1: Lightweight Metadata Scan
# ─────────────────────────────────────────────────────────

def scan_metadata(input_dir: str) -> List[dict]:
    """
    Scan .npz files for metadata only. Does NOT load feature arrays.
    """
    in_path = Path(input_dir)
    npz_files = sorted(in_path.glob("*_features.npz"))

    if not npz_files:
        raise FileNotFoundError(f"No *_features.npz in {in_path}. Run Stage 2 first.")

    catalog = []

    for fpath in npz_files:
        key = fpath.stem.replace("_features", "")
        base = extract_base_instrument(key)
        resolution = key[len(base) + 1:] if len(key) > len(base) else "unknown"

        with np.load(fpath, allow_pickle=True, mmap_mode="r") as data:
            timestamps = pd.to_datetime(data["timestamps"])
            n_samples = data["features"].shape[0]
            n_features = data["features"].shape[2]
            feature_names = list(data["feature_names"])
            regimes = data["regimes"].copy()

        catalog.append({
            "file": str(fpath),
            "key": key,
            "base_instrument": base,
            "resolution": resolution,
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_names": feature_names,
            "date_min": timestamps.min(),
            "date_max": timestamps.max(),
            "timestamps": timestamps,
            "regimes": regimes,
        })

    return catalog


# ─────────────────────────────────────────────────────────
# Cross-Resolution Groups
# ─────────────────────────────────────────────────────────

def get_cross_resolution_groups(catalog: List[dict]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for c in catalog:
        groups[c["base_instrument"]].append(c["key"])
    return dict(groups)


# ─────────────────────────────────────────────────────────
# Fold Boundary Computation
# ─────────────────────────────────────────────────────────

def compute_fold_boundaries(catalog: List[dict], max_folds: int) -> List[dict]:
    global_start = min(c["date_min"] for c in catalog)
    global_end = max(c["date_max"] for c in catalog)
    total_years = (global_end - global_start).days / 365.25

    print(f"\n  Date range: {global_start.date()} → {global_end.date()} "
          f"({total_years:.1f} years)")

    folds = []
    fold_num = 1
    current_start = global_start

    while True:
        train_start = pd.Timestamp(current_start)
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        purge_end = train_end + pd.DateOffset(days=PURGE_DAYS)
        val_start = purge_end
        val_end = val_start + pd.DateOffset(years=VAL_YEARS)
        test_start = val_end
        test_end = test_start + pd.DateOffset(years=TEST_YEARS)

        if test_end > global_end + pd.DateOffset(days=30):
            break

        folds.append({
            "fold": fold_num,
            "train_start": train_start, "train_end": train_end,
            "purge_end": purge_end,
            "val_start": val_start, "val_end": val_end,
            "test_start": test_start, "test_end": test_end,
        })

        fold_num += 1
        current_start += pd.DateOffset(years=SLIDE_YEARS)

    total_possible = len(folds)

    # Cap folds — use the LAST N folds (most recent data is most relevant)
    if len(folds) > max_folds:
        print(f"    {total_possible} possible folds, capping to last {max_folds} "
              f"(most recent data)")
        folds = folds[-max_folds:]
        # Renumber
        for i, f in enumerate(folds, 1):
            f["fold"] = i

    return folds


# ─────────────────────────────────────────────────────────
# Sample Index Computation (metadata only)
# ─────────────────────────────────────────────────────────

def compute_split_indices(catalog: List[dict], fold: dict) -> Dict[str, dict]:
    """
    For each instrument, compute which sample indices fall in
    train/val/test for this fold. No feature loading.
    """
    result = {}

    for c in catalog:
        ts = c["timestamps"]
        train_idx = np.where((ts >= fold["train_start"]) & (ts < fold["train_end"]))[0]
        val_idx = np.where((ts >= fold["val_start"]) & (ts < fold["val_end"]))[0]
        test_idx = np.where((ts >= fold["test_start"]) & (ts < fold["test_end"]))[0]

        result[c["key"]] = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

    return result


# ─────────────────────────────────────────────────────────
# Instrument Capping (Section 3.3)
# ─────────────────────────────────────────────────────────

def compute_capping(
    split_indices: Dict[str, dict],
    max_share: float = MAX_INSTRUMENT_SHARE,
) -> Dict[str, int]:
    """
    Compute how many training samples each instrument is allowed.
    """
    counts = {k: len(v["train"]) for k, v in split_indices.items()}

    allowed = dict(counts)
    for _ in range(10):
        total = sum(allowed.values())
        if total == 0:
            break
        capped = False
        for key in allowed:
            if allowed[key] / total > max_share:
                new_count = int(total * max_share)
                if new_count < allowed[key]:
                    allowed[key] = new_count
                    capped = True
        if not capped:
            break

    return allowed


# ─────────────────────────────────────────────────────────
# Per-Fold Assembly: Stream to Disk
# ─────────────────────────────────────────────────────────

def assemble_and_save_fold(
    catalog: List[dict],
    fold: dict,
    split_indices: Dict[str, dict],
    cap_limits: Dict[str, int],
    unified_dim: int,
    fold_dir: Path,
    seed: int = 42,
) -> dict:
    """
    For one fold: load each instrument one at a time, slice its
    contribution, save as a shard file, free memory. Never holds
    more than one instrument's data in RAM.

    Saves:
      fold_dir/train/INSTRUMENT_KEY.npz
      fold_dir/val/INSTRUMENT_KEY.npz
      fold_dir/test/INSTRUMENT_KEY.npz
    """
    rng = np.random.RandomState(seed)

    for split_name in ["train", "val", "test"]:
        (fold_dir / split_name).mkdir(parents=True, exist_ok=True)

    manifest = {
        "fold": fold["fold"],
        "train_start": str(fold["train_start"].date()),
        "train_end": str(fold["train_end"].date()),
        "purge_end": str(fold["purge_end"].date()),
        "val_start": str(fold["val_start"].date()),
        "val_end": str(fold["val_end"].date()),
        "test_start": str(fold["test_start"].date()),
        "test_end": str(fold["test_end"].date()),
        "feature_dim": int(unified_dim),
        "shards": {"train": {}, "val": {}, "test": {}},
        "sample_counts": {"train": 0, "val": 0, "test": 0},
        "regime_counts": {"train": {}, "val": {}, "test": {}},
        "cap_info": {},
    }

    for c in catalog:
        key = c["key"]
        indices = split_indices[key]

        # Skip if this instrument has nothing for this fold
        has_data = any(len(indices[s]) > 0 for s in ["train", "val", "test"])
        if not has_data:
            continue

        # ── Load this one file ──
        data = np.load(c["file"], allow_pickle=True)
        features = data["features"]
        targets = data["targets"]
        timestamps = data["timestamps"]
        regimes = data["regimes"]

        n_feat = features.shape[2]

        # ── Process each split ──
        for split_name in ["train", "val", "test"]:
            idx = indices[split_name]
            if len(idx) == 0:
                continue

            f = features[idx]
            t = targets[idx]
            ts = timestamps[idx]
            r = regimes[idx]

            # Pad features if needed
            if n_feat < unified_dim:
                f = np.pad(
                    f,
                    pad_width=((0, 0), (0, 0), (0, unified_dim - n_feat)),
                    mode="constant", constant_values=0.0,
                )

            # Apply capping for training only
            if split_name == "train":
                cap_limit = cap_limits.get(key, len(f))
                if cap_limit < len(f):
                    keep = rng.choice(len(f), size=cap_limit, replace=False)
                    keep.sort()
                    f, t, ts, r = f[keep], t[keep], ts[keep], r[keep]
                    manifest["cap_info"][key] = {
                        "original": int(len(idx)),
                        "capped_to": int(cap_limit),
                    }

            # Save shard
            shard_path = fold_dir / split_name / f"{key}.npz"
            np.savez_compressed(
                shard_path,
                features=f.astype(np.float32),
                targets=t.astype(np.float32),
                timestamps=ts,
                regimes=r,
            )

            n_saved = len(f)
            manifest["shards"][split_name][key] = {
                "file": str(shard_path),
                "n_samples": int(n_saved),
                "n_features": int(unified_dim),
            }
            manifest["sample_counts"][split_name] += n_saved

            # Regime tracking
            for regime_label in r:
                rl = str(regime_label)
                manifest["regime_counts"][split_name][rl] = \
                    manifest["regime_counts"][split_name].get(rl, 0) + 1

        # ── Free memory ──
        del data, features, targets, timestamps, regimes
        gc.collect()

    return manifest


# ─────────────────────────────────────────────────────────
# Cross-Resolution Validation
# ─────────────────────────────────────────────────────────

def validate_cross_resolution(
    catalog: List[dict],
    split_indices: Dict[str, dict],
    cross_res_groups: Dict[str, List[str]],
    fold_num: int,
) -> bool:
    """
    Verify no cross-resolution leakage using metadata only (no loading).
    """
    violations = []
    catalog_map = {c["key"]: c for c in catalog}

    for base, keys in cross_res_groups.items():
        if len(keys) <= 1:
            continue

        for i, key_a in enumerate(keys):
            for key_b in keys[i + 1:]:
                if key_a not in split_indices or key_b not in split_indices:
                    continue

                # Training dates for key_a
                idx_a_train = split_indices[key_a]["train"]
                if len(idx_a_train) == 0:
                    continue
                dates_a_train = set(
                    catalog_map[key_a]["timestamps"][idx_a_train].normalize()
                )

                # Val/test dates for key_b
                for check_split in ["val", "test"]:
                    idx_b = split_indices[key_b][check_split]
                    if len(idx_b) == 0:
                        continue
                    dates_b = set(
                        catalog_map[key_b]["timestamps"][idx_b].normalize()
                    )
                    overlap = dates_a_train & dates_b
                    if overlap:
                        violations.append(
                            f"Fold {fold_num}: {key_a} train ↔ "
                            f"{key_b} {check_split}: {len(overlap)} days"
                        )

                # Reverse: training dates for key_b vs val/test for key_a
                idx_b_train = split_indices[key_b]["train"]
                if len(idx_b_train) == 0:
                    continue
                dates_b_train = set(
                    catalog_map[key_b]["timestamps"][idx_b_train].normalize()
                )

                for check_split in ["val", "test"]:
                    idx_a = split_indices[key_a][check_split]
                    if len(idx_a) == 0:
                        continue
                    dates_a = set(
                        catalog_map[key_a]["timestamps"][idx_a].normalize()
                    )
                    overlap = dates_b_train & dates_a
                    if overlap:
                        violations.append(
                            f"Fold {fold_num}: {key_b} train ↔ "
                            f"{key_a} {check_split}: {len(overlap)} days"
                        )

    if violations:
        for v in violations:
            print(f"        ✗ {v}")
        return False
    return True


# ─────────────────────────────────────────────────────────
# Regime Balance Report
# ─────────────────────────────────────────────────────────

def regime_balance_report(regime_counts: dict, label: str) -> dict:
    # Filter out non-regime labels
    skip = {"insufficient_history", "unknown", ""}
    filtered = {k: v for k, v in regime_counts.items() if k not in skip}

    total = sum(filtered.values())
    if total == 0:
        return {"total": 0, "distribution": {}, "warnings": []}

    distribution = {k: round(v / total * 100, 2) for k, v in filtered.items()}

    warn = []
    expected = {"trending_low_vol", "trending_normal_vol", "trending_high_vol",
                "range_bound_low_vol", "range_bound_normal_vol", "range_bound_high_vol"}

    for regime, pct in distribution.items():
        if pct < MIN_REGIME_SHARE * 100:
            warn.append(f"⚠ {label}: '{regime}' = {pct}% (< {MIN_REGIME_SHARE*100}%)")

    missing = expected - set(filtered.keys())
    if missing:
        warn.append(f"⚠ {label}: Missing: {missing}")

    return {"total": total, "distribution": distribution, "warnings": warn}


# ─────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────

def run_stage3(
    input_dir: str = "data/processed",
    output_dir: str = "data/splits",
    dry_run: bool = False,
    max_folds: int = DEFAULT_MAX_FOLDS,
):
    out_path = Path(output_dir)

    print(f"\n{'='*65}")
    print(f"  STAGE 3: Walk-Forward Validation (Disk-Streaming)")
    print(f"{'='*65}")
    print(f"  Train window:  {TRAIN_YEARS} years")
    print(f"  Purge gap:     {PURGE_DAYS} calendar days")
    print(f"  Val window:    {VAL_YEARS} year")
    print(f"  Test window:   {TEST_YEARS} year")
    print(f"  Slide:         {SLIDE_YEARS} year")
    print(f"  Max folds:     {max_folds}")
    print(f"  Max instrument share: {MAX_INSTRUMENT_SHARE*100:.0f}%")
    print(f"{'='*65}")

    # ═══════════════════════════════════════════════════════
    # PASS 1: Metadata scan
    # ═══════════════════════════════════════════════════════
    print(f"\n  Pass 1: Scanning metadata...")
    catalog = scan_metadata(input_dir)
    total_samples = sum(c["n_samples"] for c in catalog)
    print(f"    {len(catalog)} instrument files, {total_samples:,} total samples")

    cross_res_groups = get_cross_resolution_groups(catalog)
    multi_res = {k: v for k, v in cross_res_groups.items() if len(v) > 1}
    print(f"    {len(cross_res_groups)} base instruments, "
          f"{len(multi_res)} multi-resolution")
    for base, keys in multi_res.items():
        res_labels = [k[len(base) + 1:] for k in keys]
        print(f"    ⚡ {base}: {', '.join(res_labels)}")

    all_dims = set(c["n_features"] for c in catalog)
    unified_dim = max(all_dims)
    if len(all_dims) > 1:
        print(f"    Feature dimensions: {sorted(all_dims)} → padding to {unified_dim}")
    else:
        print(f"    Feature dimension: {unified_dim}")

    # ═══════════════════════════════════════════════════════
    # PASS 2: Fold boundaries
    # ═══════════════════════════════════════════════════════
    print(f"\n  Pass 2: Computing fold boundaries...")
    folds = compute_fold_boundaries(catalog, max_folds)

    if not folds:
        print(f"\n  ✗ No valid folds.")
        return

    print(f"    Using {len(folds)} fold(s):\n")
    for fold in folds:
        print(f"      Fold {fold['fold']}: "
              f"Train {fold['train_start'].date()} → {fold['train_end'].date()} | "
              f"Val {fold['val_start'].date()} → {fold['val_end'].date()} | "
              f"Test {fold['test_start'].date()} → {fold['test_end'].date()}")

    if len(folds) < MIN_FOLDS:
        print(f"\n  ⚠ Only {len(folds)} fold(s) (blueprint wants ≥{MIN_FOLDS}).")

    if dry_run:
        print(f"\n  DRY RUN — sample counts:\n")
        for fold in folds:
            si = compute_split_indices(catalog, fold)
            n_tr = sum(len(v["train"]) for v in si.values())
            n_va = sum(len(v["val"]) for v in si.values())
            n_te = sum(len(v["test"]) for v in si.values())
            n_inst = sum(1 for v in si.values() if len(v["train"]) > 0)
            est_gb = n_tr * 60 * unified_dim * 4 / 1024**3
            print(f"    Fold {fold['fold']}: train={n_tr:,} (~{est_gb:.1f}GB), "
                  f"val={n_va:,}, test={n_te:,} ({n_inst} instruments)")
        print(f"\n  Run without --dry-run to save.")
        return folds

    # ═══════════════════════════════════════════════════════
    # PASS 3: Assemble each fold (stream to disk)
    # ═══════════════════════════════════════════════════════
    print(f"\n  Pass 3: Assembling folds (streaming to disk)...")
    out_path.mkdir(parents=True, exist_ok=True)
    fold_summaries = []

    for fold in folds:
        fold_num = fold["fold"]
        print(f"\n    ── Fold {fold_num} ──")

        # Compute indices
        split_indices = compute_split_indices(catalog, fold)
        total_train_raw = sum(len(v["train"]) for v in split_indices.values())

        if total_train_raw == 0:
            print(f"      ✗ No training data. Skipping.")
            continue

        # Capping
        cap_limits = compute_capping(split_indices, MAX_INSTRUMENT_SHARE)
        n_capped = 0
        for key in sorted(cap_limits.keys()):
            original = len(split_indices[key]["train"])
            limit = cap_limits[key]
            if limit < original and original > 0:
                n_capped += 1
                if n_capped <= 10:
                    print(f"      Cap {key}: {original:,} → {limit:,}")
        if n_capped > 10:
            print(f"      ... and {n_capped - 10} more instruments capped")

        # Cross-resolution check (metadata only — no loading)
        print(f"      Cross-resolution check:")
        is_safe = validate_cross_resolution(
            catalog, split_indices, cross_res_groups, fold_num
        )
        if is_safe:
            print(f"        ✓ PASSED")

        # Assemble and save (one instrument at a time to disk)
        fold_dir = out_path / f"fold_{fold_num}"
        manifest = assemble_and_save_fold(
            catalog, fold, split_indices, cap_limits,
            unified_dim=unified_dim,
            fold_dir=fold_dir,
            seed=42 + fold_num,
        )

        manifest["cross_resolution_safe"] = is_safe

        # Purge gap check
        n_train = manifest["sample_counts"]["train"]
        n_val = manifest["sample_counts"]["val"]
        n_test = manifest["sample_counts"]["test"]

        print(f"      Samples: train={n_train:,}, val={n_val:,}, test={n_test:,}")
        print(f"      Shards: train={len(manifest['shards']['train'])}, "
              f"val={len(manifest['shards']['val'])}, "
              f"test={len(manifest['shards']['test'])}")

        # Instrument distribution (top 10)
        if manifest["shards"]["train"]:
            print(f"      Training distribution (top 10):")
            shard_counts = {
                k: v["n_samples"]
                for k, v in manifest["shards"]["train"].items()
            }
            for inst_key, count in sorted(shard_counts.items(),
                                          key=lambda x: -x[1])[:10]:
                share = count / n_train * 100 if n_train > 0 else 0
                bar = "█" * int(share / 2)
                print(f"        {inst_key:40s} {count:7,} ({share:5.1f}%) {bar}")
            if len(shard_counts) > 10:
                print(f"        ... and {len(shard_counts) - 10} more")

        # Regime balance
        print(f"      Regime balance:")
        for split_name in ["train", "val", "test"]:
            balance = regime_balance_report(
                manifest["regime_counts"][split_name], split_name
            )
            manifest[f"regime_balance_{split_name}"] = balance["distribution"]
            if balance["total"] > 0:
                top = sorted(balance["distribution"].items(),
                             key=lambda x: -x[1])[:3]
                dist_str = ", ".join(f"{k}:{v:.1f}%" for k, v in top)
                print(f"        {split_name:5s}: {balance['total']:7,} — {dist_str}")
                for w in balance["warnings"][:3]:
                    print(f"          {w}")

        # Save manifest
        with open(fold_dir / "fold_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        print(f"      ✓ Saved to {fold_dir}/")
        fold_summaries.append(manifest)
        gc.collect()

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  STAGE 3 COMPLETE")
    print(f"{'='*65}")
    print(f"  Folds: {len(fold_summaries)}")

    if fold_summaries:
        t_tr = sum(m["sample_counts"]["train"] for m in fold_summaries)
        t_va = sum(m["sample_counts"]["val"] for m in fold_summaries)
        t_te = sum(m["sample_counts"]["test"] for m in fold_summaries)
        print(f"  Total train: {t_tr:,}")
        print(f"  Total val:   {t_va:,}")
        print(f"  Total test:  {t_te:,}")
        print(f"  Feature dim: {unified_dim}")
        all_safe = all(m.get("cross_resolution_safe", False) for m in fold_summaries)
        print(f"  Cross-res:   {'✓ ALL SAFE' if all_safe else '✗ LEAKAGE'}")

    print(f"  Output: {out_path.resolve()}")
    print(f"\n  Next: Stage 4 — Model Architecture")
    print(f"{'='*65}\n")

    # Cleanup
    for c in catalog:
        c["timestamps"] = None
        c["regimes"] = None
    gc.collect()

    return fold_summaries


# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Walk-forward validation (disk-streaming)."
    )
    parser.add_argument("--input-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan and estimated sizes without saving")
    parser.add_argument("--max-folds", type=int, default=DEFAULT_MAX_FOLDS,
                        help=f"Maximum number of folds (default: {DEFAULT_MAX_FOLDS})")

    args = parser.parse_args()

    run_stage3(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        max_folds=args.max_folds,
    )