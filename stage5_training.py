"""
Stage 5: Training Loop & Optimisation (v4 - Fast + Fixed)
============================================================
Fixes over previous versions:
  SPEED:
    - num_workers=0 for preloaded data (workers add IPC overhead, not speed)
    - float32 storage (eliminates 100K per-sample f16->f32 casts per epoch)
    - torch.compile() support for 2-3x model speedup
  QUALITY:
    - LR schedule based on expected ~50 epochs, not max 200
      (warmup was taking 10 epochs when overfitting starts at epoch 4)

Blueprint Sections 5.1-5.5 all implemented.

Usage:
    python stage5_training.py                # Train all folds
    python stage5_training.py --folds 1      # Single fold
    python stage5_training.py --amp          # Mixed precision (CUDA)
    python stage5_training.py --no-compile   # Disable torch.compile
    python stage5_training.py --dry-run      # Config only
"""

import os, sys, gc, json, time, copy, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional

# ---- Stage 4 Compatibility Imports ----
_this_dir = Path(__file__).resolve().parent
for _p in [_this_dir, _this_dir.parent / "models"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from stage4_model import TradingModel as _ModelClass
except ImportError:
    try:
        from stage4_model import TradingTCN as _ModelClass
    except ImportError:
        _ModelClass = None

try:
    from stage4_model import TradingLoss as _LossClass
except ImportError:
    try:
        from stage4_model import CombinedLoss as _LossClass
    except ImportError:
        _LossClass = None

try:
    from stage4_model import build_model as _build_fn
except ImportError:
    try:
        from stage4_model import create_model as _build_fn
    except ImportError:
        _build_fn = None

try:
    from stage4_model import FORWARD_HORIZONS
except ImportError:
    FORWARD_HORIZONS = [1, 5, 20]

if _ModelClass is None:
    print("ERROR: Cannot import model from stage4_model.py")
    sys.exit(1)

API_STYLE = "v1" if _ModelClass.__name__ == "TradingModel" else "v2"

# ---- Unified Wrappers ----
def make_model_and_loss(n_features, device="auto"):
    if device == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    if _build_fn:
        result = _build_fn(n_features, device=device)
        if isinstance(result, tuple) and len(result) == 3: return result
        if isinstance(result, tuple): return result[0], result[1], device
        loss_fn = _LossClass().to(device) if _LossClass else _FallbackLoss().to(device)
        return result, loss_fn, device
    model = _ModelClass(n_features=n_features).to(device)
    loss_fn = _LossClass().to(device) if _LossClass else _FallbackLoss().to(device)
    return model, loss_fn, device

def model_forward(model, features):
    out = model(features)
    if isinstance(out, dict): return out["direction"], out["magnitude"]
    return out[0], out[1]

def compute_loss(loss_fn, pred_dirs, pred_mags, target_dirs, target_mags):
    try:
        r = loss_fn(pred_dirs, pred_mags, target_dirs, target_mags)
        if isinstance(r, tuple): return r[0], r[1], r[2]
        return r["total"], r["directional"], r["magnitude"]
    except TypeError:
        targets = torch.cat([target_dirs, target_mags], dim=1)
        r = loss_fn({"direction": pred_dirs, "magnitude": pred_mags}, targets)
        if isinstance(r, dict): return r["total"], r["directional"], r["magnitude"]
        return r[0], r[1], r[2]

def get_param_count(model):
    r = model.count_parameters()
    return r.get("total", r) if isinstance(r, dict) else r

class _FallbackLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=0.01, reduction="mean")
    def forward(self, pd, pm, td, tm):
        s = td * 0.95 + 0.025
        bce = F.binary_cross_entropy(pd.clamp(1e-7, 1-1e-7), s, reduction="mean")
        h = self.huber(pm, tm)
        return 0.5*bce + 0.5*h, bce, h

# ---- Configuration ----
LEARNING_RATE = 1e-4          # Reduced from 3e-4 — less aggressive, less overfitting
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
GRAD_CLIP_NORM = 1.0
WARMUP_FRACTION = 0.05
LR_MIN = 1e-6                # Lower floor for gentler annealing
MAX_EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 0.0001
GAP_THRESHOLD = 0.15
GAP_GROWING_EPOCHS = 3
BATCH_SIZE = 256              # 256 is optimal for CPU throughput
EXPECTED_EPOCHS = 80          # Longer schedule — LR decays more gradually

# ---- Preloaded Dataset (float32, zero-overhead) ----
class PreloadedDataset(Dataset):
    def __init__(self, fold_dir, split="train"):
        fold_path = Path(fold_dir)
        manifest = json.load(open(fold_path / "fold_manifest.json"))
        shard_info = manifest["shards"].get(split, {})
        n_h = len(FORWARD_HORIZONS)
        feat_list, dir_list, mag_list = [], [], []
        for key in sorted(shard_info.keys()):
            shard_path = fold_path / split / f"{key}.npz"
            if not shard_path.exists(): continue
            data = np.load(shard_path)
            feat_list.append(data["features"])
            dir_list.append(data["targets"][:, :n_h])
            mag_list.append(data["targets"][:, n_h:])
            del data
        if not feat_list:
            self.features = torch.empty(0)
            self.dirs = torch.empty(0)
            self.mags = torch.empty(0)
            self.n_samples = 0
            return
        self.features = torch.from_numpy(np.concatenate(feat_list, axis=0)).float()
        self.dirs = torch.from_numpy(np.concatenate(dir_list, axis=0)).float()
        self.mags = torch.from_numpy(np.concatenate(mag_list, axis=0)).float()
        self.n_samples = len(self.features)
        del feat_list, dir_list, mag_list
        gc.collect()

    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        return self.features[idx], self.dirs[idx], self.mags[idx]

def preload_data(fold_dir, split, verbose=True):
    t0 = time.time()
    ds = PreloadedDataset(fold_dir, split)
    elapsed = time.time() - t0
    if verbose and ds.n_samples > 0:
        mem_mb = ds.features.nelement() * 4 / 1024**2
        print(f"    {split:5s}: {ds.n_samples:,} samples, {mem_mb:.0f} MB, {elapsed:.1f}s")
    return ds

# ---- LR Schedule (warmup based on expected epochs, not max) ----
class WarmupCosineScheduler:
    def __init__(self, optimizer, expected_steps, warmup_fraction=WARMUP_FRACTION,
                 lr_max=LEARNING_RATE, lr_min=LR_MIN):
        self.optimizer = optimizer
        self.expected_steps = max(expected_steps, 1)
        self.warmup_steps = int(expected_steps * warmup_fraction)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_step = 0
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups: pg["lr"] = lr
    def get_lr(self):
        if self.current_step <= self.warmup_steps:
            return self.lr_max * (self.current_step / max(self.warmup_steps, 1))
        if self.current_step >= self.expected_steps:
            return self.lr_min
        decay_steps = self.expected_steps - self.warmup_steps
        progress = (self.current_step - self.warmup_steps) / max(decay_steps, 1)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + math.cos(math.pi * progress))

# ---- Early Stopping + Gap Monitor ----
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.best_epoch, self.best_state, self.counter = float("inf"), 0, None, 0
    def check(self, val_loss, epoch, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch = val_loss, epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0; return False
        self.counter += 1; return self.counter >= self.patience
    def restore_best(self, model):
        if self.best_state: model.load_state_dict(self.best_state)

class GapMonitor:
    def __init__(self, threshold=GAP_THRESHOLD, consecutive=GAP_GROWING_EPOCHS):
        self.threshold, self.consecutive, self.gap_history = threshold, consecutive, []
    def check(self, train_loss, val_loss):
        gap = val_loss - train_loss
        self.gap_history.append(gap)
        if gap <= self.threshold or len(self.gap_history) < self.consecutive + 1: return False
        recent = self.gap_history[-(self.consecutive + 1):]
        return all(recent[i+1] > recent[i] for i in range(len(recent)-1))

# ---- Metrics ----
def compute_dir_accuracy(pred, target):
    with torch.no_grad():
        return [round(a.item()*100, 2) for a in ((pred > 0.5).float() == target).float().mean(dim=0)]

# ---- Train / Validate ----
def train_one_epoch(model, loss_fn, optimizer, scheduler, loader, device, scaler=None):
    model.train()
    total_s, dir_s, mag_s, grad_s, n = 0., 0., 0., 0., 0
    all_p, all_t = [], []
    use_amp = scaler is not None
    for feat, t_dir, t_mag in loader:
        feat = feat.to(device, non_blocking=True)
        t_dir = t_dir.to(device, non_blocking=True)
        t_mag = t_mag.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                p_dir, p_mag = model_forward(model, feat)
                tl, dl, ml = compute_loss(loss_fn, p_dir, p_mag, t_dir, t_mag)
            scaler.scale(tl).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).item()
            scaler.step(optimizer); scaler.update()
        else:
            p_dir, p_mag = model_forward(model, feat)
            tl, dl, ml = compute_loss(loss_fn, p_dir, p_mag, t_dir, t_mag)
            tl.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).item()
            optimizer.step()
        scheduler.step()
        total_s += tl.item(); dir_s += dl.item(); mag_s += ml.item(); grad_s += gn; n += 1
        all_p.append(p_dir.detach().cpu()); all_t.append(t_dir.detach().cpu())
    n = max(n, 1)
    return {"total_loss": total_s/n, "dir_loss": dir_s/n, "mag_loss": mag_s/n,
            "dir_accuracy": compute_dir_accuracy(torch.cat(all_p), torch.cat(all_t)),
            "grad_norm": grad_s/n, "lr": scheduler.get_lr(), "n_batches": n}

@torch.no_grad()
def validate(model, loss_fn, loader, device, use_amp=False):
    model.eval()
    total_s, dir_s, mag_s, n = 0., 0., 0., 0
    all_p, all_t = [], []
    for feat, t_dir, t_mag in loader:
        feat = feat.to(device, non_blocking=True)
        t_dir = t_dir.to(device, non_blocking=True)
        t_mag = t_mag.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                p_dir, p_mag = model_forward(model, feat)
                tl, dl, ml = compute_loss(loss_fn, p_dir, p_mag, t_dir, t_mag)
        else:
            p_dir, p_mag = model_forward(model, feat)
            tl, dl, ml = compute_loss(loss_fn, p_dir, p_mag, t_dir, t_mag)
        total_s += tl.item(); dir_s += dl.item(); mag_s += ml.item(); n += 1
        all_p.append(p_dir.cpu()); all_t.append(t_dir.cpu())
    n = max(n, 1)
    return {"total_loss": total_s/n, "dir_loss": dir_s/n, "mag_loss": mag_s/n,
            "dir_accuracy": compute_dir_accuracy(torch.cat(all_p), torch.cat(all_t)), "n_batches": n}

# ---- Train One Fold ----
def train_fold(fold_dir, model_save_dir="models/base", log_dir="logs/training",
               batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
               expected_epochs=EXPECTED_EPOCHS, device="auto",
               use_amp=False, use_compile=True, verbose=True):
    fold_path = Path(fold_dir)
    manifest = json.load(open(fold_path / "fold_manifest.json"))
    n_features = manifest["feature_dim"]
    n_train = manifest["sample_counts"]["train"]
    n_val = manifest["sample_counts"]["val"]
    fold_num = manifest["fold"]
    if device == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"

    if verbose:
        print(f"\n{'~'*65}")
        print(f"  Training fold_{fold_num}")
        print(f"{'~'*65}")
        print(f"  Samples: train={n_train:,}, val={n_val:,}, Features: {n_features}")
        print(f"  Preloading data...")

    train_ds = preload_data(fold_dir, "train", verbose=verbose)
    val_ds = preload_data(fold_dir, "val", verbose=verbose)
    # num_workers=0: data is in RAM, workers only add IPC overhead
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    steps_per_epoch = len(train_loader)
    expected_steps = steps_per_epoch * expected_epochs
    warmup_ep = int(expected_steps * WARMUP_FRACTION) / steps_per_epoch

    if verbose:
        print(f"  Steps/epoch: {steps_per_epoch}, "
              f"LR schedule: {expected_epochs}ep expected, warmup ~{warmup_ep:.1f}ep")

    model, loss_fn, device = make_model_and_loss(n_features, device)
    compiled = False
    if use_compile and hasattr(torch, "compile"):
        try: model = torch.compile(model); compiled = True
        except Exception: pass

    if verbose:
        print(f"  Params: {get_param_count(model):,}, Device: {device}"
              + (", compiled" if compiled else ""))

    decay_p, no_decay_p = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        (no_decay_p if ("bias" in name or "norm" in name) else decay_p).append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay_p, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_p, "weight_decay": 0.0},
    ], lr=LEARNING_RATE, betas=BETAS)

    scheduler = WarmupCosineScheduler(optimizer, expected_steps)
    early_stop = EarlyStopping()
    gap_monitor = GapMonitor()
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device == "cuda") else None

    training_log = {"fold": fold_num, "config": {
        "n_features": n_features, "n_train": n_train, "n_val": n_val,
        "batch_size": batch_size, "max_epochs": max_epochs,
        "expected_epochs": expected_epochs, "lr": LEARNING_RATE,
        "device": device, "compiled": compiled, "use_amp": use_amp,
        "n_parameters": get_param_count(model)}, "epochs": []}

    if verbose:
        h_str = "/".join(f"h{h}" for h in FORWARD_HORIZONS)
        print(f"\n  {'Ep':>4s}  {'TrLoss':>8s}  {'VaLoss':>8s}  "
              f"{'Gap':>7s}  {'ValAcc('+h_str+')':>18s}  "
              f"{'GrdN':>6s}  {'LR':>10s}  {'Time':>6s} {'Note'}")
        print(f"  {'~'*86}")

    t_start = time.time()
    final_epoch = 0
    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        final_epoch = epoch
        train_m = train_one_epoch(model, loss_fn, optimizer, scheduler, train_loader, device, scaler)
        val_m = validate(model, loss_fn, val_loader, device, use_amp and device == "cuda")
        gap = val_m["total_loss"] - train_m["total_loss"]
        ep_time = time.time() - t_ep

        entry = {"epoch": epoch, "train_loss": round(train_m["total_loss"], 6),
                 "train_dir_loss": round(train_m["dir_loss"], 6),
                 "train_mag_loss": round(train_m["mag_loss"], 6),
                 "val_loss": round(val_m["total_loss"], 6),
                 "val_dir_loss": round(val_m["dir_loss"], 6),
                 "val_mag_loss": round(val_m["mag_loss"], 6),
                 "gap": round(gap, 6),
                 "train_dir_acc": train_m["dir_accuracy"],
                 "val_dir_acc": val_m["dir_accuracy"],
                 "grad_norm": round(train_m["grad_norm"], 4),
                 "lr": round(train_m["lr"], 8),
                 "epoch_seconds": round(ep_time, 1)}
        training_log["epochs"].append(entry)

        if verbose:
            va = val_m["dir_accuracy"]
            is_best = val_m["total_loss"] < early_stop.best_loss - MIN_DELTA
            note = "**" if is_best else ""
            if train_m["grad_norm"] > 5.0: note += " !grd"
            print(f"  {epoch:4d}  {train_m['total_loss']:8.5f}  "
                  f"{val_m['total_loss']:8.5f}  {gap:+7.4f}  "
                  f"{va[0]:5.1f}/{va[1]:5.1f}/{va[2]:5.1f}  "
                  f"{train_m['grad_norm']:6.3f}  {train_m['lr']:.2e}  "
                  f"{ep_time:5.1f}s {note}")

        should_stop = early_stop.check(val_m["total_loss"], epoch, model)
        gap_halt = gap_monitor.check(train_m["total_loss"], val_m["total_loss"])

        if gap_halt:
            training_log["stop_reason"] = f"Gap halt at epoch {epoch}"
            if verbose: print(f"\n  X Gap halt: gap > {GAP_THRESHOLD} growing for {GAP_GROWING_EPOCHS}ep")
            break
        if should_stop:
            training_log["stop_reason"] = f"Early stop at epoch {epoch} (best: {early_stop.best_epoch})"
            if verbose: print(f"\n  - Early stop (best: epoch {early_stop.best_epoch})")
            break
    else:
        training_log["stop_reason"] = f"Max epochs ({max_epochs})"

    total_time = time.time() - t_start
    early_stop.restore_best(model)
    training_log["best_epoch"] = early_stop.best_epoch
    training_log["best_val_loss"] = round(early_stop.best_loss, 6)
    training_log["total_epochs_run"] = final_epoch
    training_log["total_time_seconds"] = round(total_time, 1)

    final_val = validate(model, loss_fn, val_loader, device)
    training_log["final_val_loss"] = round(final_val["total_loss"], 6)
    training_log["final_val_dir_acc"] = final_val["dir_accuracy"]

    if verbose:
        a = final_val["dir_accuracy"]
        print(f"\n  Best: epoch {early_stop.best_epoch}, val={early_stop.best_loss:.6f}")
        print(f"  Acc: h1={a[0]:.1f}%, h5={a[1]:.1f}%, h20={a[2]:.1f}%")
        print(f"  Time: {total_time:.0f}s ({total_time/max(final_epoch,1):.1f}s/epoch)")

    # Save checkpoint - handle torch.compile prefix stripping
    save_dir = Path(model_save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"checkpoint_base_fold_{fold_num}.pt"
    state_dict = model.state_dict()
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    _model_ref = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model_state_dict": clean_state,
                "n_features": n_features,
                "hidden_dim": _model_ref.hidden_dim,
                "horizons": _model_ref.horizons,
                "best_epoch": early_stop.best_epoch,
                "best_val_loss": early_stop.best_loss,
                "fold": fold_num,
                "training_config": training_log["config"]}, ckpt_path)

    log_save = Path(log_dir); log_save.mkdir(parents=True, exist_ok=True)
    log_path = log_save / f"fold_{fold_num}_training_log.json"
    with open(log_path, "w") as f: json.dump(training_log, f, indent=2)
    if verbose:
        print(f"  -> Checkpoint: {ckpt_path}")
        print(f"  -> Log: {log_path}")

    early_stop.best_state = None
    del model, loss_fn, optimizer, scheduler, early_stop, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return training_log

# ---- Train All Folds ----
def run_stage5(splits_dir="data/splits", model_dir="models/base",
               log_dir="logs/training", batch_size=BATCH_SIZE,
               max_epochs=MAX_EPOCHS, expected_epochs=EXPECTED_EPOCHS,
               device="auto", fold_numbers=None, dry_run=False,
               use_amp=False, use_compile=True):
    fold_dirs = sorted(Path(splits_dir).glob("fold_*"))
    if not fold_dirs:
        print(f"\n  X No folds in {splits_dir}."); return
    if fold_numbers:
        fold_dirs = [d for d in fold_dirs if int(d.name.split("_")[1]) in fold_numbers]

    print(f"\n{'='*65}")
    print(f"  STAGE 5: Training (v4 - Fast)")
    print(f"{'='*65}")
    print(f"  API:         {API_STYLE} ({_ModelClass.__name__})")
    print(f"  Folds:       {len(fold_dirs)}")
    print(f"  Batch:       {batch_size}")
    print(f"  Max epochs:  {max_epochs}")
    print(f"  LR schedule: {expected_epochs} expected epochs "
          f"(warmup ~{expected_epochs*WARMUP_FRACTION:.0f}ep)")
    print(f"  LR:          {LEARNING_RATE} -> {LR_MIN}")
    print(f"  Early stop:  patience={PATIENCE}")
    print(f"  Gap halt:    >{GAP_THRESHOLD} for {GAP_GROWING_EPOCHS}ep")
    print(f"  AMP:         {'yes' if use_amp else 'no'}")
    print(f"  Compile:     {'yes' if use_compile else 'no'}")
    print(f"  Data:        preload f32, num_workers=0")
    print(f"{'='*65}")

    if dry_run:
        print(f"\n  DRY RUN:\n")
        for fd in fold_dirs:
            m = json.load(open(fd / "fold_manifest.json"))
            nt = m["sample_counts"]["train"]
            mem_mb = nt * 60 * m["feature_dim"] * 4 / 1024**2
            steps = nt // batch_size
            print(f"    {fd.name}: {nt:,} train, {steps} steps/ep, ~{mem_mb:.0f}MB RAM")
        return

    all_results = []
    for fold_dir in fold_dirs:
        result = train_fold(str(fold_dir), model_dir, log_dir, batch_size,
                            max_epochs, expected_epochs, device,
                            use_amp=use_amp, use_compile=use_compile)
        all_results.append(result)
        gc.collect()

    print(f"\n{'='*65}")
    print(f"  STAGE 5 COMPLETE")
    print(f"{'='*65}")
    for r in all_results:
        a = r["final_val_dir_acc"]
        print(f"  Fold {r['fold']}: ep {r['best_epoch']}/{r['total_epochs_run']}, "
              f"val={r['best_val_loss']:.6f}, acc=[{a[0]:.1f}, {a[1]:.1f}, {a[2]:.1f}]%, "
              f"{r['total_time_seconds']:.0f}s")
    if len(all_results) > 1:
        avg_loss = np.mean([r["best_val_loss"] for r in all_results])
        avg_acc = np.mean([r["final_val_dir_acc"] for r in all_results], axis=0)
        std_acc = np.std([r["final_val_dir_acc"] for r in all_results], axis=0)
        print(f"\n  Cross-fold averages:")
        print(f"    Val loss: {avg_loss:.6f}")
        for i, h in enumerate(FORWARD_HORIZONS):
            print(f"    H{h:2d}: {avg_acc[i]:.1f}% +/- {std_acc[i]:.1f}%")
    total_t = sum(r["total_time_seconds"] for r in all_results)
    print(f"\n  Total: {total_t:.0f}s ({total_t/60:.1f}min)")
    print(f"  Checkpoints: {Path(model_dir).resolve()}/")
    print(f"  Next: Stage 6 (Fine-Tuning) or Stage 7 (Evaluation)")
    print(f"{'='*65}\n")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Training (fast).")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--model-dir", type=str, default="models/base")
    parser.add_argument("--log-dir", type=str, default="logs/training")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--expected-epochs", type=int, default=EXPECTED_EPOCHS,
                        help=f"Expected training length for LR schedule (default: {EXPECTED_EPOCHS})")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cuda","mps","cpu"])
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (CUDA)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()
    run_stage5(splits_dir=args.splits_dir, model_dir=args.model_dir,
               log_dir=args.log_dir, batch_size=args.batch_size,
               max_epochs=args.max_epochs, expected_epochs=args.expected_epochs,
               device=args.device, fold_numbers=args.folds, dry_run=args.dry_run,
               use_amp=args.amp, use_compile=not args.no_compile)