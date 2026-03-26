"""
Stage 5: Training Loop (v5 — EMA + Augmentation)
==================================================
Key anti-overfitting mechanisms:
  1. EMA (Exponential Moving Average) of weights — the saved checkpoint
     uses averaged weights, not single-epoch weights. This smooths out
     noise-fitting across hundreds of gradient steps.
  2. Training-time augmentation — noise injection, feature masking,
     time masking make each epoch see different data.
  3. Early stopping monitors EMA validation loss, not raw model loss.

Usage:
    python stage5_training.py                # Train all folds
    python stage5_training.py --folds 1      # Single fold
    python stage5_training.py --dry-run      # Config only
    python stage5_training.py --no-compile   # Disable torch.compile
"""

import os, sys, gc, json, time, copy, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional

# ── Stage 4 Imports ──
_this_dir = Path(__file__).resolve().parent
for _p in [_this_dir, _this_dir.parent / "models"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from stage4_model import TradingModel as _ModelClass
except ImportError:
    from stage4_model import TradingTCN as _ModelClass

try:
    from stage4_model import TradingLoss as _LossClass
except ImportError:
    from stage4_model import CombinedLoss as _LossClass

try:
    from stage4_model import build_model as _build_fn
except ImportError:
    _build_fn = None

try:
    from stage4_model import FORWARD_HORIZONS
except ImportError:
    FORWARD_HORIZONS = [1, 5, 20]

API_STYLE = "v1" if _ModelClass.__name__ == "TradingModel" else "v2"


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 5e-4
BETAS = (0.9, 0.999)
GRAD_CLIP_NORM = 1.0
WARMUP_FRACTION = 0.02
LR_MIN = 1e-6
MAX_EPOCHS = 30               # Model peaks at ~10, no point going past 30
PATIENCE = 12                 # Stop 12 epochs after best
MIN_DELTA = 0.00002
GAP_THRESHOLD = 0.15
GAP_GROWING_EPOCHS = 8
BATCH_SIZE = 256
EXPECTED_EPOCHS = 25          # LR schedule matched to actual training length

# Augmentation
NOISE_STD = 0.15
FEAT_MASK_PROB = 0.20
TIME_MASK_PROB = 0.10

# EMA
EMA_DECAY = 0.9995


# ─────────────────────────────────────────────────────────
# Wrappers
# ─────────────────────────────────────────────────────────

def make_model_and_loss(n_features, device="auto"):
    if device == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    if _build_fn:
        result = _build_fn(n_features, device=device)
        if isinstance(result, tuple) and len(result) == 3: return result
        if isinstance(result, tuple): return result[0], result[1], device
    model = _ModelClass(n_features=n_features).to(device)
    loss_fn = _LossClass().to(device)
    return model, loss_fn, device

def model_forward(model, x):
    out = model(x)
    if isinstance(out, dict): return out["direction"], out["magnitude"]
    return out[0], out[1]

def compute_loss(loss_fn, pd, pm, td, tm):
    try:
        r = loss_fn(pd, pm, td, tm)
        if isinstance(r, tuple): return r[0], r[1], r[2]
        return r["total"], r["directional"], r["magnitude"]
    except TypeError:
        targets = torch.cat([td, tm], dim=1)
        r = loss_fn({"direction": pd, "magnitude": pm}, targets)
        if isinstance(r, dict): return r["total"], r["directional"], r["magnitude"]
        return r[0], r[1], r[2]

def get_param_count(model):
    r = model.count_parameters()
    return r.get("total", r) if isinstance(r, dict) else r


# ─────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ─────────────────────────────────────────────────────────

class EMAModel:
    """
    Maintains an exponential moving average of model parameters.
    
    After each optimizer step, call ema.update(model).
    For validation, call ema.apply(model) to load EMA weights,
    then ema.restore(model) to put original weights back.
    
    The EMA weights average over hundreds of gradient steps,
    smoothing out noise-fitting while retaining signal.
    """
    
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Call after each optimizer.step()"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )
    
    def apply(self, model):
        """Swap model weights with EMA weights for evaluation."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        """Get EMA weights for saving."""
        return {k: v.clone() for k, v in self.shadow.items()}


# ─────────────────────────────────────────────────────────
# Preloaded Dataset
# ─────────────────────────────────────────────────────────

class PreloadedDataset(Dataset):
    def __init__(self, fold_dir, split="train"):
        fp = Path(fold_dir)
        m = json.load(open(fp / "fold_manifest.json"))
        si = m["shards"].get(split, {})
        n_h = len(FORWARD_HORIZONS)
        af, ad, am = [], [], []
        for k in sorted(si.keys()):
            sp = fp / split / f"{k}.npz"
            if not sp.exists(): continue
            d = np.load(sp)
            af.append(d["features"]); ad.append(d["targets"][:,:n_h])
            am.append(d["targets"][:,n_h:]); del d
        if not af:
            self.features=torch.empty(0); self.dirs=torch.empty(0)
            self.mags=torch.empty(0); self.n_samples=0; return
        self.features = torch.from_numpy(np.concatenate(af)).float()
        self.dirs = torch.from_numpy(np.concatenate(ad)).float()
        self.mags = torch.from_numpy(np.concatenate(am)).float()
        self.n_samples = len(self.features)
        del af, ad, am; gc.collect()

    def __len__(self): return self.n_samples
    def __getitem__(self, i): return self.features[i], self.dirs[i], self.mags[i]


def preload_data(fold_dir, split, verbose=True):
    t0 = time.time()
    ds = PreloadedDataset(fold_dir, split)
    if verbose and ds.n_samples > 0:
        print(f"    {split:5s}: {ds.n_samples:,} samples, "
              f"{ds.features.nbytes//1024//1024} MB, {time.time()-t0:.1f}s")
    return ds


# ─────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────

def augment_batch(features):
    """Apply noise + masking to prevent memorisation."""
    # Gaussian noise
    if NOISE_STD > 0:
        features = features + torch.randn_like(features) * NOISE_STD
    # Feature masking
    if FEAT_MASK_PROB > 0:
        mask = torch.rand(features.shape[0], 1, features.shape[2],
                         device=features.device) > FEAT_MASK_PROB
        features = features * mask.float()
    # Time masking
    if TIME_MASK_PROB > 0:
        mask = torch.rand(features.shape[0], features.shape[1], 1,
                         device=features.device) > TIME_MASK_PROB
        features = features * mask.float()
    return features


# ─────────────────────────────────────────────────────────
# LR / Early Stopping / Gap Monitor
# ─────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, opt, expected_steps):
        self.opt = opt
        self.expected = max(expected_steps, 1)
        self.warmup = int(expected_steps * WARMUP_FRACTION)
        self.step_n = 0

    def step(self):
        self.step_n += 1
        lr = self.get_lr()
        for pg in self.opt.param_groups: pg["lr"] = lr

    def get_lr(self):
        if self.step_n <= self.warmup:
            return LEARNING_RATE * (self.step_n / max(self.warmup, 1))
        if self.step_n >= self.expected:
            return LR_MIN
        p = (self.step_n - self.warmup) / max(self.expected - self.warmup, 1)
        return LR_MIN + 0.5 * (LEARNING_RATE - LR_MIN) * (1 + math.cos(math.pi * p))


class EarlyStopping:
    def __init__(self):
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0

    def check(self, val_loss, epoch, ema_state):
        """Track best EMA weights based on EMA validation loss."""
        if val_loss < self.best_loss - MIN_DELTA:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = {k: v.clone() for k, v in ema_state.items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= PATIENCE


class GapMonitor:
    def __init__(self):
        self.h = []

    def check(self, train_loss, val_loss):
        g = val_loss - train_loss
        self.h.append(g)
        if g <= GAP_THRESHOLD or len(self.h) < GAP_GROWING_EPOCHS + 1:
            return False
        r = self.h[-(GAP_GROWING_EPOCHS+1):]
        return all(r[i+1] > r[i] for i in range(len(r)-1))


# ─────────────────────────────────────────────────────────
# Directional Accuracy
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def dir_acc(p, t):
    return [round(a.item()*100, 2) for a in ((p>0.5).float()==t).float().mean(dim=0)]


# ─────────────────────────────────────────────────────────
# Train / Validate
# ─────────────────────────────────────────────────────────

def train_one_epoch(model, loss_fn, opt, sched, loader, device, ema, scaler=None):
    model.train()
    ts, ds, ms, gs, n = 0., 0., 0., 0., 0
    ap, at = [], []
    use_amp = scaler is not None

    for feat, t_dir, t_mag in loader:
        feat = feat.to(device, non_blocking=True)
        t_dir = t_dir.to(device, non_blocking=True)
        t_mag = t_mag.to(device, non_blocking=True)

        # Augment
        feat = augment_batch(feat)

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                pd, pm = model_forward(model, feat)
                tl, dl, ml = compute_loss(loss_fn, pd, pm, t_dir, t_mag)
            scaler.scale(tl).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).item()
            scaler.step(opt); scaler.update()
        else:
            pd, pm = model_forward(model, feat)
            tl, dl, ml = compute_loss(loss_fn, pd, pm, t_dir, t_mag)
            tl.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).item()
            opt.step()

        # Update EMA after each step
        ema.update(model)

        sched.step()
        ts += tl.item(); ds += dl.item(); ms += ml.item(); gs += gn; n += 1
        ap.append(pd.detach().cpu()); at.append(t_dir.detach().cpu())

    n = max(n, 1)
    return {"total_loss": ts/n, "dir_loss": ds/n, "mag_loss": ms/n,
            "dir_accuracy": dir_acc(torch.cat(ap), torch.cat(at)),
            "grad_norm": gs/n, "lr": sched.get_lr(), "n_batches": n}


@torch.no_grad()
def validate(model, loss_fn, loader, device, use_amp=False):
    model.eval()
    ts, ds, ms, n = 0., 0., 0., 0
    ap, at = [], []
    for feat, t_dir, t_mag in loader:
        feat = feat.to(device, non_blocking=True)
        t_dir = t_dir.to(device, non_blocking=True)
        t_mag = t_mag.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                pd, pm = model_forward(model, feat)
                tl, dl, ml = compute_loss(loss_fn, pd, pm, t_dir, t_mag)
        else:
            pd, pm = model_forward(model, feat)
            tl, dl, ml = compute_loss(loss_fn, pd, pm, t_dir, t_mag)
        ts += tl.item(); ds += dl.item(); ms += ml.item(); n += 1
        ap.append(pd.cpu()); at.append(t_dir.cpu())
    n = max(n, 1)
    return {"total_loss": ts/n, "dir_loss": ds/n, "mag_loss": ms/n,
            "dir_accuracy": dir_acc(torch.cat(ap), torch.cat(at)), "n_batches": n}


# ─────────────────────────────────────────────────────────
# Train One Fold
# ─────────────────────────────────────────────────────────

def train_fold(fold_dir, model_save_dir="models/base", log_dir="logs/training",
               batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
               expected_epochs=EXPECTED_EPOCHS, device="auto",
               use_amp=False, use_compile=True, verbose=True):

    fp = Path(fold_dir)
    m = json.load(open(fp / "fold_manifest.json"))
    nf, nt, nv, fn = m["feature_dim"], m["sample_counts"]["train"], m["sample_counts"]["val"], m["fold"]

    if device == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"

    if verbose:
        print(f"\n{'~'*65}")
        print(f"  Training fold_{fn}")
        print(f"{'~'*65}")
        print(f"  Samples: train={nt:,}, val={nv:,}")
        print(f"  Preloading...")

    tds = preload_data(fold_dir, "train", verbose)
    vds = preload_data(fold_dir, "val", verbose)
    tl = DataLoader(tds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(vds, batch_size=batch_size, shuffle=False, num_workers=0)

    spe = len(tl)
    exp_steps = spe * expected_epochs

    if verbose:
        print(f"  Steps/ep: {spe}, EMA decay: {EMA_DECAY}")
        print(f"  Augment: noise={NOISE_STD}, feat_mask={FEAT_MASK_PROB}, time_mask={TIME_MASK_PROB}")

    model, loss_fn, device = make_model_and_loss(nf, device)

    compiled = False
    if use_compile and hasattr(torch, "compile"):
        try: model = torch.compile(model); compiled = True
        except: pass

    np_ = get_param_count(model)
    if verbose:
        print(f"  Params: {np_:,}, Device: {device}" + (", compiled" if compiled else ""))

    # Optimizer
    dp, ndp = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (ndp if "bias" in name or "norm" in name else dp).append(p)
    opt = torch.optim.AdamW([
        {"params": dp, "weight_decay": WEIGHT_DECAY},
        {"params": ndp, "weight_decay": 0.0},
    ], lr=LEARNING_RATE, betas=BETAS)

    sched = WarmupCosineScheduler(opt, exp_steps)
    ema = EMAModel(model, decay=EMA_DECAY)
    es = EarlyStopping()
    gm = GapMonitor()
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device == "cuda") else None

    log = {"fold": fn, "config": {
        "n_features": nf, "n_train": nt, "n_val": nv, "batch_size": batch_size,
        "max_epochs": max_epochs, "lr": LEARNING_RATE, "ema_decay": EMA_DECAY,
        "noise_std": NOISE_STD, "feat_mask": FEAT_MASK_PROB, "time_mask": TIME_MASK_PROB,
        "device": device, "n_parameters": np_}, "epochs": []}

    if verbose:
        hs = "/".join(f"h{h}" for h in FORWARD_HORIZONS)
        print(f"\n  {'Ep':>4s}  {'TrLoss':>8s}  {'EmaVal':>8s}  "
              f"{'Gap':>7s}  {'EmaAcc('+hs+')':>18s}  "
              f"{'GrdN':>6s}  {'LR':>10s}  {'Time':>6s} {'Note'}")
        print(f"  {'~'*86}")

    t0_all = time.time()
    final_ep = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        final_ep = epoch

        # Train (updates EMA internally)
        tm = train_one_epoch(model, loss_fn, opt, sched, tl, device, ema, scaler)

        # Validate using EMA weights
        ema.apply(model)
        vm = validate(model, loss_fn, vl, device, use_amp and device == "cuda")
        ema.restore(model)

        gap = vm["total_loss"] - tm["total_loss"]
        et = time.time() - t0

        log["epochs"].append({
            "epoch": epoch,
            "train_loss": round(tm["total_loss"], 6),
            "ema_val_loss": round(vm["total_loss"], 6),
            "gap": round(gap, 6),
            "train_dir_acc": tm["dir_accuracy"],
            "ema_val_dir_acc": vm["dir_accuracy"],
            "grad_norm": round(tm["grad_norm"], 4),
            "lr": round(tm["lr"], 8),
            "epoch_seconds": round(et, 1),
        })

        if verbose:
            va = vm["dir_accuracy"]
            ib = vm["total_loss"] < es.best_loss - MIN_DELTA
            note = "★" if ib else ""
            if tm["grad_norm"] > 5.0: note += " ⚠grd"
            print(f"  {epoch:4d}  {tm['total_loss']:8.5f}  "
                  f"{vm['total_loss']:8.5f}  {gap:+7.4f}  "
                  f"{va[0]:5.1f}/{va[1]:5.1f}/{va[2]:5.1f}  "
                  f"{tm['grad_norm']:6.3f}  {tm['lr']:.2e}  "
                  f"{et:5.1f}s {note}")

        # Early stopping tracks EMA validation loss and saves EMA state
        ss = es.check(vm["total_loss"], epoch, ema.shadow)
        gh = gm.check(tm["total_loss"], vm["total_loss"])

        if gh:
            log["stop_reason"] = f"Gap halt epoch {epoch}"
            if verbose: print(f"\n  ✗ Gap halt")
            break
        if ss:
            log["stop_reason"] = f"Early stop (best ep {es.best_epoch})"
            if verbose: print(f"\n  ■ Early stop (best: ep {es.best_epoch})")
            break
    else:
        log["stop_reason"] = f"Max epochs ({max_epochs})"

    tt = time.time() - t0_all

    # Load best EMA weights into model for final save
    if es.best_state:
        for name, param in model.named_parameters():
            if name in es.best_state:
                param.data.copy_(es.best_state[name])

    log["best_epoch"] = es.best_epoch
    log["best_val_loss"] = round(es.best_loss, 6)
    log["total_epochs"] = final_ep
    log["total_seconds"] = round(tt, 1)

    fv = validate(model, loss_fn, vl, device)
    log["final_val_loss"] = round(fv["total_loss"], 6)
    log["final_val_dir_acc"] = fv["dir_accuracy"]

    if verbose:
        a = fv["dir_accuracy"]
        print(f"\n  Best EMA: ep {es.best_epoch}, val={es.best_loss:.6f}")
        print(f"  Acc: h1={a[0]:.1f}%, h5={a[1]:.1f}%, h20={a[2]:.1f}%")
        print(f"  Time: {tt:.0f}s ({tt/max(final_ep,1):.1f}s/ep)")

    # Save checkpoint with EMA weights
    sd = Path(model_save_dir); sd.mkdir(parents=True, exist_ok=True)
    cp = sd / f"checkpoint_base_fold_{fn}.pt"
    state = model.state_dict()
    clean = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    _ref = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model_state_dict": clean, "n_features": nf,
                "hidden_dim": _ref.hidden_dim, "horizons": _ref.horizons,
                "best_epoch": es.best_epoch, "best_val_loss": es.best_loss,
                "fold": fn, "training_config": log["config"]}, cp)

    ld = Path(log_dir); ld.mkdir(parents=True, exist_ok=True)
    lp = ld / f"fold_{fn}_training_log.json"
    with open(lp, "w") as f: json.dump(log, f, indent=2)
    if verbose:
        print(f"  ✓ Checkpoint: {cp}\n  ✓ Log: {lp}")

    es.best_state = None; ema.shadow.clear()
    del model, loss_fn, opt, sched, ema, es, tds, vds, tl, vl
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return log


# ─────────────────────────────────────────────────────────
# Train All Folds
# ─────────────────────────────────────────────────────────

def run_stage5(splits_dir="data/splits", model_dir="models/base",
               log_dir="logs/training", batch_size=BATCH_SIZE,
               max_epochs=MAX_EPOCHS, expected_epochs=EXPECTED_EPOCHS,
               device="auto", fold_numbers=None, dry_run=False,
               use_amp=False, use_compile=True, retrain=False):

    fds = sorted(Path(splits_dir).glob("fold_*"))
    if not fds: print(f"\n  ✗ No folds in {splits_dir}."); return
    if fold_numbers:
        fds = [d for d in fds if int(d.name.split("_")[1]) in fold_numbers]

    # Optimise CPU
    nc = os.cpu_count() or 4
    torch.set_num_threads(nc)

    print(f"\n{'='*65}")
    print(f"  STAGE 5: Training (v5 — EMA + Augmentation)")
    print(f"{'='*65}")
    print(f"  API:         {API_STYLE} ({_ModelClass.__name__})")
    print(f"  Folds:       {len(fds)}")
    print(f"  Batch:       {batch_size}")
    print(f"  Max epochs:  {max_epochs}")
    print(f"  LR:          {LEARNING_RATE} → {LR_MIN}")
    print(f"  EMA decay:   {EMA_DECAY}")
    print(f"  Augment:     noise={NOISE_STD}, feat_mask={FEAT_MASK_PROB}, time_mask={TIME_MASK_PROB}")
    print(f"  Early stop:  patience={PATIENCE}")
    print(f"  Gap halt:    >{GAP_THRESHOLD} for {GAP_GROWING_EPOCHS}ep")
    print(f"  CPU cores:   {nc}")
    print(f"{'='*65}")

    if dry_run:
        print(f"\n  DRY RUN:\n")
        for fd in fds:
            mm = json.load(open(fd / "fold_manifest.json"))
            nt = mm["sample_counts"]["train"]; s = nt // batch_size
            ram = nt * 60 * mm["feature_dim"] * 4 / 1024**2
            fn = int(fd.name.split("_")[1])
            ckpt = Path(model_dir) / f"checkpoint_base_fold_{fn}.pt"
            status = " ✓ exists" if ckpt.exists() else ""
            print(f"    {fd.name}: {nt:,} train, {s} steps/ep, ~{ram:.0f}MB RAM{status}")
        return

    results = []
    skipped = []
    for fd in fds:
        fold_num = int(fd.name.split("_")[1])
        ckpt_path = Path(model_dir) / f"checkpoint_base_fold_{fold_num}.pt"
        if ckpt_path.exists() and not retrain:
            print(f"\n  ⊘ fold_{fold_num}: checkpoint exists ({ckpt_path.name}), skipping. "
                  f"Use --retrain to force.")
            skipped.append(fold_num)
            continue
        r = train_fold(str(fd), model_dir, log_dir, batch_size, max_epochs,
                       expected_epochs, device, use_amp, use_compile)
        results.append(r); gc.collect()

    print(f"\n{'='*65}")
    print(f"  STAGE 5 COMPLETE")
    print(f"{'='*65}")
    if skipped:
        print(f"  Skipped (existing checkpoints): {skipped}")
    for r in results:
        a = r["final_val_dir_acc"]
        print(f"  Fold {r['fold']}: ep {r['best_epoch']}/{r['total_epochs']}, "
              f"val={r['best_val_loss']:.6f}, acc=[{a[0]:.1f},{a[1]:.1f},{a[2]:.1f}]%, "
              f"{r['total_seconds']:.0f}s")

    if len(results) > 1:
        al = np.mean([r["best_val_loss"] for r in results])
        aa = np.mean([r["final_val_dir_acc"] for r in results], axis=0)
        sa = np.std([r["final_val_dir_acc"] for r in results], axis=0)
        print(f"\n  Averages: val={al:.6f}")
        for i, h in enumerate(FORWARD_HORIZONS):
            print(f"    H{h:2d}: {aa[i]:.1f}% ± {sa[i]:.1f}%")

    print(f"\n  Total: {sum(r['total_seconds'] for r in results):.0f}s")
    print(f"  Checkpoints: {Path(model_dir).resolve()}/")
    print(f"{'='*65}\n")
    return results


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Stage 5: Training (v5)")
    pa.add_argument("--splits-dir", default="data/splits")
    pa.add_argument("--model-dir", default="models/base")
    pa.add_argument("--log-dir", default="logs/training")
    pa.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    pa.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    pa.add_argument("--expected-epochs", type=int, default=EXPECTED_EPOCHS)
    pa.add_argument("--device", default="auto", choices=["auto","cuda","mps","cpu"])
    pa.add_argument("--folds", type=int, nargs="+", default=None)
    pa.add_argument("--dry-run", action="store_true")
    pa.add_argument("--amp", action="store_true")
    pa.add_argument("--no-compile", action="store_true")
    pa.add_argument("--retrain", action="store_true",
                    help="Force retrain folds that already have checkpoints")
    a = pa.parse_args()
    run_stage5(a.splits_dir, a.model_dir, a.log_dir, a.batch_size, a.max_epochs,
               a.expected_epochs, a.device, a.folds, a.dry_run, a.amp, not a.no_compile,
               a.retrain)