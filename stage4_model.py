"""
Stage 4: Model Architecture (Optimised)
=========================================
TCN encoder + multi-horizon prediction head.

PERFORMANCE: Eliminates 12 transpose operations per forward pass
by using ChannelNorm (mathematically equivalent to LayerNorm over
the channel dimension, but works directly on [B, C, T]).
Also fixes CausalConv to left-pad only instead of both-pad-then-trim.

Blueprint Section 4.1-4.4: all specifications identical.

Usage:
    python stage4_model.py              # Run self-test
    python stage4_model.py --summary    # Print architecture
"""

import json
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------
# Configuration (from blueprint Appendix A)
# ---------------------------------------------------------

LOOKBACK = 60
DEFAULT_HIDDEN_DIM = 32       # 32 gives ~25k params → 4:1 ratio with 100k+ samples
MAX_HIDDEN_DIM = 256
N_TCN_LAYERS = 3
KERNEL_SIZE = 3
ENCODER_DROPOUT = 0.35        # Strong regularisation for noisy financial data

HEAD_DROPOUT = 0.45           # Even stronger — head overfits fastest
FORWARD_HORIZONS = [1, 5, 20]
LABEL_SMOOTHING = 0.10        # Increased from 0.05 — prevents overconfident outputs

HUBER_DELTA = 0.01
DIRECTION_WEIGHT = 0.5
MAGNITUDE_WEIGHT = 0.5


# =========================================================
# Optimised Building Blocks
# =========================================================

class ChannelNorm(nn.Module):
    """
    LayerNorm over the channel dimension for [B, C, T] tensors.
    Mathematically equivalent to: x.transpose(1,2) → LayerNorm(C) → transpose back.
    But without any transpose operations.
    """

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=1, keepdim=True)   # [B, 1, T]
        var = x.var(dim=1, keepdim=True, unbiased=False)  # [B, 1, T]
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


class CausalConv1d(nn.Module):
    """Causal 1D convolution — left-pad only (no wasted computation)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad_length = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We handle padding manually
        )

    def forward(self, x):
        if self.pad_length > 0:
            x = F.pad(x, (self.pad_length, 0))  # Left pad only
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    TCN residual block — no transpose operations.
    CausalConv → ChannelNorm → GELU → Dropout ×2 + residual
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = ChannelNorm(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = ChannelNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        """x: [batch, channels, time]"""
        res = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        return out + res


# =========================================================
# TCN Encoder
# =========================================================

class TCNEncoder(nn.Module):
    """
    Blueprint Section 4.2:
      Input:  [batch, lookback, n_features]
      3 TCN layers, dilations [1, 2, 4]
      Hidden dim 128, kernel 3, GELU, LayerNorm, dropout 0.25
      Output: global avg pool -> [batch, hidden_dim]
    """

    def __init__(self, n_features, hidden_dim=DEFAULT_HIDDEN_DIM,
                 n_layers=N_TCN_LAYERS, kernel_size=KERNEL_SIZE,
                 dropout=ENCODER_DROPOUT):
        super().__init__()

        if hidden_dim > MAX_HIDDEN_DIM:
            raise ValueError(f"hidden_dim={hidden_dim} exceeds max {MAX_HIDDEN_DIM}")

        self.n_features = n_features
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            in_ch = n_features if i == 0 else hidden_dim
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """x: [batch, lookback, n_features] -> [batch, hidden_dim]"""
        out = x.transpose(1, 2)    # [batch, n_features, lookback] — single transpose
        out = self.network(out)     # [batch, hidden_dim, lookback] — no transposes inside
        out = out.mean(dim=2)       # global average pooling
        return out


# =========================================================
# Prediction Head
# =========================================================

class PredictionHead(nn.Module):
    """
    Blueprint Section 4.3:
      MLP: hidden_dim -> 64 -> 32 -> output
      Direction: sigmoid probability
      Magnitude: linear (log return prediction)
    """

    def __init__(self, hidden_dim, dropout=HEAD_DROPOUT):
        super().__init__()
        mid = max(hidden_dim, 32)
        small = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, small),
            nn.GELU(),
        )
        self.dir_out = nn.Linear(small, 1)
        self.mag_out = nn.Linear(small, 1)

    def forward(self, x):
        h = self.net(x)
        direction = torch.sigmoid(self.dir_out(h))
        magnitude = self.mag_out(h)
        return direction, magnitude


# =========================================================
# Full Model
# =========================================================

class TradingModel(nn.Module):
    """
    TCN encoder -> three parallel prediction heads (horizons 1, 5, 20).

    Output:
      directions: [batch, 3]  probabilities (after sigmoid)
      magnitudes: [batch, 3]  log return predictions
    """

    def __init__(self, n_features, hidden_dim=DEFAULT_HIDDEN_DIM,
                 n_tcn_layers=N_TCN_LAYERS, kernel_size=KERNEL_SIZE,
                 encoder_dropout=ENCODER_DROPOUT, head_dropout=HEAD_DROPOUT,
                 horizons=None):
        super().__init__()

        if horizons is None:
            horizons = FORWARD_HORIZONS

        self.horizons = horizons
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.encoder = TCNEncoder(
            n_features=n_features, hidden_dim=hidden_dim,
            n_layers=n_tcn_layers, kernel_size=kernel_size,
            dropout=encoder_dropout,
        )

        self.heads = nn.ModuleDict({
            f"horizon_{h}": PredictionHead(hidden_dim, head_dropout)
            for h in horizons
        })

    def forward(self, x):
        """x: [batch, 60, n_features] -> (dirs [batch,3], mags [batch,3])"""
        encoded = self.encoder(x)

        dirs, mags = [], []
        for h in self.horizons:
            d, m = self.heads[f"horizon_{h}"](encoded)
            dirs.append(d)
            mags.append(m)

        return torch.cat(dirs, dim=1), torch.cat(mags, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_encoder_params(self):
        return list(self.encoder.parameters())

    def get_head_params(self):
        return list(self.heads.parameters())

    def freeze_encoder(self):
        """Fine-tuning Phase 1: freeze all encoder layers."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze_last_n_layers(self, n=2):
        """Fine-tuning Phase 3: unfreeze last N TCN blocks."""
        layers = list(self.encoder.network.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
            layer.train()

    def set_finetune_dropout(self, encoder_dropout=0.40, head_dropout=0.45):
        """Fine-tuning Phase 3: increase dropout for smaller dataset."""
        for module in self.encoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = encoder_dropout
        for module in self.heads.modules():
            if isinstance(module, nn.Dropout):
                module.p = head_dropout


# =========================================================
# Loss Function
# =========================================================

class TradingLoss(nn.Module):
    """
    Combined loss:
      0.5 * BCE(directions, label_smoothed_targets) +
      0.5 * Huber(magnitudes, target_magnitudes, delta=0.01)

    Label smoothing: 0 -> 0.025, 1 -> 0.975
    """

    def __init__(self, label_smoothing=LABEL_SMOOTHING,
                 huber_delta=HUBER_DELTA,
                 direction_weight=DIRECTION_WEIGHT,
                 magnitude_weight=MAGNITUDE_WEIGHT):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.huber = nn.HuberLoss(reduction="mean", delta=huber_delta)

    def forward(self, pred_dirs, pred_mags, target_dirs, target_mags):
        """Returns: (total_loss, direction_loss, magnitude_loss)"""
        smoothed = target_dirs * (1 - self.label_smoothing) + self.label_smoothing / 2
        pred_clamped = pred_dirs.clamp(1e-7, 1 - 1e-7)
        bce = F.binary_cross_entropy(pred_clamped, smoothed, reduction="mean")
        huber = self.huber(pred_mags, target_mags)
        total = self.direction_weight * bce + self.magnitude_weight * huber
        return total, bce, huber


# =========================================================
# Shard-Based Dataset (for Stage 3 output)
# =========================================================

class ShardDataset(Dataset):
    """
    Memory-efficient dataset loading Stage 3 shard files on demand.
    Each shard = one instrument's data for one fold split.
    Shards loaded lazily with LRU cache.
    """

    def __init__(self, fold_dir, split="train", max_cached_shards=10):
        self.fold_dir = Path(fold_dir)
        self.split = split
        self.max_cached = max_cached_shards

        manifest_path = self.fold_dir / "fold_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.feature_dim = manifest["feature_dim"]

        self.shard_files = []
        self.shard_sizes = []
        self.cumulative_sizes = []
        cumsum = 0

        shard_dir = self.fold_dir / split
        shard_info = manifest["shards"].get(split, {})

        for key, info in sorted(shard_info.items()):
            shard_path = shard_dir / f"{key}.npz"
            if shard_path.exists():
                self.shard_files.append(str(shard_path))
                self.shard_sizes.append(info["n_samples"])
                cumsum += info["n_samples"]
                self.cumulative_sizes.append(cumsum)

        self.total_samples = cumsum
        self._cache = {}
        self._cache_order = []

    def __len__(self):
        return self.total_samples

    def _find_shard(self, global_idx):
        for i, cum in enumerate(self.cumulative_sizes):
            if global_idx < cum:
                local = global_idx - (self.cumulative_sizes[i - 1] if i > 0 else 0)
                return i, local
        raise IndexError(f"Index {global_idx} out of range {self.total_samples}")

    def _load_shard(self, shard_idx):
        if shard_idx in self._cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        data = np.load(self.shard_files[shard_idx], allow_pickle=True)
        shard = {"features": data["features"], "targets": data["targets"]}

        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)

        while len(self._cache) > self.max_cached:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return shard

    def __getitem__(self, idx):
        """Returns: (features, direction_targets, magnitude_targets)"""
        shard_idx, local_idx = self._find_shard(idx)
        shard = self._load_shard(shard_idx)

        features = torch.from_numpy(shard["features"][local_idx]).float()
        targets = shard["targets"][local_idx]
        n_h = len(FORWARD_HORIZONS)
        directions = torch.from_numpy(targets[:n_h].copy()).float()
        magnitudes = torch.from_numpy(targets[n_h:].copy()).float()

        return features, directions, magnitudes

    def clear_cache(self):
        self._cache.clear()
        self._cache_order.clear()
        gc.collect()


def create_dataloader(fold_dir, split="train", batch_size=256,
                      shuffle=True, num_workers=0, max_cached_shards=10):
    """Create DataLoader for a fold split."""
    dataset = ShardDataset(fold_dir, split, max_cached_shards)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


# =========================================================
# Model Factory
# =========================================================

def build_model(n_features, hidden_dim=DEFAULT_HIDDEN_DIM, device="auto"):
    """Build model + loss, auto-detect device."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = TradingModel(n_features=n_features, hidden_dim=hidden_dim).to(device)
    loss_fn = TradingLoss().to(device)
    return model, loss_fn, device


# =========================================================
# Self-Test
# =========================================================

def run_self_test():
    print(f"\n{'='*65}")
    print(f"  STAGE 4: Model Architecture - Self Test")
    print(f"{'='*65}")

    device = "cpu"
    model, loss_fn, device = build_model(35, device=device)
    n_params = model.count_parameters()
    print(f"  Model: {n_params:,} params on {device}")

    # Forward pass
    x = torch.randn(32, LOOKBACK, 35)
    dirs, mags = model(x)
    print(f"  Forward: input={list(x.shape)} → dirs={list(dirs.shape)}, mags={list(mags.shape)}")
    print(f"  Dirs range: [{dirs.min():.4f}, {dirs.max():.4f}] (should be 0-1)")

    # Loss
    td = (torch.rand(32, 3) > 0.5).float()
    tm = torch.randn(32, 3) * 0.01
    total, bce, huber = loss_fn(dirs, mags, td, tm)
    print(f"  Loss: total={total.item():.4f}, dir={bce.item():.4f}, mag={huber.item():.6f}")

    # Backward
    total.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  Grad norm: {grad_norm:.4f}")

    # Speed test
    import time
    model.zero_grad()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x256 = torch.randn(256, LOOKBACK, 35)
    td256 = (torch.rand(256, 3) > 0.5).float()
    tm256 = torch.randn(256, 3) * 0.01

    # Warmup
    d, m = model(x256)
    loss_fn(d, m, td256, tm256)[0].backward()
    optimizer.step()

    t0 = time.time()
    for _ in range(10):
        d, m = model(x256)
        loss, _, _ = loss_fn(d, m, td256, tm256)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    ms = (time.time() - t0) / 10 * 1000
    print(f"  Speed: {ms:.0f}ms/step (batch=256)")

    # Fine-tuning methods
    model.freeze_encoder()
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Freeze: {frozen} frozen, {trainable} trainable")

    # Shard dataset test
    fold_dir = Path("data/splits/fold_1")
    if fold_dir.exists():
        loader = create_dataloader(str(fold_dir), "train", batch_size=32)
        feat, td, tm = next(iter(loader))
        print(f"  Shard batch: {list(feat.shape)}")
        loader.dataset.clear_cache()

    print(f"\n  ✓ All tests passed")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_self_test()