"""
Stage 2: Feature Engineering & Normalisation
=============================================
Transforms raw OHLCV data from Stage 1 into model-ready feature arrays.

Blueprint Sections 2.1–2.5:
  2.1  No raw prices — everything is returns, ratios, or z-scores.
  2.2  OHLC-derived features (8 features).
  2.3  Technical indicator features (19 features).
  2.4  Calendar & session features (up to 11 features).
  2.5  Normalisation pipeline (rolling z-score → clip → stack).

Usage:
    python stage2_feature_engineering.py                    # Process all Stage 1 outputs
    python stage2_feature_engineering.py --demo             # Run on synthetic demo data
    python stage2_feature_engineering.py --input FILE.parquet  # Process a single file

Output:
    data/processed/{instrument}_{resolution}_features.npz
    Each .npz contains:
      - features:  np.array  [n_samples, lookback(60), n_features]
      - targets:   np.array  [n_samples, 6]  (direction×3 + magnitude×3 for horizons 1,5,20)
      - timestamps: np.array [n_samples]     (timestamp of each sample's final bar)
      - regimes:   np.array  [n_samples]     (regime label for each sample)
      - feature_names: np.array              (ordered list of feature names)
"""

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────
# Constants (from blueprint)
# ─────────────────────────────────────────────────────────

LOOKBACK_WINDOW = 60        # Bars per training sample
ZSCORE_WINDOW = 60          # Rolling z-score window (backward-looking)
ZSCORE_CLIP = 3.0           # Clip z-scores to [-3, +3]
EPSILON = 1e-10             # Prevent division by zero
FORWARD_HORIZONS = [1, 5, 20]  # Prediction horizons in bars

# ATR periods for volatility features
ATR_PERIODS = [7, 14, 28]

# RSI periods
RSI_PERIODS = [14, 28]

# Rate of change lookback periods
ROC_PERIODS = [5, 10, 20, 60]

# Bollinger Band / SMA periods
BB_PERIOD = 20
SMA_PERIODS = [20, 50]

# Rolling correlation window
CORR_WINDOW = 20

# Range percentile window
RANGE_PERCENTILE_WINDOW = 60


# ─────────────────────────────────────────────────────────
# Section 2.2: OHLC-Derived Features
# ─────────────────────────────────────────────────────────

def compute_ohlc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 8 OHLC-derived features. All dimensionless.

    Features:
      1. log_return       — ln(close[t] / close[t-1])
      2. body_ratio       — (close - open) / (high - low + ε), range [-1, +1]
      3. upper_wick_ratio — upper rejection signal
      4. lower_wick_ratio — lower support signal
      5. range_zscore     — (high - low) z-scored over 20 bars
      6. volume_zscore    — volume z-scored over 20 bars
      7. volume_delta     — volume[t] / volume[t-1] - 1
      8. gap              — (open[t] - close[t-1]) / close[t-1]
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    feats = pd.DataFrame(index=df.index)

    # 1. Log return
    feats["log_return"] = np.log(c / c.shift(1))

    # 2. Body ratio
    bar_range = h - l + EPSILON
    feats["body_ratio"] = (c - o) / bar_range

    # 3. Upper wick ratio
    max_oc = pd.concat([o, c], axis=1).max(axis=1)
    feats["upper_wick_ratio"] = (h - max_oc) / bar_range

    # 4. Lower wick ratio
    min_oc = pd.concat([o, c], axis=1).min(axis=1)
    feats["lower_wick_ratio"] = (min_oc - l) / bar_range

    # 5. Range z-score (20-bar rolling)
    raw_range = h - l
    range_mean = raw_range.rolling(window=20, min_periods=20).mean()
    range_std = raw_range.rolling(window=20, min_periods=20).std()
    feats["range_zscore"] = (raw_range - range_mean) / (range_std + EPSILON)

    # 6. Volume z-score (20-bar rolling)
    vol_mean = v.rolling(window=20, min_periods=20).mean()
    vol_std = v.rolling(window=20, min_periods=20).std()
    feats["volume_zscore"] = (v - vol_mean) / (vol_std + EPSILON)

    # 7. Volume delta
    feats["volume_delta"] = v / (v.shift(1) + EPSILON) - 1

    # 8. Gap
    feats["gap"] = (o - c.shift(1)) / (c.shift(1) + EPSILON)

    return feats


# ─────────────────────────────────────────────────────────
# Section 2.3: Technical Indicator Features
# ─────────────────────────────────────────────────────────

def compute_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    """Average True Range."""
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index, normalised to [0, 1]."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + EPSILON)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi / 100.0  # Normalise to [0, 1]


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_macd_histogram(close: pd.Series, atr14: pd.Series) -> pd.Series:
    """
    MACD histogram normalised by ATR(14).
    MACD = EMA(12) - EMA(26); Signal = EMA(9) of MACD.
    Histogram = MACD - Signal, then / ATR(14).
    """
    ema12 = compute_ema(close, 12)
    ema26 = compute_ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = compute_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return histogram / (atr14 + EPSILON)


def compute_bollinger_position(close: pd.Series, period: int = BB_PERIOD) -> pd.Series:
    """
    Bollinger Band position: (close - BB_lower) / (BB_upper - BB_lower).
    Ranges ~0 to ~1, values outside indicate breakout.
    """
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    return (close - bb_lower) / (bb_upper - bb_lower + EPSILON)


def compute_sma_distance(close: pd.Series, atr14: pd.Series, period: int) -> pd.Series:
    """Distance from SMA, normalised by ATR(14). Signed."""
    sma = close.rolling(window=period, min_periods=period).mean()
    return (close - sma) / (atr14 + EPSILON)


def compute_range_percentile(h: pd.Series, l: pd.Series,
                             window: int = RANGE_PERCENTILE_WINDOW) -> pd.Series:
    """
    Where the current bar's range sits in the distribution of the
    last `window` bars' ranges. Percentile in [0, 1].
    """
    bar_range = h - l
    result = bar_range.copy() * np.nan

    for i in range(window, len(bar_range)):
        past_ranges = bar_range.iloc[i - window:i]
        current = bar_range.iloc[i]
        percentile = (past_ranges < current).sum() / window
        result.iloc[i] = percentile

    return result


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 19 technical indicator features. All dimensionless.

    Volatility (4):
      atr7_norm, atr14_norm, atr28_norm  — ATR / close
      atr_ratio                          — ATR(7) / ATR(28)

    Momentum (7):
      rsi14, rsi28                       — RSI normalised to [0,1]
      macd_histogram                     — normalised by ATR(14)
      roc_5, roc_10, roc_20, roc_60      — log return over N bars

    Mean Reversion (3):
      bb_position                        — Bollinger Band position [0,1]
      sma20_distance                     — distance from SMA20 / ATR(14)
      sma50_distance                     — distance from SMA50 / ATR(14)

    Context (2):
      range_percentile                   — current range percentile [0,1]
      (rolling_corr_index skipped — requires index data alignment;
       added as placeholder with zeros, replaced in cross-instrument step)

    Total: 16 computed + 1 placeholder = 17
    Note: rolling_corr_index is computed separately when index data is available.
    """
    h, l, c = df["high"], df["low"], df["close"]

    feats = pd.DataFrame(index=df.index)

    # ── Volatility features ──
    atr7 = compute_atr(h, l, c, 7)
    atr14 = compute_atr(h, l, c, 14)
    atr28 = compute_atr(h, l, c, 28)

    feats["atr7_norm"] = atr7 / (c + EPSILON)
    feats["atr14_norm"] = atr14 / (c + EPSILON)
    feats["atr28_norm"] = atr28 / (c + EPSILON)
    feats["atr_ratio"] = atr7 / (atr28 + EPSILON)

    # ── Momentum features ──
    feats["rsi14"] = compute_rsi(c, 14)
    feats["rsi28"] = compute_rsi(c, 28)
    feats["macd_histogram"] = compute_macd_histogram(c, atr14)

    for period in ROC_PERIODS:
        feats[f"roc_{period}"] = np.log(c / (c.shift(period) + EPSILON))

    # ── Mean reversion features ──
    feats["bb_position"] = compute_bollinger_position(c, BB_PERIOD)
    feats["sma20_distance"] = compute_sma_distance(c, atr14, 20)
    feats["sma50_distance"] = compute_sma_distance(c, atr14, 50)

    # ── Context features ──
    feats["range_percentile"] = compute_range_percentile(h, l, RANGE_PERCENTILE_WINDOW)

    # Placeholder for rolling correlation to index
    # Will be filled in if index data is available; otherwise stays 0
    feats["rolling_corr_index"] = 0.0

    return feats


# ─────────────────────────────────────────────────────────
# Section 2.4: Calendar & Session Features
# ─────────────────────────────────────────────────────────

def compute_calendar_features(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """
    Compute calendar and session features using sine/cosine encoding.

    Always computed (4 features):
      dow_sin, dow_cos         — day of week (0=Mon, 4=Fri)
      month_sin, month_cos     — month of year (1–12)

    Intraday only — 1H, 4H (2 features):
      hour_sin, hour_cos       — hour of day (0–23)

    Forex only — intraday (4 features):
      session_asian, session_london, session_newyork, session_overlap

    Always (1 feature):
      quarter_end_flag         — 1 if within 5 trading days of quarter end
    """
    times = pd.to_datetime(df["time"])
    feats = pd.DataFrame(index=df.index)

    # Day of week (Monday=0 to Friday=4)
    dow = times.dt.dayofweek  # 0–6, but trading data is mostly 0–4
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Month of year
    month = times.dt.month
    feats["month_sin"] = np.sin(2 * np.pi * month / 12)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Hour of day (intraday only)
    is_intraday = resolution in ("1H", "4H")
    if is_intraday:
        hour = times.dt.hour
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Session flags (forex intraday only)
    # Detect if this is forex by checking the exchange/ticker
    is_forex = False
    if "exchange" in df.columns:
        is_forex = df["exchange"].iloc[0] in ("FOREXCOM", "FX_IDC", "OANDA", "FX")
    elif "instrument" in df.columns:
        is_forex = "FOREXCOM" in df["instrument"].iloc[0]

    if is_intraday and is_forex:
        hour = times.dt.hour
        feats["session_asian"] = ((hour >= 0) & (hour < 8)).astype(float)
        feats["session_london"] = ((hour >= 8) & (hour < 16)).astype(float)
        feats["session_newyork"] = ((hour >= 13) & (hour < 21)).astype(float)
        feats["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)

    # Quarter-end flag: 1 if within 5 trading days of quarter end
    quarter_ends = pd.to_datetime(
        [f"{y}-{m:02d}-{d}" for y in range(times.dt.year.min(), times.dt.year.max() + 1)
         for m, d in [(3, 31), (6, 30), (9, 30), (12, 31)]]
    )

    feats["quarter_end_flag"] = 0.0
    for qe in quarter_ends:
        # Within 5 trading days before quarter end
        mask = (times >= qe - pd.Timedelta(days=9)) & (times <= qe)
        feats.loc[mask, "quarter_end_flag"] = 1.0

    return feats


# ─────────────────────────────────────────────────────────
# Section 2.5: Normalisation Pipeline
# ─────────────────────────────────────────────────────────

def apply_rolling_zscore(features_df: pd.DataFrame,
                         calendar_cols: List[str],
                         window: int = ZSCORE_WINDOW,
                         clip: float = ZSCORE_CLIP) -> pd.DataFrame:
    """
    Apply rolling z-score normalisation to all features EXCEPT calendar/session
    features (which are already in normalised ranges by design).

    Blueprint Section 2.5:
      1. For each feature, compute rolling z-score with backward-looking window.
      2. Clip all z-scores to [-3, +3].
      3. Calendar features pass through unchanged.

    CRITICAL: Window uses only past bars (no look-ahead leakage).
    """
    result = pd.DataFrame(index=features_df.index)

    for col in features_df.columns:
        if col in calendar_cols:
            # Calendar/session features pass through unchanged
            result[col] = features_df[col]
        else:
            series = features_df[col]
            roll_mean = series.rolling(window=window, min_periods=window).mean()
            roll_std = series.rolling(window=window, min_periods=window).std()

            z = (series - roll_mean) / (roll_std + EPSILON)
            z = z.clip(-clip, clip)
            result[col] = z

    return result


def create_training_samples(normalised_df: pd.DataFrame,
                            df_original: pd.DataFrame,
                            lookback: int = LOOKBACK_WINDOW,
                            horizons: List[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack normalised features into 3D training arrays.

    Each sample: [lookback, n_features] — the preceding `lookback` bars.

    Targets (per blueprint Section 4.3):
      - Directional: 1 if close[t+horizon] > close[t], else 0
      - Magnitude: log return over horizon = ln(close[t+h] / close[t])

    For 3 horizons (1, 5, 20), targets shape: [n_samples, 6]
      columns: [dir_1, dir_5, dir_20, mag_1, mag_5, mag_20]

    Returns:
      features:   [n_samples, lookback, n_features]
      targets:    [n_samples, 6]
      timestamps: [n_samples] — timestamp of each sample's final bar
      regimes:    [n_samples] — regime label string
    """
    if horizons is None:
        horizons = FORWARD_HORIZONS

    max_horizon = max(horizons)
    close = df_original["close"].values
    n_bars = len(normalised_df)
    n_features = normalised_df.shape[1]
    feat_values = normalised_df.values

    # Determine valid range: need `lookback` bars behind and `max_horizon` bars ahead
    # Also need the rolling z-score window behind the lookback
    start_idx = lookback + ZSCORE_WINDOW  # Ensure all rolling stats are computed
    end_idx = n_bars - max_horizon  # Ensure forward targets exist

    if end_idx <= start_idx:
        warnings.warn(f"Insufficient data: need at least "
                       f"{lookback + ZSCORE_WINDOW + max_horizon} bars, have {n_bars}.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    samples = []
    targets = []
    timestamps = []
    regimes = []

    # Get regime labels if available
    has_regimes = "regime" in df_original.columns
    regime_values = df_original["regime"].values if has_regimes else None

    # Get timestamps
    time_values = pd.to_datetime(df_original["time"]).values

    for i in range(start_idx, end_idx):
        # Extract the lookback window: bars [i-lookback, i) — NOT including bar i
        window_start = i - lookback
        window_end = i
        sample = feat_values[window_start:window_end]  # shape [lookback, n_features]

        # Check for NaN/Inf in this sample
        if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
            continue

        # Compute targets at bar i
        current_close = close[i]
        target_row = []

        valid_target = True
        for h in horizons:
            future_idx = i + h
            if future_idx >= n_bars:
                valid_target = False
                break
            future_close = close[future_idx]

            # Directional: 1 if future > current
            direction = 1.0 if future_close > current_close else 0.0

            # Magnitude: log return
            magnitude = np.log(future_close / (current_close + EPSILON))

            target_row.append(direction)
            target_row.append(magnitude)

        if not valid_target:
            continue

        samples.append(sample)
        # Reorder to [dir_1, dir_5, dir_20, mag_1, mag_5, mag_20]
        # Currently it's [dir_1, mag_1, dir_5, mag_5, dir_20, mag_20]
        dirs = [target_row[j * 2] for j in range(len(horizons))]
        mags = [target_row[j * 2 + 1] for j in range(len(horizons))]
        targets.append(dirs + mags)

        timestamps.append(time_values[i])
        if has_regimes:
            regimes.append(regime_values[i])
        else:
            regimes.append("unknown")

    features_array = np.array(samples, dtype=np.float32)
    targets_array = np.array(targets, dtype=np.float32)
    timestamps_array = np.array(timestamps)
    regimes_array = np.array(regimes)

    return features_array, targets_array, timestamps_array, regimes_array


# ─────────────────────────────────────────────────────────
# Rolling Correlation to Index (Context Feature)
# ─────────────────────────────────────────────────────────

def compute_rolling_correlation_to_index(
    instrument_returns: pd.Series,
    index_returns: pd.Series,
    window: int = CORR_WINDOW
) -> pd.Series:
    """
    Compute rolling correlation between an instrument's log returns
    and a broad index's log returns.

    If index data is not available, returns zeros.
    """
    if index_returns is None or len(index_returns) == 0:
        return pd.Series(0.0, index=instrument_returns.index)

    # Align the two series by index (timestamp)
    aligned = pd.DataFrame({
        "inst": instrument_returns,
        "idx": index_returns,
    }).dropna()

    if len(aligned) < window:
        return pd.Series(0.0, index=instrument_returns.index)

    corr = aligned["inst"].rolling(window=window, min_periods=window).corr(aligned["idx"])

    # Reindex back to original index, fill missing with 0
    return corr.reindex(instrument_returns.index).fillna(0.0)


# ─────────────────────────────────────────────────────────
# Full Feature Engineering Pipeline
# ─────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame,
                      resolution: str = None,
                      index_returns: pd.Series = None,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray,
                                                      List[str]]:
    """
    Run the complete feature engineering pipeline for one instrument.

    Steps:
      1. Compute OHLC features (Section 2.2)
      2. Compute technical features (Section 2.3)
      3. Compute calendar features (Section 2.4)
      4. Fill rolling correlation if index data available
      5. Apply rolling z-score normalisation (Section 2.5)
      6. Create windowed training samples with targets

    Args:
      df: DataFrame from Stage 1 with OHLCV + regime labels
      resolution: "1H", "4H", "1D", "1W" (auto-detected if column exists)
      index_returns: Optional Series of index log returns for correlation
      verbose: Print progress

    Returns:
      features:      [n_samples, 60, n_features]
      targets:       [n_samples, 6]
      timestamps:    [n_samples]
      regimes:       [n_samples]
      feature_names: list of feature column names
    """
    if resolution is None:
        if "resolution" in df.columns:
            resolution = df["resolution"].iloc[0]
        else:
            resolution = "1D"  # Default assumption

    instrument_name = "unknown"
    if "instrument" in df.columns:
        instrument_name = df["instrument"].iloc[0]

    if verbose:
        print(f"\n  Processing {instrument_name} @ {resolution} ({len(df)} bars)...")

    # ── Step 1: OHLC features ──
    if verbose:
        print(f"    Step 1: OHLC features...")
    ohlc_feats = compute_ohlc_features(df)

    # ── Step 2: Technical features ──
    if verbose:
        print(f"    Step 2: Technical indicator features...")
    tech_feats = compute_technical_features(df)

    # ── Step 2b: Fill rolling correlation if index data available ──
    if index_returns is not None:
        log_returns = np.log(df["close"] / df["close"].shift(1))
        tech_feats["rolling_corr_index"] = compute_rolling_correlation_to_index(
            log_returns, index_returns, CORR_WINDOW
        )

    # ── Step 3: Calendar features ──
    if verbose:
        print(f"    Step 3: Calendar & session features...")
    cal_feats = compute_calendar_features(df, resolution)

    # ── Combine all features ──
    all_feats = pd.concat([ohlc_feats, tech_feats, cal_feats], axis=1)
    feature_names = list(all_feats.columns)
    calendar_cols = list(cal_feats.columns)

    if verbose:
        print(f"    Total features: {len(feature_names)}")
        print(f"      OHLC:      {ohlc_feats.shape[1]}")
        print(f"      Technical: {tech_feats.shape[1]}")
        print(f"      Calendar:  {cal_feats.shape[1]}")

    # ── Step 4: Rolling z-score normalisation ──
    if verbose:
        print(f"    Step 4: Rolling z-score normalisation "
              f"(window={ZSCORE_WINDOW}, clip=±{ZSCORE_CLIP})...")
    normalised = apply_rolling_zscore(all_feats, calendar_cols,
                                      window=ZSCORE_WINDOW, clip=ZSCORE_CLIP)

    # ── Step 5: Create training samples ──
    if verbose:
        print(f"    Step 5: Creating training samples "
              f"(lookback={LOOKBACK_WINDOW}, horizons={FORWARD_HORIZONS})...")
    features, targets, timestamps, regimes = create_training_samples(
        normalised, df, lookback=LOOKBACK_WINDOW, horizons=FORWARD_HORIZONS
    )

    if verbose:
        if len(features) > 0:
            print(f"    ✓ Output shape: features={features.shape}, targets={targets.shape}")
            print(f"      Samples: {len(features)}")
            print(f"      Feature vector: {features.shape[1]} bars × {features.shape[2]} features")

            # Quick sanity checks
            nan_count = np.isnan(features).sum()
            inf_count = np.isinf(features).sum()
            if nan_count > 0:
                print(f"    ⚠ WARNING: {nan_count} NaN values in features!")
            if inf_count > 0:
                print(f"    ⚠ WARNING: {inf_count} Inf values in features!")
            if nan_count == 0 and inf_count == 0:
                print(f"    ✓ No NaN or Inf values — clean output")

            # Feature statistics
            feat_means = features.reshape(-1, features.shape[2]).mean(axis=0)
            feat_stds = features.reshape(-1, features.shape[2]).std(axis=0)
            print(f"    Feature means range: [{feat_means.min():.4f}, {feat_means.max():.4f}]")
            print(f"    Feature stds range:  [{feat_stds.min():.4f}, {feat_stds.max():.4f}]")

            # Target balance
            for i, h in enumerate(FORWARD_HORIZONS):
                up_pct = targets[:, i].mean() * 100
                print(f"    Horizon {h:2d} direction: {up_pct:.1f}% up / "
                      f"{100 - up_pct:.1f}% down")
        else:
            print(f"    ⚠ No valid samples generated!")

    return features, targets, timestamps, regimes, feature_names


# ─────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────

def run_stage2(input_dir: str = "data/processed",
               output_dir: str = "data/processed",
               file_pattern: str = "*_labelled.parquet",
               single_file: str = None):
    """
    Run Stage 2 on all labelled parquet files from Stage 1.

    Saves .npz files with features, targets, timestamps, regimes, feature_names.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if single_file:
        parquet_files = [Path(single_file)]
    else:
        parquet_files = sorted(in_path.glob(file_pattern))

    if not parquet_files:
        print(f"\n✗ No labelled parquet files found matching '{file_pattern}' in {in_path}")
        print(f"  Run Stage 1 first: python stage1_data_collection.py")
        return

    print(f"\n{'='*65}")
    print(f"  STAGE 2: Feature Engineering & Normalisation")
    print(f"{'='*65}")
    print(f"  Input files: {len(parquet_files)}")
    print(f"  Lookback window: {LOOKBACK_WINDOW} bars")
    print(f"  Z-score window: {ZSCORE_WINDOW} bars (backward-looking)")
    print(f"  Z-score clip: ±{ZSCORE_CLIP}")
    print(f"  Forward horizons: {FORWARD_HORIZONS} bars")
    print(f"{'='*65}")

    total_samples = 0
    results = []

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)

        # Extract instrument and resolution from the data
        instrument = df["instrument"].iloc[0] if "instrument" in df.columns else "unknown"
        resolution = df["resolution"].iloc[0] if "resolution" in df.columns else "1D"

        # Run feature engineering
        features, targets, timestamps, regimes, feature_names = engineer_features(
            df, resolution=resolution, index_returns=None, verbose=True
        )

        if len(features) == 0:
            print(f"    ⚠ Skipping {parquet_file.name} — no valid samples")
            continue

        # Save as .npz
        save_name = f"{instrument}_{resolution}_features.npz"
        save_path = out_path / save_name

        np.savez_compressed(
            save_path,
            features=features,
            targets=targets,
            timestamps=timestamps,
            regimes=regimes,
            feature_names=np.array(feature_names),
        )

        total_samples += len(features)
        results.append({
            "file": save_name,
            "instrument": instrument,
            "resolution": resolution,
            "n_samples": len(features),
            "n_features": features.shape[2],
            "shape": features.shape,
        })

        print(f"    ✓ Saved: {save_path}")

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  STAGE 2 COMPLETE")
    print(f"{'='*65}")
    print(f"  Files processed: {len(results)}")
    print(f"  Total samples:   {total_samples}")

    if results:
        n_feat = results[0]["n_features"]
        print(f"  Feature vector:  {LOOKBACK_WINDOW} bars × {n_feat} features")
        print(f"  Target vector:   {len(FORWARD_HORIZONS) * 2} "
              f"(direction + magnitude × {len(FORWARD_HORIZONS)} horizons)")

        print(f"\n  Feature names ({n_feat} total):")
        for i, name in enumerate(feature_names):
            print(f"    {i+1:3d}. {name}")

    print(f"\n  Output directory: {out_path.resolve()}")
    print(f"\n  Next step: Stage 3 — Walk-Forward Validation splits")
    print(f"{'='*65}\n")

    return results


# ─────────────────────────────────────────────────────────
# Demo Mode
# ─────────────────────────────────────────────────────────

def run_demo():
    """
    Run Stage 2 on the demo data generated by Stage 1's --demo mode.
    If no demo data exists, generate it first.
    """
    processed_dir = Path("data/processed")
    parquet_files = list(processed_dir.glob("*_labelled.parquet"))

    if not parquet_files:
        print("No Stage 1 data found. Running Stage 1 demo first...\n")
        from stage1_data_collection import run_demo as stage1_demo
        stage1_demo()
        parquet_files = list(processed_dir.glob("*_labelled.parquet"))

    if not parquet_files:
        print("✗ Could not generate Stage 1 demo data.")
        return

    # Process only a subset for demo speed
    demo_files = parquet_files[:4]
    print(f"\nRunning Stage 2 demo on {len(demo_files)} files...")

    results = []
    total_samples = 0

    print(f"\n{'='*65}")
    print(f"  STAGE 2 DEMO: Feature Engineering & Normalisation")
    print(f"{'='*65}")

    for pf in demo_files:
        df = pd.read_parquet(pf)
        instrument = df["instrument"].iloc[0] if "instrument" in df.columns else "unknown"
        resolution = df["resolution"].iloc[0] if "resolution" in df.columns else "1D"

        features, targets, timestamps, regimes, feature_names = engineer_features(
            df, resolution=resolution, verbose=True
        )

        if len(features) > 0:
            save_name = f"{instrument}_{resolution}_features.npz"
            save_path = processed_dir / save_name
            np.savez_compressed(
                save_path,
                features=features,
                targets=targets,
                timestamps=timestamps,
                regimes=regimes,
                feature_names=np.array(feature_names),
            )
            total_samples += len(features)
            results.append({"file": save_name, "samples": len(features),
                            "shape": features.shape})
            print(f"    ✓ Saved: {save_path}")

    print(f"\n{'='*65}")
    print(f"  DEMO COMPLETE — {total_samples} total samples across {len(results)} files")
    print(f"{'='*65}\n")

    return results


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Feature engineering and normalisation pipeline."
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run on demo/synthetic data")
    parser.add_argument("--input", type=str, default=None,
                        help="Process a single parquet file")
    parser.add_argument("--input-dir", type=str, default="data/processed",
                        help="Input directory with labelled parquet files")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for feature arrays")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.input:
        run_stage2(single_file=args.input)
    else:
        run_stage2(input_dir=args.input_dir, output_dir=args.output_dir)