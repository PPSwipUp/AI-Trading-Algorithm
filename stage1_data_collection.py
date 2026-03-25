"""
Stage 1: Data Collection & Instrument Diversity
================================================
- Loads raw TradingView CSV exports from data/raw/
- Validates file naming, columns, row counts, and gap detection
- Labels each 60-bar window with trend regime and volatility regime
- Monitors regime class balance and flags imbalances
- Saves a master catalogue (data/processed/instrument_catalogue.csv)

Expected CSV naming convention:
    {EXCHANGE}_{TICKER}_{RESOLUTION}_{STARTDATE}_{ENDDATE}.csv
    e.g. NYSE_AAPL_1D_20150101_20241231.csv

Expected CSV columns (TradingView export):
    time, open, high, low, close, volume
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

VALID_RESOLUTIONS = {"1H", "4H", "1D", "1W"}
REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "volume"}
MIN_HISTORY_YEARS = 5  # Exclude instruments below this from base training
PREFERRED_HISTORY_YEARS = 8  # Blueprint minimum
LOOKBACK_WINDOW = 60  # Bars per training sample
ATR_PERIOD = 14  # For regime labelling

# Expected approximate row counts per year by resolution
APPROX_ROWS_PER_YEAR = {
    "1H": 252 * 7,    # ~1764 trading hours/year (equities), more for forex
    "4H": 252 * 2,    # ~504
    "1D": 252,         # ~252 trading days
    "1W": 52,          # ~52 weeks
}

# Regime thresholds (from blueprint Section 1.5)
TREND_ATR_MULTIPLIER = 1.5
VOL_LOW_THRESHOLD = 0.01
VOL_HIGH_THRESHOLD = 0.025
MAX_REGIME_CLASS_SHARE = 0.40


# ─────────────────────────────────────────────────────────
# File Naming Parser
# ─────────────────────────────────────────────────────────

def parse_filename(filename: str) -> dict:
    """
    Parse a TradingView CSV filename into its components.
    
    Expected format: {EXCHANGE}_{TICKER}_{RESOLUTION}_{STARTDATE}_{ENDDATE}.csv
    
    Returns dict with keys: exchange, ticker, resolution, start_date, end_date
    Raises ValueError if the filename does not match the expected pattern.
    """
    name = Path(filename).stem
    # Pattern: EXCHANGE_TICKER_RESOLUTION_YYYYMMDD_YYYYMMDD
    # EXCHANGE and TICKER can contain letters/numbers
    pattern = r'^([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d{8})_(\d{8})$'
    match = re.match(pattern, name)
    
    if not match:
        raise ValueError(
            f"Filename '{filename}' does not match expected pattern: "
            f"{{EXCHANGE}}_{{TICKER}}_{{RESOLUTION}}_{{STARTDATE}}_{{ENDDATE}}.csv"
        )
    
    exchange, ticker, resolution, start_str, end_str = match.groups()
    
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(
            f"Resolution '{resolution}' in '{filename}' is not valid. "
            f"Expected one of: {VALID_RESOLUTIONS}"
        )
    
    try:
        start_date = datetime.strptime(start_str, "%Y%m%d")
        end_date = datetime.strptime(end_str, "%Y%m%d")
    except ValueError:
        raise ValueError(f"Could not parse dates in '{filename}'. Expected YYYYMMDD format.")
    
    if end_date <= start_date:
        raise ValueError(f"End date must be after start date in '{filename}'.")
    
    return {
        "exchange": exchange,
        "ticker": ticker,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
        "filename": filename,
    }


# ─────────────────────────────────────────────────────────
# CSV Loader & Validator
# ─────────────────────────────────────────────────────────

def load_and_validate_csv(filepath: str) -> pd.DataFrame:
    """
    Load a TradingView CSV export and run validation checks.
    
    Checks:
    1. Required columns present
    2. No duplicate timestamps
    3. Chronological ordering
    4. No negative prices or volumes
    5. Row count sanity (warns if too few)
    6. Gap detection (warns about large gaps)
    
    Returns a cleaned DataFrame with 'time' parsed as datetime index.
    """
    filepath = Path(filepath)
    info = parse_filename(filepath.name)
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Check required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"[{filepath.name}] Missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    
    # Parse timestamps
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    # Check for duplicates
    n_dupes = df["time"].duplicated().sum()
    if n_dupes > 0:
        warnings.warn(f"[{filepath.name}] {n_dupes} duplicate timestamps found. Dropping duplicates.")
        df = df.drop_duplicates(subset="time", keep="first").reset_index(drop=True)
    
    # Check chronological order (should be guaranteed after sort, but verify)
    time_diffs = df["time"].diff().dropna()
    if (time_diffs < pd.Timedelta(0)).any():
        raise ValueError(f"[{filepath.name}] Timestamps are not monotonically increasing after sort.")
    
    # Check for negative values
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            n_bad = (df[col] <= 0).sum()
            warnings.warn(f"[{filepath.name}] {n_bad} non-positive values in '{col}'. Rows will be dropped.")
            df = df[df[col] > 0].reset_index(drop=True)
    
    if (df["volume"] < 0).any():
        n_bad = (df["volume"] < 0).sum()
        warnings.warn(f"[{filepath.name}] {n_bad} negative volume values. Setting to 0.")
        df.loc[df["volume"] < 0, "volume"] = 0
    
    # Row count sanity check
    resolution = info["resolution"]
    years_of_data = (info["end_date"] - info["start_date"]).days / 365.25
    expected_approx = APPROX_ROWS_PER_YEAR.get(resolution, 252) * years_of_data
    actual_rows = len(df)
    
    if actual_rows < expected_approx * 0.7:
        warnings.warn(
            f"[{filepath.name}] Row count ({actual_rows}) is significantly below "
            f"expected (~{int(expected_approx)}). Possible data gaps."
        )
    
    # Gap detection — flag gaps larger than expected
    expected_gap = {
        "1H": pd.Timedelta(hours=4),    # Allow weekend gaps
        "4H": pd.Timedelta(hours=16),
        "1D": pd.Timedelta(days=5),     # Allow weekends + holidays
        "1W": pd.Timedelta(days=14),
    }
    max_gap = expected_gap.get(resolution, pd.Timedelta(days=5))
    large_gaps = time_diffs[time_diffs > max_gap]
    
    if len(large_gaps) > 0:
        warnings.warn(
            f"[{filepath.name}] {len(large_gaps)} unusually large time gaps detected. "
            f"Largest: {large_gaps.max()}. These may indicate missing data periods."
        )
    
    # History depth check
    if years_of_data < MIN_HISTORY_YEARS:
        warnings.warn(
            f"[{filepath.name}] Only {years_of_data:.1f} years of data. "
            f"Minimum for base training is {PREFERRED_HISTORY_YEARS} years. "
            f"This instrument should be excluded from base training."
        )
    
    # Add metadata columns
    df["exchange"] = info["exchange"]
    df["ticker"] = info["ticker"]
    df["resolution"] = info["resolution"]
    df["instrument"] = f"{info['exchange']}_{info['ticker']}"
    
    print(f"  ✓ {filepath.name}: {len(df)} rows, "
          f"{years_of_data:.1f} years, resolution={resolution}")
    
    return df


# ─────────────────────────────────────────────────────────
# Regime Labelling
# ─────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Compute Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    
    return atr


def label_regimes(df: pd.DataFrame, window: int = LOOKBACK_WINDOW) -> pd.DataFrame:
    """
    Label each bar with its regime class based on the surrounding window.
    
    Two dimensions (from blueprint Section 1.5):
    - Trend regime: Trending vs Range-bound (based on 1.5 × ATR)
    - Volatility regime: Low / Normal / High (based on ATR/price thresholds)
    
    Produces 6 combined regime classes.
    
    Labels are assigned per bar based on the PRECEDING window of data
    (backward-looking only — no future leakage).
    """
    df = df.copy()
    
    # Compute ATR
    atr = compute_atr(df, period=ATR_PERIOD)
    df["atr"] = atr
    
    # For each bar, look at the preceding 'window' bars
    # Trend: did price move > 1.5 × ATR directionally over the window?
    price_move = df["close"] - df["close"].shift(window)
    avg_atr_window = atr.rolling(window=window, min_periods=window).mean()
    
    directional_threshold = TREND_ATR_MULTIPLIER * avg_atr_window
    
    df["trend_regime"] = np.where(
        price_move.abs() > directional_threshold,
        "trending",
        "range_bound"
    )
    
    # Volatility: ATR / price ratio over the window
    atr_price_ratio = avg_atr_window / df["close"]
    
    df["vol_regime"] = np.select(
        [
            atr_price_ratio < VOL_LOW_THRESHOLD,
            atr_price_ratio > VOL_HIGH_THRESHOLD,
        ],
        ["low_vol", "high_vol"],
        default="normal_vol"
    )
    
    # Combined regime label
    df["regime"] = df["trend_regime"] + "_" + df["vol_regime"]
    
    # Mark rows without enough history as unlabelled
    df.loc[:window + ATR_PERIOD, "regime"] = "insufficient_history"
    df.loc[:window + ATR_PERIOD, "trend_regime"] = "insufficient_history"
    df.loc[:window + ATR_PERIOD, "vol_regime"] = "insufficient_history"
    
    return df


# ─────────────────────────────────────────────────────────
# Regime Balance Monitor
# ─────────────────────────────────────────────────────────

def check_regime_balance(df: pd.DataFrame) -> dict:
    """
    Check regime class distribution and flag imbalances.
    
    Rules (from blueprint):
    - No single class > 40% of total samples
    - Flag any class < 5% of total samples
    
    Returns a dict with class counts, percentages, and warnings.
    """
    # Only consider labelled bars
    labelled = df[df["regime"] != "insufficient_history"]
    
    if len(labelled) == 0:
        return {"counts": {}, "percentages": {}, "warnings": ["No labelled samples found."]}
    
    counts = labelled["regime"].value_counts().to_dict()
    total = len(labelled)
    percentages = {k: round(v / total * 100, 2) for k, v in counts.items()}
    
    warnings_list = []
    
    for regime, pct in percentages.items():
        if pct > MAX_REGIME_CLASS_SHARE * 100:
            warnings_list.append(
                f"⚠ Regime '{regime}' accounts for {pct}% of samples "
                f"(max allowed: {MAX_REGIME_CLASS_SHARE * 100}%). Consider downsampling."
            )
        if pct < 5.0:
            warnings_list.append(
                f"⚠ Regime '{regime}' accounts for only {pct}% of samples. "
                f"Model will be blind to this regime. Consider adding instruments."
            )
    
    # Check all 6 expected regimes are present
    expected_regimes = {
        "trending_low_vol", "trending_normal_vol", "trending_high_vol",
        "range_bound_low_vol", "range_bound_normal_vol", "range_bound_high_vol",
    }
    missing = expected_regimes - set(counts.keys())
    if missing:
        warnings_list.append(
            f"⚠ Missing regime classes: {missing}. "
            f"These regimes are not represented in the data."
        )
    
    return {
        "counts": counts,
        "percentages": percentages,
        "total_labelled": total,
        "warnings": warnings_list,
    }


# ─────────────────────────────────────────────────────────
# Instrument Diversity Checker
# ─────────────────────────────────────────────────────────

def check_instrument_diversity(catalogue: pd.DataFrame) -> list:
    """
    Verify that the instrument set meets the blueprint's minimum requirements.
    
    Returns a list of warnings/notes about diversity gaps.
    """
    notes = []
    
    # Count unique instruments by asset class heuristics
    # (The user would need to tag asset classes — here we check what we can)
    instruments = catalogue["instrument"].unique()
    n_instruments = len(instruments)
    
    resolutions_present = catalogue["resolution"].unique()
    
    notes.append(f"Total unique instruments: {n_instruments}")
    notes.append(f"Resolutions present: {sorted(resolutions_present)}")
    
    expected_resolutions = {"1H", "4H", "1D", "1W"}
    missing_res = expected_resolutions - set(resolutions_present)
    if missing_res:
        notes.append(f"⚠ Missing resolutions: {missing_res}")
    
    # Check per-instrument sample contribution
    if "n_rows" in catalogue.columns:
        total_rows = catalogue["n_rows"].sum()
        for _, row in catalogue.iterrows():
            share = row["n_rows"] / total_rows * 100
            if share > 15:
                notes.append(
                    f"⚠ {row['instrument']} ({row['resolution']}) contributes {share:.1f}% "
                    f"of samples (max 15%). Consider downsampling."
                )
    
    return notes


# ─────────────────────────────────────────────────────────
# Main Pipeline: Load All → Validate → Label → Report
# ─────────────────────────────────────────────────────────

def run_stage1(raw_dir: str = "data/raw", output_dir: str = "data/processed"):
    """
    Run the full Stage 1 pipeline:
    1. Scan raw_dir for CSV files
    2. Load and validate each file
    3. Label regimes for each instrument
    4. Check regime balance across all data
    5. Save instrument catalogue and regime-labelled data
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(raw_path.glob("*.csv"))
    
    if not csv_files:
        print(f"\n✗ No CSV files found in {raw_path.resolve()}")
        print(f"  Place your TradingView CSV exports in: {raw_path.resolve()}/")
        print(f"  Expected naming: {{EXCHANGE}}_{{TICKER}}_{{RESOLUTION}}_{{STARTDATE}}_{{ENDDATE}}.csv")
        print(f"  Example: NYSE_AAPL_1D_20150101_20241231.csv")
        return
    
    print(f"\n{'='*60}")
    print(f"STAGE 1: Data Collection & Validation")
    print(f"{'='*60}")
    print(f"Found {len(csv_files)} CSV files in {raw_path.resolve()}\n")
    
    # ── Step 1: Load and validate all files ──
    print("Step 1: Loading and validating CSV files...")
    all_data = []
    catalogue_rows = []
    errors = []
    
    for csv_file in csv_files:
        try:
            df = load_and_validate_csv(csv_file)
            info = parse_filename(csv_file.name)
            
            catalogue_rows.append({
                "filename": csv_file.name,
                "exchange": info["exchange"],
                "ticker": info["ticker"],
                "instrument": f"{info['exchange']}_{info['ticker']}",
                "resolution": info["resolution"],
                "start_date": info["start_date"].strftime("%Y-%m-%d"),
                "end_date": info["end_date"].strftime("%Y-%m-%d"),
                "n_rows": len(df),
                "years": round((info["end_date"] - info["start_date"]).days / 365.25, 1),
            })
            
            all_data.append(df)
            
        except Exception as e:
            errors.append(f"  ✗ {csv_file.name}: {e}")
    
    if errors:
        print(f"\nErrors encountered:")
        for err in errors:
            print(err)
    
    if not all_data:
        print("\n✗ No valid data loaded. Fix the errors above and re-run.")
        return
    
    # ── Step 2: Regime labelling ──
    print(f"\nStep 2: Labelling regimes (window={LOOKBACK_WINDOW} bars)...")
    labelled_data = []
    
    for df in all_data:
        instrument = df["instrument"].iloc[0]
        resolution = df["resolution"].iloc[0]
        
        labelled = label_regimes(df, window=LOOKBACK_WINDOW)
        labelled_data.append(labelled)
        
        # Quick regime summary for this instrument
        valid = labelled[labelled["regime"] != "insufficient_history"]
        if len(valid) > 0:
            top_regime = valid["regime"].value_counts().index[0]
            top_pct = valid["regime"].value_counts().iloc[0] / len(valid) * 100
            print(f"  ✓ {instrument} ({resolution}): {len(valid)} labelled bars, "
                  f"dominant regime: {top_regime} ({top_pct:.1f}%)")
    
    # ── Step 3: Check overall regime balance ──
    print(f"\nStep 3: Checking regime balance across all instruments...")
    combined = pd.concat(labelled_data, ignore_index=True)
    balance = check_regime_balance(combined)
    
    print(f"\n  Regime distribution ({balance['total_labelled']} total labelled samples):")
    for regime, pct in sorted(balance["percentages"].items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 2)
        print(f"    {regime:<30s} {pct:6.2f}% {bar}")
    
    if balance["warnings"]:
        print(f"\n  Warnings:")
        for w in balance["warnings"]:
            print(f"    {w}")
    else:
        print(f"\n  ✓ Regime balance is within acceptable limits.")
    
    # ── Step 4: Instrument diversity check ──
    print(f"\nStep 4: Checking instrument diversity...")
    catalogue = pd.DataFrame(catalogue_rows)
    diversity_notes = check_instrument_diversity(catalogue)
    for note in diversity_notes:
        print(f"  {note}")
    
    # ── Step 5: Save outputs ──
    print(f"\nStep 5: Saving outputs...")
    
    # Save catalogue
    catalogue_path = out_path / "instrument_catalogue.csv"
    catalogue.to_csv(catalogue_path, index=False)
    print(f"  ✓ Instrument catalogue: {catalogue_path}")
    
    # Save regime-labelled data per instrument/resolution
    for df in labelled_data:
        instrument = df["instrument"].iloc[0]
        resolution = df["resolution"].iloc[0]
        save_path = out_path / f"{instrument}_{resolution}_labelled.parquet"
        df.to_parquet(save_path, index=False)
        print(f"  ✓ Labelled data: {save_path}")
    
    # Save combined regime balance report
    balance_df = pd.DataFrame([
        {"regime": k, "count": balance["counts"].get(k, 0), "percentage": v}
        for k, v in balance["percentages"].items()
    ])
    balance_path = out_path / "regime_balance_report.csv"
    balance_df.to_csv(balance_path, index=False)
    print(f"  ✓ Regime balance report: {balance_path}")
    
    print(f"\n{'='*60}")
    print(f"STAGE 1 COMPLETE")
    print(f"{'='*60}")
    print(f"  Instruments loaded: {len(all_data)}")
    print(f"  Total labelled samples: {balance['total_labelled']}")
    print(f"  Errors: {len(errors)}")
    print(f"  Output directory: {out_path.resolve()}")
    
    return combined, catalogue


# ─────────────────────────────────────────────────────────
# Demo / Test Mode (runs with synthetic data)
# ─────────────────────────────────────────────────────────

def generate_demo_csv(output_dir: str, exchange: str, ticker: str,
                      resolution: str, years: int = 10):
    """
    Generate a synthetic TradingView-style CSV for testing.
    Creates realistic OHLCV data with multiple market regimes.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2024 - years, 1, 1)
    
    # Generate timestamps based on resolution
    if resolution == "1D":
        dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    elif resolution == "1W":
        dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
    elif resolution == "4H":
        # Simulate 4H bars during market hours (6 bars per day)
        daily = pd.bdate_range(start=start_date, end=end_date, freq="B")
        dates = []
        for d in daily:
            for h in [9, 13, 17]:  # Simplified 4H bars
                dates.append(d + timedelta(hours=h))
        dates = pd.DatetimeIndex(dates)
    elif resolution == "1H":
        daily = pd.bdate_range(start=start_date, end=end_date, freq="B")
        dates = []
        for d in daily:
            for h in range(9, 17):  # 8 hours per day
                dates.append(d + timedelta(hours=h))
        dates = pd.DatetimeIndex(dates)
    else:
        raise ValueError(f"Unknown resolution: {resolution}")
    
    n = len(dates)
    np.random.seed(hash(f"{exchange}_{ticker}_{resolution}") % 2**32)
    
    # Generate price series with regime changes
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n):
        # Switch regimes roughly every 250 bars
        regime_phase = (i // 250) % 4
        if regime_phase == 0:  # Bull trend
            drift = 0.0003
            vol = 0.012
        elif regime_phase == 1:  # High vol correction
            drift = -0.0002
            vol = 0.025
        elif regime_phase == 2:  # Range bound
            drift = 0.0
            vol = 0.008
        else:  # Recovery
            drift = 0.0002
            vol = 0.015
        
        ret = np.random.normal(drift, vol)
        prices.append(prices[-1] * np.exp(ret))
    
    prices = np.array(prices)
    
    # Generate OHLC from close prices
    daily_vol = np.abs(np.random.normal(0, 0.005, n))
    highs = prices * (1 + daily_vol)
    lows = prices * (1 - daily_vol)
    opens = prices * (1 + np.random.normal(0, 0.002, n))
    
    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, prices))
    lows = np.minimum(lows, np.minimum(opens, prices))
    
    # Volume with realistic patterns
    base_volume = np.random.lognormal(mean=15, sigma=0.5, size=n)
    volume = (base_volume * (1 + daily_vol * 20)).astype(int)
    
    df = pd.DataFrame({
        "time": dates[:n],
        "open": np.round(opens, 4),
        "high": np.round(highs, 4),
        "low": np.round(lows, 4),
        "close": np.round(prices, 4),
        "volume": volume,
    })
    
    filename = f"{exchange}_{ticker}_{resolution}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    filepath = out_path / filename
    df.to_csv(filepath, index=False)
    print(f"  Generated: {filename} ({len(df)} rows)")
    
    return filepath


def run_demo():
    """
    Generate synthetic data and run the full Stage 1 pipeline.
    Use this to verify the code works before using real TradingView data.
    """
    print("="*60)
    print("STAGE 1 DEMO: Generating synthetic data for testing")
    print("="*60)
    
    demo_instruments = [
        # Equities
        ("NYSE", "AAPL", "1D", 10),
        ("NYSE", "MSFT", "1D", 10),
        ("LSE", "SHEL", "1D", 10),
        ("ASX", "BHP", "1D", 9),
        ("TSX", "RY", "1D", 8),
        # Forex
        ("FOREXCOM", "EURUSD", "4H", 10),
        ("FOREXCOM", "GBPUSD", "4H", 10),
        ("FOREXCOM", "USDJPY", "4H", 10),
        # Commodities
        ("COMEX", "GOLD", "1D", 10),
        ("NYMEX", "WTI", "1D", 10),
        # Indices
        ("INDEX", "SPX", "1D", 10),
        ("INDEX", "DAX", "1D", 10),
        # Multi-resolution example
        ("NYSE", "AAPL", "4H", 10),
        ("NYSE", "AAPL", "1W", 10),
    ]
    
    print(f"\nGenerating {len(demo_instruments)} synthetic CSV files...\n")
    
    for exchange, ticker, resolution, years in demo_instruments:
        generate_demo_csv("data/raw", exchange, ticker, resolution, years)
    
    print(f"\n{'─'*60}\n")
    
    # Run the full pipeline
    result = run_stage1("data/raw", "data/processed")
    
    return result


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        run_stage1()