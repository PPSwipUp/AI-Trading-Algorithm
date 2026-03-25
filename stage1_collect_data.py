"""
Stage 1: Automated Data Collection from TradingView
====================================================
Uses the tvdatafeed library to programmatically download OHLCV data
for all instruments specified in the blueprint.

Setup:
    pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git

Usage:
    # No login (limited symbols/history):
    python stage1_collect_data.py

    # With TradingView login (recommended — more symbols, more history):
    python stage1_collect_data.py --username YOUR_USERNAME --password YOUR_PASSWORD

    # Collect only specific asset classes:
    python stage1_collect_data.py --assets equities forex

    # Dry run (show what would be downloaded without downloading):
    python stage1_collect_data.py --dry-run

    # After collection, run the validation pipeline:
    python stage1_data_collection.py
"""

import os
import sys
import time
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# ─────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Instrument Universe
# ─────────────────────────────────────────────────────────
# Blueprint Section 1.2: Minimum instrument requirements
# Each entry: (symbol, exchange, asset_class, description)
# The exchange names must match TradingView's exchange identifiers.

INSTRUMENTS = {
    "equities": [
        # US Markets
        ("AAPL",    "NASDAQ",   "equities", "Apple — US tech"),
        ("MSFT",    "NASDAQ",   "equities", "Microsoft — US tech"),
        ("GOOGL",   "NASDAQ",   "equities", "Alphabet — US tech"),
        ("AMZN",    "NASDAQ",   "equities", "Amazon — US consumer/tech"),
        ("NVDA",    "NASDAQ",   "equities", "Nvidia — US semiconductors"),
        ("META",    "NASDAQ",   "equities", "Meta — US tech"),
        ("TSLA",    "NASDAQ",   "equities", "Tesla — US consumer/auto"),
        ("JPM",     "NYSE",     "equities", "JPMorgan — US financials"),
        ("JNJ",     "NYSE",     "equities", "Johnson & Johnson — US healthcare"),
        ("XOM",     "NYSE",     "equities", "ExxonMobil — US energy"),
        ("PFE",     "NYSE",     "equities", "Pfizer — US healthcare"),
        ("BAC",     "NYSE",     "equities", "Bank of America — US financials"),
        ("KO",      "NYSE",     "equities", "Coca-Cola — US consumer staples"),
        ("DIS",     "NYSE",     "equities", "Disney — US media"),
        ("WMT",     "NYSE",     "equities", "Walmart — US consumer"),
        ("PG",      "NYSE",     "equities", "Procter & Gamble — US consumer"),
        ("CVX",     "NYSE",     "equities", "Chevron — US energy"),
        ("ABBV",    "NYSE",     "equities", "AbbVie — US pharma"),
        ("HD",      "NYSE",     "equities", "Home Depot — US retail"),
        ("MRK",     "NYSE",     "equities", "Merck — US pharma"),
        # Non-US Markets (blueprint requires at least 5)
        ("SHEL",    "LSE",      "equities", "Shell — UK energy"),
        ("HSBA",    "LSE",      "equities", "HSBC — UK financials"),
        ("AZN",     "LSE",      "equities", "AstraZeneca — UK pharma"),
        ("BP",      "LSE",      "equities", "BP — UK energy"),
        ("ULVR",    "LSE",      "equities", "Unilever — UK consumer"),
        ("RIO",     "ASX",      "equities", "Rio Tinto — AU mining"),
        ("BHP",     "ASX",      "equities", "BHP Group — AU mining"),
        ("CSL",     "ASX",      "equities", "CSL Limited — AU healthcare"),
        ("RY",      "TSX",      "equities", "Royal Bank — CA financials"),
        ("TD",      "TSX",      "equities", "TD Bank — CA financials"),
        ("SAN",     "EURONEXT", "equities", "Santander — EU financials"),
        ("MC",      "EURONEXT", "equities", "LVMH — EU luxury/consumer"),
        ("SAP",     "XETR",     "equities", "SAP — DE tech"),
        ("SIE",     "XETR",     "equities", "Siemens — DE industrials"),
        ("7203",    "TSE",      "equities", "Toyota — JP auto"),
    ],
    "forex": [
        # Majors
        ("EURUSD",  "FOREXCOM", "forex", "Euro / US Dollar"),
        ("GBPUSD",  "FOREXCOM", "forex", "British Pound / US Dollar"),
        ("USDJPY",  "FOREXCOM", "forex", "US Dollar / Japanese Yen"),
        ("AUDUSD",  "FOREXCOM", "forex", "Australian Dollar / US Dollar"),
        ("USDCAD",  "FOREXCOM", "forex", "US Dollar / Canadian Dollar"),
        ("USDCHF",  "FOREXCOM", "forex", "US Dollar / Swiss Franc"),
        ("NZDUSD",  "FOREXCOM", "forex", "New Zealand Dollar / US Dollar"),
        # Minors
        ("EURGBP",  "FOREXCOM", "forex", "Euro / British Pound"),
        ("GBPJPY",  "FOREXCOM", "forex", "British Pound / Japanese Yen"),
        ("AUDNZD",  "FOREXCOM", "forex", "Australian Dollar / NZ Dollar"),
        ("EURCHF",  "FOREXCOM", "forex", "Euro / Swiss Franc"),
        ("EURJPY",  "FOREXCOM", "forex", "Euro / Japanese Yen"),
    ],
    "commodities": [
        ("XAUUSD",  "FOREXCOM", "commodities", "Gold"),
        ("XAGUSD",  "FOREXCOM", "commodities", "Silver"),
        ("USOIL",   "FOREXCOM", "commodities", "WTI Crude Oil"),
        ("UKOIL",   "FOREXCOM", "commodities", "Brent Crude Oil"),
        ("NGAS",    "FOREXCOM", "commodities", "Natural Gas"),
        ("COPPER",  "FOREXCOM", "commodities", "Copper"),
        ("WHEAT",   "CBOT",     "commodities", "Wheat"),
    ],
    "indices": [
        ("SPX",     "SP",       "indices", "S&P 500"),
        ("DJI",     "DJ",       "indices", "Dow Jones Industrial"),
        ("NDX",     "NASDAQ",   "indices", "Nasdaq 100"),
        ("FTSE",    "FTSE",     "indices", "FTSE 100"),
        ("DEU40",   "PEPPERSTONE","indices", "DAX 40"),
        ("NI225",   "TVC",      "indices", "Nikkei 225"),
        ("HSI",     "HSI",      "indices", "Hang Seng Index"),
    ],
    "crypto": [
        ("BTCUSD",  "COINBASE", "crypto", "Bitcoin / USD"),
        ("ETHUSD",  "COINBASE", "crypto", "Ethereum / USD"),
        ("BTCUSD",  "BITSTAMP", "crypto", "Bitcoin / USD (Bitstamp)"),
    ],
}

# ─────────────────────────────────────────────────────────
# Resolution mapping
# ─────────────────────────────────────────────────────────
# Blueprint Section 1.3: Pull 1H, 4H, 1D, 1W per instrument.
# tvdatafeed Interval enum values and bar limits.

RESOLUTIONS = {
    "1H": {
        "n_bars": 5000,  # Max allowed — covers ~2.5 years of hourly
        "min_years_expected": 2,
    },
    "4H": {
        "n_bars": 5000,  # Covers ~8 years of 4H
        "min_years_expected": 5,
    },
    "1D": {
        "n_bars": 5000,  # Covers ~20 years of daily
        "min_years_expected": 8,
    },
    "1W": {
        "n_bars": 5000,  # Covers ~96 years of weekly
        "min_years_expected": 8,
    },
}


def get_interval_enum(resolution: str):
    """Map our resolution string to tvdatafeed Interval enum."""
    from tvDatafeed import Interval
    mapping = {
        "1H": Interval.in_1_hour,
        "4H": Interval.in_4_hour,
        "1D": Interval.in_daily,
        "1W": Interval.in_weekly,
    }
    return mapping[resolution]


# ─────────────────────────────────────────────────────────
# Data Collection
# ─────────────────────────────────────────────────────────

def download_instrument(tv, symbol: str, exchange: str, resolution: str,
                        output_dir: str, retry_count: int = 3,
                        delay: float = 2.0) -> dict:
    """
    Download OHLCV data for a single instrument/resolution from TradingView.

    Returns a result dict with status, filepath, row count, etc.
    """
    interval = get_interval_enum(resolution)
    n_bars = RESOLUTIONS[resolution]["n_bars"]
    result = {
        "symbol": symbol,
        "exchange": exchange,
        "resolution": resolution,
        "status": "pending",
        "filepath": None,
        "n_rows": 0,
        "start_date": None,
        "end_date": None,
        "error": None,
    }

    for attempt in range(1, retry_count + 1):
        try:
            log.info(f"  Downloading {exchange}:{symbol} @ {resolution} "
                     f"(attempt {attempt}/{retry_count}, requesting {n_bars} bars)...")

            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                n_bars=n_bars,
            )

            if df is None or len(df) == 0:
                raise ValueError("No data returned (symbol may not exist on this exchange)")

            # tvdatafeed returns a DataFrame with datetime index and columns:
            # symbol, open, high, low, close, volume
            df = df.reset_index()

            # Normalise column names
            df.columns = [c.strip().lower() for c in df.columns]

            # The index column is typically named 'datetime'
            time_col = None
            for candidate in ["datetime", "date", "time", "timestamp"]:
                if candidate in df.columns:
                    time_col = candidate
                    break

            if time_col is None:
                # Fall back to first column if it looks like a datetime
                if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                    time_col = df.columns[0]
                else:
                    raise ValueError(f"Cannot identify time column. Columns: {list(df.columns)}")

            # Rename to standard 'time' column
            df = df.rename(columns={time_col: "time"})
            df["time"] = pd.to_datetime(df["time"])

            # Keep only OHLCV columns
            required = ["time", "open", "high", "low", "close", "volume"]
            available = [c for c in required if c in df.columns]
            if len(available) < 5:  # time + at least OHLC
                raise ValueError(f"Missing columns. Available: {list(df.columns)}")

            df = df[available].copy()

            # Sort chronologically
            df = df.sort_values("time").reset_index(drop=True)

            # Drop duplicates
            df = df.drop_duplicates(subset="time", keep="first").reset_index(drop=True)

            # If volume column is missing (some indices), fill with 0
            if "volume" not in df.columns:
                df["volume"] = 0

            # Determine date range
            start_date = df["time"].iloc[0]
            end_date = df["time"].iloc[-1]

            # Build filename: {EXCHANGE}_{TICKER}_{RESOLUTION}_{STARTDATE}_{ENDDATE}.csv
            filename = (
                f"{exchange}_{symbol}_{resolution}_"
                f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            )
            filepath = Path(output_dir) / filename

            # Save CSV
            df.to_csv(filepath, index=False)

            result["status"] = "success"
            result["filepath"] = str(filepath)
            result["n_rows"] = len(df)
            result["start_date"] = start_date.strftime("%Y-%m-%d")
            result["end_date"] = end_date.strftime("%Y-%m-%d")
            result["years"] = round((end_date - start_date).days / 365.25, 1)

            log.info(f"  ✓ {filename}: {len(df)} rows, "
                     f"{result['years']} years ({start_date.date()} → {end_date.date()})")

            # Check history depth
            min_years = RESOLUTIONS[resolution]["min_years_expected"]
            if result["years"] < min_years:
                log.warning(
                    f"  ⚠ {symbol}: Only {result['years']} years "
                    f"(expected ≥{min_years} for {resolution}). "
                    f"May be insufficient for base training."
                )

            return result

        except Exception as e:
            error_msg = str(e)
            log.warning(f"  ⚠ Attempt {attempt} failed for {exchange}:{symbol} "
                        f"@ {resolution}: {error_msg}")
            result["error"] = error_msg

            if attempt < retry_count:
                wait = delay * attempt
                log.info(f"    Retrying in {wait:.0f}s...")
                time.sleep(wait)

    result["status"] = "failed"
    log.error(f"  ✗ FAILED: {exchange}:{symbol} @ {resolution} after {retry_count} attempts")
    return result


# ─────────────────────────────────────────────────────────
# Alternative exchange fallbacks
# ─────────────────────────────────────────────────────────
# Some symbols may not be available on the primary exchange.
# These are fallbacks to try if the primary fails.

EXCHANGE_FALLBACKS = {
    # Equities that might need different exchange IDs
    ("SHEL", "LSE"):     [("SHEL", "NYSE")],
    ("BP", "LSE"):       [("BP.", "LSE"), ("BP", "NYSE")],
    ("ULVR", "LSE"):     [("ULVR.", "LSE")],
    ("HSBA", "LSE"):     [("HSBA.", "LSE"), ("HSBC", "NYSE")],
    ("AZN", "LSE"):      [("AZN.", "LSE"), ("AZN", "NASDAQ")],
    ("RIO", "ASX"):      [("RIO", "NYSE"), ("RIO", "LSE")],
    ("BHP", "ASX"):      [("BHP", "NYSE")],
    ("CSL", "ASX"):      [("CSL.", "ASX")],
    ("RY", "TSX"):       [("RY", "NYSE")],
    ("TD", "TSX"):       [("TD", "NYSE")],
    ("SAN", "EURONEXT"): [("SAN", "BME")],
    ("MC", "EURONEXT"):  [("MC", "EURONEXT_PARIS")],
    ("7203", "TSE"):     [("TM", "NYSE")],  # Toyota ADR
    # Commodities
    ("USOIL", "FOREXCOM"):  [("CL1!", "NYMEX"), ("USOIL", "TVC")],
    ("UKOIL", "FOREXCOM"):  [("BRN1!", "ICEEUR"), ("UKOIL", "TVC")],
    ("NGAS", "FOREXCOM"):   [("NG1!", "NYMEX"), ("NGAS", "TVC")],
    ("COPPER", "FOREXCOM"): [("HG1!", "COMEX"), ("COPPER", "TVC")],
    ("WHEAT", "CBOT"):      [("ZW1!", "CBOT"), ("WHEAT", "TVC")],
    # Indices
    ("SPX", "SP"):          [("SPX", "TVC"), ("SPX500USD", "OANDA")],
    ("DJI", "DJ"):          [("DJI", "TVC"), ("US30", "FOREXCOM")],
    ("NDX", "NASDAQ"):      [("NDX", "TVC"), ("NAS100USD", "OANDA")],
    ("FTSE", "FTSE"):       [("FTSE", "TVC"), ("UK100", "FOREXCOM")],
    ("DEU40", "PEPPERSTONE"):[("DEU40", "FX"), ("GER40", "FOREXCOM")],
    ("NI225", "TVC"):       [("NI225", "CAPITALCOM"), ("JP225USD", "OANDA")],
    ("HSI", "HSI"):         [("HSI", "TVC"), ("HK50", "FOREXCOM")],
    # Forex (fallbacks)
    ("EURUSD", "FOREXCOM"): [("EURUSD", "FX_IDC"), ("EURUSD", "OANDA")],
    ("GBPUSD", "FOREXCOM"): [("GBPUSD", "FX_IDC"), ("GBPUSD", "OANDA")],
    ("USDJPY", "FOREXCOM"): [("USDJPY", "FX_IDC"), ("USDJPY", "OANDA")],
    # Crypto
    ("BTCUSD", "COINBASE"): [("BTCUSD", "BITSTAMP"), ("BTCUSD", "CRYPTO")],
    ("ETHUSD", "COINBASE"): [("ETHUSD", "BITSTAMP"), ("ETHUSD", "CRYPTO")],
}


def download_with_fallbacks(tv, symbol: str, exchange: str, resolution: str,
                            output_dir: str) -> dict:
    """
    Try to download from the primary exchange, then try fallbacks if it fails.
    """
    result = download_instrument(tv, symbol, exchange, resolution, output_dir)

    if result["status"] == "success":
        return result

    # Try fallbacks
    fallback_key = (symbol, exchange)
    if fallback_key in EXCHANGE_FALLBACKS:
        for alt_symbol, alt_exchange in EXCHANGE_FALLBACKS[fallback_key]:
            log.info(f"  → Trying fallback: {alt_exchange}:{alt_symbol}...")
            result = download_instrument(
                tv, alt_symbol, alt_exchange, resolution, output_dir
            )
            if result["status"] == "success":
                return result

    return result


# ─────────────────────────────────────────────────────────
# Main Collection Pipeline
# ─────────────────────────────────────────────────────────

def run_collection(
    username: str = None,
    password: str = None,
    asset_classes: list = None,
    resolutions: list = None,
    output_dir: str = "data/raw",
    dry_run: bool = False,
    delay_between: float = 1.5,
):
    """
    Run the full data collection pipeline.

    Args:
        username: TradingView username (None for no-login mode)
        password: TradingView password
        asset_classes: Which asset classes to collect (default: all)
        resolutions: Which resolutions to collect (default: all four)
        output_dir: Where to save CSV files
        dry_run: If True, just show what would be downloaded
        delay_between: Seconds to wait between downloads (rate limiting)
    """
    from tvDatafeed import TvDatafeed

    if asset_classes is None:
        asset_classes = ["equities", "forex", "commodities", "indices"]
    if resolutions is None:
        resolutions = ["1D", "4H", "1W", "1H"]

    # ── Build download plan ──
    download_plan = []
    for asset_class in asset_classes:
        if asset_class not in INSTRUMENTS:
            log.warning(f"Unknown asset class: {asset_class}. Skipping.")
            continue
        for symbol, exchange, _, description in INSTRUMENTS[asset_class]:
            for res in resolutions:
                download_plan.append({
                    "symbol": symbol,
                    "exchange": exchange,
                    "resolution": res,
                    "asset_class": asset_class,
                    "description": description,
                })

    total = len(download_plan)
    print(f"\n{'='*65}")
    print(f"  STAGE 1: TradingView Data Collection")
    print(f"{'='*65}")
    print(f"  Asset classes : {', '.join(asset_classes)}")
    print(f"  Resolutions   : {', '.join(resolutions)}")
    print(f"  Instruments   : {sum(len(INSTRUMENTS.get(ac, [])) for ac in asset_classes)}")
    print(f"  Total downloads: {total}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Login mode    : {'Authenticated' if username else 'No-login (limited)'}")
    print(f"{'='*65}\n")

    if dry_run:
        print("DRY RUN — showing download plan:\n")
        for i, item in enumerate(download_plan, 1):
            print(f"  {i:3d}. {item['exchange']}:{item['symbol']} "
                  f"@ {item['resolution']} — {item['description']}")
        print(f"\nTotal: {total} downloads. Run without --dry-run to execute.")
        return

    # ── Initialise TradingView connection ──
    log.info("Connecting to TradingView...")
    if username and password:
        tv = TvDatafeed(username, password)
        log.info("✓ Authenticated connection established")
    else:
        tv = TvDatafeed()
        log.warning("Using no-login mode — some symbols/history may be limited")

    # ── Create output directory ──
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Download loop ──
    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, item in enumerate(download_plan, 1):
        symbol = item["symbol"]
        exchange = item["exchange"]
        resolution = item["resolution"]

        # Check if file already exists
        existing = list(out_path.glob(f"{exchange}_{symbol}_{resolution}_*.csv"))
        if existing:
            log.info(f"[{i}/{total}] {exchange}:{symbol} @ {resolution} — "
                     f"already exists: {existing[0].name}. Skipping.")
            skip_count += 1
            results.append({
                "symbol": symbol, "exchange": exchange,
                "resolution": resolution, "status": "skipped",
                "filepath": str(existing[0]),
            })
            continue

        log.info(f"[{i}/{total}] {item['description']}")
        result = download_with_fallbacks(tv, symbol, exchange, resolution, output_dir)
        results.append(result)

        if result["status"] == "success":
            success_count += 1
        else:
            fail_count += 1

        # Rate limiting — be polite to TradingView
        if i < total:
            time.sleep(delay_between)

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'='*65}")
    print(f"  ✓ Successful : {success_count}")
    print(f"  ⊘ Skipped    : {skip_count} (already existed)")
    print(f"  ✗ Failed     : {fail_count}")
    print(f"  Total        : {total}")

    if fail_count > 0:
        print(f"\n  Failed downloads:")
        for r in results:
            if r["status"] == "failed":
                print(f"    ✗ {r['exchange']}:{r['symbol']} @ {r['resolution']}"
                      f" — {r.get('error', 'unknown error')}")

    # ── Save collection log ──
    log_path = out_path / "_collection_log.json"
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "login_mode": "authenticated" if username else "nologin",
        "asset_classes": asset_classes,
        "resolutions": resolutions,
        "total": total,
        "success": success_count,
        "skipped": skip_count,
        "failed": fail_count,
        "results": results,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"\n  Collection log saved: {log_path}")

    print(f"\n  Next step: run stage1_data_collection.py to validate and label regimes.")
    print(f"{'='*65}\n")

    return results


# ─────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download instrument data from TradingView for the trading algorithm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                     # No-login, all assets, all resolutions
  %(prog)s --username USER --password PASS      # Authenticated (recommended)
  %(prog)s --assets equities forex              # Only equities and forex
  %(prog)s --resolutions 1D 4H                  # Only daily and 4-hour
  %(prog)s --dry-run                            # Show plan without downloading
  %(prog)s --output data/raw                    # Custom output directory
        """,
    )

    parser.add_argument("--username", "-u", type=str, default=None,
                        help="TradingView username (recommended for more data)")
    parser.add_argument("--password", "-p", type=str, default=None,
                        help="TradingView password")
    parser.add_argument("--assets", "-a", nargs="+", default=None,
                        choices=["equities", "forex", "commodities", "indices", "crypto"],
                        help="Asset classes to collect (default: all except crypto)")
    parser.add_argument("--resolutions", "-r", nargs="+", default=None,
                        choices=["1H", "4H", "1D", "1W"],
                        help="Resolutions to collect (default: 1D, 4H, 1W, 1H)")
    parser.add_argument("--output", "-o", type=str, default="data/raw",
                        help="Output directory for CSV files (default: data/raw)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show download plan without actually downloading")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between downloads (rate limiting, default: 1.5)")
    parser.add_argument("--include-crypto", action="store_true",
                        help="Include crypto (excluded by default per blueprint advice)")

    args = parser.parse_args()

    # Default asset classes (crypto excluded unless explicitly included)
    if args.assets is None:
        args.assets = ["equities", "forex", "commodities", "indices"]
        if args.include_crypto:
            args.assets.append("crypto")

    run_collection(
        username=args.username,
        password=args.password,
        asset_classes=args.assets,
        resolutions=args.resolutions,
        output_dir=args.output,
        dry_run=args.dry_run,
        delay_between=args.delay,
    )


if __name__ == "__main__":
    main()