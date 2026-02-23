"""
AlphaSignal — Tick-Level Market Data Loader
Processes 10+ years of large-scale, tick-level market data using Pandas,
preparing clean time-series datasets for feature engineering and model training.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_tick_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load tick-level data for a given equity symbol.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., 'AAPL').
    start_date, end_date : str, optional
        ISO-format date filters.

    Returns
    -------
    pd.DataFrame with columns: timestamp, open, high, low, close, volume
    """
    filepath = RAW_DIR / f"{symbol.upper()}_ticks.csv"

    if not filepath.exists():
        logger.warning("File %s not found. Generating synthetic data.", filepath)
        return _generate_synthetic_ticks(symbol, start_date, end_date)

    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    logger.info("Loaded %d ticks for %s", len(df), symbol)
    return df


def _generate_synthetic_ticks(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Generate synthetic tick-level OHLCV data for development."""
    np.random.seed(hash(symbol) % 2**31)

    start = pd.Timestamp(start_date or "2013-01-01")
    end = pd.Timestamp(end_date or "2023-12-31")
    trading_days = pd.bdate_range(start, end)

    n = len(trading_days)
    base_price = np.random.uniform(50, 300)
    returns = np.random.normal(0.0003, 0.02, n)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            "close": prices,
            "volume": np.random.randint(100_000, 50_000_000, n),
        },
        index=trading_days,
    )
    df.index.name = "timestamp"

    logger.info("Generated %d synthetic trading days for %s", n, symbol)
    return df


def resample_to_intervals(
    df: pd.DataFrame, interval: str = "1h"
) -> pd.DataFrame:
    """Resample tick data to a coarser time interval."""
    resampled = df.resample(interval).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    resampled.dropna(inplace=True)
    return resampled


def save_processed(df: pd.DataFrame, symbol: str, tag: str = "clean") -> Path:
    """Persist processed DataFrame to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{symbol.upper()}_{tag}.parquet"
    df.to_parquet(out)
    logger.info("Saved %d rows to %s", len(df), out)
    return out


if __name__ == "__main__":
    for sym in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
        ticks = load_tick_data(sym, "2013-01-01", "2023-12-31")
        save_processed(ticks, sym)
