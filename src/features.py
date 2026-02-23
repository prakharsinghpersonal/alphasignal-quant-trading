"""
AlphaSignal — Rolling-Window Statistical Feature Engineering
Custom technical indicators and statistical features computed over
10+ years of tick-level data to feed LSTM input tensors.
"""

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a comprehensive set of rolling-window features from OHLCV data.

    Returns a DataFrame with original columns plus engineered features.
    """
    df = df.copy()

    # --- Price-based returns ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_change"] = df["close"].pct_change()

    # --- Moving averages ---
    for window in [5, 10, 20, 50, 200]:
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()

    # --- Bollinger Bands (20-day) ---
    bb_window = 20
    df["bb_mid"] = df["close"].rolling(bb_window).mean()
    bb_std = df["close"].rolling(bb_window).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    # --- RSI (14-day) ---
    df["rsi_14"] = _compute_rsi(df["close"], 14)

    # --- MACD ---
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # --- Volatility clustering ---
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = df["log_return"].rolling(window).std() * np.sqrt(
            252
        )

    # --- Volume features ---
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # --- Average True Range (ATR) ---
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    # --- Momentum ---
    for period in [5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

    # --- Target: next-day direction (1 = up, 0 = down) ---
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df.dropna(inplace=True)
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excluding target and raw OHLCV)."""
    exclude = {"open", "high", "low", "close", "volume", "target"}
    return [c for c in df.columns if c not in exclude]
