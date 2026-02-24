"""
AlphaSignal Configuration
=========================
Configuration for the LSTM-based quantitative equities prediction pipeline.
"""
import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data ingestion and feature engineering configuration."""
    ticker_symbols: list = None
    lookback_window: int = 60  # trading days
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    rolling_windows: list = None
    data_dir: str = os.getenv("DATA_DIR", "data/")

    def __post_init__(self):
        if self.ticker_symbols is None:
            self.ticker_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50]


@dataclass
class ModelConfig:
    """LSTM model hyperparameters."""
    input_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    patience: int = 15
    device: str = "cuda"


@dataclass
class TradingConfig:
    """Backtesting and strategy parameters."""
    initial_capital: float = 1_000_000
    transaction_cost_bps: float = 5.0  # basis points
    slippage_bps: float = 2.0
    position_size: float = 0.1  # fraction of portfolio
    risk_free_rate: float = 0.05
    confidence_threshold: float = 0.55
