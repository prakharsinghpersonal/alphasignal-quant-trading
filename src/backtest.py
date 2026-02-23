"""
AlphaSignal — Backtesting Engine with Transaction Costs
Realistic trading strategy simulations incorporating real-world
transaction costs and slippage, computing Sharpe ratio and drawdown.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "initial_capital": 100_000,
    "transaction_cost_bps": 10,    # 10 basis points per trade
    "slippage_bps": 5,             # 5 basis points slippage
    "risk_free_rate": 0.04,        # annualized
    "trading_days_per_year": 252,
}


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Backtest a directional trading strategy with realistic costs."""

    def __init__(self, config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
    ) -> dict:
        """
        Execute backtest.

        Parameters
        ----------
        prices : pd.Series
            Daily close prices indexed by date.
        signals : pd.Series
            Binary signals: 1 = long, 0 = flat. Aligned with prices.

        Returns
        -------
        dict with strategy_returns, benchmark_returns, metrics, and equity_curve.
        """
        prices = prices.copy()
        signals = signals.copy()

        daily_returns = prices.pct_change().fillna(0)

        # Detect position changes (trades)
        trades = signals.diff().abs().fillna(0)

        # Cost per trade (transaction cost + slippage)
        cost_per_trade = (
            self.config["transaction_cost_bps"] + self.config["slippage_bps"]
        ) / 10_000

        # Strategy returns = signal * market return - trade costs
        strategy_returns = (signals * daily_returns) - (trades * cost_per_trade)

        # Equity curves
        capital = self.config["initial_capital"]
        strategy_equity = capital * (1 + strategy_returns).cumprod()
        benchmark_equity = capital * (1 + daily_returns).cumprod()

        # Metrics
        metrics = self._compute_metrics(strategy_returns, daily_returns, trades)

        return {
            "strategy_returns": strategy_returns,
            "benchmark_returns": daily_returns,
            "strategy_equity": strategy_equity,
            "benchmark_equity": benchmark_equity,
            "metrics": metrics,
        }

    def _compute_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        trades: pd.Series,
    ) -> dict:
        """Compute comprehensive performance metrics."""
        rf_daily = self.config["risk_free_rate"] / self.config["trading_days_per_year"]
        n_days = self.config["trading_days_per_year"]

        # Annualized returns
        strat_annual = strategy_returns.mean() * n_days
        bench_annual = benchmark_returns.mean() * n_days

        # Annualized volatility
        strat_vol = strategy_returns.std() * np.sqrt(n_days)
        bench_vol = benchmark_returns.std() * np.sqrt(n_days)

        # Sharpe ratio
        sharpe = (
            (strategy_returns.mean() - rf_daily) / strategy_returns.std()
        ) * np.sqrt(n_days) if strategy_returns.std() > 0 else 0

        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_days = (strategy_returns > 0).sum()
        trading_days = (strategy_returns != 0).sum()
        win_rate = winning_days / trading_days if trading_days > 0 else 0

        # Trade count
        total_trades = int(trades.sum())

        return {
            "annualized_return": round(strat_annual, 4),
            "annualized_volatility": round(strat_vol, 4),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
            "total_trades": total_trades,
            "benchmark_return": round(bench_annual, 4),
            "benchmark_sharpe": round(
                (benchmark_returns.mean() - rf_daily) / benchmark_returns.std()
                * np.sqrt(n_days)
                if benchmark_returns.std() > 0 else 0,
                2,
            ),
        }


def generate_signals_from_predictions(
    predictions: np.ndarray, threshold: float = 0.5
) -> pd.Series:
    """Convert model probability outputs to binary trading signals."""
    return pd.Series((predictions > threshold).astype(int))


if __name__ == "__main__":
    from data_loader import load_tick_data
    from features import compute_features

    # Load data
    df = load_tick_data("AAPL", "2013-01-01", "2023-12-31")
    df = compute_features(df)

    prices = df["close"]

    # Demo: random signals as placeholder for model predictions
    np.random.seed(42)
    signals = pd.Series(np.random.choice([0, 1], len(prices)), index=prices.index)

    bt = Backtester()
    results = bt.run(prices, signals)

    logger.info("Strategy Metrics:")
    for k, v in results["metrics"].items():
        logger.info("  %s: %s", k, v)
