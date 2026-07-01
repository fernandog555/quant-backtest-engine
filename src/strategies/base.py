"""Strategy base class. All strategies operate on a DataFrame of bars and
emit a Series of target positions (not raw signals) — this makes backtesting
and live execution share the same interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """
    A strategy consumes historical bars up to "now" and outputs a target
    position for each timestamp: +1 (fully long), 0 (flat), -1 (fully short).

    This is a vectorized, backtest-friendly interface. Position sizing and
    risk limits are applied downstream by the RiskManager — strategies only
    express *direction and conviction*, not position size in shares/dollars.
    """

    name: str = "unnamed_strategy"

    @abstractmethod
    def generate_positions(self, bars: pd.DataFrame) -> pd.Series:
        """
        bars: DataFrame with columns [open, high, low, close, volume], indexed by timestamp.
        Returns: Series indexed the same as bars, values in [-1, 1].
        """
        raise NotImplementedError

    def generate_latest_signal(self, bars: pd.DataFrame) -> float:
        """
        Convenience for live trading: run generate_positions on the full
        history and return only the most recent target position.
        Strategies with lookahead-safe logic will return a value consistent
        with what the backtester would have produced at this point in time.
        """
        positions = self.generate_positions(bars)
        return float(positions.iloc[-1]) if len(positions) else 0.0
