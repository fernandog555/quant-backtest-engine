from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    """
    Classic trend-following strategy: long when fast MA > slow MA,
    flat/short otherwise. Simple, well-understood, easy to sanity-check —
    good baseline to compare fancier strategies against.
    """

    def __init__(self, fast_window: int = 20, slow_window: int = 50, allow_short: bool = False):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.allow_short = allow_short
        self.name = f"MA_Crossover_{fast_window}_{slow_window}"

    def generate_positions(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        fast_ma = close.rolling(self.fast_window).mean()
        slow_ma = close.rolling(self.slow_window).mean()

        positions = pd.Series(0.0, index=bars.index)
        long_mask = fast_ma > slow_ma
        positions[long_mask] = 1.0

        if self.allow_short:
            short_mask = fast_ma < slow_ma
            positions[short_mask] = -1.0

        # No signal until slow_ma has enough data
        positions[slow_ma.isna()] = 0.0
        return positions
