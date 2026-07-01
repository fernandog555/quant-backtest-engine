from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


class BuyAndHold(Strategy):
    """Benchmark strategy: go long on the first valid bar and never exit.
    Every other strategy should be compared against this — if you can't
    beat buy-and-hold on a risk-adjusted basis, the added complexity
    probably isn't worth it."""

    name = "BuyAndHold"

    def generate_positions(self, bars: pd.DataFrame) -> pd.Series:
        positions = pd.Series(1.0, index=bars.index)
        return positions
