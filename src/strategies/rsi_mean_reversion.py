from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


class RSIMeanReversion(Strategy):
    """
    Mean-reversion strategy: go long when RSI signals oversold, exit/go flat
    when RSI recovers past the midpoint. Works best in range-bound markets;
    tends to get hurt in strong trends — a good complement to the MA
    crossover strategy when comparing regimes in the backtester.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        exit_level: float = 50.0,
        allow_short: bool = False,
        overbought: float = 70.0,
    ):
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level
        self.overbought = overbought
        self.allow_short = allow_short
        self.name = f"RSI_MeanReversion_{period}"

    def generate_positions(self, bars: pd.DataFrame) -> pd.Series:
        rsi = _rsi(bars["close"], self.period)
        positions = pd.Series(0.0, index=bars.index)

        in_long = False
        in_short = False

        rsi_vals = rsi.to_numpy()
        pos_vals = positions.to_numpy().copy()

        for i in range(len(rsi_vals)):
            r = rsi_vals[i]
            if pd.isna(r):
                continue

            if in_long:
                if r >= self.exit_level:
                    in_long = False
                else:
                    pos_vals[i] = 1.0
                    continue

            if in_short:
                if r <= self.exit_level:
                    in_short = False
                else:
                    pos_vals[i] = -1.0
                    continue

            if r <= self.oversold:
                in_long = True
                pos_vals[i] = 1.0
            elif self.allow_short and r >= self.overbought:
                in_short = True
                pos_vals[i] = -1.0

        return pd.Series(pos_vals, index=bars.index)
