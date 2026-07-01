"""
Turns a raw fill log (one row per order execution) into round-trip trades
(entry -> exit, with realized P&L) and computes trade-level statistics.

Separated from the backtest engine itself so this same logic can eventually
be reused on live/paper fill logs pulled from Webull, not just backtest output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RoundTripTrade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    holding_bars: int


def pair_trades(fills: pd.DataFrame, bar_index: pd.Index) -> list[RoundTripTrade]:
    """
    Walks the fill log per symbol and pairs entries with exits using FIFO
    matching. Handles the common cases: full close, flip (close + reverse in
    one fill, e.g. long -> short), and partial size changes within a
    position (weighted-average entry price).

    fills: DataFrame with columns [timestamp, symbol, shares, fill_price, reason]
           where `shares` is signed (positive = bought, negative = sold).
    bar_index: the backtest's bar index, used to compute holding_bars.
    """
    if fills.empty:
        return []

    trades: list[RoundTripTrade] = []
    index_pos = {ts: i for i, ts in enumerate(bar_index)}

    for symbol, group in fills.groupby("symbol"):
        group = group.sort_values("timestamp")
        position_qty = 0.0
        avg_entry_price = 0.0
        entry_time = None

        for _, fill in group.iterrows():
            fill_shares = fill["shares"]
            fill_price = fill["fill_price"]
            ts = fill["timestamp"]

            if position_qty == 0:
                # Opening a fresh position
                position_qty = fill_shares
                avg_entry_price = fill_price
                entry_time = ts
                continue

            same_direction = np.sign(fill_shares) == np.sign(position_qty)

            if same_direction:
                # Adding to the position — update weighted average entry price
                new_qty = position_qty + fill_shares
                avg_entry_price = (
                    avg_entry_price * abs(position_qty) + fill_price * abs(fill_shares)
                ) / abs(new_qty)
                position_qty = new_qty
                continue

            # Opposite direction: this fill closes some or all of the
            # existing position, and may also open a new one in the other
            # direction if the fill size exceeds the current position.
            closing_qty = min(abs(fill_shares), abs(position_qty))
            side = "LONG" if position_qty > 0 else "SHORT"
            pnl = (
                (fill_price - avg_entry_price) * closing_qty
                if side == "LONG"
                else (avg_entry_price - fill_price) * closing_qty
            )
            pnl_pct = pnl / (avg_entry_price * closing_qty) if avg_entry_price > 0 else 0.0

            holding_bars = index_pos.get(ts, 0) - index_pos.get(entry_time, 0)

            trades.append(
                RoundTripTrade(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=ts,
                    side=side,
                    entry_price=avg_entry_price,
                    exit_price=fill_price,
                    quantity=closing_qty,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=fill.get("reason", "SIGNAL"),
                    holding_bars=max(holding_bars, 0),
                )
            )

            remaining_fill = abs(fill_shares) - closing_qty
            if remaining_fill > 1e-9:
                # Fill was large enough to flip the position to the other side
                position_qty = remaining_fill * np.sign(fill_shares)
                avg_entry_price = fill_price
                entry_time = ts
            else:
                position_qty = position_qty + fill_shares  # closes exactly, or partially
                if abs(position_qty) < 1e-9:
                    position_qty = 0.0
                    avg_entry_price = 0.0
                    entry_time = None
                # else: partial close, avg_entry_price and entry_time unchanged

    return trades


def compute_trade_stats(trades: list[RoundTripTrade]) -> dict:
    """Standard trade-level statistics used to evaluate a strategy beyond
    just the equity curve — win rate, payoff ratio, expectancy, etc."""
    if not trades:
        return {
            "num_round_trips": 0,
            "win_rate_pct": None,
            "avg_win": None,
            "avg_loss": None,
            "payoff_ratio": None,
            "expectancy": None,
            "avg_holding_bars": None,
            "largest_win": None,
            "largest_loss": None,
        }

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = len(wins) / len(pnls) if len(pnls) else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "num_round_trips": len(trades),
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio is not None else None,
        "expectancy": round(expectancy, 2),
        "avg_holding_bars": round(float(np.mean([t.holding_bars for t in trades])), 1),
        "largest_win": round(float(pnls.max()), 2),
        "largest_loss": round(float(pnls.min()), 2),
    }


def trades_to_dataframe(trades: list[RoundTripTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol", "entry_time", "exit_time", "side", "entry_price",
                "exit_price", "quantity", "pnl", "pnl_pct", "exit_reason", "holding_bars",
            ]
        )
    return pd.DataFrame([t.__dict__ for t in trades])
