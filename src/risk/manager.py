"""
Risk management: converts a strategy's directional signal (-1 to 1) into an
actual position size, and enforces hard limits that override the strategy
entirely when tripped. This sits between strategy and execution in both
backtest and live paths, so risk logic is never accidentally bypassed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_position_pct: float = 0.20       # max % of equity in a single symbol
    max_gross_exposure_pct: float = 1.0  # max % of equity deployed across all positions
    max_daily_loss_pct: float = 0.03     # halt new entries for the day past this drawdown
    max_drawdown_pct: float = 0.15       # halt trading entirely past this drawdown from peak
    per_trade_stop_loss_pct: float = 0.05  # exit a position if it moves this far against entry


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self._peak_equity: float | None = None
        self._day_start_equity: float | None = None
        self._trading_halted = False

    def reset_day(self, equity: float) -> None:
        self._day_start_equity = equity

    def update_peak(self, equity: float) -> None:
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    def check_halt(self, equity: float) -> bool:
        """Returns True if trading should be halted given current equity."""
        self.update_peak(equity)

        if self._peak_equity:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.limits.max_drawdown_pct:
                self._trading_halted = True

        if self._day_start_equity:
            daily_loss = (self._day_start_equity - equity) / self._day_start_equity
            if daily_loss >= self.limits.max_daily_loss_pct:
                self._trading_halted = True

        return self._trading_halted

    def size_position(
        self,
        signal_strength: float,
        equity: float,
        price: float,
        current_gross_exposure_pct: float = 0.0,
    ) -> float:
        """
        Converts a directional signal into a target quantity (in shares).
        signal_strength: -1 to 1 (direction and conviction from the strategy)
        Returns 0 if trading is halted or limits are already breached.
        """
        if self._trading_halted:
            return 0.0

        if abs(signal_strength) < 1e-9:
            return 0.0

        # Cap this position's own allocation
        target_pct = min(abs(signal_strength) * self.limits.max_position_pct, self.limits.max_position_pct)

        # Cap remaining room under gross exposure limit
        remaining_room = max(self.limits.max_gross_exposure_pct - current_gross_exposure_pct, 0.0)
        target_pct = min(target_pct, remaining_room)

        target_dollars = target_pct * equity
        quantity = target_dollars / price if price > 0 else 0.0

        return quantity if signal_strength > 0 else -quantity

    def stop_loss_triggered(self, entry_price: float, current_price: float, side: int) -> bool:
        """side: 1 for long, -1 for short."""
        if entry_price <= 0:
            return False
        pct_move = (current_price - entry_price) / entry_price
        if side > 0:
            return pct_move <= -self.limits.per_trade_stop_loss_pct
        else:
            return pct_move >= self.limits.per_trade_stop_loss_pct
