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
        self._day_start_date: str | None = None
        self._trading_halted = False

    def reset_day(self, equity: float) -> None:
        from datetime import date
        self._day_start_equity = equity
        self._day_start_date = date.today().isoformat()

    def to_state(self):
        from src.risk.state_store import RiskManagerState
        return RiskManagerState(
            peak_equity=self._peak_equity,
            day_start_equity=self._day_start_equity,
            day_start_date=self._day_start_date,
            trading_halted=self._trading_halted,
        )

    def load_state(self, state) -> None:
        """Restore state from a prior process (see RiskStateStore). If the
        loaded state is from a previous calendar day, day_start_equity is
        NOT restored as-is — call reset_day() with fresh equity afterward
        to start the new day's tracking correctly. peak_equity and the
        halted flag persist across days by design (a halt should require
        deliberate human review to clear, not just a date rollover)."""
        self._peak_equity = state.peak_equity
        self._trading_halted = state.trading_halted

        from src.risk.state_store import RiskStateStore
        if not RiskStateStore.is_new_day(state):
            self._day_start_equity = state.day_start_equity
            self._day_start_date = state.day_start_date
        # else: leave day fields unset; caller should invoke reset_day()

    def manually_clear_halt(self) -> None:
        """Explicit, deliberately-named method to resume trading after a
        halt. Not called automatically anywhere — a human should decide
        this, ideally after understanding why the halt triggered."""
        self._trading_halted = False

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
