"""
Orchestrator that ties strategy signals, risk sizing, and execution together
for live or sandbox trading. This mirrors the backtester's logic as closely
as possible so behavior doesn't silently diverge between backtest and live.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.execution.webull_client import WebullExecutionClient
from src.risk.manager import RiskManager
from src.strategies.base import Strategy
from src.utils.models import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        execution_client: WebullExecutionClient,
        symbol: str,
    ):
        self.strategy = strategy
        self.risk = risk_manager
        self.execution = execution_client
        self.symbol = symbol

    def run_once(self, bars: pd.DataFrame) -> Order | None:
        """
        Call this on a schedule (e.g. once per bar close) with the latest
        historical bars including the current one. Computes the target
        signal, sizes it through risk management, and submits an order if
        a change in position is warranted.
        """
        signal_strength = self.strategy.generate_latest_signal(bars)
        latest_price = float(bars["close"].iloc[-1])

        buying_power = self.execution.get_buying_power()
        positions = self.execution.get_account_positions()
        current_position = next((p for p in positions if p.get("symbol") == self.symbol), None)
        current_shares = float(current_position["quantity"]) if current_position else 0.0

        equity = buying_power + sum(
            float(p.get("quantity", 0)) * float(p.get("last_price", 0)) for p in positions
        )

        halted = self.risk.check_halt(equity)
        if halted:
            logger.warning("Trading halted by risk manager. No orders will be submitted.")
            return None

        target_qty = self.risk.size_position(
            signal_strength=signal_strength,
            equity=equity,
            price=latest_price,
        )

        delta = target_qty - current_shares
        if abs(delta) < 1e-6:
            logger.info("No position change required (target=%s, current=%s)", target_qty, current_shares)
            return None

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order = Order(
            symbol=self.symbol,
            side=side,
            quantity=abs(delta),
            order_type=OrderType.MARKET,
        )

        preview = self.execution.preview_order(order)
        logger.info("Order preview: %s", preview)

        submitted = self.execution.submit_order(order)
        logger.info("Order submitted: %s", submitted)
        return submitted
