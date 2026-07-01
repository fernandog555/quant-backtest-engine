"""Core data models shared across the project."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Bar:
    """A single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    """Emitted by a strategy: what it wants to do, not what actually happens."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    strength: float = 1.0  # 0-1, lets strategies express conviction / position sizing hints
    reason: str = ""


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    timestamp: datetime | None = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    order_id: str | None = None


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-9


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    positions_value: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def equity(self) -> float:
        return self.cash + self.positions_value
