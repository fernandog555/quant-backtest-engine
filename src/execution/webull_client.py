"""
Execution layer wrapping Webull's OpenAPI.

Design principles:
- Defaults to sandbox (UAT). Live trading requires an explicit, separate
  opt-in — never inferred from config file presence alone.
- Every order goes through preview before submission.
- This module intentionally does NOT read strategy signals directly; it
  only accepts already-sized Orders from the risk manager, so risk logic
  can never be bypassed by a strategy calling execution directly.

Requires: pip install webull-openapi-python-sdk
Credentials via environment variables (see .env.example) — never hardcode
WEBULL_APP_KEY / WEBULL_APP_SECRET in source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.utils.models import Order, OrderSide, OrderStatus, OrderType


class LiveTradingNotEnabledError(Exception):
    """Raised when an order would execute against production but the
    explicit safety opt-in has not been set."""


@dataclass
class WebullConfig:
    app_key: str
    app_secret: str
    region_id: str = "us"
    environment: str = "sandbox"  # "sandbox" or "prod" — sandbox is the safe default
    # Second, independent confirmation required to ever route to prod.
    # Prevents a single mistyped env var from accidentally going live.
    confirm_live_trading: bool = False

    @classmethod
    def from_env(cls) -> "WebullConfig":
        return cls(
            app_key=os.environ["WEBULL_APP_KEY"],
            app_secret=os.environ["WEBULL_APP_SECRET"],
            region_id=os.environ.get("WEBULL_REGION_ID", "us"),
            environment=os.environ.get("WEBULL_ENVIRONMENT", "sandbox"),
            confirm_live_trading=os.environ.get("WEBULL_CONFIRM_LIVE", "false").lower() == "true",
        )

    @property
    def is_live(self) -> bool:
        return self.environment == "prod"


class WebullExecutionClient:
    """
    Thin wrapper around the Webull OpenAPI SDK. Kept separate from strategy
    and risk code so it can be swapped for a different broker without
    touching anything upstream.
    """

    def __init__(self, config: WebullConfig):
        self.config = config

        if config.is_live and not config.confirm_live_trading:
            raise LiveTradingNotEnabledError(
                "WEBULL_ENVIRONMENT=prod is set but WEBULL_CONFIRM_LIVE=true was not. "
                "This is a deliberate double-confirmation to prevent accidental live trading. "
                "Set WEBULL_CONFIRM_LIVE=true only when you intend to trade real capital."
            )

        self._client = self._build_sdk_client()

    def _build_sdk_client(self):
        # Import kept local so the rest of the package works without the SDK
        # installed (useful for pure backtesting environments).
        from webull_openapi.client import WebullClient  # type: ignore

        return WebullClient(
            app_key=self.config.app_key,
            app_secret=self.config.app_secret,
            region_id=self.config.region_id,
            env=self.config.environment,
        )

    def get_account_positions(self) -> list[dict]:
        return self._client.account.get_positions()

    def get_buying_power(self) -> float:
        account = self._client.account.get_balance()
        return float(account.get("buying_power", 0.0))

    def preview_order(self, order: Order) -> dict:
        """Always call before submit_order. Returns estimated fill details,
        fees, and any warnings from Webull before anything goes live."""
        return self._client.trading.preview_stock_order(
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
        )

    def submit_order(self, order: Order, skip_preview: bool = False) -> Order:
        if not skip_preview:
            self.preview_order(order)

        response = self._client.trading.place_stock_order(
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
        )

        order.order_id = response.get("order_id")
        order.status = OrderStatus.PENDING
        return order

    def cancel_order(self, order_id: str) -> bool:
        result = self._client.trading.cancel_order(order_id=order_id)
        return bool(result.get("success", False))
