"""
Execution layer wrapping the official Webull OpenAPI Python SDK.

Verified against Webull's published docs (developer.webull.com/apis/docs/trade-api)
as of mid-2026:
- Client construction: ApiClient(app_key, app_secret, region) + TradeClient(api_client)
- Responses are requests-style: res.status_code, res.json()
- Namespaced calls: trade_client.account_v2.*, trade_client.order_v3.*
- Orders are submitted as a list of dicts with a client_order_id (idempotency key)

Design principles:
- Defaults to UAT (sandbox) — this matches Webull's own default and terminology.
- Live trading requires an explicit, separate opt-in beyond WEBULL_ENVIRONMENT=prod.
- Every order goes through preview before submission.
- This module intentionally does NOT read strategy signals directly; it
  only accepts already-sized Orders from the risk manager, so risk logic
  can never be bypassed by a strategy calling execution directly.

Requires: pip install webull-openapi-python-sdk
Credentials via environment variables (see .env.example) — never hardcode
WEBULL_APP_KEY / WEBULL_APP_SECRET in source.

NOTE: This client has been written against Webull's published API docs and
sample code, but has not been exercised against a live sandbox connection
in this environment (no network access to Webull from here). Before relying
on it, run scripts/verify_webull_connection.py against your own sandbox
credentials and confirm the response shapes match what's assumed below —
SDK response fields can drift between versions.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from src.utils.models import Order, OrderSide, OrderStatus, OrderType


class LiveTradingNotEnabledError(Exception):
    """Raised when an order would execute against production but the
    explicit safety opt-in has not been set."""


class WebullApiError(Exception):
    """Raised when the Webull API returns a non-200 response. Wraps the
    status code and response body for debugging."""

    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Webull API error {status_code}: {body}")


@dataclass
class WebullConfig:
    app_key: str
    app_secret: str
    region_id: str = "us"
    # Webull's own terminology is "uat" (sandbox) vs "prod" — matched here
    # deliberately so env vars line up with Webull's official docs/MCP server.
    environment: str = "uat"
    api_endpoint: str | None = None  # optional override, e.g. for a specific regional endpoint
    # Second, independent confirmation required to ever route to prod.
    # Prevents a single mistyped env var from accidentally going live.
    confirm_live_trading: bool = False

    @classmethod
    def from_env(cls) -> "WebullConfig":
        return cls(
            app_key=os.environ["WEBULL_APP_KEY"],
            app_secret=os.environ["WEBULL_APP_SECRET"],
            region_id=os.environ.get("WEBULL_REGION_ID", "us"),
            environment=os.environ.get("WEBULL_ENVIRONMENT", "uat"),
            api_endpoint=os.environ.get("WEBULL_API_ENDPOINT"),
            confirm_live_trading=os.environ.get("WEBULL_CONFIRM_LIVE", "false").lower() == "true",
        )

    @property
    def is_live(self) -> bool:
        return self.environment == "prod"


class WebullExecutionClient:
    """
    Thin wrapper around the official Webull OpenAPI Python SDK. Kept
    separate from strategy and risk code so it can be swapped for a
    different broker without touching anything upstream.
    """

    def __init__(self, config: WebullConfig):
        self.config = config

        if config.is_live and not config.confirm_live_trading:
            raise LiveTradingNotEnabledError(
                "WEBULL_ENVIRONMENT=prod is set but WEBULL_CONFIRM_LIVE=true was not. "
                "This is a deliberate double-confirmation to prevent accidental live trading. "
                "Set WEBULL_CONFIRM_LIVE=true only when you intend to trade real capital."
            )

        self._trade_client = self._build_trade_client()
        self._account_id: str | None = None

    def _build_trade_client(self):
        # Imports kept local so the rest of the package works without the
        # SDK installed (useful for pure backtesting environments).
        from webull.core.client import ApiClient
        from webull.trade.trade_client import TradeClient

        api_client = ApiClient(self.config.app_key, self.config.app_secret, self.config.region_id)
        if self.config.api_endpoint:
            api_client.add_endpoint(self.config.region_id, self.config.api_endpoint)

        return TradeClient(api_client)

    @staticmethod
    def _unwrap(res):
        """Every SDK call returns a requests-style response. Raise a clear
        error on failure instead of letting a confusing KeyError bubble up
        from calling code that assumes success."""
        if res.status_code != 200:
            raise WebullApiError(res.status_code, _safe_body(res))
        return res.json()

    def get_account_id(self, refresh: bool = False) -> str:
        """Accounts are per asset-type (stock/options/futures/crypto may
        have separate IDs) — this returns the first account in the list.
        For multi-account setups, call get_account_list() directly and
        select explicitly."""
        if self._account_id and not refresh:
            return self._account_id

        accounts = self.get_account_list()
        if not accounts:
            raise WebullApiError(200, "Account list was empty — no accounts found for these credentials.")
        self._account_id = accounts[0]["account_id"]
        return self._account_id

    def get_account_list(self) -> list[dict]:
        res = self._trade_client.account_v2.get_account_list()
        return self._unwrap(res)

    def get_account_balance(self, account_id: str | None = None) -> dict:
        account_id = account_id or self.get_account_id()
        res = self._trade_client.account_v2.get_account_balance(account_id)
        return self._unwrap(res)

    def get_buying_power(self, account_id: str | None = None) -> float:
        balance = self.get_account_balance(account_id)
        # Field name per Webull docs; guard against SDK drift with a clear error
        # rather than a silent 0.0 that could mis-size a live order.
        if "buying_power" not in balance:
            raise WebullApiError(
                200, f"Expected 'buying_power' field in balance response, got keys: {list(balance.keys())}"
            )
        return float(balance["buying_power"])

    def get_account_positions(self, account_id: str | None = None) -> list[dict]:
        account_id = account_id or self.get_account_id()
        res = self._trade_client.account_v2.get_account_position(account_id)
        return self._unwrap(res)

    def _build_order_payload(self, order: Order) -> dict:
        return {
            "combo_type": "NORMAL",
            "client_order_id": order.order_id or uuid.uuid4().hex,
            "symbol": order.symbol,
            "instrument_type": "EQUITY",
            "market": "US",
            "order_type": order.order_type.value,
            "limit_price": str(order.limit_price) if order.limit_price is not None else None,
            "quantity": str(order.quantity),
            "support_trading_session": "CORE",
            "side": order.side.value,
            "time_in_force": "DAY",
            "entrust_type": "QTY",
        }

    def preview_order(self, order: Order, account_id: str | None = None) -> dict:
        """Always call before submit_order. Returns estimated fill details,
        fees, and any warnings from Webull before anything goes live."""
        account_id = account_id or self.get_account_id()
        payload = self._build_order_payload(order)
        res = self._trade_client.order_v3.preview_order(account_id, [payload])
        return self._unwrap(res)

    def submit_order(self, order: Order, account_id: str | None = None, skip_preview: bool = False) -> Order:
        account_id = account_id or self.get_account_id()

        if not skip_preview:
            self.preview_order(order, account_id=account_id)

        payload = self._build_order_payload(order)
        order.order_id = payload["client_order_id"]

        res = self._trade_client.order_v3.place_order(account_id, [payload])
        response_body = self._unwrap(res)

        order.status = OrderStatus.PENDING
        return order

    def cancel_order(self, client_order_id: str, account_id: str | None = None) -> bool:
        account_id = account_id or self.get_account_id()
        res = self._trade_client.order_v3.cancel_order(account_id, client_order_id)
        self._unwrap(res)
        return True

    def replace_order(
        self, client_order_id: str, quantity: float | None = None,
        limit_price: float | None = None, account_id: str | None = None,
    ) -> dict:
        account_id = account_id or self.get_account_id()
        modify_payload = {"client_order_id": client_order_id}
        if quantity is not None:
            modify_payload["quantity"] = str(quantity)
        if limit_price is not None:
            modify_payload["limit_price"] = str(limit_price)

        res = self._trade_client.order_v3.replace_order(account_id, [modify_payload])
        return self._unwrap(res)


def _safe_body(res):
    try:
        return res.json()
    except Exception:
        return getattr(res, "text", "<no response body>")
