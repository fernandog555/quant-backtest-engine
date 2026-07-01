import pytest

from src.execution.webull_client import (
    WebullConfig,
    WebullExecutionClient,
    LiveTradingNotEnabledError,
    WebullApiError,
)
from src.utils.models import Order, OrderSide, OrderType


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class TestWebullConfig:
    def test_defaults_to_uat(self):
        config = WebullConfig(app_key="k", app_secret="s")
        assert config.environment == "uat"
        assert config.is_live is False

    def test_is_live_true_only_for_prod(self):
        config = WebullConfig(app_key="k", app_secret="s", environment="prod")
        assert config.is_live is True

    def test_from_env_reads_variables(self, monkeypatch):
        monkeypatch.setenv("WEBULL_APP_KEY", "test_key")
        monkeypatch.setenv("WEBULL_APP_SECRET", "test_secret")
        monkeypatch.setenv("WEBULL_REGION_ID", "hk")
        monkeypatch.setenv("WEBULL_ENVIRONMENT", "prod")
        monkeypatch.setenv("WEBULL_CONFIRM_LIVE", "true")

        config = WebullConfig.from_env()
        assert config.app_key == "test_key"
        assert config.region_id == "hk"
        assert config.is_live is True
        assert config.confirm_live_trading is True

    def test_from_env_defaults_confirm_live_false(self, monkeypatch):
        monkeypatch.setenv("WEBULL_APP_KEY", "k")
        monkeypatch.setenv("WEBULL_APP_SECRET", "s")
        monkeypatch.delenv("WEBULL_CONFIRM_LIVE", raising=False)
        config = WebullConfig.from_env()
        assert config.confirm_live_trading is False


class TestLiveTradingGuard:
    def test_raises_when_prod_without_confirmation(self):
        config = WebullConfig(app_key="k", app_secret="s", environment="prod", confirm_live_trading=False)
        with pytest.raises(LiveTradingNotEnabledError):
            WebullExecutionClient(config)

    def test_uat_does_not_require_confirmation(self, monkeypatch):
        # UAT should attempt to build the SDK client (which will fail here
        # since the SDK isn't installed / no real creds) rather than raising
        # LiveTradingNotEnabledError -- confirm we get past the guard.
        config = WebullConfig(app_key="k", app_secret="s", environment="uat")
        with pytest.raises(ImportError):
            # Expected: SDK not installed in test environment. The important
            # assertion is that this is NOT a LiveTradingNotEnabledError.
            WebullExecutionClient(config)


class TestOrderPayloadBuilding:
    @pytest.fixture
    def client(self, monkeypatch):
        # Bypass SDK construction entirely to test pure payload-building logic
        config = WebullConfig(app_key="k", app_secret="s", environment="uat")
        instance = WebullExecutionClient.__new__(WebullExecutionClient)
        instance.config = config
        instance._trade_client = None
        instance._account_id = "acct123"
        return instance

    def test_market_order_payload(self, client):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET)
        payload = client._build_order_payload(order)
        assert payload["symbol"] == "AAPL"
        assert payload["side"] == "BUY"
        assert payload["quantity"] == "10"
        assert payload["order_type"] == "MARKET"
        assert payload["limit_price"] is None

    def test_limit_order_payload_includes_price(self, client):
        order = Order(
            symbol="TSLA", side=OrderSide.SELL, quantity=5,
            order_type=OrderType.LIMIT, limit_price=250.50,
        )
        payload = client._build_order_payload(order)
        assert payload["limit_price"] == "250.5"
        assert payload["order_type"] == "LIMIT"

    def test_payload_generates_client_order_id_if_missing(self, client):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1)
        payload = client._build_order_payload(order)
        assert payload["client_order_id"] is not None
        assert len(payload["client_order_id"]) > 0

    def test_payload_reuses_existing_order_id(self, client):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1, order_id="existing-id-123")
        payload = client._build_order_payload(order)
        assert payload["client_order_id"] == "existing-id-123"


class TestResponseUnwrapping:
    def test_unwrap_success_returns_json(self):
        res = FakeResponse(200, {"foo": "bar"})
        assert WebullExecutionClient._unwrap(res) == {"foo": "bar"}

    def test_unwrap_failure_raises_webull_api_error(self):
        res = FakeResponse(401, {"error": "unauthorized"})
        with pytest.raises(WebullApiError) as exc_info:
            WebullExecutionClient._unwrap(res)
        assert exc_info.value.status_code == 401

    def test_get_buying_power_raises_clear_error_on_missing_field(self, monkeypatch):
        config = WebullConfig(app_key="k", app_secret="s", environment="uat")
        instance = WebullExecutionClient.__new__(WebullExecutionClient)
        instance.config = config
        instance._account_id = "acct123"
        instance.get_account_balance = lambda account_id=None: {"unexpected_field": 123}

        with pytest.raises(WebullApiError) as exc_info:
            instance.get_buying_power()
        assert "buying_power" in str(exc_info.value)
