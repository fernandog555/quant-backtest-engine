import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Backtester, BacktestConfig
from src.risk.manager import RiskLimits
from src.strategies.base import Strategy
from src.strategies.buy_and_hold import BuyAndHold


class AlwaysFlat(Strategy):
    name = "AlwaysFlat"

    def generate_positions(self, bars):
        return pd.Series(0.0, index=bars.index)


@pytest.fixture
def flat_price_bars():
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    close = np.full(50, 100.0)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000}, index=dates
    )


@pytest.fixture
def uptrend_bars():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    close = np.linspace(100, 130, 100)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000}, index=dates
    )


class TestBacktesterBasics:
    def test_flat_strategy_preserves_capital(self, flat_price_bars):
        bt = Backtester(BacktestConfig(initial_capital=100_000))
        result = bt.run(flat_price_bars, AlwaysFlat(), symbol="TEST")
        assert result.equity_curve.iloc[-1] == pytest.approx(100_000, rel=1e-6)
        assert result.metrics["num_trades"] == 0

    def test_no_lookahead_bias(self, uptrend_bars):
        # BuyAndHold's first executed trade should happen on bar index 1,
        # not bar index 0 -- since signal at t=0 is only actable at t=1.
        bt = Backtester(BacktestConfig(initial_capital=100_000))
        result = bt.run(uptrend_bars, BuyAndHold(), symbol="TEST")
        first_trade_ts = result.trades.iloc[0]["timestamp"]
        assert first_trade_ts == uptrend_bars.index[1]

    def test_equity_curve_same_length_as_input(self, uptrend_bars):
        bt = Backtester(BacktestConfig())
        result = bt.run(uptrend_bars, BuyAndHold(), symbol="TEST")
        assert len(result.equity_curve) == len(uptrend_bars)

    def test_slippage_reduces_returns_vs_zero_slippage(self, uptrend_bars):
        bt_no_slip = Backtester(BacktestConfig(slippage_bps=0))
        bt_with_slip = Backtester(BacktestConfig(slippage_bps=50))

        result_no_slip = bt_no_slip.run(uptrend_bars, BuyAndHold(), symbol="TEST")
        result_with_slip = bt_with_slip.run(uptrend_bars, BuyAndHold(), symbol="TEST")

        assert result_with_slip.equity_curve.iloc[-1] < result_no_slip.equity_curve.iloc[-1]


class TestBacktesterRiskIntegration:
    def test_drawdown_halt_stops_new_trading(self, uptrend_bars):
        # Force an artificially tiny drawdown limit so it trips immediately,
        # and confirm equity flatlines (no further trading) after the halt.
        crash_bars = uptrend_bars.copy()
        crash_close = np.concatenate([np.full(20, 100.0), np.linspace(100, 50, 80)])
        crash_bars["close"] = crash_close
        crash_bars["open"] = crash_close
        crash_bars["high"] = crash_close
        crash_bars["low"] = crash_close

        config = BacktestConfig(
            initial_capital=100_000,
            risk_limits=RiskLimits(max_position_pct=1.0, max_gross_exposure_pct=1.0, max_drawdown_pct=0.10),
        )
        bt = Backtester(config)
        result = bt.run(crash_bars, BuyAndHold(), symbol="TEST")

        assert result.metrics["max_drawdown_pct"] <= -9  # confirms drawdown did occur
        # Once halted, shares_held should stop changing (no re-entry)
        halt_idx = (result.equity_curve / result.equity_curve.cummax() - 1 <= -0.10).idxmax()
        post_halt_positions = result.positions.loc[halt_idx:]
        assert post_halt_positions.nunique() <= 1
