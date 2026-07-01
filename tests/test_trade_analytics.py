import pandas as pd
import pytest

from src.backtest.trade_analytics import pair_trades, compute_trade_stats, trades_to_dataframe


def make_fills(rows):
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "shares", "fill_price", "reason"])


@pytest.fixture
def bar_index():
    return pd.date_range("2023-01-01", periods=10, freq="D")


class TestSimpleRoundTrip:
    def test_single_long_entry_and_exit(self, bar_index):
        fills = make_fills([
            (bar_index[0], "AAPL", 100, 50.0, "SIGNAL"),
            (bar_index[5], "AAPL", -100, 55.0, "SIGNAL"),
        ])
        trades = pair_trades(fills, bar_index)
        assert len(trades) == 1
        t = trades[0]
        assert t.side == "LONG"
        assert t.entry_price == 50.0
        assert t.exit_price == 55.0
        assert t.pnl == pytest.approx((55.0 - 50.0) * 100)
        assert t.holding_bars == 5

    def test_single_short_entry_and_exit(self, bar_index):
        fills = make_fills([
            (bar_index[0], "AAPL", -100, 50.0, "SIGNAL"),
            (bar_index[3], "AAPL", 100, 45.0, "SIGNAL"),
        ])
        trades = pair_trades(fills, bar_index)
        assert len(trades) == 1
        t = trades[0]
        assert t.side == "SHORT"
        # Short profits when price drops
        assert t.pnl == pytest.approx((50.0 - 45.0) * 100)

    def test_losing_trade_has_negative_pnl(self, bar_index):
        fills = make_fills([
            (bar_index[0], "AAPL", 100, 50.0, "SIGNAL"),
            (bar_index[2], "AAPL", -100, 45.0, "STOP_LOSS"),
        ])
        trades = pair_trades(fills, bar_index)
        assert trades[0].pnl < 0
        assert trades[0].exit_reason == "STOP_LOSS"


class TestPositionFlip:
    def test_long_to_short_flip_in_one_fill(self, bar_index):
        # Long 100 shares, then a single -200 fill both closes the long
        # and opens a new short of 100
        fills = make_fills([
            (bar_index[0], "AAPL", 100, 50.0, "SIGNAL"),
            (bar_index[2], "AAPL", -200, 55.0, "SIGNAL"),
            (bar_index[4], "AAPL", 100, 52.0, "SIGNAL"),  # closes the short
        ])
        trades = pair_trades(fills, bar_index)
        assert len(trades) == 2
        assert trades[0].side == "LONG"
        assert trades[1].side == "SHORT"
        # Short entered at 55, closed at 52 -> profit
        assert trades[1].pnl == pytest.approx((55.0 - 52.0) * 100)


class TestPositionScaling:
    def test_adding_to_position_updates_avg_entry(self, bar_index):
        fills = make_fills([
            (bar_index[0], "AAPL", 100, 50.0, "SIGNAL"),
            (bar_index[1], "AAPL", 100, 60.0, "SIGNAL"),  # add to position
            (bar_index[2], "AAPL", -200, 70.0, "SIGNAL"),  # close all
        ])
        trades = pair_trades(fills, bar_index)
        assert len(trades) == 1
        # avg entry should be (50*100 + 60*100) / 200 = 55
        assert trades[0].entry_price == pytest.approx(55.0)

    def test_partial_close_keeps_position_open(self, bar_index):
        fills = make_fills([
            (bar_index[0], "AAPL", 100, 50.0, "SIGNAL"),
            (bar_index[1], "AAPL", -50, 55.0, "SIGNAL"),  # partial close
            (bar_index[2], "AAPL", -50, 60.0, "SIGNAL"),  # close remainder
        ])
        trades = pair_trades(fills, bar_index)
        assert len(trades) == 2
        assert trades[0].quantity == 50
        assert trades[1].quantity == 50


class TestEmptyAndEdgeCases:
    def test_empty_fills_returns_empty(self, bar_index):
        assert pair_trades(pd.DataFrame(), bar_index) == []

    def test_open_position_with_no_exit_produces_no_trade(self, bar_index):
        fills = make_fills([(bar_index[0], "AAPL", 100, 50.0, "SIGNAL")])
        trades = pair_trades(fills, bar_index)
        assert trades == []  # no exit yet, nothing to report


class TestTradeStats:
    def test_stats_on_no_trades(self):
        stats = compute_trade_stats([])
        assert stats["num_round_trips"] == 0
        assert stats["win_rate_pct"] is None

    def test_win_rate_and_payoff_ratio(self, bar_index):
        fills = make_fills([
            (bar_index[0], "AAPL", 100, 50.0, "SIGNAL"),
            (bar_index[1], "AAPL", -100, 60.0, "SIGNAL"),  # win: +1000
            (bar_index[2], "AAPL", 100, 60.0, "SIGNAL"),
            (bar_index[3], "AAPL", -100, 55.0, "SIGNAL"),  # loss: -500
        ])
        trades = pair_trades(fills, bar_index)
        stats = compute_trade_stats(trades)
        assert stats["num_round_trips"] == 2
        assert stats["win_rate_pct"] == 50.0
        assert stats["avg_win"] == pytest.approx(1000.0)
        assert stats["avg_loss"] == pytest.approx(-500.0)
        assert stats["payoff_ratio"] == pytest.approx(2.0)

    def test_trades_to_dataframe_empty(self):
        df = trades_to_dataframe([])
        assert df.empty
        assert "pnl" in df.columns

    def test_trades_to_dataframe_populated(self, bar_index):
        fills = make_fills([
            (bar_index[0], "AAPL", 100, 50.0, "SIGNAL"),
            (bar_index[1], "AAPL", -100, 55.0, "SIGNAL"),
        ])
        trades = pair_trades(fills, bar_index)
        df = trades_to_dataframe(trades)
        assert len(df) == 1
        assert df.iloc[0]["pnl"] == pytest.approx(500.0)
