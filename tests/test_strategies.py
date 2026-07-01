import numpy as np
import pandas as pd
import pytest

from src.strategies.buy_and_hold import BuyAndHold
from src.strategies.moving_average_crossover import MovingAverageCrossover
from src.strategies.rsi_mean_reversion import RSIMeanReversion


@pytest.fixture
def trending_bars():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    # Clean uptrend so MA crossover behavior is deterministic
    close = np.linspace(100, 150, 100)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1_000_000},
        index=dates,
    )


@pytest.fixture
def choppy_bars():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(1)
    close = 100 + np.cumsum(np.random.normal(0, 1, 100))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1_000_000},
        index=dates,
    )


class TestBuyAndHold:
    def test_always_fully_long(self, trending_bars):
        positions = BuyAndHold().generate_positions(trending_bars)
        assert (positions == 1.0).all()


class TestMovingAverageCrossover:
    def test_rejects_invalid_windows(self):
        with pytest.raises(ValueError):
            MovingAverageCrossover(fast_window=50, slow_window=20)

    def test_no_signal_before_slow_window_warms_up(self, trending_bars):
        strat = MovingAverageCrossover(fast_window=5, slow_window=20)
        positions = strat.generate_positions(trending_bars)
        assert (positions.iloc[:19] == 0.0).all()

    def test_goes_long_in_uptrend(self, trending_bars):
        strat = MovingAverageCrossover(fast_window=5, slow_window=20)
        positions = strat.generate_positions(trending_bars)
        # After warmup, a clean uptrend should be long
        assert positions.iloc[-1] == 1.0

    def test_short_disabled_by_default(self, trending_bars):
        # Reverse the trend to a downtrend
        down_bars = trending_bars.copy()
        down_bars["close"] = down_bars["close"].to_numpy()[::-1]
        strat = MovingAverageCrossover(fast_window=5, slow_window=20, allow_short=False)
        positions = strat.generate_positions(down_bars)
        assert (positions >= 0).all()  # never short when disabled


class TestRSIMeanReversion:
    def test_no_lookahead_length_mismatch(self, choppy_bars):
        strat = RSIMeanReversion()
        positions = strat.generate_positions(choppy_bars)
        assert len(positions) == len(choppy_bars)

    def test_positions_bounded(self, choppy_bars):
        strat = RSIMeanReversion(allow_short=True)
        positions = strat.generate_positions(choppy_bars)
        assert positions.isin([-1.0, 0.0, 1.0]).all()

    def test_does_not_mutate_input(self, choppy_bars):
        # Regression test for the read-only numpy array bug found during dev
        original = choppy_bars.copy()
        RSIMeanReversion().generate_positions(choppy_bars)
        pd.testing.assert_frame_equal(choppy_bars, original)
