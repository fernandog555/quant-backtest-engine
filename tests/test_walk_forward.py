import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig
from src.backtest.walk_forward import WalkForwardValidator
from src.risk.manager import RiskLimits
from src.strategies.buy_and_hold import BuyAndHold
from src.strategies.moving_average_crossover import MovingAverageCrossover


@pytest.fixture
def long_bars():
    dates = pd.date_range("2021-01-01", periods=400, freq="B")
    np.random.seed(3)
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, 400)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995, "close": close, "volume": 1_000_000},
        index=dates,
    )


class TestWindowSplitting:
    def test_rolling_windows_have_fixed_train_size(self, long_bars):
        wf = WalkForwardValidator(train_bars=100, test_bars=25, anchored=False)
        windows = wf.split_windows(long_bars)
        for train_start, train_end, test_start, test_end in windows:
            assert train_end - train_start == 100
            assert test_end - test_start == 25

    def test_anchored_windows_grow_train_size(self, long_bars):
        wf = WalkForwardValidator(train_bars=100, test_bars=25, anchored=True)
        windows = wf.split_windows(long_bars)
        train_sizes = [train_end - train_start for train_start, train_end, _, _ in windows]
        assert train_sizes == sorted(train_sizes)  # non-decreasing
        assert train_sizes[0] == 0 or train_sizes[0] >= 100  # anchored always starts at 0

    def test_windows_are_non_overlapping_in_test_period(self, long_bars):
        wf = WalkForwardValidator(train_bars=100, test_bars=25, anchored=False)
        windows = wf.split_windows(long_bars)
        for i in range(len(windows) - 1):
            assert windows[i][3] == windows[i + 1][2]  # test_end of one == test_start of next

    def test_raises_if_not_enough_data(self):
        wf = WalkForwardValidator(train_bars=1000, test_bars=100)
        tiny_bars = pd.DataFrame(
            {"open": [1] * 10, "high": [1] * 10, "low": [1] * 10, "close": [1] * 10, "volume": [1] * 10},
            index=pd.date_range("2023-01-01", periods=10),
        )
        with pytest.raises(ValueError):
            wf.run(tiny_bars, lambda: BuyAndHold())


class TestWalkForwardRun:
    def test_produces_one_row_per_window(self, long_bars):
        wf = WalkForwardValidator(
            backtest_config=BacktestConfig(risk_limits=RiskLimits(max_position_pct=0.9)),
            train_bars=100,
            test_bars=25,
        )
        report = wf.run(long_bars, lambda: BuyAndHold(), symbol="TEST")
        assert len(report.per_window_metrics) == len(wf.split_windows(long_bars))

    def test_combined_metrics_present(self, long_bars):
        wf = WalkForwardValidator(train_bars=100, test_bars=25)
        report = wf.run(long_bars, lambda: BuyAndHold(), symbol="TEST")
        assert "pct_windows_profitable" in report.combined_metrics
        assert "num_windows" in report.combined_metrics
        assert 0 <= report.combined_metrics["pct_windows_profitable"] <= 100

    def test_test_windows_do_not_overlap_across_report(self, long_bars):
        wf = WalkForwardValidator(train_bars=100, test_bars=25)
        report = wf.run(long_bars, lambda: MovingAverageCrossover(10, 30), symbol="TEST")
        for i in range(len(report.windows) - 1):
            assert report.windows[i].test_end < report.windows[i + 1].test_start

    def test_strategy_factory_receiving_train_slice(self, long_bars):
        # Strategies that want to fit params should receive only the train
        # slice, never data from the test window (which would be lookahead
        # at the walk-forward level).
        seen_train_slices = []

        def factory(train_slice):
            seen_train_slices.append(train_slice)
            return BuyAndHold()

        wf = WalkForwardValidator(train_bars=100, test_bars=25)
        wf.run(long_bars, factory, symbol="TEST")

        windows = wf.split_windows(long_bars)
        assert len(seen_train_slices) == len(windows)
        for (train_start, train_end, test_start, _), slice_ in zip(windows, seen_train_slices):
            assert len(slice_) == train_end - train_start
            # confirm no overlap with test data
            assert slice_.index[-1] < long_bars.index[test_start]
