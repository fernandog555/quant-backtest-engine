"""
Walk-forward validation: splits historical data into rolling train/test
windows and runs the backtester on each out-of-sample test window.

This is the single most important tool for catching overfitting. A strategy
that looks great on one static backtest window might just be tuned to that
window's specific noise. Walk-forward forces the strategy to prove itself
on data it was never "seen" fitting against, repeated across multiple
non-overlapping periods.

Two modes:
- Anchored: train window always starts at the beginning of the data and grows
- Rolling: train window is a fixed size and slides forward

Note: the strategies in this project (MA crossover, RSI) have fixed,
externally-specified parameters — they don't "fit" to the train window
internally. For those, walk-forward here mainly validates that performance
is consistent across different market regimes/periods, not that the
strategy avoided overfitting its own parameters. If you build a strategy
that *does* fit parameters (e.g. grid-searching MA windows), the train
window is where that search should happen — using only the train slice,
then validating on the untouched test slice.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import Backtester, BacktestConfig, BacktestResult
from src.backtest.trade_analytics import pair_trades
from src.strategies.base import Strategy


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: BacktestResult


@dataclass
class WalkForwardReport:
    windows: list[WalkForwardWindow]
    combined_metrics: dict
    per_window_metrics: pd.DataFrame


class WalkForwardValidator:
    def __init__(
        self,
        backtest_config: BacktestConfig | None = None,
        train_bars: int = 252,
        test_bars: int = 63,
        anchored: bool = False,
    ):
        """
        train_bars: size of the training window (informational for strategies
                    that don't use it, required for parameter-fitting ones)
        test_bars: size of each out-of-sample evaluation window
        anchored: if True, train window always starts at bar 0 and grows;
                  if False, it's a fixed-size window that slides forward
        """
        self.backtest_config = backtest_config or BacktestConfig()
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.anchored = anchored

    def split_windows(self, bars: pd.DataFrame) -> list[tuple[int, int, int, int]]:
        """Returns list of (train_start_idx, train_end_idx, test_start_idx, test_end_idx)."""
        n = len(bars)
        windows = []
        test_start = self.train_bars

        while test_start + self.test_bars <= n:
            train_start = 0 if self.anchored else test_start - self.train_bars
            train_end = test_start
            test_end = test_start + self.test_bars
            windows.append((train_start, train_end, test_start, test_end))
            test_start += self.test_bars

        return windows

    def run(
        self,
        bars: pd.DataFrame,
        strategy_factory,
        symbol: str = "SYMBOL",
    ) -> WalkForwardReport:
        """
        strategy_factory: zero-arg callable returning a fresh Strategy instance
                           (or a function of the train slice, for strategies that
                           fit parameters — signature: (train_bars_df) -> Strategy)
        """
        idx_windows = self.split_windows(bars)
        if not idx_windows:
            raise ValueError(
                f"Not enough bars ({len(bars)}) for train_bars={self.train_bars} + "
                f"test_bars={self.test_bars}. Provide more historical data or shrink windows."
            )

        bt = Backtester(self.backtest_config)
        windows: list[WalkForwardWindow] = []

        for train_start, train_end, test_start, test_end in idx_windows:
            train_slice = bars.iloc[train_start:train_end]
            test_slice = bars.iloc[test_start:test_end]

            strategy = self._build_strategy(strategy_factory, train_slice)

            # generate_positions needs enough lookback for indicators (e.g.
            # a 50-bar MA needs 50 bars before the test window even starts)
            # so we run the backtest over train+test but only report metrics
            # from the test portion.
            context_and_test = bars.iloc[train_start:test_end]
            full_result = bt.run(context_and_test, strategy, symbol=symbol)

            test_equity = full_result.equity_curve.loc[test_slice.index]
            # Rebase test-window metrics so each window is judged on its own
            # return, not inflated by gains already made during the "context" span
            rebased_equity = test_equity / test_equity.iloc[0] * self.backtest_config.initial_capital

            test_trades = (
                full_result.trades[full_result.trades["timestamp"].isin(test_slice.index)]
                if not full_result.trades.empty
                else full_result.trades
            )

            test_round_trips = pair_trades(test_trades, test_slice.index) if not test_trades.empty else []

            window_metrics = Backtester._compute_metrics(rebased_equity, test_trades, test_round_trips)

            window_result = BacktestResult(
                equity_curve=rebased_equity,
                positions=full_result.positions.loc[test_slice.index],
                trades=test_trades,
                round_trip_trades=full_result.round_trip_trades,
                metrics=window_metrics,
            )

            windows.append(
                WalkForwardWindow(
                    train_start=train_slice.index[0] if len(train_slice) else test_slice.index[0],
                    train_end=train_slice.index[-1] if len(train_slice) else test_slice.index[0],
                    test_start=test_slice.index[0],
                    test_end=test_slice.index[-1],
                    result=window_result,
                )
            )

        return self._build_report(windows)

    @staticmethod
    def _build_strategy(strategy_factory, train_slice: pd.DataFrame) -> Strategy:
        import inspect

        sig = inspect.signature(strategy_factory)
        if len(sig.parameters) == 0:
            return strategy_factory()
        return strategy_factory(train_slice)

    @staticmethod
    def _build_report(windows: list[WalkForwardWindow]) -> WalkForwardReport:
        rows = []
        for w in windows:
            row = {"test_start": w.test_start, "test_end": w.test_end}
            row.update(w.result.metrics)
            rows.append(row)
        per_window_df = pd.DataFrame(rows)

        # Combined metrics: average across windows, plus consistency measures
        numeric_cols = [
            c
            for c in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"]
            if c in per_window_df.columns
        ]
        combined = {}
        for col in numeric_cols:
            vals = per_window_df[col].dropna()
            if len(vals):
                combined[f"avg_{col}"] = round(vals.mean(), 2)
                combined[f"std_{col}"] = round(vals.std(), 2)

        if "total_return_pct" in per_window_df.columns:
            positive_windows = (per_window_df["total_return_pct"] > 0).sum()
            combined["pct_windows_profitable"] = round(100 * positive_windows / len(per_window_df), 1)

        combined["num_windows"] = len(windows)

        return WalkForwardReport(
            windows=windows, combined_metrics=combined, per_window_metrics=per_window_df
        )
