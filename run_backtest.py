"""
CLI to run a backtest against historical data and print/compare results.

Example:
    python run_backtest.py --symbol AAPL --start 2022-01-01 --end 2024-01-01 --strategy ma_crossover
"""
from __future__ import annotations

import argparse

from src.backtest.engine import Backtester, BacktestConfig
from src.backtest.walk_forward import WalkForwardValidator
from src.data.loader import HistoricalDataLoader
from src.risk.manager import RiskLimits
from src.strategies.buy_and_hold import BuyAndHold
from src.strategies.moving_average_crossover import MovingAverageCrossover
from src.strategies.rsi_mean_reversion import RSIMeanReversion

STRATEGY_REGISTRY = {
    "buy_and_hold": lambda: BuyAndHold(),
    "ma_crossover": lambda: MovingAverageCrossover(fast_window=20, slow_window=50),
    "rsi_mean_reversion": lambda: RSIMeanReversion(period=14, oversold=30, exit_level=50),
}


def main():
    parser = argparse.ArgumentParser(description="Run a backtest against historical stock data.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--interval", default="1d", help="Bar interval (default: 1d)")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which strategy to run (default: run all and compare)",
    )
    parser.add_argument("--capital", type=float, default=100_000, help="Starting capital")
    parser.add_argument(
        "--walk-forward", action="store_true", help="Run walk-forward validation instead of a single backtest"
    )
    parser.add_argument("--train-bars", type=int, default=252, help="Walk-forward train window size (bars)")
    parser.add_argument("--test-bars", type=int, default=63, help="Walk-forward test window size (bars)")
    args = parser.parse_args()

    loader = HistoricalDataLoader()
    bars = loader.load(args.symbol, start=args.start, end=args.end, interval=args.interval)
    print(f"Loaded {len(bars)} bars for {args.symbol} ({bars.index[0].date()} to {bars.index[-1].date()})\n")

    config = BacktestConfig(
        initial_capital=args.capital,
        slippage_bps=5,
        risk_limits=RiskLimits(max_position_pct=0.9, max_gross_exposure_pct=1.0),
    )

    strategy_names = [args.strategy] if args.strategy != "all" else list(STRATEGY_REGISTRY.keys())

    if args.walk_forward:
        wf = WalkForwardValidator(
            backtest_config=config, train_bars=args.train_bars, test_bars=args.test_bars
        )
        for name in strategy_names:
            factory = STRATEGY_REGISTRY[name]
            report = wf.run(bars, factory, symbol=args.symbol)
            print(f"--- Walk-forward: {name} ({report.combined_metrics['num_windows']} windows) ---")
            for k, v in report.combined_metrics.items():
                print(f"  {k}: {v}")
            print()
            print(report.per_window_metrics[
                ["test_start", "test_end", "total_return_pct", "sharpe_ratio", "max_drawdown_pct"]
            ].to_string(index=False))
            print()
        return

    bt = Backtester(config)
    for name in strategy_names:
        strategy = STRATEGY_REGISTRY[name]()
        result = bt.run(bars, strategy, symbol=args.symbol)
        print(f"--- {strategy.name} ---")
        for k, v in result.metrics.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
