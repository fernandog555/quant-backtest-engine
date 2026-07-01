"""Background worker that runs data loading + backtesting off the UI thread.

yfinance downloads and multi-window walk-forward runs can take several
seconds; doing them on the GUI thread would freeze the window. This wraps
the existing engine interfaces in a QThread and reports progress/results via
signals.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from src.backtest.engine import Backtester, BacktestConfig
from src.backtest.walk_forward import WalkForwardValidator
from src.data.loader import HistoricalDataLoader
from src.risk.manager import RiskLimits


@dataclass
class RunParams:
    symbol: str
    start: str
    end: str | None
    interval: str
    strategy: str          # a registry key, or "all"
    capital: float
    slippage_bps: float
    walk_forward: bool
    train_bars: int
    test_bars: int


class BacktestWorker(QThread):
    """Runs one backtest (or walk-forward) job and emits the result.

    Signals:
        progress(str)   -- human-readable status updates for the UI
        finished_ok(object) -- a result payload dict (see run())
        failed(str)     -- an error message to surface in a dialog
    """

    progress = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, params: RunParams, strategy_registry: dict):
        super().__init__()
        self.params = params
        self.strategy_registry = strategy_registry

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            payload = self._execute()
            self.finished_ok.emit(payload)
        except Exception as exc:  # surfaced as a dialog, not a crash
            self.failed.emit(str(exc))

    # -- internals ---------------------------------------------------------

    def _execute(self) -> dict:
        p = self.params

        self.progress.emit(f"Loading data for {p.symbol}...")
        loader = HistoricalDataLoader()
        bars = loader.load(p.symbol, start=p.start, end=p.end, interval=p.interval)
        span = f"{bars.index[0].date()} to {bars.index[-1].date()}"
        self.progress.emit(f"Loaded {len(bars)} bars ({span}).")

        config = BacktestConfig(
            initial_capital=p.capital,
            slippage_bps=p.slippage_bps,
            risk_limits=RiskLimits(max_position_pct=0.9, max_gross_exposure_pct=1.0),
        )

        names = (
            [p.strategy]
            if p.strategy != "all"
            else list(self.strategy_registry.keys())
        )

        if p.walk_forward:
            return self._run_walk_forward(bars, config, names, span)
        return self._run_single(bars, config, names, span)

    def _run_single(self, bars, config, names, span) -> dict:
        bt = Backtester(config)
        results = {}
        for name in names:
            self.progress.emit(f"Running backtest: {name}...")
            strategy = self.strategy_registry[name]()
            results[strategy.name] = bt.run(bars, strategy, symbol=self.params.symbol)
        self.progress.emit("Done.")
        return {"mode": "single", "results": results, "span": span, "bars": len(bars)}

    def _run_walk_forward(self, bars, config, names, span) -> dict:
        wf = WalkForwardValidator(
            backtest_config=config,
            train_bars=self.params.train_bars,
            test_bars=self.params.test_bars,
        )
        reports = {}
        for name in names:
            self.progress.emit(f"Running walk-forward: {name}...")
            factory = self.strategy_registry[name]
            reports[name] = wf.run(bars, factory, symbol=self.params.symbol)
        self.progress.emit("Done.")
        return {"mode": "walk_forward", "reports": reports, "span": span, "bars": len(bars)}
