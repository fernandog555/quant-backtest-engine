"""
Scheduled runner for live/paper trading against Webull.

This is the actual entry point for running the bot continuously — it wraps
TradingOrchestrator with:
- A polling loop that only acts once per new bar (avoids duplicate orders
  if the loop runs more frequently than the bar interval)
- Persistent risk state across restarts (see src/risk/state_store.py)
- Pluggable alerting so halts and errors don't fail silently
- Basic market-hours awareness (skips outside 9:30-16:00 ET on weekdays;
  does not account for market holidays — extend AlertingHooks/is_market_open
  for production use)

Usage:
    python run_live.py --symbol AAPL --strategy ma_crossover --interval 300

This defaults to Webull's UAT (sandbox) environment. See README for the
double-confirmation required before this can route to production.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from src.data.loader import HistoricalDataLoader
from src.execution.orchestrator import TradingOrchestrator
from src.execution.webull_client import WebullConfig, WebullExecutionClient
from src.risk.manager import RiskLimits, RiskManager
from src.risk.state_store import RiskStateStore
from src.strategies.buy_and_hold import BuyAndHold
from src.strategies.moving_average_crossover import MovingAverageCrossover
from src.strategies.rsi_mean_reversion import RSIMeanReversion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_live")

STRATEGY_REGISTRY = {
    "buy_and_hold": lambda: BuyAndHold(),
    "ma_crossover": lambda: MovingAverageCrossover(fast_window=20, slow_window=50),
    "rsi_mean_reversion": lambda: RSIMeanReversion(period=14, oversold=30, exit_level=50),
}


class AlertingHooks:
    """
    Central place for anything that should notify a human, not just a log
    line. Default implementation just logs loudly — wire in email/Slack/SMS
    here for real use. Kept as a simple class (not a callback param threaded
    through everything) so it's easy to swap the whole notification strategy
    in one place.
    """

    def on_halt(self, reason: str, equity: float) -> None:
        logger.critical("TRADING HALTED: %s (equity=%.2f). Manual review required.", reason, equity)
        # TODO: send email/Slack/SMS here

    def on_order_submitted(self, order) -> None:
        logger.info("Order submitted: %s %s x%s", order.side.value, order.symbol, order.quantity)

    def on_error(self, error: Exception) -> None:
        logger.error("Error in trading loop: %s", error, exc_info=True)
        # TODO: send email/Slack/SMS here — errors in a live trading loop
        # should never fail silently.


def is_market_open(now: datetime | None = None) -> bool:
    """Basic US equity market hours check (9:30-16:00 ET, weekdays).
    Does NOT account for market holidays — extend this for production use,
    or query Webull's market calendar endpoint if available."""
    now = now or datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    return dtime(9, 30) <= now.time() <= dtime(16, 0)


def run_loop(
    symbol: str,
    strategy_name: str,
    poll_interval_seconds: int,
    lookback_bars: int,
    state_path: str,
    risk_limits: RiskLimits,
    skip_market_hours_check: bool = False,
):
    config = WebullConfig.from_env()
    logger.info(
        "Starting live runner: symbol=%s strategy=%s environment=%s region=%s",
        symbol, strategy_name, config.environment, config.region_id,
    )
    if config.is_live:
        logger.warning("*** LIVE TRADING ENABLED — real capital is at risk. ***")

    execution_client = WebullExecutionClient(config)
    strategy = STRATEGY_REGISTRY[strategy_name]()
    risk_manager = RiskManager(risk_limits)
    alerts = AlertingHooks()

    state_store = RiskStateStore(state_path)
    prior_state = state_store.load()
    if prior_state:
        risk_manager.load_state(prior_state)
        logger.info("Restored risk state from %s", state_path)

    orchestrator = TradingOrchestrator(strategy, risk_manager, execution_client, symbol)
    loader = HistoricalDataLoader()

    last_bar_timestamp = None

    while True:
        try:
            if not skip_market_hours_check and not is_market_open():
                logger.info("Market closed — sleeping.")
                time.sleep(poll_interval_seconds)
                continue

            if RiskStateStore.is_new_day(risk_manager.to_state()):
                equity = execution_client.get_buying_power()
                risk_manager.reset_day(equity)
                logger.info("New trading day — reset daily risk tracking (equity=%.2f)", equity)

            bars = loader.load(
                symbol,
                start=_lookback_start_date(lookback_bars),
                use_cache=False,
            )

            latest_bar_ts = bars.index[-1]
            if latest_bar_ts == last_bar_timestamp:
                logger.debug("No new bar yet, sleeping.")
                time.sleep(poll_interval_seconds)
                continue
            last_bar_timestamp = latest_bar_ts

            equity_now = execution_client.get_buying_power()
            was_halted = risk_manager._trading_halted
            halted = risk_manager.check_halt(equity_now)
            if halted and not was_halted:
                alerts.on_halt("Risk limit breached", equity_now)

            order = orchestrator.run_once(bars)
            if order:
                alerts.on_order_submitted(order)

            state_store.save(risk_manager.to_state())

        except Exception as e:  # noqa: BLE001 - top-level loop must not crash silently
            alerts.on_error(e)

        time.sleep(poll_interval_seconds)


def _lookback_start_date(lookback_bars: int) -> str:
    from datetime import timedelta
    # Rough calendar-day buffer for weekends/holidays; loader trims to actual bars
    days_back = int(lookback_bars * 1.6) + 10
    return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(description="Run the trading bot against Webull (defaults to UAT/sandbox).")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--strategy", choices=list(STRATEGY_REGISTRY.keys()), default="ma_crossover")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval in seconds")
    parser.add_argument("--lookback-bars", type=int, default=100, help="Bars of history to fetch each poll")
    parser.add_argument("--state-file", default="risk_state.json")
    parser.add_argument("--max-position-pct", type=float, default=0.20)
    parser.add_argument("--max-drawdown-pct", type=float, default=0.15)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.03)
    parser.add_argument(
        "--skip-market-hours-check", action="store_true",
        help="Useful for testing against UAT outside market hours",
    )
    args = parser.parse_args()

    limits = RiskLimits(
        max_position_pct=args.max_position_pct,
        max_drawdown_pct=args.max_drawdown_pct,
        max_daily_loss_pct=args.max_daily_loss_pct,
    )

    run_loop(
        symbol=args.symbol,
        strategy_name=args.strategy,
        poll_interval_seconds=args.interval,
        lookback_bars=args.lookback_bars,
        state_path=args.state_file,
        risk_limits=limits,
        skip_market_hours_check=args.skip_market_hours_check,
    )


if __name__ == "__main__":
    main()
