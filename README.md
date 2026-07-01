# quant-backtest-engine

![CI](https://github.com/<you>/quant-backtest-engine/actions/workflows/ci.yml/badge.svg)

A strategy research, backtesting, and (optional) live execution framework for
US equities, built on [Webull's OpenAPI/MCP](https://developer.webull.com/apis/docs/AI-friendly-Resources/mcp/)
for market data and order execution.

This project is primarily a **research and engineering exercise**: building a
backtester that avoids the common mistakes (lookahead bias, ignoring
slippage, no risk management) and pairing it with a real broker integration
that defaults to safe, sandboxed behavior.

**This is not investment advice, and nothing here is a recommendation to
trade any particular strategy or asset.** See [Disclaimer](#disclaimer).

## Why this project

Most "trading bot" tutorials skip the parts that actually matter:
- Backtests that leak future information into past decisions
- No accounting for slippage, commissions, or realistic fills
- No position sizing — "buy" and "sell" with no notion of how much
- No risk limits — nothing stops a bad strategy from blowing up an account

This project treats those as first-class concerns, not afterthoughts.

## Architecture

```
src/
├── data/          # Historical data loading (yfinance) + caching
├── strategies/     # Strategy interface + implementations
│   ├── base.py                    # Abstract Strategy class
│   ├── buy_and_hold.py            # Benchmark strategy
│   ├── moving_average_crossover.py
│   └── rsi_mean_reversion.py
├── risk/          # Position sizing, drawdown halts, stop losses
│   ├── manager.py       # Core risk logic
│   └── state_store.py   # Persists risk state (peak equity, halts) across restarts
├── backtest/      # Vectorized backtesting engine
│   ├── engine.py           # Core backtest loop
│   ├── trade_analytics.py  # Round-trip trade pairing, win rate, payoff ratio
│   ├── walk_forward.py     # Rolling/anchored train-test split validation
│   └── plotting.py         # Equity curve, drawdown, and comparison charts
├── execution/     # Webull OpenAPI wrapper + live/paper orchestrator
│   ├── webull_client.py  # SDK wrapper (verified against official API docs)
│   └── orchestrator.py   # Ties strategy -> risk -> execution together
└── utils/         # Shared data models (Order, Position, Bar, etc.)

scripts/
└── verify_webull_connection.py  # Read-only sandbox connection check

run_backtest.py    # CLI: single backtest or walk-forward validation
run_live.py        # CLI: scheduled live/paper trading loop
```

**Key design decision:** strategies, the risk manager, and the backtester
share the exact same interfaces used by the live orchestrator. A strategy
that's been backtested is running the same code path live — the only thing
that changes is where bars and orders come from.

## Backtesting engine

The backtester (`src/backtest/engine.py`) is built around a few rules that
are easy to get wrong:

1. **No lookahead bias.** A signal generated from bar `t`'s close is only
   actable at bar `t+1` — you can't trade on information you didn't have yet.
   This is enforced by shifting signals forward by one bar before execution,
   and tested explicitly (`test_no_lookahead_bias`).
2. **Slippage and commissions are modeled**, not ignored. Every fill is
   adjusted against you by a configurable number of basis points.
3. **Risk management runs bar-by-bar**, not just as a post-hoc check. A
   strategy that would blow through a max-drawdown limit mid-backtest
   actually gets halted mid-backtest, matching what would happen live.
4. **Every strategy is compared against buy-and-hold.** If a strategy can't
   beat the naive benchmark on a risk-adjusted basis, added complexity
   probably isn't worth it. Note: the same `RiskLimits` (stop loss, max
   drawdown halt) apply to every strategy run through the backtester,
   including `BuyAndHold` — so its equity curve reflects "buy and hold,
   subject to the configured risk controls," not literally never selling.
   Set `RiskLimits(per_trade_stop_loss_pct=1.0, max_drawdown_pct=1.0)` if
   you want a true unconstrained buy-and-hold comparison.

## Walk-forward validation

A single backtest window can't tell you whether a strategy has a real edge
or just got lucky on that particular slice of history. `src/backtest/walk_forward.py`
splits data into repeated train/test windows (rolling or anchored) and
evaluates the strategy only on data outside its own training window:

```bash
python run_backtest.py --symbol AAPL --start 2020-01-01 --walk-forward \
    --strategy ma_crossover --train-bars 252 --test-bars 63
```

This reports per-window returns plus aggregate consistency metrics
(`pct_windows_profitable`, mean/std of Sharpe across windows) — a strategy
that's profitable in 90% of windows tells a very different story than one
that's profitable overall only because of one outlier window.

For strategies that fit parameters (not the fixed-parameter ones included
here), pass a factory function of the train slice instead of a zero-arg one,
so parameter search only ever sees the train portion of each window —
never the test portion it's about to be judged on.

## Trade-level analytics

Beyond the portfolio-level equity curve, `src/backtest/trade_analytics.py`
pairs individual fills into round-trip trades (handling partial closes and
position flips) and computes standard trade statistics: win rate, average
win/loss, payoff ratio, and expectancy. Available on every `BacktestResult`
as `.round_trip_trades` (a DataFrame) and folded into `.metrics`.

## Visualization

`src/backtest/plotting.py` provides equity curve + drawdown charts,
multi-strategy comparison overlays, and walk-forward window bar charts:

```python
from src.backtest.plotting import plot_equity_curve, plot_walk_forward_windows

plot_equity_curve(result, benchmark=benchmark_result.equity_curve, save_path="equity.png")
plot_walk_forward_windows(report, save_path="walk_forward.png")
```

## Risk management

`src/risk/manager.py` sits between every strategy and the execution layer —
strategies emit a direction and conviction (-1 to 1), never a trade size.
The risk manager is what actually decides "how much."

Configurable limits:
- **Max position size** — cap on how much equity goes into one symbol
- **Max gross exposure** — cap on total capital deployed at once
- **Max daily loss** — halts new entries for the day past a threshold, resets each calendar day
- **Max drawdown** — halts trading entirely and force-closes open positions (sticky — requires manual review to resume)
- **Per-trade stop loss** — exits a position automatically if it moves too far against entry

Risk state (peak equity, daily start equity, halt status) can be persisted
across process restarts via `src/risk/state_store.py` — see the live
execution section below.

## Development notes

This section stays intentionally honest about bugs found during development
rather than presenting the code as having been correct from the start —
the debugging process is itself part of what a backtesting engine needs to
get right.

**Bugs found and fixed via testing, not just code review:**

- **Read-only numpy array crash** in the RSI strategy: `.to_numpy()` on a
  pandas Series can return a read-only view; in-place mutation crashed
  only on certain pandas/numpy version combinations. Fixed by explicit
  `.copy()`, caught by running the backtester end-to-end rather than
  unit-testing the strategy in isolation.
- **Buy-and-hold rebalancing on every bar**: floating-point drift in
  target position sizing caused the "buy and hold" strategy to submit a
  trade on nearly every bar instead of once. Fixed with a meaningful-drift
  threshold before rebalancing.
- **Drawdown halt not actually limiting drawdown**: a strategy configured
  with `max_drawdown_pct=0.30` produced an actual max drawdown of -47.5%.
  Root cause was two compounding issues: (1) `reset_day()` was called once
  at backtest start instead of once per calendar day, silently turning
  `max_daily_loss_pct` into "max loss since backtest start" over a
  multi-year run, and (2) once a halt fired, existing positions were left
  open and continued to mark-to-market indefinitely instead of being
  force-closed. Both are now covered by regression tests
  (`test_daily_loss_limit_resets_each_calendar_day`,
  `test_max_drawdown_halt_force_closes_existing_position`).
- **Wrong Webull SDK interface**: the execution client was initially
  written against a guessed interface (`WebullClient(...)` constructor,
  `.account.get_positions()`-style calls). After checking Webull's actual
  published docs, the real SDK uses `ApiClient` + `TradeClient` with
  `requests`-style responses (`res.status_code`, `res.json()`) and
  namespaced calls like `trade_client.account_v2.get_account_balance(...)`.
  Rewritten to match; still needs verification against a live sandbox
  connection (see `scripts/verify_webull_connection.py`).

## Execution layer (Webull)

`src/execution/webull_client.py` wraps the official
[Webull OpenAPI Python SDK](https://developer.webull.com/apis/docs/trade-api/getting-started)
with a few deliberate safety choices:

- **Defaults to UAT (sandbox)**, matching Webull's own terminology. Production
  trading requires setting `WEBULL_ENVIRONMENT=prod`.
- **Double-confirmation for live trading.** Even with `prod` set, a second
  independent flag (`WEBULL_CONFIRM_LIVE=true`) must also be set, so a single
  mistyped environment variable can't send real orders.
- **Preview before submit.** Every order is previewed before being placed,
  unless explicitly skipped.
- Credentials are read from environment variables only — never hardcoded,
  never logged.

**Important caveat:** this client is written against Webull's published API
docs and sample code, but has not been exercised against a live sandbox
connection (no network access to Webull from the environment this was built
in). Before trusting it with anything, run:

```bash
cp .env.example .env   # fill in your UAT credentials
python scripts/verify_webull_connection.py
```

This makes only read-only calls (account list, balance, positions) and
prints the raw response shape so you can confirm field names match what
the code assumes — SDK response fields can drift between versions.

### Running live/paper

`run_live.py` wraps the execution layer in a polling loop with market-hours
awareness, persistent risk state across restarts, and alerting hooks for
halts/errors:

```bash
python run_live.py --symbol AAPL --strategy ma_crossover --interval 300
```

Risk state (peak equity, daily starting equity, halt status) persists to
`risk_state.json` between runs — without this, restarting the process mid-day
would silently reset drawdown tracking and defeat the point of the halt.

## Getting started

```bash
git clone <this-repo> quant-backtest-engine
cd quant-backtest-engine
pip install -r requirements.txt

# Run a backtest comparing all strategies on a symbol
python run_backtest.py --symbol AAPL --start 2022-01-01 --end 2024-01-01

# Run a single strategy
python run_backtest.py --symbol AAPL --start 2022-01-01 --strategy ma_crossover
```

For live/paper execution against Webull:

```bash
pip install webull-openapi-python-sdk
cp .env.example .env
# Fill in WEBULL_APP_KEY / WEBULL_APP_SECRET from developer.webull.com
# Leave WEBULL_ENVIRONMENT=uat until you've validated the connection

# Confirm the SDK response shapes match what the code expects (read-only)
python scripts/verify_webull_connection.py

# Run walk-forward validation instead of a single backtest
python run_backtest.py --symbol AAPL --start 2020-01-01 --walk-forward --strategy ma_crossover

# Run the live/paper trading loop
python run_live.py --symbol AAPL --strategy ma_crossover --interval 300
```

## Running tests

```bash
pytest tests/ -v
```

69 tests covering strategy logic, risk manager edge cases (drawdown halts,
daily-loss reset, position sizing, stop losses), risk state persistence
across restarts, trade-pairing/P&L analytics, walk-forward window splitting,
Webull client payload construction and error handling, and backtester
correctness (lookahead bias, slippage impact, risk integration, force-close
on halt).

## Roadmap / ideas for extension

- [x] Walk-forward analysis / out-of-sample validation
- [x] Trade-level P&L attribution and win-rate tracking
- [x] Live paper-trading loop with scheduled execution
- [x] Equity curve / drawdown / walk-forward visualization
- [x] CI pipeline (GitHub Actions)
- [ ] Verify execution layer against a live Webull UAT connection (untested
      due to no network access during development — see `scripts/verify_webull_connection.py`)
- [ ] Multi-symbol portfolio backtesting with correlation-aware sizing
- [ ] Additional strategies (momentum, pairs trading, volatility breakout)
- [ ] Parameter sensitivity / grid-search analysis (distinct from walk-forward)
- [ ] Dashboard (Streamlit) for visualizing equity curves and comparing runs
- [ ] Market holiday calendar awareness in the live trading loop

## Disclaimer

This project is for educational and research purposes. It is not investment
advice, and the author is not a licensed financial advisor. Backtested
performance does not guarantee future results. The Webull OpenAPI/MCP
integration is provided "as is"; trading involves substantial risk of loss.
Anyone using the live execution layer is solely responsible for reviewing
and confirming all orders before they are placed.
