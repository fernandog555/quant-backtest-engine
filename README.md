# quant-backtest-engine

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
├── backtest/      # Vectorized backtesting engine
├── execution/     # Webull OpenAPI wrapper + live/paper orchestrator
└── utils/         # Shared data models (Order, Position, Bar, etc.)
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
   probably isn't worth it.

## Risk management

`src/risk/manager.py` sits between every strategy and the execution layer —
strategies emit a direction and conviction (-1 to 1), never a trade size.
The risk manager is what actually decides "how much."

Configurable limits:
- **Max position size** — cap on how much equity goes into one symbol
- **Max gross exposure** — cap on total capital deployed at once
- **Max daily loss** — halts new entries for the day past a threshold
- **Max drawdown** — halts trading entirely (sticky — requires manual review to resume)
- **Per-trade stop loss** — exits a position automatically if it moves too far against entry

## Execution layer (Webull)

`src/execution/webull_client.py` wraps the
[Webull OpenAPI](https://developer.webull.com/apis/docs/AI-friendly-Resources/mcp/)
with a few deliberate safety choices:

- **Defaults to sandbox.** Production trading requires setting
  `WEBULL_ENVIRONMENT=prod`.
- **Double-confirmation for live trading.** Even with `prod` set, a second
  independent flag (`WEBULL_CONFIRM_LIVE=true`) must also be set, so a single
  mistyped environment variable can't send real orders.
- **Preview before submit.** Every order is previewed via Webull's
  `preview_stock_order` before being placed, unless explicitly skipped.
- Credentials are read from environment variables only — never hardcoded,
  never logged.

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
# Leave WEBULL_ENVIRONMENT=sandbox until you've validated behavior thoroughly
```

## Running tests

```bash
pytest tests/ -v
```

25 tests covering strategy logic, risk manager edge cases (drawdown halts,
position sizing, stop losses), and backtester correctness (lookahead bias,
slippage impact, risk integration).

## Roadmap / ideas for extension

- [ ] Walk-forward analysis / out-of-sample validation
- [ ] Multi-symbol portfolio backtesting with correlation-aware sizing
- [ ] Additional strategies (momentum, pairs trading, volatility breakout)
- [ ] Trade-level P&L attribution and win-rate tracking
- [ ] Dashboard (Streamlit) for visualizing equity curves and comparing runs
- [ ] Live paper-trading loop with scheduled execution (cron / APScheduler)

## Disclaimer

This project is for educational and research purposes. It is not investment
advice, and the author is not a licensed financial advisor. Backtested
performance does not guarantee future results. The Webull OpenAPI/MCP
integration is provided "as is"; trading involves substantial risk of loss.
Anyone using the live execution layer is solely responsible for reviewing
and confirming all orders before they are placed.
