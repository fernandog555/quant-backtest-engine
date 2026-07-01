"""
Vectorized backtesting engine with realistic frictions.

Key design choices that matter for credibility:
- Signals generated at close of bar t are executed at close of bar t+1
  (no lookahead bias — you can't trade on information you didn't have yet).
- Slippage and commission are modeled explicitly, not ignored.
- Risk manager sizing and drawdown halts are applied bar-by-bar, not
  just at the end — a strategy that blows through max_drawdown mid-backtest
  actually stops trading, same as it would live.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.risk.manager import RiskManager, RiskLimits
from src.strategies.base import Strategy


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_per_share: float = 0.0  # Webull is commission-free on US stocks; kept configurable
    slippage_bps: float = 5.0  # basis points of price, applied against you on every fill
    risk_limits: RiskLimits = field(default_factory=RiskLimits)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict


class Backtester:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, bars: pd.DataFrame, strategy: Strategy, symbol: str = "SYMBOL") -> BacktestResult:
        bars = bars.copy()
        target_positions = strategy.generate_positions(bars)

        # Lag by 1 bar: signal computed on bar t's close is acted on at bar t+1's open/close
        executed_target = target_positions.shift(1).fillna(0.0)

        risk = RiskManager(self.config.risk_limits)
        cash = self.config.initial_capital
        shares_held = 0.0
        entry_price = 0.0

        equity_curve = []
        position_history = []
        trades = []

        risk.reset_day(cash)

        for i, (ts, row) in enumerate(bars.iterrows()):
            price = row["close"]
            current_equity = cash + shares_held * price

            halted = risk.check_halt(current_equity)

            # Per-trade stop loss, checked every bar regardless of new signals
            if shares_held != 0 and not halted:
                side = 1 if shares_held > 0 else -1
                if risk.stop_loss_triggered(entry_price, price, side):
                    fill_price = self._apply_slippage(price, sell=shares_held > 0)
                    proceeds = shares_held * fill_price
                    cash += proceeds - abs(shares_held) * self.config.commission_per_share
                    trades.append(self._trade_record(ts, symbol, -shares_held, fill_price, "STOP_LOSS"))
                    shares_held = 0.0
                    entry_price = 0.0

            desired_signal = 0.0 if halted else executed_target.iloc[i]
            gross_exposure_pct = abs(shares_held * price) / current_equity if current_equity > 0 else 0.0

            target_qty = risk.size_position(
                signal_strength=desired_signal,
                equity=current_equity,
                price=price,
                current_gross_exposure_pct=0.0,  # single-symbol backtest; no prior exposure to net against
            )

            delta = target_qty - shares_held
            # Rebalance only on meaningful drift (>1% of target position or a
            # full entry/exit) — otherwise small equity fluctuations cause
            # constant no-op-ish trading and inflate trade counts/costs.
            meaningful_drift = abs(delta) > max(abs(target_qty) * 0.01, 1e-6)
            direction_flip = np.sign(target_qty) != np.sign(shares_held)

            if (meaningful_drift or direction_flip) and abs(delta) > 1e-6 and not halted:
                fill_price = self._apply_slippage(price, sell=delta < 0)
                cost = delta * fill_price
                commission = abs(delta) * self.config.commission_per_share
                cash -= cost + commission

                if shares_held == 0 or np.sign(target_qty) != np.sign(shares_held):
                    entry_price = fill_price

                trades.append(self._trade_record(ts, symbol, delta, fill_price, "SIGNAL"))
                shares_held = target_qty

            current_equity = cash + shares_held * price
            equity_curve.append(current_equity)
            position_history.append(shares_held)

        equity_series = pd.Series(equity_curve, index=bars.index, name="equity")
        position_series = pd.Series(position_history, index=bars.index, name="shares_held")
        trades_df = pd.DataFrame(trades)

        metrics = self._compute_metrics(equity_series, trades_df)

        return BacktestResult(
            equity_curve=equity_series,
            positions=position_series,
            trades=trades_df,
            metrics=metrics,
        )

    def _apply_slippage(self, price: float, sell: bool) -> float:
        slip = price * (self.config.slippage_bps / 10_000)
        return price - slip if sell else price + slip

    @staticmethod
    def _trade_record(ts, symbol, delta_shares, fill_price, reason) -> dict:
        return {
            "timestamp": ts,
            "symbol": symbol,
            "shares": delta_shares,
            "fill_price": fill_price,
            "reason": reason,
        }

    @staticmethod
    def _compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
        returns = equity.pct_change().dropna()

        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 1 else 0.0

        # Assumes daily bars for annualization; adjust externally if using a different interval
        ann_factor = 252
        ann_return = (1 + total_return) ** (ann_factor / max(len(equity), 1)) - 1 if len(equity) > 1 else 0.0
        ann_vol = returns.std() * np.sqrt(ann_factor) if len(returns) > 1 else 0.0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) else 0.0

        win_rate = None
        if not trades.empty and "reason" in trades.columns:
            closing_trades = trades[trades["shares"] != 0]
            win_rate = None  # requires trade-pairing logic; left for the notebook-level analysis

        return {
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(ann_return * 100, 2),
            "annualized_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "num_trades": int((trades["shares"] != 0).sum()) if not trades.empty else 0,
            "final_equity": round(equity.iloc[-1], 2) if len(equity) else None,
        }
