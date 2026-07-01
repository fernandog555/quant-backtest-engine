"""
Persistence for RiskManager state across process restarts.

Without this, a scheduled bot that restarts mid-day loses track of its
peak equity and daily starting equity — which defeats the purpose of the
drawdown and daily-loss halts. This stores state as simple JSON so it's
easy to inspect/reset manually if needed.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date


@dataclass
class RiskManagerState:
    peak_equity: float | None
    day_start_equity: float | None
    day_start_date: str | None  # ISO date string; used to detect day rollover
    trading_halted: bool


class RiskStateStore:
    def __init__(self, path: str = "risk_state.json"):
        self.path = path

    def load(self) -> RiskManagerState | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r") as f:
            data = json.load(f)
        return RiskManagerState(**data)

    def save(self, state: RiskManagerState) -> None:
        with open(self.path, "w") as f:
            json.dump(asdict(state), f, indent=2)

    @staticmethod
    def is_new_day(state: "RiskManagerState | None") -> bool:
        if state is None or state.day_start_date is None:
            return True
        return state.day_start_date != date.today().isoformat()
