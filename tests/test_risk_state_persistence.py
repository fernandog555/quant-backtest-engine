import os
import tempfile
from datetime import date, timedelta

import pytest

from src.risk.manager import RiskManager, RiskLimits
from src.risk.state_store import RiskStateStore, RiskManagerState


@pytest.fixture
def tmp_state_path():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "risk_state.json")


class TestRiskStateStore:
    def test_load_returns_none_when_no_file(self, tmp_state_path):
        store = RiskStateStore(tmp_state_path)
        assert store.load() is None

    def test_save_and_load_roundtrip(self, tmp_state_path):
        store = RiskStateStore(tmp_state_path)
        state = RiskManagerState(
            peak_equity=105_000, day_start_equity=100_000,
            day_start_date=date.today().isoformat(), trading_halted=False,
        )
        store.save(state)
        loaded = store.load()
        assert loaded == state

    def test_is_new_day_true_for_none_state(self):
        assert RiskStateStore.is_new_day(None) is True

    def test_is_new_day_false_for_today(self):
        state = RiskManagerState(
            peak_equity=100, day_start_equity=100,
            day_start_date=date.today().isoformat(), trading_halted=False,
        )
        assert RiskStateStore.is_new_day(state) is False

    def test_is_new_day_true_for_yesterday(self):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        state = RiskManagerState(
            peak_equity=100, day_start_equity=100,
            day_start_date=yesterday, trading_halted=False,
        )
        assert RiskStateStore.is_new_day(state) is True


class TestRiskManagerPersistence:
    def test_peak_equity_survives_restart(self, tmp_state_path):
        rm1 = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        rm1.update_peak(100_000)
        rm1.reset_day(100_000)

        store = RiskStateStore(tmp_state_path)
        store.save(rm1.to_state())

        # Simulate a fresh process
        rm2 = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        rm2.load_state(store.load())

        # Should immediately halt since peak_equity carried over
        assert rm2.check_halt(88_000) is True

    def test_halt_flag_survives_restart(self, tmp_state_path):
        rm1 = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        rm1.update_peak(100_000)
        rm1.check_halt(80_000)  # trips halt
        assert rm1._trading_halted is True

        store = RiskStateStore(tmp_state_path)
        store.save(rm1.to_state())

        rm2 = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        rm2.load_state(store.load())
        assert rm2.size_position(1.0, equity=90_000, price=50) == 0.0

    def test_day_start_equity_not_carried_across_day_boundary(self, tmp_state_path):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        stale_state = RiskManagerState(
            peak_equity=100_000, day_start_equity=95_000,
            day_start_date=yesterday, trading_halted=False,
        )
        rm = RiskManager(RiskLimits())
        rm.load_state(stale_state)
        # day_start_equity should NOT have been restored since it's stale
        assert rm._day_start_equity is None

    def test_manually_clear_halt(self, tmp_state_path):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        rm.update_peak(100_000)
        rm.check_halt(80_000)
        assert rm.size_position(1.0, equity=80_000, price=50) == 0.0

        rm.manually_clear_halt()
        # Note: check_halt will immediately re-halt if still in drawdown --
        # this only clears the flag, it doesn't fix the underlying condition
        assert rm._trading_halted is False
