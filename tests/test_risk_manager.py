from src.risk.manager import RiskManager, RiskLimits


class TestPositionSizing:
    def test_zero_signal_gives_zero_position(self):
        rm = RiskManager()
        assert rm.size_position(0.0, equity=100_000, price=50) == 0.0

    def test_respects_max_position_pct(self):
        rm = RiskManager(RiskLimits(max_position_pct=0.10))
        qty = rm.size_position(1.0, equity=100_000, price=100)
        # 10% of 100k = 10k = 100 shares at $100
        assert qty == pytest_approx(100)

    def test_negative_signal_gives_negative_quantity(self):
        rm = RiskManager(RiskLimits(max_position_pct=0.10))
        qty = rm.size_position(-1.0, equity=100_000, price=100)
        assert qty < 0

    def test_respects_remaining_gross_exposure_room(self):
        rm = RiskManager(RiskLimits(max_position_pct=0.50, max_gross_exposure_pct=0.30))
        qty = rm.size_position(1.0, equity=100_000, price=100, current_gross_exposure_pct=0.25)
        # Only 5% of room left even though max_position_pct allows 50%
        assert qty * 100 <= 100_000 * 0.05 + 1e-6


class TestDrawdownHalt:
    def test_no_halt_under_threshold(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.15))
        rm.update_peak(100_000)
        assert rm.check_halt(90_000) is False

    def test_halts_past_max_drawdown(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.15))
        rm.update_peak(100_000)
        assert rm.check_halt(80_000) is True

    def test_halt_is_sticky(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.15))
        rm.update_peak(100_000)
        rm.check_halt(80_000)  # trips halt
        # Even if equity recovers, halt should remain (no un-halt logic --
        # by design, a human should review before resuming)
        assert rm.check_halt(99_000) is True

    def test_halted_positions_are_zeroed(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.15))
        rm.update_peak(100_000)
        rm.check_halt(80_000)
        assert rm.size_position(1.0, equity=80_000, price=50) == 0.0

    def test_daily_loss_halt(self):
        rm = RiskManager(RiskLimits(max_daily_loss_pct=0.03))
        rm.reset_day(100_000)
        assert rm.check_halt(96_000) is True


class TestStopLoss:
    def test_long_stop_triggers_on_drop(self):
        rm = RiskManager(RiskLimits(per_trade_stop_loss_pct=0.05))
        assert rm.stop_loss_triggered(entry_price=100, current_price=94, side=1) is True

    def test_long_stop_not_triggered_within_range(self):
        rm = RiskManager(RiskLimits(per_trade_stop_loss_pct=0.05))
        assert rm.stop_loss_triggered(entry_price=100, current_price=97, side=1) is False

    def test_short_stop_triggers_on_rise(self):
        rm = RiskManager(RiskLimits(per_trade_stop_loss_pct=0.05))
        assert rm.stop_loss_triggered(entry_price=100, current_price=106, side=-1) is True


def pytest_approx(value, rel=1e-6):
    import pytest
    return pytest.approx(value, rel=rel)
