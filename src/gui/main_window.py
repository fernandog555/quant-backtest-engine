"""Main application window for the backtesting desktop GUI.

Layout: an input panel on the left (symbol, dates, strategy, capital,
walk-forward toggle) and a results area on the right (embedded matplotlib
chart + a metrics table). Backtests run on a background thread so the UI
stays responsive.
"""
from __future__ import annotations

from datetime import date

import matplotlib

matplotlib.use("QtAgg")  # ensure the Qt canvas backend, regardless of env default

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.backtest.plotting import (
    plot_equity_curve,
    plot_strategy_comparison,
    plot_walk_forward_windows,
)
from src.gui.worker import BacktestWorker, RunParams

# Single source of truth for available strategies, shared with the CLI.
from run_backtest import STRATEGY_REGISTRY


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quant Backtest Engine")
        self.resize(1200, 760)
        self._worker: BacktestWorker | None = None

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 860])

        root = QVBoxLayout(self)
        root.addWidget(splitter)

    # -- input panel -------------------------------------------------------

    def _build_input_panel(self) -> QWidget:
        panel = QGroupBox("Backtest configuration")
        form = QFormLayout(panel)

        self.symbol_edit = QLineEdit("AAPL")
        form.addRow("Symbol", self.symbol_edit)

        self.start_edit = QLineEdit("2022-01-01")
        self.start_edit.setPlaceholderText("YYYY-MM-DD")
        form.addRow("Start date", self.start_edit)

        self.end_edit = QLineEdit()
        self.end_edit.setPlaceholderText("YYYY-MM-DD (blank = today)")
        form.addRow("End date", self.end_edit)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1d", "1wk", "1mo"])
        form.addRow("Interval", self.interval_combo)

        self.strategy_combo = QComboBox()
        for key in STRATEGY_REGISTRY:
            self.strategy_combo.addItem(key)
        self.strategy_combo.addItem("all")
        form.addRow("Strategy", self.strategy_combo)

        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1_000, 100_000_000)
        self.capital_spin.setSingleStep(10_000)
        self.capital_spin.setValue(100_000)
        self.capital_spin.setGroupSeparatorShown(True)
        self.capital_spin.setPrefix("$ ")
        form.addRow("Capital", self.capital_spin)

        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 100)
        self.slippage_spin.setValue(5)
        self.slippage_spin.setSuffix(" bps")
        form.addRow("Slippage", self.slippage_spin)

        self.wf_check = QCheckBox("Walk-forward validation")
        self.wf_check.toggled.connect(self._on_wf_toggled)
        form.addRow(self.wf_check)

        self.train_spin = QSpinBox()
        self.train_spin.setRange(10, 5000)
        self.train_spin.setValue(252)
        self.train_row_label = QLabel("Train bars")
        form.addRow(self.train_row_label, self.train_spin)

        self.test_spin = QSpinBox()
        self.test_spin.setRange(5, 5000)
        self.test_spin.setValue(63)
        self.test_row_label = QLabel("Test bars")
        form.addRow(self.test_row_label, self.test_spin)

        self.run_button = QPushButton("Run backtest")
        self.run_button.clicked.connect(self._on_run)
        form.addRow(self.run_button)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        form.addRow(self.status_label)

        self._set_wf_fields_visible(False)
        return panel

    def _on_wf_toggled(self, checked: bool) -> None:
        self._set_wf_fields_visible(checked)
        self.run_button.setText("Run walk-forward" if checked else "Run backtest")

    def _set_wf_fields_visible(self, visible: bool) -> None:
        for w in (self.train_row_label, self.train_spin, self.test_row_label, self.test_spin):
            w.setVisible(visible)

    # -- results panel -----------------------------------------------------

    def _build_results_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.chart_container = QVBoxLayout()
        chart_host = QWidget()
        chart_host.setLayout(self.chart_container)

        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        results_splitter = QSplitter(Qt.Vertical)
        results_splitter.addWidget(chart_host)
        results_splitter.addWidget(self.table)
        results_splitter.setSizes([500, 240])

        layout.addWidget(results_splitter)
        self._placeholder = QLabel("Configure a backtest and press Run.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self.chart_container.addWidget(self._placeholder)
        return panel

    # -- run flow ----------------------------------------------------------

    def _on_run(self) -> None:
        symbol = self.symbol_edit.text().strip().upper()
        start = self.start_edit.text().strip()
        if not symbol or not start:
            QMessageBox.warning(self, "Missing input", "Symbol and start date are required.")
            return

        params = RunParams(
            symbol=symbol,
            start=start,
            end=self.end_edit.text().strip() or None,
            interval=self.interval_combo.currentText(),
            strategy=self.strategy_combo.currentText(),
            capital=self.capital_spin.value(),
            slippage_bps=self.slippage_spin.value(),
            walk_forward=self.wf_check.isChecked(),
            train_bars=self.train_spin.value(),
            test_bars=self.test_spin.value(),
        )

        self.run_button.setEnabled(False)
        self.status_label.setText("Starting...")

        self._worker = BacktestWorker(params, STRATEGY_REGISTRY)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.status_label.setText("Failed.")
        QMessageBox.critical(self, "Backtest failed", message)

    def _on_finished(self, payload: dict) -> None:
        self.run_button.setEnabled(True)
        try:
            if payload["mode"] == "single":
                self._render_single(payload)
            else:
                self._render_walk_forward(payload)
        except Exception as exc:  # rendering issues shouldn't kill the app
            QMessageBox.critical(self, "Display error", str(exc))

    # -- rendering ---------------------------------------------------------

    def _swap_chart(self, fig) -> None:
        # Clear whatever chart/placeholder is currently shown, then embed fig.
        while self.chart_container.count():
            item = self.chart_container.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        self.chart_container.addWidget(toolbar)
        self.chart_container.addWidget(canvas)
        canvas.draw()

    def _render_single(self, payload: dict) -> None:
        results = payload["results"]
        if len(results) == 1:
            (name, result), = results.items()
            fig = plot_equity_curve(result, title=f"{name} — Equity Curve")
            self._populate_metrics_single(name, result.metrics)
        else:
            # "all": overlay every strategy (buy-and-hold included) so the
            # benchmark comparison the README calls for is visible directly.
            fig = plot_strategy_comparison(results, title="Strategy Comparison")
            self._populate_metrics_multi({n: r.metrics for n, r in results.items()})
        self._swap_chart(fig)

    def _render_walk_forward(self, payload: dict) -> None:
        reports = payload["reports"]
        # Chart the first report's windows; table shows per-window + combined.
        first_name, first_report = next(iter(reports.items()))
        fig = plot_walk_forward_windows(
            first_report, title=f"{first_name} — Walk-Forward Windows"
        )
        self._swap_chart(fig)
        self._populate_metrics_walk_forward(reports)

    # -- metrics tables ----------------------------------------------------

    def _populate_metrics_single(self, name: str, metrics: dict) -> None:
        self.table.clear()
        rows = list(metrics.items())
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Metric", name])
        for r, (k, v) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.table.setItem(r, 1, QTableWidgetItem(_fmt(v)))

    def _populate_metrics_multi(self, metrics_by_strategy: dict) -> None:
        self.table.clear()
        strategies = list(metrics_by_strategy.keys())
        # Union of metric keys, preserving first-seen order.
        keys: list[str] = []
        for m in metrics_by_strategy.values():
            for k in m:
                if k not in keys:
                    keys.append(k)
        self.table.setRowCount(len(keys))
        self.table.setColumnCount(1 + len(strategies))
        self.table.setHorizontalHeaderLabels(["Metric", *strategies])
        for r, key in enumerate(keys):
            self.table.setItem(r, 0, QTableWidgetItem(key))
            for c, strat in enumerate(strategies, start=1):
                self.table.setItem(r, c, QTableWidgetItem(_fmt(metrics_by_strategy[strat].get(key))))

    def _populate_metrics_walk_forward(self, reports: dict) -> None:
        self.table.clear()
        strategies = list(reports.keys())
        keys: list[str] = []
        for rep in reports.values():
            for k in rep.combined_metrics:
                if k not in keys:
                    keys.append(k)
        self.table.setRowCount(len(keys))
        self.table.setColumnCount(1 + len(strategies))
        self.table.setHorizontalHeaderLabels(["Aggregate metric", *strategies])
        for r, key in enumerate(keys):
            self.table.setItem(r, 0, QTableWidgetItem(key))
            for c, strat in enumerate(strategies, start=1):
                self.table.setItem(r, c, QTableWidgetItem(_fmt(reports[strat].combined_metrics.get(key))))


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)
