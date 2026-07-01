"""Launch the desktop GUI for the backtesting engine.

    python run_gui.py

Requires PySide6 (see requirements.txt). This is a thin entry point; all the
UI lives in src/gui/. The core engine has no GUI dependency — the CLI
(run_backtest.py) still works without PySide6 installed.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.stderr.write(
            "PySide6 is required for the GUI. Install it with:\n"
            "    pip install PySide6\n"
        )
        return 1

    from src.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
