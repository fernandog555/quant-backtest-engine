"""Desktop GUI for the backtesting engine (PySide6).

Kept separate from the core library so the engine itself has no hard GUI
dependency — only running `run_gui.py` requires PySide6. Everything here is
a pure consumer of the existing Backtester / WalkForwardValidator interfaces.
"""
