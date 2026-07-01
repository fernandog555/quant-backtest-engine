"""
Historical data loader.

Uses yfinance for free historical OHLCV bars — good enough for backtesting
and strategy research. Webull's OpenAPI is used later for the live/paper
execution layer, not for bulk historical research data.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


class HistoricalDataLoader:
    """Fetches and caches historical OHLCV data for a symbol."""

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)

    def load(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by timestamp with columns:
        open, high, low, close, volume
        """
        cache_path = f"{self.cache_dir}/{symbol}_{start}_{end}_{interval}.parquet"

        if use_cache:
            try:
                return pd.read_parquet(cache_path)
            except FileNotFoundError:
                pass

        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            raise ValueError(f"No data returned for {symbol} between {start} and {end}")

        # yfinance sometimes returns MultiIndex columns for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "timestamp"

        df.to_parquet(cache_path)
        return df

    def load_multiple(
        self, symbols: list[str], start: str, end: str | None = None, interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        return {sym: self.load(sym, start, end, interval) for sym in symbols}
