import yfinance as yf
import pandas as pd


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns yfinance sometimes returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_historical(stock: str) -> pd.DataFrame:
    return _flatten(yf.download(stock, period="5y", interval="1d", auto_adjust=True))


def get_daily_recent(stock: str) -> pd.DataFrame:
    return _flatten(yf.download(stock, period="6mo", interval="1d", auto_adjust=True))


def get_live(stock: str) -> pd.DataFrame:
    return _flatten(yf.download(stock, period="5d", interval="5m", auto_adjust=True))


def get_monthly_history(stock: str) -> pd.DataFrame:
    return _flatten(yf.download(stock, period="5y", interval="1mo", auto_adjust=True))
