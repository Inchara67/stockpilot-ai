import pandas as pd


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Builds all ML features from a daily OHLCV DataFrame.
    Used by both model.py (training) and utils.py (prediction)
    to ensure consistent feature scales.
    """
    df = data.copy()

    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(10).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Volume_MA'] = df['Volume'].rolling(10).mean()

    return df.dropna()


FEATURE_COLS = ['Close', 'MA10', 'MA50', 'Return',
                'Volatility', 'Momentum', 'Volume_MA']
