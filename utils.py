from data_loader import get_daily_recent
from model import load_model
from features import build_features, FEATURE_COLS


def predict(stock: str):
    """
    Predict next closing price using the saved model.
    Uses DAILY data (same interval as training) to avoid feature scale mismatch.
    Previously used 5-min intraday data which caused incorrect MA/volatility values.
    """
    model = load_model(stock)
    if model is None:
        return None

    data = get_daily_recent(stock)

    if data.empty:
        return None

    df = build_features(data)

    # Need at least 50 rows for MA50 to be meaningful
    if len(df) < 50:
        return None

    latest = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)

    return float(model.predict(latest)[0])
