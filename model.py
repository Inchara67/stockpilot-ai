import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from features import build_features, FEATURE_COLS


def train_model(data, stock: str):
    """Train RandomForest on daily historical data and persist to disk."""
    df = build_features(data)

    X = df[FEATURE_COLS]
    y = df['Close'].shift(-1).dropna()
    X = X.iloc[:-1]  # align with shifted target

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{stock}.pkl")

    return model


def load_model(stock: str):
    path = f"models/{stock}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None
