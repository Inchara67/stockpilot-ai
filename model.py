import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from features import build_features, FEATURE_COLS


def train_model(data, stock: str):
    """Train RandomForest on daily historical data and persist to disk."""
    df = build_features(data)

    X = df[FEATURE_COLS]
    y = df['Close'].shift(-1).dropna()
    X = X.iloc[:-1]  # align with shifted target

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    eval_model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1)
    eval_model.fit(X_train, y_train)
    preds = eval_model.predict(X_test)

    baseline = X_test['Close'].values

    model_mae = mean_absolute_error(y_test, preds)
    baseline_mae = mean_absolute_error(y_test, baseline)

    actual_dir = np.sign(y_test.values - X_test['Close'].values)
    pred_dir = np.sign(preds - X_test['Close'].values)
    dir_acc = float(np.mean(actual_dir == pred_dir))

    print(f"[model] {stock} | Model MAE: {model_mae:.2f} | "
          f"Baseline MAE: {baseline_mae:.2f} | Directional acc: {dir_acc:.1%}")

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
