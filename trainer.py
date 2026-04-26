import threading
import os
import time
from data_loader import get_historical
from model import train_model

RETRAIN_INTERVAL_HOURS = 24  # only retrain once per day


def _should_retrain(stock: str) -> bool:
    """Returns True if model doesn't exist or is older than RETRAIN_INTERVAL_HOURS."""
    progress_path = f"models/{stock}_progress.txt"
    model_path = f"models/{stock}.pkl"

    if not os.path.exists(model_path):
        return True

    if os.path.exists(progress_path):
        age_seconds = time.time() - os.path.getmtime(progress_path)
        if age_seconds < RETRAIN_INTERVAL_HOURS * 3600:
            return False  # trained recently, skip

    return True


def _write_progress(path: str, value: int):
    with open(path, "w") as f:
        f.write(str(value))


def background_train(stock: str):
    os.makedirs("models", exist_ok=True)
    progress_path = f"models/{stock}_progress.txt"

    try:
        _write_progress(progress_path, 5)

        # Fetch historical data (~20% of work)
        data = get_historical(stock)
        _write_progress(progress_path, 30)

        if data.empty:
            _write_progress(progress_path, -1)  # signal failure
            return

        # Train model (~70% of work — this is real, no fake sleep)
        train_model(data, stock)
        _write_progress(progress_path, 100)

        print(f"[trainer] {stock} model updated successfully")

    except Exception as e:
        _write_progress(progress_path, -1)
        print(f"[trainer] Failed to train {stock}: {e}")


def start_training(stock: str):
    """Spawn background training thread only if retraining is needed."""
    if not _should_retrain(stock):
        return  # skip — model is fresh

    thread = threading.Thread(target=background_train,
                              args=(stock,), daemon=True)
    thread.start()


def get_training_progress(stock: str) -> int:
    """Returns training progress 0–100, or -1 on failure, or 100 if complete."""
    path = f"models/{stock}_progress.txt"
    if not os.path.exists(path):
        return 0
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return 0
