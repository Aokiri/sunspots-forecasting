from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

DATA_URL: str = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
# Built from the location of this file, so it works from any notebook folder.
DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "monthly-sunspots.csv"

# Last 20% of the months for test, and the last 20% of the rest for validation (section 5.1).
TEST_FRACTION: float = 0.2
VAL_FRACTION: float = 0.2

# Chosen on the validation set in Notebook 01, section 5.4.
WINDOW: int = 36
USE_SQRT: bool = True


def load_sunspots(path: Path = DATA_PATH, url: str = DATA_URL) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(url).to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["Month"], index_col="Month")


def split_bounds(n_months: int, test_fraction: float = TEST_FRACTION,
                 val_fraction: float = VAL_FRACTION) -> dict[str, tuple[int, int]]:
    # Positions (start, end) of each part; the parts keep their order in time.
    test_start = int(n_months * (1 - test_fraction))
    val_start = int(test_start * (1 - val_fraction))
    return {"train": (0, val_start), "validation": (val_start, test_start), "test": (test_start, n_months)}


def clipped_square(values: np.ndarray) -> np.ndarray:
    # A model can predict slightly below zero, and squaring that would turn it into a
    # positive count, so I clip at zero first.
    return np.square(np.clip(values, 0, None))


def make_scaler(use_sqrt: bool = USE_SQRT) -> Pipeline:
    steps = [FunctionTransformer(np.sqrt, inverse_func=clipped_square)] if use_sqrt else []
    return make_pipeline(*steps, MinMaxScaler())


def make_windows(values: np.ndarray, window: int, horizon: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    # Each window predicts the months t, ..., t + horizon - 1 from the `window` months before t.
    # Only t has to be inside [start, end); the inputs can come from the previous part.
    first_targets = range(max(start, window), end - horizon + 1)
    X = np.stack([values[t - window:t] for t in first_targets])
    y = np.stack([values[t:t + horizon] for t in first_targets])
    # The RNN layers expect (windows, time steps, features), with a single feature here.
    return X[..., np.newaxis], y


def to_sunspots(scaler: Pipeline, values: np.ndarray) -> np.ndarray:
    # The scaler works on a single column, so predictions with several steps ahead
    # are flattened, converted back and given their original shape again.
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def rmse_sunspots(scaler: Pipeline, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # The models work in scaled units, but every error is reported in sunspots.
    return root_mean_squared_error(to_sunspots(scaler, y_true), to_sunspots(scaler, y_pred))