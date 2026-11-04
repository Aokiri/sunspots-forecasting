import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import rmse_sunspots, to_sunspots

Windows = dict[str, tuple[np.ndarray, np.ndarray]]
Predictions = dict[str, dict[str, np.ndarray]]


def error_table(scaler: Pipeline, windows: Windows, baselines: Predictions, networks: Predictions) -> pd.DataFrame:
    # Baselines hold one prediction per part; networks hold one per seed, (seeds, windows, horizon).
    records: list[dict] = []
    for part in ("validation", "test"):
        y = windows[part][1]
        for name, preds in baselines.items():
            records.append({"model": name, "part": part, "RMSE": rmse_sunspots(scaler, y, preds[part])})
        for name, preds in networks.items():
            for seed_prediction in preds[part]:
                records.append({"model": name, "part": part, "RMSE": rmse_sunspots(scaler, y, seed_prediction)})

    errors = pd.DataFrame(records).groupby(["model", "part"], sort=False)["RMSE"]
    results = pd.DataFrame({
        "validation RMSE": errors.mean().xs("validation", level="part"),
        "test RMSE": errors.mean().xs("test", level="part"),
        # The baselines are deterministic, so their spread over seeds is NaN.
        "test std over seeds": errors.std().xs("test", level="part"),
        "test sem": errors.sem().xs("test", level="part"),
    })
    # Dividing by persistence on the same period lets validation and test be compared.
    for part in ("validation", "test"):
        results[f"{part} / persistence"] = results[f"{part} RMSE"] / results.loc["persistence", f"{part} RMSE"]
    return results


def step_rmse(scaler: Pipeline, y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    # RMSE in sunspots of each step ahead, one value per column.
    return np.sqrt(np.mean((to_sunspots(scaler, y) - to_sunspots(scaler, prediction)) ** 2, axis=0))