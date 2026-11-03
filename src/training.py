import copy
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Built from the location of this file, like DATA_PATH in src/data.py
MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 32
MAX_EPOCHS: int = 300
# The validation loss is noisy from one epoch to the next, so a short patience
# would stop at the first bump.
PATIENCE: int = 20
# Limits exploding gradients through the 36 steps of a recurrent network.
CLIP_NORM: float = 1.0

History = dict[str, list[float]]


def to_tensor(values: np.ndarray) -> torch.Tensor: 
    # The windows are float64, but the layers hold float32 parameters.
    return torch.as_tensor(values, dtype=torch.float32)
    

def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(to_tensor(X)).numpy()
    
    
def predict_recursive(model: nn.Module, X: np.ndarray, horizon: int) -> np.ndarray:
    # Each one-step forecast becomes  the newest month of the window and the oldest month
    # is dropped, so the window keeps its length and later steps build on earlier forecasts.
    model.eval()
    window = to_tensor(X)
    steps = []
    with torch.no_grad():
        for _ in range(horizon):
            step = model(window)    # (windows, 1)
            steps.append(step)
            window = torch.cat([window[:, 1:, :], step[:, :, None]], dim=1)
    return torch.cat(steps, dim=1).numpy()
    
    
def fit(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> History:
    # Shuffling the windows is fine: the time order is kept inside each window.
    loader = DataLoader(TensorDataset(to_tensor(X_train), to_tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    history: History = {"train_loss": [], "val_loss": []}
    best_loss, best_weights, wait = np.inf, None, 0
    
    for _ in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        
        history["train_loss"].append(total_loss / len(X_train))
        history["val_loss"].append(float(np.mean((predict(model, X_val) - y_val) ** 2))) # 1/N * sum(predicted - true)^2
        
        # Early stopping
        if history["val_loss"][-1] < best_loss:
            best_loss, best_weights, wait = history["val_loss"][-1], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
            
    model.load_state_dict(best_weights)
    return history

    
def train_or_load(
    make_model: Callable[[], nn.Module],
    seed: int,
    path: Path,
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    retrain: bool = False,
) -> tuple[nn.Module, History]:
    # The seed is set before the model is built, so it fixes both the initial
    # weights and the order in which the batches are shuffled.
    torch.manual_seed(seed)
    model = make_model()
    history_path = path.with_suffix(".json")
    
    if path.exists() and history_path.exists() and not retrain:
        model.load_state_dict(torch.load(path, weights_only=True))
        history = json.loads(history_path.read_text())
    else:
        history = fit(model, *train, *validation)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        history_path.write_text(json.dumps(history))
    return model, history


def best_epoch(history: History) -> int:
    # Epochs are counted from 1, as in the training log.
    return int(np.argmin(history["val_loss"])) + 1