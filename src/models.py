from typing import Literal

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn

RECURRENT_CELLS: dict[str, type[nn.RNNBase]] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
RNN_HIDDEN: int = 32
DENSE_HIDDEN: int = 64
# Chosen on validation in notebook 04, section 5.1.
LSTM_HIDDEN: int = 16
GRU_HIDDEN: int = 16
# Chosen on validation in notebook 04, section 6.2 (direct models with 6 outputs).
LSTM_DIRECT_HIDDEN: int = 64
GRU_DIRECT_HIDDEN: int = 16


class RecurrentForecaster(nn.Module):
    def __init__(self, hidden: int, horizon: int = 1, cell: Literal["rnn", "lstm", "gru"] = "rnn") -> None:
        super().__init__()
        self._recurrent = RECURRENT_CELLS[cell](input_size=1, hidden_size=hidden, batch_first=True)
        # Linear head: the test period goes above the scaled training range, so the
        # output cannot be squashed by an activation.
        self._head = nn.Linear(hidden, horizon)
        
    def last_hidden(self, x: torch.Tensor) -> torch.Tensor:
        # Hidden state after the last month of each window, the input of the linear head.
        states, _ = self._recurrent(x)
        return states[:, -1, :]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(self.last_hidden(x))
        

class DenseForecaster(nn.Module):
    def __init__(self, window: int, hidden: int, horizon: int = 1) -> None:
        super().__init__()
        # The same window but flattened: the network sees the 36-month window at once
        # and has zero notion of their order.
        self._layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)
        

def count_parameters(model: nn.Module) -> int: 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def baseline_forecasts(windows: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    # Persistence repeats the last month of each window for every step ahead; the linear AR
    # model is a linear regression on the window, fitted on train only, with one output per step.
    X_train, y_train = windows["train"]
    horizon = y_train.shape[1]
    linear_ar = LinearRegression().fit(X_train[..., 0], y_train)
    return {
        "persistence": {part: np.repeat(X[:, -1, :], horizon, axis=1) for part, (X, _) in windows.items()},
        "linear AR": {part: linear_ar.predict(X[..., 0]) for part, (X, _) in windows.items()},
    }