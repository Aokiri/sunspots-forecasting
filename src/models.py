from typing import Literal

import torch
from torch import nn

RECURRENT_CELLS: dict[str, type[nn.RNNBase]] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
RNN_HIDDEN: int = 32
DENSE_HIDDEN: int = 64


class RecurrentForecaster(nn.Module):
    def __init__(self, hidden: int, horizon: int = 1, cell: Literal["rnn", "lstm", "gru"] = "rnn") -> None:
        super().__init__()
        self._recurrent = RECURRENT_CELLS[cell](input_size=1, hidden_size=hidden, batch_first=True)
        # Linear head: the test period goes above the scaled training range, so the
        # output cannot be squashed by an activation.
        self._head = nn.Linear(hidden, horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, 1). Only hidden state after the last month is used.
        states, _ = self._recurrent(x)
        return self._head(states[:, -1, :])
        

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