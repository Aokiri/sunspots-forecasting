# Sunspots forecasting

Forecasting the monthly mean sunspot number (1749 to 1983) with recurrent neural networks: simple RNN, LSTM and GRU, compared with dense networks, a linear autoregressive model and a persistence baseline.

## Executed notebooks

All notebooks are already executed. The `html/` folder has an HTML export of each one, so the results and plots can be read without installing anything.

## Installation

Python 3.13 was used. From the project root:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is stored in `data/monthly-sunspots.csv`, so no internet connection is needed. When opening a notebook, select the kernel of this environment.

## Run order

1. `01_data_preparation.ipynb`
2. `02_one_step.ipynb`
3. `03_multi_step.ipynb`
4. `04_gated_cells.ipynb`

Each notebook runs on its own, since the shared code and the decisions of earlier notebooks are in `src/` and the trained models in `models/`.

## Saved models

Every network is trained with 10 seeds, and the weights and loss histories of all runs are saved in `models/`, one folder per notebook. By default the notebooks load these weights, so a full run takes about a minute per notebook. Setting `RETRAIN = True` in the setup cell of notebooks 02 to 04 trains everything again; on a laptop CPU this takes about 2.5 hours in total, most of it for the GRU in notebook 04.

All models were trained on CPU. On the same machine, retraining gives exactly the same weights; on other hardware, or on a GPU, the numbers can differ slightly.

## Code in `src/`

- `data.py`: loading, temporal split, scaler, windows and RMSE in sunspots.
- `models.py`: the recurrent and dense networks, the baselines and the chosen hidden sizes.
- `training.py`: training loop with early stopping, saving and loading of the weights, and recursive forecasts.
- `evaluation.py`: the error tables used in the test sections.
- `style.py`: plot style and colour palettes, with a colourblind-safe alternative.

## Course context

Developed for the Advanced Machine Learning course. Mapping to the course tasks:
- Tasks 1 and 2 in `01_data_preparation.ipynb`
- Task 3 in `02_one_step.ipynb`
- Task 4 in `03_multi_step.ipynb`
- Task 5 in `04_gated_cells.ipynb`