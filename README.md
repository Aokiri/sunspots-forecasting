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

The dataset is stored in `data/monthly-sunspots.csv`, so no internet connection is needed.

## Run order

1. `01_data_preparation.ipynb`
2. `02_one_step.ipynb`
3. `03_multi_step.ipynb`
4. `04_gated_cells.ipynb`

## Course context

Developed for the Advanced Machine Learning course. Mapping to the course tasks: 
- Tasks 1 and 2 in `01_data_preparation.ipynb`
- Task 3 in `02_one_step.ipynb`
- Task 4 in `03_multi_step.ipynb`
- Task 5 in `04_gated_cells.ipynb`