# Sunspots forecasting

Midterm project of Advanced Machine Learning and Deep Learning: forecasting the monthly mean sunspot number with recurrent neural networks.

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

1. `01_data_preparation.ipynb`: Task 1 and Task 2
2. `02_one_step.ipynb`: Task 3
3. `03_multi_step.ipynb`: Task 4
4. `04_gated_cells.ipynb`: Task 5
