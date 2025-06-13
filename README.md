# BDA 2025 Final

資工三 詹挹辰 B11902057

## Environment

- Python 3.9.6
- Required packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

## Usage

Run `main.py` to generate the submission file.

```bash
python main.py
```

This script will generate `public_submission.csv` and `private_submission.csv` files in the current directory. Also, it will plot the graphs of the data and save to `plots/` directory.

There is also another script `eda.py`, which will generate the feature pairs plots. You can run it with:

```bash
python eda.py
```