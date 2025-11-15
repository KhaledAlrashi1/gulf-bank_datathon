# ATM Cash Demand – Sub-Task 1 (Baseline Forecasting)

## Objective

Forecast ATM cash withdrawals for the next 14 days for each `(dt, atm_id)` in `atm_transactions_test.csv`, predicting:

- `withdrawn_kwd` – total withdrawal amount in KWD
- `withdraw_count` – number of withdrawal transactions

## Data

- `atm_transactions_train.csv` – daily withdrawals per ATM.
- `atm_transactions_test.csv` – `(dt, atm_id)` pairs for the forecast horizon.

In the code, raw columns are renamed to internal names:

- `total_withdrawn_amount_kwd` → `withdrawn_kwd`
- `total_withdraw_txn_count`   → `withdraw_count`

and aggregated to one row per `(atm_id, dt)`.

## Baseline Models

All baselines are computed separately for each `atm_id`:

1. **Naive last-day (`naive_last`)**  
   Prediction for all future days = last observed value in the training period.

2. **7-day moving average (`ma_7`)**  
   Prediction = mean of the last (up to) 7 days of history.

3. **14-day moving average (`ma_14`)**  
   Prediction = mean of the last (up to) 14 days of history.

4. **28-day moving average (`ma_28`)**  
   Prediction = mean of the last (up to) 28 days of history.

These are simple and fully interpretable benchmarks.

## Validation

We use a time-based holdout on the training data:

- Training calibration: all dates *before* the last 14 days.
- Validation: the last 14 days of the training period.

For each baseline we compute RMSE for both targets:

| model       | RMSE (withdrawn_kwd) | RMSE (withdraw_count) | Average RMSE |
|------------|---------------------:|----------------------:|-------------:|
| naive_last | 349.36               | 10.76                 | 180.06       |
| ma_7       | 292.40               |  9.09                 | 150.74       |
| ma_14      | 285.03               |  8.87                 | 146.95       |
| ma_28      | **263.12**           | **8.10**              | **135.61**   |

The **28-day moving average** (`ma_28`) gives the lowest RMSE on both targets and is used for the final `predictions.csv`.

## How to Run

```bash
pip install -r requirements.txt

python train.py    # evaluates baselines and prints RMSEs
python predict.py  # generates predictions.csv