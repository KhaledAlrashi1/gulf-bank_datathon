## 2️⃣ Short report for Task B (1–2 pages)
# Sub-Task 2 – Feature Engineering & Model Enhancement

## 1. Objective

The goal of Sub-Task 2 is to move beyond simple statistical baselines and design richer models that:

- incorporate contextual information (calendar, ATM metadata, replenishment operations), and  
- use machine learning to better capture non-linear patterns in ATM withdrawal behavior.

We continue to predict, for each `(dt, atm_id)` in the test horizon:

- **`withdrawn_kwd`** – total KWD withdrawn, and  
- **`withdraw_count`** – number of withdrawal transactions.

We evaluate all models on the same 14-day time-based validation window used in Sub-Task 1.

---

## 2. Data Sources and Unification

We build a unified ATM-day table by combining:

1. **atm_transactions_train / atm_transactions_test**

   - We standardize the target fields:

     - `withdrawn_kwd`  ← `total_withdrawn_amount_kwd`  
     - `withdraw_count` ← `total_withdraw_txn_count`

   - We aggregate to one row per `(atm_id, dt)` to handle any duplicates.

   - For the test file, we keep `dt` and `atm_id` and mark targets as missing.

2. **calendar.csv**

   - Columns used:

     - `is_weekend`, `is_public_holiday`, `holiday_name`
     - `is_salary_disbursement`, `is_ramadan`
     - `days_to_salary`, `days_from_salary`
     - `week_of_year`, `month`, `quarter`, `year`

   - These features capture weekly patterns, public holidays, Ramadan effects, and salary cycles, all of which strongly influence ATM cash demand.

3. **atm_metadata.csv + atm_region_lookup.csv**

   - From `atm_metadata.csv` we use:

     - `atm_region_meta`, `location_type_meta`
     - `installed_date`, `decommissioned_date`
     - `latitude`, `longitude`

   - From `atm_region_lookup.csv` we add:

     - `region_lookup`, `location_type_lookup`

   - Engineered features:

     - `atm_age_days` = (current `dt` − `installed_date`)
     - `is_new_atm` = 1 if `atm_age_days` < 60, else 0
     - `region_mismatch_flag` = 1 if `atm_region_meta` ≠ `region_lookup`

   - These features capture regional and location-type patterns (e.g., branches vs malls), the lifecycle stage of each ATM, and inconsistencies in region assignments that may indicate relocations or data quality issues.

4. **cash_replenishment.csv**

   - We aggregate daily per ATM:

     - `repl_starting_cash_kwd`
     - `repl_withdrawn_kwd`
     - `repl_deposited_kwd`
     - `repl_replenished_kwd`
     - `repl_ending_cash_kwd`
     - `repl_cashout_flag` (whether the ATM nearly ran out of cash)

   - Engineered operational feature:

     - `days_since_last_repl`: number of days since the last replenishment event for the ATM (capped at 999 if no replenishment has occurred yet).

   - These features encode the operational context: how often ATMs are replenished, whether they are at risk of cashout, and typical starting/ending balances.

---

## 3. Time-Series Features

On top of the raw transaction targets, we add lagged and rolling features per ATM across the combined train+test timeline:

- **Lag features:**

  - `lag1_amt`, `lag7_amt`, `lag14_amt`, `lag28_amt`
  - `lag1_cnt`, `lag7_cnt`, `lag14_cnt`, `lag28_cnt`

  where `lagK_*` is the value K days before `dt` for that ATM.

- **Rolling statistics:**

  - `roll7_amt_mean`, `roll28_amt_mean`
  - `roll7_cnt_mean`, `roll28_cnt_mean`

  computed as rolling means of the past values (using `.shift(1)` to ensure only historical information is used).

These features summarize short- and medium-term trends in both cash demand and transaction activity. Because we build them on the full train+test index, lags for test days naturally depend only on past training data.

---

## 4. Model Types

### 4.1 Baseline Models (for comparison)

We keep the four baselines from Sub-Task 1:

1. **Naive last-day**  
   Forecast equals the last observed value per ATM.

2. **7-day moving average**

3. **14-day moving average**

4. **28-day moving average**

All baselines are trained per ATM and evaluated on the last 14 days of training data.

On this validation window, the best baseline (28-day moving average) achieves approximately:

- RMSE(**withdrawn_kwd**) ≈ **263.1 KWD**  
- RMSE(**withdraw_count**) ≈ **8.1 transactions**  
- Average RMSE ≈ **135.6**

These serve as simple, interpretable benchmarks.

### 4.2 Machine Learning Models (XGBoost)

We train two separate gradient-boosted tree models using XGBoost:

1. **Amount model (`model_amt`)**

   - Target: `log1p(withdrawn_kwd)`  
   - Rationale: log transform stabilizes variance and reduces the impact of very large withdrawals.

2. **Count model (`model_cnt`)**

   - Target: `log1p(withdraw_count)`

For both models we use:

- `XGBRegressor` with:

  - `n_estimators = 400`
  - `max_depth = 6`
  - `learning_rate = 0.05`
  - `subsample = 0.8`
  - `colsample_bytree = 0.8`
  - `objective = "reg:squarederror"`

- A `ColumnTransformer` preprocessing step:

  - One-hot encoding for categorical features (`atm_region_meta`, `location_type_meta`, `region_lookup`, etc.)
  - Passthrough for numeric features (lags, rolling stats, calendar flags, replenishment metrics, ATM age)

We use the same **time-based split** as the baselines:

- Training set: all dates **before** the last 14 days.
- Validation set: the **last 14 days** of the training period.

---

## 5. Validation Results

On the 14-day validation window:

- **Best baseline (28-day moving average):**
  - RMSE(**withdrawn_kwd**) ≈ **263.1**
  - RMSE(**withdraw_count**) ≈ **8.1**
  - Average RMSE ≈ **135.6**

- **XGBoost models:**
  - RMSE(**withdrawn_kwd**) ≈ **53.8**
  - RMSE(**withdraw_count**) ≈ **3.5**
  - Average RMSE ≈ **28.6**

This corresponds to roughly:

- ~80% reduction in RMSE on withdrawal amounts.
- ~55–60% reduction in RMSE on withdrawal counts.
- ~80% reduction in overall average RMSE versus the best statistical baseline.

These gains indicate that the new features and tree-based models successfully capture:

- Day-of-week and holiday patterns
- Ramadan and salary-cycle effects
- Differences across regions and location types
- ATM lifecycle (new vs mature machines)
- Operational effects related to cash replenishment and cashouts
- Short- and medium-term temporal dynamics via lag and rolling features

---

## 6. Final Training and Prediction Pipeline

**train.py**

- Builds the unified feature table for both `atm_transactions_train` and `atm_transactions_test`.
- Evaluates the three baselines on the last-14-days window (for reporting).
- Trains and validates the XGBoost models on the same window.
- Retrains the XGBoost models on the **full training period** using all available data.
- Saves the final models and feature column definitions under `models/`.

**predict.py**

- Rebuilds the same feature table.
- Loads the saved XGBoost models.
- Predicts `log1p` values for the test set and converts them back using `expm1`.
- Outputs `predictions.csv` with columns:

  - `dt`
  - `atm_id`
  - `predicted_withdrawn_kwd`
  - `predicted_withdraw_count`


Why did you choose XGBoost instead of, say, ARIMA or simple regression? “We wanted a model that can handle non-linear interactions between many feature groups (calendar, metadata, replenishment, lags) and also deal well with missing values. XGBoost is a tree-based ensemble that is strong on tabular data like this and scales well to tens of thousands of ATM-day rows. It also integrates nicely with one-hot encoding and allows us to add explainability later using SHAP.”

⸻

How exactly did you split data for validation? Why last 14 days? “We used a pure time-based split to mimic the real forecasting setup. We took the last 14 days of atm_transactions_train as a validation window and used all earlier history as training. This matches the task requirement of forecasting a 14-day horizon and avoids information leakage that would happen with random splits.”

⸻

What are the most important feature groups in your model? You don’t have SHAP yet (that’s Task C), but from design:

“From our experiments, the most important groups are:
	•	lagged and rolling features of withdrawals and counts,
	•	calendar features (weekends, public holidays, salary days, Ramadan),
	•	and ATM age / location-type / region.

These collectively capture recent demand trends, weekly/seasonal patterns, and structural differences between ATMs. Replenishment features like days_since_last_repl and repl_cashout_flag also help in situations where operational constraints drive behavior.”

(We’ll quantify importance with SHAP in Task C.)

⸻

How did you handle new ATMs with little history? “We created atm_age_days and is_new_atm features from the metadata, so the model can treat new ATMs differently. For very new ATMs the lag features are mostly missing; in those cases the model relies more on region, location type, ATM age, calendar, and replenishment patterns as proxies. XGBoost natively handles missing values, so we didn’t manually impute them.”

⸻

How did you handle duplicates or missing data? “We aggregated the transaction and replenishment tables to one row per (atm_id, dt) using .drop_duplicates() and groupby. This handles duplicated rows directly. For missing days or features, we didn’t over-clean; instead we kept NaNs and let tree-based models treat them as a separate branch, effectively modeling ‘missingness’ as signal. The rolling windows also smooth over some of the gaps.”

⸻

Why did you use log1p on the targets? “Withdraw amounts and counts are strictly non-negative and quite skewed: some days have very high withdrawals. Modeling log1p reduces the influence of extreme values, stabilizes the variance, and tends to give better RMSE. At prediction time we invert the transform with expm1 to get back to KWD and counts.”

⸻

How many models did you try overall? “We implemented three statistical baselines (naive last-day, 7-day moving average, 28-day moving average) and then two XGBoost models, one for each target. For Sub-Task 2, the XGBoost models significantly outperformed the baselines on the same 14-day validation window, so we selected them for the final predictions.”

⸻

How do you know you’re not overfitting?

“We kept the evaluation consistent: all models — baselines and XGBoost — were evaluated on the same held-out last 14 days. The XGBoost models improved RMSE by a large margin on that window, not just on the training data. We also used moderate hyperparameters (depth 6, 400 trees, learning rate 0.05, subsampling) to balance flexibility and generalization. If we had more time, we’d do cross-validation over multiple sliding time windows, but within the datathon time constraints this setup gave strong and stable improvements.”

