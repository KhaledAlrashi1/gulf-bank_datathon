# Scenario Visualization – Summary (Task 1.D)

## Objective

Provide an interactive view of how different business scenarios affect ATM cash demand, using the trained XGBoost models from Sub-Task 2.

The dashboard is implemented in **Streamlit** (`app.py`) and uses the same feature-engineering pipeline and models as the main solution.

## Scenarios

The dashboard supports three what-if scenarios, which can be combined:

1. **Salary-day shift**

   - Control: slider from −5 to +5 days.
   - Implementation: adjusts `days_to_salary`, `days_from_salary` and `is_salary_disbursement` features in the calendar data for the test horizon, then re-runs the models.
   - Use case: explore the effect of paying salaries earlier or later in the month on total cash withdrawals and transaction counts.

2. **Holiday / Ramadan effect**

   - Controls: checkboxes to *remove public holiday effect* and *remove Ramadan effect*.
   - Implementation: sets `is_public_holiday` and/or `is_ramadan` to zero in the feature table before prediction.
   - Use case: compare demand patterns on normal days versus periods with holidays or Ramadan, and understand how much extra cash should be allocated during those periods.

3. **New ATMs in a region**

   - Controls: region selector and a slider for the number of new ATMs (0–10).
   - Implementation: for the selected region, estimates the average per-ATM forecast and scales it by the number of new ATMs, adding this to regional totals.
   - Use case: estimate how adding more ATMs in a region would increase total regional cash demand and transaction volume.

## Visual Outputs

For the selected region (or all regions), the app displays:

- **Daily withdrawal amount** – baseline vs scenario line chart.
- **Daily withdrawal count** – baseline vs scenario line chart.
- Summary metrics:
  - Baseline total KWD and total transactions
  - Scenario totals
  - Absolute and percentage change versus baseline

These charts are designed to be easy for business stakeholders to read: the blue line is the **baseline forecast**, and the orange line is the **scenario forecast**.

## Business Value

The dashboard links model outputs directly to decisions such as:

- how much extra cash to load around salary days or holidays,
- how sensitive demand is to changing salary dates,
- and how regional cash needs change when new ATMs are added.