Sub-Task 3 – Explainability & Business Insight Report

1. Objective

The goal of Sub-Task 3 is to understand why the ATM cash-demand models make their predictions and translate those findings into actionable recommendations for cash planning and replenishment.

We focus on the two XGBoost models developed in Sub-Task 2:
	•	Model 1: withdrawn_kwd (total KWD withdrawn per ATM-day)
	•	Model 2: withdraw_count (number of withdrawal transactions per ATM-day)

Both models were trained on the full 2020–2025 training period and validated on the last 14 days of data, matching the competition’s forecast horizon.

⸻

2. Explainability Method

We use permutation importance from sklearn.inspection as an explainability tool:
	1.	Hold out the same 14-day validation window used for model selection.
	2.	For each feature:
	•	Randomly shuffle its values across the validation set,
	•	Measure how much the model’s error (on log-transformed targets) increases,
	3.	The larger the average error increase, the more important the feature.

We compute importance both:
	•	at the individual feature level, and
	•	at the feature-group level, grouping features into:
  	•	Replenishment operations (repl_*, days_since_last_repl, cashout flags)
  	•	Time history (lags and rolling means: lag*, roll*)
  	•	Calendar (salary disbursements, weekends, holidays, Ramadan, month/quarter/year)
  	•	ATM metadata (region, location type, ATM age, region mismatch)
	  •	Other (remaining features)

Figures:
	•	Figure 1. Top 15 features – withdrawal amount
(analysis_outputs/perm_importance_amount_top15.png)
	•	Figure 2. Top 15 features – withdrawal count
(analysis_outputs/perm_importance_count_top15.png)
	•	Figure 3. Group importance – withdrawal amount
(analysis_outputs/perm_importance_amount_groups.png)
	•	Figure 4. Group importance – withdrawal count
(analysis_outputs/perm_importance_count_groups.png)

⸻

3. Key Drivers of Withdrawal Amount

3.1. Top features
The most important features for withdrawal amount are:
	1.	repl_withdrawn_kwd – cash actually drawn from the vault
	2.	repl_starting_cash_kwd – starting cash in the ATM on that day
	3.	roll28_amt_mean – 28-day rolling mean of withdrawn KWD
	4.	repl_replenished_kwd – cash loaded into the ATM by CIT
	5.	repl_ending_cash_kwd – end-of-day cash balance
	6.	is_salary_disbursement – salary pay-day indicator
	7.	repl_deposited_kwd – cash deposited by customers
	8.	roll7_amt_mean – 7-day rolling mean of withdrawn KWD
	9.	days_since_last_repl – days since last replenishment
	10.	roll7_cnt_mean – 7-day average withdrawal count

At the group level (Figure 3):
	•	Replenishment operations dominate by a large margin.
	•	Time-history features are the second most important group.
	•	Calendar features (salary/holidays) have modest but non-zero importance.
	•	ATM metadata and other features contribute only small additional signal.

3.2. Interpretation
	•	The model views the replenishment cycle itself as the main driver of cash withdrawals: how much cash is loaded, how rapidly it is withdrawn, and how long it has been since the last refill.
	•	Recent historical volume (especially 28-day and 7-day averages) forms a stable baseline per ATM.
	•	Salary disbursement days act as consistent positive shocks on top of the operational and historical patterns, but they are not the primary driver.

⸻

4. Key Drivers of Withdrawal Count

4.1. Top features
The most important features for withdrawal count are:
	1.	repl_withdrawn_kwd
	2.	roll28_cnt_mean – 28-day average transaction count
	3.	location_type_meta – branch vs mall vs off-site
	4.	is_salary_disbursement
	5.	repl_deposited_kwd
	6.	roll7_cnt_mean
	7.	repl_replenished_kwd
	8.	location_type_lookup
	9.	repl_starting_cash_kwd
	10.	roll28_amt_mean

At the group level (Figure 4):
	•	Replenishment operations again rank first.
	•	Time-history is second.
	•	ATM metadata (especially location type) becomes much more important than for the amount model.
	•	Calendar still matters but is weaker than the above groups.

4.2. Interpretation
	•	Operational flows (repl_withdrawn_kwd, repl_deposited_kwd) and recent transaction history determine how many customers are likely to use the ATM on a given day.
	•	Location type is a key differentiator for counts:
	•	Branch ATMs vs mall ATMs vs off-site ATMs show distinct customer-traffic patterns.
	•	Salary days reliably increase transaction counts, but the magnitude of the increase depends strongly on ATM type and its historical volume.

⸻

5. Business Insights & Recommendations

Grounded in the feature-importance results, we propose the following operational insights.

5.1. Segment ATMs by replenishment intensity
Insight: Replenishment features are the most important drivers for both amount and count.

Action:
	•	Classify ATMs into high, medium, and low intensity segments based on:
	•	average repl_withdrawn_kwd,
	•	frequency of replenishments,
	•	days_since_last_repl behavior.
	•	For high-intensity ATMs:
	•	Maintain higher cash buffers,
	•	Shorten replenishment intervals,
	•	Prioritize them at the start of CIT routes.
	•	For low-intensity ATMs:
	•	Reduce starting cash levels,
	•	Consider longer replenishment cycles,
	•	Potentially redeploy capacity if volumes remain low.

Benefit: Frees idle cash from quiet ATMs and reallocates it to high-demand machines, improving liquidity usage while reducing cash-out risk.

⸻

5.2. Use rolling history as dynamic baselines
Insight: 7-day and 28-day rolling averages of withdrawals and counts are the second most important feature group.

Action:
	•	For each ATM, maintain:
	•	a 28-day “normal” baseline for both amount and count,
	•	a 7-day short-term trend indicator.
	•	Trigger alerts when:
	•	roll7 significantly exceeds roll28 (sudden surge),
	•	or roll7 drops sharply below roll28 (potential issue or demand shift).
	•	Incorporate these metrics into the replenishment algorithm so that routes and cash levels update automatically in response to recent usage.

Benefit: The bank moves from static rules to adaptive, data-driven planning, better capturing local surges and declines.

⸻

5.3. Target salary days where they matter most
Insight: is_salary_disbursement is a top feature, but calendar effects are weaker than operations and history.

Action:
	•	On salary disbursement days:
	•	Focus extra cash and additional replenishment visits on ATMs that are:
	•	already high intensity (from 5.1), and
	•	currently above their 28-day baseline.
	•	For low-intensity ATMs, only modest top-ups are needed; history and replenishment logs show they do not convert salary days into large spikes.

Benefit: Avoid over-loading low-usage machines on salary days while ensuring high-traffic ATMs never run dry during peak demand.

⸻

5.4. Differentiate strategies by ATM location type
Insight: location_type_meta and location_type_lookup are among the most important features for withdrawal count.

Action:
	•	Design separate policies for branches, malls, and off-site ATMs:
	•	Branch ATMs – high transaction counts but often smaller amounts per transaction:
	•	Focus on visit frequency and uptime.
	•	Mall / high-footfall ATMs – strong peaks on evenings/weekends:
	•	Align replenishment schedules with these peaks.
	•	Off-site ATMs – fewer but potentially larger withdrawals:
	•	Maintain sufficient buffer to cover occasional large draws, but routes can be more flexible.
	•	Use the model’s forecasts by segment to trial different cash limits and thresholds per location type.

Benefit: Align cash strategies with real customer-traffic patterns rather than treating all ATMs uniformly.

⸻

5.5. Manage cash-out risk with “days since last replenishment”
Insight: days_since_last_repl appears in the top drivers for withdrawal amount and belongs to the dominant replenishment group.

Action:
	•	Combine days_since_last_repl with predicted demand to compute a cash-out risk score per ATM-day.
	•	Use this score to:
	•	Prioritize ATMs in daily routing,
	•	Flag ATMs that are predicted to approach their lower cash threshold before the next scheduled visit.

Benefit: Move from fixed replenishment intervals to risk-based scheduling, reducing emergency cash-outs and unnecessary visits.

⸻

6. Limitations and Next Steps
	•	Due to time constraints, we focused on global permutation importance. As a next step, we could:
	•	Use SHAP values to analyze local explanations for specific ATMs and dates (e.g., why a particular ATM is forecasted to spike during Ramadan).
	•	Create per-region or per-segment models to capture finer-grained behavior.
	•	The current importance analysis is based on the last 14 days of the training period; extending it to multiple sliding windows would further validate stability of the drivers.

⸻

Summary:
Permutation importance shows that replenishment behavior, recent historical demand, and location type are the dominant drivers of ATM cash withdrawals and transaction counts, with salary days and calendar effects as important modifiers. These findings directly support practical improvements in cash-loading limits, route prioritization, and differentiated strategies by ATM segment, helping Gulf Bank reduce idle cash while maintaining high service availability.