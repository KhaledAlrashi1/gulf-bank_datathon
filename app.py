# dashboard/app.py
"""
Streamlit dashboard for "what-if" scenario visualization (Task 1.D).

Scenarios:
- Shift salary payment date earlier/later and see impact on demand.
- Toggle holiday / Ramadan effects on or off.
- Add new ATMs in a region and estimate extra regional demand.

The app:
- rebuilds the feature table using train.build_feature_table_train_test()
- uses the trained XGBoost models to predict for the test horizon
- aggregates results by day and region
- plots baseline vs scenario lines for easy comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from joblib import load

# Import the feature-building utilities and paths from train.py
from train import (
    build_feature_table_train_test,
    TRAIN_TX_PATH,
    TEST_TX_PATH,
    CAL_PATH,
    META_PATH,
    REGION_PATH,
    REPL_PATH,
)


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"


# ------------------------------------------------------------------
# Caching helpers
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_feature_table():
    """Build full feature table for train + test, then return test rows only."""
    df_all = build_feature_table_train_test(
        TRAIN_TX_PATH,
        TEST_TX_PATH,
        CAL_PATH,
        META_PATH,
        REGION_PATH,
        REPL_PATH,
    )
    df_test = df_all[df_all["is_train"] == 0].copy()
    return df_test


@st.cache_resource(show_spinner=False)
def load_models_and_features():
    feature_cols = np.load(MODELS_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    model_amt = load(MODELS_DIR / "model_amt_xgb.pkl")
    model_cnt = load(MODELS_DIR / "model_cnt_xgb.pkl")
    return feature_cols, model_amt, model_cnt


def predict_for_features(df_features, feature_cols, model_amt, model_cnt):
    """Run both models and return a DataFrame with predictions."""
    X = df_features[feature_cols]
    y_amt_log = model_amt.predict(X)
    y_cnt_log = model_cnt.predict(X)

    out = df_features[["dt", "atm_id"]].copy()
    # Meta for grouping
    for col in ["atm_region_meta", "location_type_meta"]:
        if col in df_features.columns:
            out[col] = df_features[col]
        else:
            out[col] = "Unknown"

    out["pred_withdrawn_kwd"] = np.expm1(y_amt_log)
    out["pred_withdraw_count"] = np.expm1(y_cnt_log)
    return out


def aggregate_daily(pred_df, region=None):
    """Aggregate predictions to daily totals, optionally for a single region."""
    df = pred_df.copy()
    if region and region != "All regions":
        df = df[df["atm_region_meta"] == region]

    daily = (
        df.groupby("dt", as_index=False)[["pred_withdrawn_kwd", "pred_withdraw_count"]]
        .sum()
        .sort_values("dt")
    )
    return daily


# ------------------------------------------------------------------
# Scenario transforms
# ------------------------------------------------------------------
def apply_salary_shift(df_features, shift_days: int) -> pd.DataFrame:
    """Shift salary date by +/- N days via calendar features."""
    if shift_days == 0:
        return df_features.copy()

    df = df_features.copy()
    if "days_to_salary" in df.columns:
        df["days_to_salary"] = df["days_to_salary"] - shift_days
    if "days_from_salary" in df.columns:
        df["days_from_salary"] = df["days_from_salary"] + shift_days
    if "is_salary_disbursement" in df.columns:
        # Recompute roughly: salary day when days_to_salary == 0
        df["is_salary_disbursement"] = (df.get("days_to_salary", 999) == 0).astype(int)
    return df


def apply_holiday_toggles(df_features, remove_holidays: bool, remove_ramadan: bool):
    """Optionally zero out holiday / Ramadan indicators."""
    df = df_features.copy()
    if remove_holidays and "is_public_holiday" in df.columns:
        df["is_public_holiday"] = 0
    if remove_ramadan and "is_ramadan" in df.columns:
        df["is_ramadan"] = 0
    return df


def compute_new_atm_boost(baseline_pred, region: str, n_new_atms: int) -> pd.DataFrame:
    """
    Approximate impact of adding N new ATMs in a region by
    assuming each new ATM behaves like the current average ATM
    in that region.

    Always returns a DataFrame with index=dt and columns:
    - boost_amt
    - boost_cnt
    """
    # Common date index
    idx = pd.Index(baseline_pred["dt"].unique(), name="dt")

    # Case 1: no scenario effect -> all zeros
    if n_new_atms <= 0:
        return pd.DataFrame(
            {
                "boost_amt": np.zeros(len(idx), dtype=float),
                "boost_cnt": np.zeros(len(idx), dtype=float),
            },
            index=idx,
        )

    # Filter region
    if region == "All regions":
        reg_df = baseline_pred.copy()
    else:
        reg_df = baseline_pred[baseline_pred["atm_region_meta"] == region]
        
    if reg_df.empty:
        return pd.DataFrame(
            {
                "boost_amt": np.zeros(len(idx), dtype=float),
                "boost_cnt": np.zeros(len(idx), dtype=float),
            },
            index=idx,
        )

    # Average per-ATM predictions per day
    per_atm = (
        reg_df.groupby(["dt", "atm_id"], as_index=False)[
            ["pred_withdrawn_kwd", "pred_withdraw_count"]
        ]
        .sum()
    )
    daily_mean = (
        per_atm.groupby("dt", as_index=False)[
            ["pred_withdrawn_kwd", "pred_withdraw_count"]
        ]
        .mean()
        .set_index("dt")
    )

    # Boost = N * avg per-ATM
    boost_amt = n_new_atms * daily_mean["pred_withdrawn_kwd"]
    boost_cnt = n_new_atms * daily_mean["pred_withdraw_count"]

    boost = pd.DataFrame(
        {"boost_amt": boost_amt, "boost_cnt": boost_cnt},
        index=daily_mean.index,
    )

    # Reindex to full date range (fill missing with 0)
    boost = boost.reindex(idx).fillna(0.0)
    return boost


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="ATM Cash Demand – Scenario Explorer",
        layout="wide",
    )

    st.title("ATM Cash Demand – Scenario Visualization")
    st.markdown(
        "Explore *what-if* scenarios on top of the trained models:\n"
        "- Shift salary payment dates\n"
        "- Remove holiday / Ramadan effects\n"
        "- Add new ATMs in a region\n\n"
        "The plots compare **baseline vs scenario** daily forecasts for the test period."
    )

    with st.sidebar:
        st.header("Scenario controls")

        # Load data & models
        df_test = load_feature_table()
        feature_cols, model_amt, model_cnt = load_models_and_features()

        # Region selection
        regions = ["All regions"] + sorted(
            df_test["atm_region_meta"].dropna().unique().tolist()
        )
        region = st.selectbox("Region", regions)

        st.subheader("Salary-day scenario")
        salary_shift = st.slider(
            "Shift salary date (days)",
            min_value=-5,
            max_value=5,
            value=0,
            help="Negative = salary paid earlier, positive = paid later",
        )

        st.subheader("Calendar effects")
        remove_holidays = st.checkbox("Remove public holiday effect", value=False)
        remove_ramadan = st.checkbox("Remove Ramadan effect", value=False)

        st.subheader("New ATM scenario")
        n_new_atms = st.slider(
            "Number of new ATMs in region",
            min_value=0,
            max_value=10,
            value=0,
            help="Assumes each new ATM behaves like the average ATM in the selected region.",
        )

    # ------------------------------------------------------------------
    # Baseline predictions
    # ------------------------------------------------------------------
    baseline_pred = predict_for_features(df_test, feature_cols, model_amt, model_cnt)
    baseline_daily = aggregate_daily(baseline_pred, region=region)

    # ------------------------------------------------------------------
    # Apply scenario transforms
    # ------------------------------------------------------------------
    df_scn = df_test.copy()
    df_scn = apply_salary_shift(df_scn, salary_shift)
    df_scn = apply_holiday_toggles(df_scn, remove_holidays, remove_ramadan)

    scenario_pred = predict_for_features(df_scn, feature_cols, model_amt, model_cnt)

    # Add approximate new ATM effect
    boost = compute_new_atm_boost(baseline_pred, region, n_new_atms)
    boost = boost.reindex(
        baseline_daily["dt"].values
    )  # align index with baseline dates

    scenario_daily = aggregate_daily(scenario_pred, region=region)
    scenario_daily["pred_withdrawn_kwd"] += boost["boost_amt"].fillna(0).values
    scenario_daily["pred_withdraw_count"] += boost["boost_cnt"].fillna(0).values

    # ------------------------------------------------------------------
    # Build comparison frames
    # ------------------------------------------------------------------
    comp_amt = pd.DataFrame(
        {
            "dt": baseline_daily["dt"],
            "Baseline": baseline_daily["pred_withdrawn_kwd"],
            "Scenario": scenario_daily["pred_withdrawn_kwd"],
        }
    )
    comp_cnt = pd.DataFrame(
        {
            "dt": baseline_daily["dt"],
            "Baseline": baseline_daily["pred_withdraw_count"],
            "Scenario": scenario_daily["pred_withdraw_count"],
        }
    )

    # Melt for Altair
    comp_amt_m = comp_amt.melt("dt", var_name="Series", value_name="withdrawn_kwd")
    comp_cnt_m = comp_cnt.melt("dt", var_name="Series", value_name="withdraw_count")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    st.subheader(
        f"Daily withdrawal amount – Baseline vs Scenario "
        f"({'All regions' if region=='All regions' else region})"
    )
    chart_amt = (
        alt.Chart(comp_amt_m)
        .mark_line()
        .encode(
            x="dt:T",
            y=alt.Y("withdrawn_kwd:Q", title="Total withdrawn KWD"),
            color="Series:N",
        )
        .properties(height=300)
    )
    st.altair_chart(chart_amt, use_container_width=True)

    # Summary metrics for amount
    col1, col2, col3 = st.columns(3)
    total_base_amt = baseline_daily["pred_withdrawn_kwd"].sum()
    total_scn_amt = scenario_daily["pred_withdrawn_kwd"].sum()
    delta_amt = total_scn_amt - total_base_amt
    pct_amt = 100 * delta_amt / total_base_amt if total_base_amt > 0 else 0.0
    col1.metric("Baseline total KWD", f"{total_base_amt:,.0f}")
    col2.metric("Scenario total KWD", f"{total_scn_amt:,.0f}")
    col3.metric("Change vs baseline", f"{delta_amt:,.0f} KWD", f"{pct_amt:+.1f}%")

    st.markdown("---")

    st.subheader(
        f"Daily withdrawal count – Baseline vs Scenario "
        f"({'All regions' if region=='All regions' else region})"
    )
    chart_cnt = (
        alt.Chart(comp_cnt_m)
        .mark_line()
        .encode(
            x="dt:T",
            y=alt.Y("withdraw_count:Q", title="Total withdrawal count"),
            color="Series:N",
        )
        .properties(height=300)
    )
    st.altair_chart(chart_cnt, use_container_width=True)

    total_base_cnt = baseline_daily["pred_withdraw_count"].sum()
    total_scn_cnt = scenario_daily["pred_withdraw_count"].sum()
    delta_cnt = total_scn_cnt - total_base_cnt
    pct_cnt = 100 * delta_cnt / total_base_cnt if total_base_cnt > 0 else 0.0

    col4, col5, col6 = st.columns(3)
    col4.metric("Baseline total count", f"{total_base_cnt:,.0f}")
    col5.metric("Scenario total count", f"{total_scn_cnt:,.0f}")
    col6.metric("Change vs baseline", f"{delta_cnt:,.0f}", f"{pct_cnt:+.1f}%")

    st.markdown("---")

    # Scenario description panel
    st.subheader("Scenario summary")
    st.write(
        f"- **Region**: {region}\n"
        f"- **Salary shift**: {salary_shift} day(s)\n"
        f"- **Remove public holidays**: {remove_holidays}\n"
        f"- **Remove Ramadan effect**: {remove_ramadan}\n"
        f"- **New ATMs in region**: {n_new_atms}\n\n"
        "The scenario line shows how these assumptions affect total cash withdrawals "
        "and transaction counts over the test horizon."
    )


if __name__ == "__main__":
    main()