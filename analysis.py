# analysis.py
"""
Explainability and business insight analysis for ATM cash demand models.

This script:
- rebuilds the feature table for train + test
- re-creates the last-14-days validation window
- loads the trained XGBoost models
- computes permutation importance for both targets
- aggregates importance by feature group (calendar, metadata, replenishment, lags, etc.)
- saves CSVs and PNG plots under analysis_outputs/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.inspection import permutation_importance

from train import (
    build_feature_table_train_test,
    get_feature_cols,
    TRAIN_TX_PATH,
    TEST_TX_PATH,
    CAL_PATH,
    META_PATH,
    REGION_PATH,
    REPL_PATH,
)


OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = Path("models")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def feature_group(name: str) -> str:
    """Roughly assign each feature to a meaningful group."""
    if name.startswith("lag") or name.startswith("roll"):
        return "time_history"  # lags & rolling windows
    if name.startswith("repl_") or "repl" in name or "cashout" in name:
        return "replenishment_ops"
    if any(k in name for k in ["salary", "weekend", "holiday", "ramadan", "week_of_year",
                               "month", "quarter", "year", "days_to_salary",
                               "days_from_salary"]):
        return "calendar"
    if any(k in name for k in ["atm_region", "location_type", "region_lookup",
                               "location_type_lookup", "atm_age", "is_new_atm",
                               "region_mismatch"]):
        return "atm_metadata"
    return "other"


def plot_top_features(importance_df: pd.DataFrame, title: str, filename: Path, top_n: int = 15):
    """Save a horizontal bar plot for the top N features by mean importance."""
    top = importance_df.sort_values("importance_mean", ascending=False).head(top_n)
    # reverse order so the most important shows at the top of the chart
    top = top.iloc[::-1]

    plt.figure(figsize=(8, max(4, 0.4 * len(top))))
    plt.barh(top["feature"], top["importance_mean"])
    plt.title(title)
    plt.xlabel("Permutation importance (mean decrease in score)")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_group_importance(group_df: pd.DataFrame, title: str, filename: Path):
    """Save a bar plot for feature group importance."""
    grp = group_df.sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(6, 4))
    plt.barh(grp["group"], grp["importance_mean"])
    plt.title(title)
    plt.xlabel("Total permutation importance")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------
def main():
    print("Rebuilding feature table for analysis...")
    df_all = build_feature_table_train_test(
        TRAIN_TX_PATH, TEST_TX_PATH,
        CAL_PATH, META_PATH, REGION_PATH, REPL_PATH,
    )

    # Use only training rows with labels
    df_train_all = df_all[df_all["is_train"] == 1].copy()
    print("Train rows:", len(df_train_all))

    # Time-based validation window: last 14 days
    max_dt = df_train_all["dt"].max()
    val_start = max_dt - pd.Timedelta(days=13)

    df_tr = df_train_all[df_train_all["dt"] < val_start].copy()
    df_val = df_train_all[df_train_all["dt"] >= val_start].copy()

    print("Train date range:", df_tr["dt"].min().date(), "->", df_tr["dt"].max().date())
    print("Val date range:  ", df_val["dt"].min().date(), "->", df_val["dt"].max().date())
    print("#Train rows:", len(df_tr), "#Val rows:", len(df_val))

    # Features
    feature_cols, cat_cols, num_cols = get_feature_cols(df_tr)
    X_val = df_val[feature_cols]

    # Targets (log1p to match training)
    y_val_amt_log = np.log1p(df_val["withdrawn_kwd"].values)
    y_val_cnt_log = np.log1p(df_val["withdraw_count"].values)

    # Load models
    print("Loading trained models from 'models/'...")
    model_amt = load(MODELS_DIR / "model_amt_xgb.pkl")
    model_cnt = load(MODELS_DIR / "model_cnt_xgb.pkl")

    # -----------------------------------------------------------------
    # Permutation importance – withdrawal amount
    # -----------------------------------------------------------------
    print("\nComputing permutation importance for WITHDRAWN_KWD (log1p target)...")
    result_amt = permutation_importance(
        model_amt,
        X_val,
        y_val_amt_log,
        n_repeats=10,
        random_state=42,
        n_jobs=4,
    )

    imp_amt = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result_amt.importances_mean,
        "importance_std": result_amt.importances_std,
    }).sort_values("importance_mean", ascending=False)

    imp_amt["group"] = imp_amt["feature"].apply(feature_group)

    # Save raw importance
    imp_amt.to_csv(OUTPUT_DIR / "perm_importance_amount_features.csv", index=False)

    # Aggregate by group
    grp_amt = (
        imp_amt.groupby("group", as_index=False)["importance_mean"]
        .sum()
        .sort_values("importance_mean", ascending=False)
    )
    grp_amt.to_csv(OUTPUT_DIR / "perm_importance_amount_groups.csv", index=False)

    # Plots
    plot_top_features(
        imp_amt,
        title="Top features – withdrawal amount (log1p) – permutation importance",
        filename=OUTPUT_DIR / "perm_importance_amount_top15.png",
        top_n=15,
    )
    plot_group_importance(
        grp_amt,
        title="Feature-group importance – withdrawal amount",
        filename=OUTPUT_DIR / "perm_importance_amount_groups.png",
    )

    # -----------------------------------------------------------------
    # Permutation importance – withdrawal count
    # -----------------------------------------------------------------
    print("\nComputing permutation importance for WITHDRAW_COUNT (log1p target)...")
    result_cnt = permutation_importance(
        model_cnt,
        X_val,
        y_val_cnt_log,
        n_repeats=10,
        random_state=42,
        n_jobs=4,
    )

    imp_cnt = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result_cnt.importances_mean,
        "importance_std": result_cnt.importances_std,
    }).sort_values("importance_mean", ascending=False)

    imp_cnt["group"] = imp_cnt["feature"].apply(feature_group)

    imp_cnt.to_csv(OUTPUT_DIR / "perm_importance_count_features.csv", index=False)

    grp_cnt = (
        imp_cnt.groupby("group", as_index=False)["importance_mean"]
        .sum()
        .sort_values("importance_mean", ascending=False)
    )
    grp_cnt.to_csv(OUTPUT_DIR / "perm_importance_count_groups.csv", index=False)

    plot_top_features(
        imp_cnt,
        title="Top features – withdrawal count (log1p) – permutation importance",
        filename=OUTPUT_DIR / "perm_importance_count_top15.png",
        top_n=15,
    )
    plot_group_importance(
        grp_cnt,
        title="Feature-group importance – withdrawal count",
        filename=OUTPUT_DIR / "perm_importance_count_groups.png",
    )

    # -----------------------------------------------------------------
    # Sanity: print top features to console
    # -----------------------------------------------------------------
    print("\nTop 10 features for WITHDRAWN_KWD (log1p):")
    print(imp_amt.head(10).to_string(index=False))

    print("\nTop 10 features for WITHDRAW_COUNT (log1p):")
    print(imp_cnt.head(10).to_string(index=False))

    print(f"\nDone. Outputs saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()