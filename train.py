# train.py
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from joblib import dump

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_DIR = Path("data")
TRAIN_TX_PATH = DATA_DIR / "atm_transactions_train.csv"
TEST_TX_PATH  = DATA_DIR / "atm_transactions_test.csv"  # adjust if needed

CAL_PATH      = DATA_DIR / "calendar.csv"
META_PATH     = DATA_DIR / "atm_metadata.csv"
REGION_PATH   = DATA_DIR / "atm_region_lookup.csv"
REPL_PATH     = DATA_DIR / "cash_replenishment.csv"

MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------
def load_clean_transactions(train_path: Path, test_path: Path):
    """Load and clean transaction data for train and test.

    For train: create withdrawn_kwd, withdraw_count and mark is_train=1.
    For test: keep dt, atm_id and mark is_train=0 (targets are NaN).
    """
    # --- Train ---
    train = pd.read_csv(train_path)

    train = train.rename(columns={
        "dt": "dt",
        "atm_id": "atm_id",
        "total_withdrawn_amount_kwd": "withdrawn_kwd",
        "total_withdraw_txn_count": "withdraw_count",
    })

    train["dt"] = pd.to_datetime(train["dt"])
    train = (
        train[["dt", "atm_id", "withdrawn_kwd", "withdraw_count"]]
        .drop_duplicates()
        .groupby(["atm_id", "dt"], as_index=False)[["withdrawn_kwd", "withdraw_count"]]
        .sum()
    )
    train["is_train"] = 1

    # --- Test ---
    test = pd.read_csv(test_path)
    test["dt"] = pd.to_datetime(test["dt"])
    test = test[["dt", "atm_id"]].copy()
    test["withdrawn_kwd"] = np.nan
    test["withdraw_count"] = np.nan
    test["is_train"] = 0

    return train, test


def load_calendar(path: Path) -> pd.DataFrame:
    cal = pd.read_csv(path)
    cal["dt"] = pd.to_datetime(cal["dt"])

    for col in ["is_weekend", "is_public_holiday",
                "is_salary_disbursement", "is_ramadan"]:
        if col in cal.columns:
            cal[col] = cal[col].astype(int)

    return cal


def load_atm_metadata(meta_path: Path, region_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)

    meta = meta.rename(columns={
        "region": "atm_region_meta",
        "location_type": "location_type_meta",
    })

    if "installed_date" in meta.columns:
        meta["installed_date"] = pd.to_datetime(meta["installed_date"])
    if "decommissioned_date" in meta.columns:
        meta["decommissioned_date"] = pd.to_datetime(meta["decommissioned_date"])

    reg = pd.read_csv(region_path)
    reg = reg.rename(columns={
        "region": "region_lookup",
        "location_type": "location_type_lookup",
    })

    meta = meta.merge(reg, on="atm_id", how="left")

    if "atm_region_meta" in meta.columns and "region_lookup" in meta.columns:
        meta["region_mismatch_flag"] = (
            meta["atm_region_meta"] != meta["region_lookup"]
        ).astype(int)

    return meta


def load_replenishment(path: Path) -> pd.DataFrame:
    repl = pd.read_csv(path)
    repl["dt"] = pd.to_datetime(repl["dt"])

    daily = repl.groupby(["atm_id", "dt"], as_index=False).agg({
        "starting_cash_kwd": "first",
        "withdrawn_kwd": "sum",
        "deposited_kwd": "sum",
        "replenished_kwd": "sum",
        "ending_cash_kwd": "last",
        "cashout_flag": "max",
    })

    daily = daily.rename(columns={
        "starting_cash_kwd": "repl_starting_cash_kwd",
        "withdrawn_kwd": "repl_withdrawn_kwd",
        "deposited_kwd": "repl_deposited_kwd",
        "replenished_kwd": "repl_replenished_kwd",
        "ending_cash_kwd": "repl_ending_cash_kwd",
        "cashout_flag": "repl_cashout_flag",
    })

    daily["repl_cashout_flag"] = daily["repl_cashout_flag"].astype(int)
    return daily


def add_lag_features(df: pd.DataFrame, lags=(1, 7, 14, 28)) -> pd.DataFrame:
    df = df.sort_values(["atm_id", "dt"]).copy()
    for lag in lags:
        df[f"lag{lag}_amt"] = df.groupby("atm_id")["withdrawn_kwd"].shift(lag)
        df[f"lag{lag}_cnt"] = df.groupby("atm_id")["withdraw_count"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows=(7, 28)) -> pd.DataFrame:
    df = df.sort_values(["atm_id", "dt"]).copy()
    for w in windows:
        df[f"roll{w}_amt_mean"] = (
            df.groupby("atm_id")["withdrawn_kwd"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"roll{w}_cnt_mean"] = (
            df.groupby("atm_id")["withdraw_count"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df


def build_feature_table_train_test(
    train_tx_path: Path,
    test_tx_path: Path,
    cal_path: Path,
    meta_path: Path,
    region_path: Path,
    repl_path: Path,
) -> pd.DataFrame:
    """Build full feature table for both train and test rows."""
    df_train_tx, df_test_tx = load_clean_transactions(train_tx_path, test_tx_path)
    cal = load_calendar(cal_path)
    meta = load_atm_metadata(meta_path, region_path)
    repl_daily = load_replenishment(repl_path)

    # Combine train + test
    df_all = pd.concat([df_train_tx, df_test_tx], ignore_index=True)
    df_all = df_all.sort_values(["atm_id", "dt"]).reset_index(drop=True)

    # Join calendar
    df_all = df_all.merge(cal, on="dt", how="left")

    # Join metadata
    df_all = df_all.merge(meta, on="atm_id", how="left")

    # Join replenishment
    df_all = df_all.merge(repl_daily, on=["atm_id", "dt"], how="left")

    # ATM age features
    if "installed_date" in df_all.columns:
        df_all["atm_age_days"] = (df_all["dt"] - df_all["installed_date"]).dt.days
        df_all["is_new_atm"] = (df_all["atm_age_days"] < 60).astype(int)
    else:
        df_all["atm_age_days"] = np.nan
        df_all["is_new_atm"] = 0

    # Days since last replenishment
    df_all = df_all.sort_values(["atm_id", "dt"]).reset_index(drop=True)
    df_all["had_repl_today"] = df_all["repl_replenished_kwd"].fillna(0) > 0
    df_all["last_repl_dt"] = (
        df_all["dt"]
        .where(df_all["had_repl_today"])
        .groupby(df_all["atm_id"])
        .ffill()
    )
    df_all["days_since_last_repl"] = (
        df_all["dt"] - df_all["last_repl_dt"]
    ).dt.days
    df_all["days_since_last_repl"] = df_all["days_since_last_repl"].fillna(999)
    df_all.drop(columns=["had_repl_today", "last_repl_dt"], inplace=True)

    # Lag & rolling features
    df_all = add_lag_features(df_all)
    df_all = add_rolling_features(df_all)

    return df_all


def get_feature_cols(df: pd.DataFrame):
    """Return feature_cols, cat_cols, num_cols given a training DataFrame."""
    drop_cols = [
        "withdrawn_kwd",
        "withdraw_count",
        "is_train",
        "dt",
        "atm_id",
        "installed_date",
        "decommissioned_date",
        "name",
        "holiday_name",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, cat_cols, num_cols


# ---------------------------------------------------------------------
# Baseline model (optional, for reporting)
# ---------------------------------------------------------------------
def make_baseline_predictions_const(
    train_df: pd.DataFrame,
    test_keys: pd.DataFrame,
    kind: str,
    window: int | None = None,
) -> pd.DataFrame:
    """Simple per-ATM baselines: 'last' or 'moving_average'."""
    train_sorted = train_df.sort_values("dt")
    gb = train_sorted.groupby("atm_id", group_keys=False)

    if kind == "last":
        stats = gb[["withdrawn_kwd", "withdraw_count"]].last()
    elif kind == "moving_average":
        if window is None:
            raise ValueError("window must be provided for moving_average")

        def agg_fn(g: pd.DataFrame) -> pd.Series:
            tail = g.sort_values("dt").tail(window)
            return tail[["withdrawn_kwd", "withdraw_count"]].mean()

        stats = gb.apply(agg_fn)
    else:
        raise ValueError(f"Unknown baseline kind: {kind}")

    stats = stats.rename(
        columns={
            "withdrawn_kwd": "predicted_withdrawn_kwd",
            "withdraw_count": "predicted_withdraw_count",
        }
    )

    preds = test_keys[["dt", "atm_id"]].merge(
        stats, left_on="atm_id", right_index=True, how="left"
    )

    if preds["predicted_withdrawn_kwd"].isna().any():
        global_means = train_df[["withdrawn_kwd", "withdraw_count"]].mean()
        preds["predicted_withdrawn_kwd"] = preds["predicted_withdrawn_kwd"].fillna(
            global_means["withdrawn_kwd"]
        )
        preds["predicted_withdraw_count"] = preds["predicted_withdraw_count"].fillna(
            global_means["withdraw_count"]
        )

    return preds


def evaluate_baselines(df_train_all: pd.DataFrame):
    """Evaluate naive + MA(7/28) on the last 14 days (for reporting only)."""
    max_dt = df_train_all["dt"].max()
    val_start = max_dt - pd.Timedelta(days=13)

    df_tr = df_train_all[df_train_all["dt"] < val_start].copy()
    df_val = df_train_all[df_train_all["dt"] >= val_start].copy()
    val_keys = df_val[["dt", "atm_id"]].drop_duplicates()

    baselines = {
        "naive_last": dict(kind="last", window=None),
        "ma_7": dict(kind="moving_average", window=7),
        "ma_28": dict(kind="moving_average", window=28),
    }

    results = []

    for name, cfg in baselines.items():
        print(f"\nEvaluating baseline: {name}")
        preds = make_baseline_predictions_const(
            df_tr, val_keys, kind=cfg["kind"], window=cfg["window"]
        )
        merged = df_val.merge(preds, on=["dt", "atm_id"], how="left")

        rmse_kwd = rmse(merged["withdrawn_kwd"], merged["predicted_withdrawn_kwd"])
        rmse_cnt = rmse(merged["withdraw_count"], merged["predicted_withdraw_count"])
        avg_rmse = (rmse_kwd + rmse_cnt) / 2.0

        print(f"  RMSE (withdrawn_kwd):   {rmse_kwd:,.2f}")
        print(f"  RMSE (withdraw_count): {rmse_cnt:,.2f}")
        print(f"  Average RMSE:           {avg_rmse:,.2f}")

        results.append((name, rmse_kwd, rmse_cnt, avg_rmse))

    results_df = pd.DataFrame(
        results, columns=["model", "rmse_kwd", "rmse_count", "rmse_avg"]
    )
    print("\nBaseline comparison:")
    print(results_df.to_string(index=False))


# ---------------------------------------------------------------------
# Main training: XGBoost models + save
# ---------------------------------------------------------------------
def main():
    print("Building full feature table for train + test...")
    df_all = build_feature_table_train_test(
        TRAIN_TX_PATH, TEST_TX_PATH,
        CAL_PATH, META_PATH, REGION_PATH, REPL_PATH,
    )

    # Separate back to train/test
    df_train_all = df_all[df_all["is_train"] == 1].copy()

    print("Train rows:", len(df_train_all))

    # --- Evaluate baselines on last 14 days ---
    evaluate_baselines(df_train_all)

    # --- Time-based split for ML model validation ---
    max_dt = df_train_all["dt"].max()
    val_start = max_dt - pd.Timedelta(days=13)

    df_tr = df_train_all[df_train_all["dt"] < val_start].copy()
    df_val = df_train_all[df_train_all["dt"] >= val_start].copy()

    print("\nML training:")
    print("Train date range:", df_tr["dt"].min().date(), "->", df_tr["dt"].max().date())
    print("Val date range:  ", df_val["dt"].min().date(), "->", df_val["dt"].max().date())
    print("#Train rows:", len(df_tr), "#Val rows:", len(df_val))

    # Feature columns
    feature_cols, cat_cols, num_cols = get_feature_cols(df_tr)
    print("Number of feature columns:", len(feature_cols))
    print("Categorical cols:", cat_cols)

    X_tr = df_tr[feature_cols]
    X_val = df_val[feature_cols]

    # Preprocessor
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # -------- Model 1: withdrawal amount (log1p) --------
    y_tr_amt  = np.log1p(df_tr["withdrawn_kwd"].values)
    y_val_amt = np.log1p(df_val["withdrawn_kwd"].values)

    model_amt = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("xgb", XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=4,
                tree_method="hist",
                random_state=42,
            )),
        ]
    )

    print("\nTraining XGB model for withdrawal amount...")
    model_amt.fit(X_tr, y_tr_amt)
    val_pred_log = model_amt.predict(X_val)
    val_pred_amt = np.expm1(val_pred_log)
    rmse_kwd_ml = rmse(df_val["withdrawn_kwd"], val_pred_amt)
    print("Validation RMSE (withdrawn_kwd) - ML model:", rmse_kwd_ml)

    # -------- Model 2: withdrawal count (log1p) --------
    y_tr_cnt  = np.log1p(df_tr["withdraw_count"].values)
    y_val_cnt = np.log1p(df_val["withdraw_count"].values)

    model_cnt = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("xgb", XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=4,
                tree_method="hist",
                random_state=42,
            )),
        ]
    )

    print("\nTraining XGB model for withdrawal count...")
    model_cnt.fit(X_tr, y_tr_cnt)
    val_pred_log_cnt = model_cnt.predict(X_val)
    val_pred_cnt = np.expm1(val_pred_log_cnt)
    rmse_cnt_ml = rmse(df_val["withdraw_count"], val_pred_cnt)
    print("Validation RMSE (withdraw_count) - ML model:", rmse_cnt_ml)

    avg_rmse_ml = (rmse_kwd_ml + rmse_cnt_ml) / 2.0
    print("Average RMSE - ML models:", avg_rmse_ml)

    # -------- Retrain on full training data and save models --------
    print("\nRetraining ML models on full training data and saving...")

    feature_cols_full, cat_cols_full, num_cols_full = get_feature_cols(df_train_all)
    X_full = df_train_all[feature_cols_full]

    preprocess_full = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_full),
            ("num", "passthrough", num_cols_full),
        ]
    )

    model_amt_full = Pipeline(
        steps=[
            ("preprocess", preprocess_full),
            ("xgb", XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=4,
                tree_method="hist",
                random_state=42,
            )),
        ]
    )

    model_cnt_full = Pipeline(
        steps=[
            ("preprocess", preprocess_full),
            ("xgb", XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=4,
                tree_method="hist",
                random_state=42,
            )),
        ]
    )

    y_full_amt = np.log1p(df_train_all["withdrawn_kwd"].values)
    y_full_cnt = np.log1p(df_train_all["withdraw_count"].values)

    print("  Fitting amount model on full data...")
    model_amt_full.fit(X_full, y_full_amt)

    print("  Fitting count model on full data...")
    model_cnt_full.fit(X_full, y_full_cnt)

    dump(model_amt_full, MODELS_DIR / "model_amt_xgb.pkl")
    dump(model_cnt_full, MODELS_DIR / "model_cnt_xgb.pkl")

    # Save feature column list for predict.py
    np.save(MODELS_DIR / "feature_cols.npy", np.array(feature_cols_full))

    print("\nModels saved in 'models/' directory.")


if __name__ == "__main__":
    main()