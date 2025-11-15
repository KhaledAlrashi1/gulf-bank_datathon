# train.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "atm_transactions_train.csv"


def load_and_clean_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {
        "dt": "dt",
        "atm_id": "atm_id",
        "total_withdrawn_amount_kwd": "withdrawn_kwd",
        "total_withdraw_txn_count": "withdraw_count",
    }
    df = df.rename(columns=rename_map)

    required = ["dt", "atm_id", "withdrawn_kwd", "withdraw_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Available: {df.columns.tolist()}")

    df["dt"] = pd.to_datetime(df["dt"])

    df = (
        df[required]
        .drop_duplicates()
        .groupby(["atm_id", "dt"], as_index=False)[["withdrawn_kwd", "withdraw_count"]]
        .sum()
    )
    return df


def make_baseline_predictions_const(
    train_df: pd.DataFrame,
    test_like_df: pd.DataFrame,
    kind: str,
    window: Optional[int] = None,
) -> pd.DataFrame:
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

    preds = test_like_df[["dt", "atm_id"]].merge(
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


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    df = load_and_clean_train(TRAIN_PATH)

    max_dt = df["dt"].max()
    val_start = max_dt - pd.Timedelta(days=13)

    df_train_calib = df[df["dt"] < val_start].copy()
    df_val = df[df["dt"] >= val_start].copy()
    val_keys = df_val[["dt", "atm_id"]].drop_duplicates()

    baselines = {
        "naive_last": dict(kind="last", window=None),
        "ma_7": dict(kind="moving_average", window=7),
        "ma_14": dict(kind="moving_average", window=14),
        "ma_28": dict(kind="moving_average", window=28),
    }

    results = []
    for name, cfg in baselines.items():
        print(f"\nEvaluating baseline: {name}")
        preds = make_baseline_predictions_const(
            df_train_calib, val_keys, kind=cfg["kind"], window=cfg["window"]
        )
        merged = df_val.merge(preds, on=["dt", "atm_id"], how="left")

        rmse_kwd = rmse(merged["withdrawn_kwd"], merged["predicted_withdrawn_kwd"])
        rmse_cnt = rmse(merged["withdraw_count"], merged["predicted_withdraw_count"])
        avg_rmse = (rmse_kwd + rmse_cnt) / 2.0

        print(f"  RMSE (withdrawn_kwd):   {rmse_kwd:,.2f}")
        print(f"  RMSE (withdraw_count): {rmse_cnt:,.2f}")
        print(f"  Average RMSE:           {avg_rmse:,.2f}")

        results.append((name, rmse_kwd, rmse_cnt, avg_rmse))

    results_df = pd.DataFrame(results, columns=["model", "rmse_kwd", "rmse_count", "rmse_avg"])
    print("\nBaseline comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()