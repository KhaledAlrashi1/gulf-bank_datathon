# predict.py
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

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

MODELS_DIR = Path("models")
OUTPUT_PATH = Path("predictions.csv")


def main():
    # Build the same feature table for train + test
    print("Rebuilding feature table for prediction...")
    df_all = build_feature_table_train_test(
        TRAIN_TX_PATH, TEST_TX_PATH,
        CAL_PATH, META_PATH, REGION_PATH, REPL_PATH,
    )

    df_train_all = df_all[df_all["is_train"] == 1].copy()
    df_test_all  = df_all[df_all["is_train"] == 0].copy()

    # Load feature columns used during training
    feature_cols = np.load(MODELS_DIR / "feature_cols.npy", allow_pickle=True).tolist()

    # Safety check: recompute feature_cols from train and compare
    feature_cols_train, _, _ = get_feature_cols(df_train_all)
    assert set(feature_cols) == set(feature_cols_train), "Feature column mismatch!"

    X_test = df_test_all[feature_cols]

    # Load models
    print("Loading trained models...")
    model_amt = load(MODELS_DIR / "model_amt_xgb.pkl")
    model_cnt = load(MODELS_DIR / "model_cnt_xgb.pkl")

    # Predict on test (remember models were trained on log1p targets)
    print("Predicting on test set...")
    pred_amt_log = model_amt.predict(X_test)
    pred_cnt_log = model_cnt.predict(X_test)

    pred_amt = np.expm1(pred_amt_log)
    pred_cnt = np.expm1(pred_cnt_log)

    # Build predictions DataFrame matching test rows exactly
    preds = pd.DataFrame({
        "dt": df_test_all["dt"].values,
        "atm_id": df_test_all["atm_id"].values,
        "predicted_withdrawn_kwd": pred_amt,
        "predicted_withdraw_count": pred_cnt,
    })

    # Sort for stability (optional)
    preds = preds.sort_values(["dt", "atm_id"]).reset_index(drop=True)

    preds.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}. "
          f"Rows: {len(preds)}")


if __name__ == "__main__":
    main()