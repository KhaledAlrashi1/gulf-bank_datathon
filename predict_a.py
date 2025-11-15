# predict.py
import pandas as pd
from pathlib import Path

from train_a import load_and_clean_train, make_baseline_predictions_const


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "atm_transactions_train.csv"
TEST_PATH = DATA_DIR / "atm_transactions_test.csv"
OUTPUT_PATH = Path("predictions.csv")


def main():
    # Load full training data
    df_train = load_and_clean_train(TRAIN_PATH)

    # Load test data
    df_test = pd.read_csv(TEST_PATH)
    df_test["dt"] = pd.to_datetime(df_test["dt"])

    # Use the test rows as-is (preserve duplicates and order)
    test_like_df = df_test[["dt", "atm_id"]].copy()

    # Use best baseline from validation: 28-day moving average
    preds = make_baseline_predictions_const(
        train_df=df_train,
        test_like_df=test_like_df,
        kind="moving_average",
        window=28,
    )

    # preds already has same number of rows and same (dt, atm_id) order as test_like_df
    predictions = preds[["dt", "atm_id", "predicted_withdrawn_kwd", "predicted_withdraw_count"]].copy()

    # Sanity check: must match test set row-for-row
    assert len(predictions) == len(df_test), (
        f"Row count mismatch: predictions={len(predictions)}, test={len(df_test)}"
    )

    predictions.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()