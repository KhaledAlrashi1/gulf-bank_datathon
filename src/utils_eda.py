import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)


def quick_overview(df: pd.DataFrame, target_col: str | None = None):
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    display(df.head())

    if target_col and target_col in df.columns:
        print(f"\nTarget '{target_col}' value counts:")
        print(df[target_col].value_counts(dropna=False))


def missing_report(df: pd.DataFrame, min_pct: float = 0.0) -> pd.DataFrame:
    missing = df.isnull().mean().sort_values(ascending=False) * 100
    missing = missing[missing >= min_pct].round(2)
    return missing.to_frame("missing_pct")


def split_num_cat(df: pd.DataFrame, exclude: list[str] | None = None):
    if exclude is None:
        exclude = []
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude).tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.difference(exclude).tolist()
    return num_cols, cat_cols


def plot_numeric_distributions(df: pd.DataFrame, num_cols: list[str], bins: int = 30):
    for col in num_cols:
        plt.figure(figsize=(4, 3))
        df[col].hist(bins=bins)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


def show_categorical_value_counts(df: pd.DataFrame, cat_cols: list[str], max_unique: int = 20):
    for col in cat_cols:
        nunique = df[col].nunique(dropna=False)
        if nunique <= max_unique:
            print(f"\nColumn: {col} (nunique={nunique})")
            print(df[col].value_counts(dropna=False))