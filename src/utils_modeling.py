import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def make_train_val(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratify if binary classification
    stratify = y if y.nunique() == 2 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_val, y_train, y_val


def evaluate_classification(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    pos_label=1,
) -> Dict[str, Any]:
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    metrics = {}

    acc = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val, pos_label=pos_label)
    metrics["accuracy"] = acc
    metrics["f1"] = f1

    if hasattr(model, "predict_proba"):
        y_proba_val = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba_val)
        metrics["auc"] = auc
    else:
        metrics["auc"] = None

    print("Validation Accuracy:", round(acc, 4))
    print("Validation F1:", round(f1, 4))
    if metrics["auc"] is not None:
        print("Validation AUC:", round(metrics["auc"], 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred_val))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val))

    return metrics