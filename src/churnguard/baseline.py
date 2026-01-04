from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

from churnguard.data import load_raw_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = load_raw_csv("telco_churn.csv")

    # ============================
    # TARGET COLUMN (CORRECT ONE)
    # ============================
    target_col = "Churn Value"  # already 0/1

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    y = df[target_col]

    # ============================
    # FEATURE SELECTION
    # ============================
    X = df.drop(columns=[target_col])

    # Drop identifiers + leakage columns
    leakage_cols = [
        "CustomerID",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
    ]
    X = X.drop(columns=[c for c in leakage_cols if c in X.columns])

    # Identify categorical vs numerical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # ============================
    # PREPROCESSING PIPELINE
    # ============================
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # ============================
    # MODEL
    # ============================
    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # ============================
    # TRAIN / TEST SPLIT
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ============================
    # TRAIN
    # ============================
    clf.fit(X_train, y_train)

    # ============================
    # EVALUATION
    # ============================
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"Baseline Logistic Regression | ROC-AUC: {roc:.3f} | F1: {f1:.3f}")

    # ============================
    # CONFUSION MATRIX
    # ============================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("ChurnGuard — Baseline Confusion Matrix")

    out_path = FIG_DIR / "baseline_confusion_matrix.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
