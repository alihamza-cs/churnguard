from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Any

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from churnguard.data import load_raw_csv


LEAKAGE_COLS = ["CustomerID", "Churn Label", "Churn Score", "Churn Reason"]
TARGET_COL = "Churn Value"


@dataclass
class ModelResult:
    name: str
    roc_auc: float
    f1: float


def make_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    df = load_raw_csv("telco_churn.csv")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    X = X.drop(columns=[c for c in LEAKAGE_COLS if c in X.columns], errors="ignore")
    return X, y


def _make_ohe_dense() -> OneHotEncoder:
    """
    Create an OneHotEncoder that returns a dense matrix.
    sklearn changed the parameter name from 'sparse' to 'sparse_output'.
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    ohe = _make_ohe_dense()

    # Dense output avoids "Sparse data was passed" errors for boosting models.
    return ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )


def evaluate_model(clf: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    return roc, f1


def main() -> None:
    print("Running churnguard.models v2 (dense one-hot + boosting)")

    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = make_preprocessor(X_train)

    models: List[tuple[str, Any]] = [
        (
            "Logistic Regression (baseline)",
            LogisticRegression(
                max_iter=8000,
                solver="saga",
                n_jobs=-1,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                random_state=42,
            ),
        ),
    ]

    results: List[ModelResult] = []

    for name, model in models:
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)
        roc, f1 = evaluate_model(clf, X_test, y_test)
        results.append(ModelResult(name=name, roc_auc=roc, f1=f1))
        print(f"{name} | ROC-AUC: {roc:.3f} | F1: {f1:.3f}")

    print("\nREADME table:")
    print("| Model | ROC-AUC | F1 |")
    print("|---|---:|---:|")
    for r in results:
        print(f"| {r.name} | {r.roc_auc:.3f} | {r.f1:.3f} |")


if __name__ == "__main__":
    main()
