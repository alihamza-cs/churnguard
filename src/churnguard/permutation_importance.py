from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from churnguard.models import make_dataset, make_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = make_preprocessor(X_train)
    model = GradientBoostingClassifier(random_state=42)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)

    result = permutation_importance(
        clf,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )

    importances = result.importances_mean
    idx = np.argsort(importances)[::-1][:15]

    feature_names = X_train.columns.to_numpy()

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(idx))[::-1], importances[idx])
    plt.yticks(range(len(idx))[::-1], feature_names[idx])
    plt.title("ChurnGuard — Top 15 Permutation Importances")
    plt.xlabel("Decrease in ROC-AUC")

    out_path = FIG_DIR / "permutation_importance_top15.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved permutation importance plot to: {out_path}")


if __name__ == "__main__":
    main()
