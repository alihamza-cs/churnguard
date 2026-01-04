from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from churnguard.models import make_dataset, make_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "churnguard_gradient_boosting.joblib"


def train_and_save() -> None:
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = make_preprocessor(X_train)
    model = GradientBoostingClassifier(random_state=42)

    clf: Pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    clf.fit(X_train, y_train)

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


def predict_from_csv(input_csv: str, output_csv: str = "predictions.csv") -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m churnguard.predict --train"
        )

    clf: Pipeline = joblib.load(MODEL_PATH)

    df = pd.read_csv(input_csv)
    probs = clf.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    out = df.copy()
    out["churn_probability"] = probs
    out["churn_prediction"] = preds

    out.to_csv(output_csv, index=False)
    print(f"Wrote predictions to: {output_csv}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train and run churn predictions.")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--input", type=str, help="Path to input CSV to score")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to output CSV")
    args = parser.parse_args()

    if args.train:
        train_and_save()
        return

    if args.input:
        predict_from_csv(args.input, args.output)
        return

    print("Nothing to do. Use --train or --input path/to/file.csv")


if __name__ == "__main__":
    main()
