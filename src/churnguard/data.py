from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def load_raw_csv(filename: str) -> pd.DataFrame:
    """Load a raw CSV dataset stored in data/raw/."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}\n"
            f"Put your dataset CSV in data/raw/ and name it '{filename}'."
        )
    return pd.read_csv(path)
