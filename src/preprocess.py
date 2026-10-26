"""
preprocess.py
Loads the Framingham dataset, cleans it, and produces train/test splits.

Modification vs. the base CardioPredictor paper:
  The paper used only 4 sensor-derived features (glucose, cholesterol,
  systolic BP, diastolic BP) plus a handful of demographics. This
  pipeline uses the full, justified Framingham feature set (expanded
  risk factors), since more of the dataset's real signal is available
  and each feature's contribution is checked via feature importance
  in train.py rather than assumed.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "framingham.csv"

# All feature columns used for training (target is TenYearCHD)
FEATURE_COLUMNS = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]
TARGET_COLUMN = "TenYearCHD"

# Physiologically plausible ranges, used both for cleaning outliers at
# training time and for input validation at prediction time (see
# src/validate.py). Values outside these ranges are treated as missing
# rather than silently trusted -- the base paper's system had no such
# safeguard.
PLAUSIBLE_RANGES = {
    "age": (18, 100),
    "totChol": (100, 700),
    "sysBP": (70, 260),
    "diaBP": (40, 160),
    "BMI": (12, 70),
    "heartRate": (30, 220),
    "glucose": (40, 500),
    "cigsPerDay": (0, 80),
}


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. 'NA' strings are parsed as missing values."""
    df = pd.read_csv(path, na_values=["NA"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and clip implausible values to NaN.

    Returns a cleaned copy; does not mutate the input.
    """
    df = df.copy()

    # Clip out-of-range values to NaN so they get imputed like any
    # other missing value, instead of corrupting training.
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            df.loc[mask, col] = pd.NA

    # Median imputation for numeric columns, mode for the few
    # ordinal/categorical ones. This is a simple, transparent choice
    # documented here rather than hidden inside a black-box pipeline
    # step, so it is easy to defend in a viva.
    numeric_cols = [
        "age", "cigsPerDay", "totChol", "sysBP", "diaBP",
        "BMI", "heartRate", "glucose",
    ]
    categorical_cols = ["education", "BPMeds"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def get_train_test_split(test_size: float = 0.2, random_state: int = 42):
    """Return X_train, X_test, y_train, y_test using a stratified split."""
    df = load_raw()
    df = clean(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Train CHD rate: {y_train.mean():.3f}, Test CHD rate: {y_test.mean():.3f}")
    print(f"Missing values remaining in train set:\n{X_train.isna().sum().sum()}")
