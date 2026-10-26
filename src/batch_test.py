"""
batch_test.py
Sanity-checks the trained model against many varied, realistic
patient profiles -- not just the one hand-picked example used in
manual API testing so far.

Approach: rather than inventing synthetic patients from scratch (which
risks generating physiologically inconsistent combinations), this
script draws real rows from the held-out test set and perturbs them
slightly (small random jitter within plausible ranges), so every
profile is grounded in a real patient but the exact combination
hasn't been seen by the model. This is a sanity check, not a formal
robustness/adversarial test suite.

Checks performed:
  1. No crashes across N varied profiles.
  2. Every probability is a valid float in [0, 1].
  3. No NaN/inf outputs.
  4. Risk category distribution isn't degenerate (e.g. everything
     landing in one bucket would suggest a bug, not a working model).
  5. Every profile passes validate.py's plausibility checks (since
     they're derived from real, plausible data).

Usage:
    python src/batch_test.py
    python src/batch_test.py --n 500
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from preprocess import get_train_test_split, FEATURE_COLUMNS, PLAUSIBLE_RANGES
from validate import validate_reading, BINARY_FIELDS, EDUCATION_VALUES

MODELS_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "models"


def make_jittered_profiles(X_test: pd.DataFrame, n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Sample rows with replacement if n exceeds the test set size.
    base = X_test.sample(n=n, replace=(n > len(X_test)), random_state=seed).reset_index(drop=True)

    jittered = base.copy()
    numeric_jitter_pct = {
        "age": 0.0,  # keep age exact -- jittering it risks pushing outside [18,100] a lot
        "totChol": 0.05, "sysBP": 0.05, "diaBP": 0.05,
        "BMI": 0.05, "heartRate": 0.08, "glucose": 0.08, "cigsPerDay": 0.10,
    }
    for col, pct in numeric_jitter_pct.items():
        if pct == 0.0 or col not in jittered.columns:
            continue
        noise = rng.normal(loc=0, scale=jittered[col] * pct)
        jittered[col] = jittered[col] + noise
        lo, hi = PLAUSIBLE_RANGES.get(col, (-np.inf, np.inf))
        jittered[col] = jittered[col].clip(lo, hi)
        if col == "cigsPerDay":
            jittered[col] = jittered[col].clip(lower=0).round().astype(int)

    return jittered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="number of profiles to test")
    args = parser.parse_args()

    model = joblib.load(MODELS_DIR / "best_model.joblib")

    _, X_test, _, _ = get_train_test_split()
    profiles = make_jittered_profiles(X_test, args.n)

    failures = []
    probabilities = []
    risk_categories = []

    for i, row in profiles.iterrows():
        record = row.to_dict()

        # Validate exactly like the API would.
        result = validate_reading(record)
        if not result.is_valid:
            failures.append((i, "validation_failed", result.errors))
            continue

        try:
            row_df = pd.DataFrame([{c: record[c] for c in FEATURE_COLUMNS}])
            proba = float(model.predict_proba(row_df)[0, 1])
        except Exception as exc:  # noqa: BLE001 -- intentionally broad for a smoke test
            failures.append((i, "prediction_crashed", str(exc)))
            continue

        if not (0.0 <= proba <= 1.0) or np.isnan(proba) or np.isinf(proba):
            failures.append((i, "invalid_probability", proba))
            continue

        probabilities.append(proba)
        if proba < 0.10:
            risk_categories.append("Low")
        elif proba < 0.20:
            risk_categories.append("Moderate")
        else:
            risk_categories.append("High")

    print(f"Tested {len(profiles)} profiles.")
    print(f"Failures: {len(failures)}")
    for f in failures[:10]:
        print(f"  {f}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")

    if probabilities:
        probs = np.array(probabilities)
        print(f"\nProbability stats: min={probs.min():.4f} max={probs.max():.4f} "
              f"mean={probs.mean():.4f} std={probs.std():.4f}")

        from collections import Counter
        counts = Counter(risk_categories)
        total = len(risk_categories)
        print("\nRisk category distribution:")
        for cat in ["Low", "Moderate", "High"]:
            n = counts.get(cat, 0)
            print(f"  {cat:9s}: {n:4d} ({100*n/total:.1f}%)")

        # Flag if the model is degenerate (everything in one bucket).
        dominant_pct = max(counts.values()) / total
        if dominant_pct > 0.95:
            print(
                f"\nWARNING: {dominant_pct*100:.1f}% of profiles landed in a single "
                f"risk category. This may indicate a bug (e.g. a frozen model or "
                f"a broken feature) rather than genuine risk distribution."
            )

    if failures:
        print(f"\nFAILED: {len(failures)}/{len(profiles)} profiles had issues.")
        sys.exit(1)
    else:
        print(f"\nPASSED: all {len(profiles)} profiles produced valid predictions.")


if __name__ == "__main__":
    main()
