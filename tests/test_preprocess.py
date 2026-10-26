"""
Tests for src/preprocess.py -- cleaning, imputation, and the
train/test split used to train the model.

Run: pytest tests/test_preprocess.py -v
"""

import pandas as pd

from preprocess import clean, get_train_test_split, FEATURE_COLUMNS, TARGET_COLUMN


def test_clean_imputes_missing_numeric_values():
    df = pd.DataFrame({
        "age": [40, None, 60],
        "cigsPerDay": [0, 5, None],
        "totChol": [200, None, 220],
        "sysBP": [120, 130, None],
        "diaBP": [80, None, 85],
        "BMI": [24.0, None, 26.0],
        "heartRate": [70, None, 75],
        "glucose": [80, None, 90],
        "education": [2, None, 3],
        "BPMeds": [0, None, 1],
    })
    cleaned = clean(df)
    assert cleaned.isna().sum().sum() == 0


def test_clean_does_not_mutate_input():
    df = pd.DataFrame({
        "age": [40, 200],  # 200 is implausible
        "cigsPerDay": [0, 5], "totChol": [200, 210], "sysBP": [120, 130],
        "diaBP": [80, 85], "BMI": [24.0, 25.0], "heartRate": [70, 75],
        "glucose": [80, 90], "education": [2, 3], "BPMeds": [0, 1],
    })
    original_age = df["age"].copy()
    clean(df)
    # clean() should return a new frame, not modify the caller's copy
    pd.testing.assert_series_equal(df["age"], original_age)


def test_clean_treats_implausible_values_as_missing():
    df = pd.DataFrame({
        "age": [40, 200],  # 200 is way outside [18, 100]
        "cigsPerDay": [0, 5], "totChol": [200, 210], "sysBP": [120, 130],
        "diaBP": [80, 85], "BMI": [24.0, 25.0], "heartRate": [70, 75],
        "glucose": [80, 90], "education": [2, 3], "BPMeds": [0, 1],
    })
    cleaned = clean(df)
    # The implausible 200 should have been replaced by the median (40),
    # not left as-is or dropped.
    assert cleaned["age"].iloc[1] == 40


def test_train_test_split_shapes_match():
    X_train, X_test, y_train, y_test = get_train_test_split(test_size=0.2)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert list(X_train.columns) == FEATURE_COLUMNS
    # Roughly an 80/20 split on ~4238 rows.
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.02


def test_train_test_split_has_no_missing_values():
    X_train, X_test, y_train, y_test = get_train_test_split()
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_train_test_split_is_stratified():
    X_train, X_test, y_train, y_test = get_train_test_split()
    # Positive rate should be similar between train and test (stratified).
    assert abs(y_train.mean() - y_test.mean()) < 0.03
