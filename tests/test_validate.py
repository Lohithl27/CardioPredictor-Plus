"""
Tests for src/validate.py -- the plausibility-check layer that the
base CardioPredictor paper doesn't have.

Run: pytest tests/test_validate.py -v
"""

from validate import validate_reading


GOOD_READING = {
    "male": 1, "age": 45, "education": 2, "currentSmoker": 0,
    "cigsPerDay": 0, "BPMeds": 0, "prevalentStroke": 0,
    "prevalentHyp": 0, "diabetes": 0, "totChol": 210, "sysBP": 120,
    "diaBP": 80, "BMI": 24.5, "heartRate": 72, "glucose": 85,
}


def test_valid_reading_passes():
    result = validate_reading(GOOD_READING)
    assert result.is_valid
    assert result.errors == []


def test_missing_required_field_fails():
    bad = dict(GOOD_READING)
    del bad["sysBP"]
    result = validate_reading(bad)
    assert not result.is_valid
    assert "sysBP" in result.errors[0]


def test_implausible_value_rejected():
    bad = dict(GOOD_READING)
    bad["sysBP"] = 400  # way outside [70, 260]
    result = validate_reading(bad)
    assert not result.is_valid
    assert any("sysBP" in e for e in result.errors)


def test_edge_case_warns_but_does_not_reject():
    edge = dict(GOOD_READING)
    edge["age"] = 96  # within [18, 100] but near the edge
    result = validate_reading(edge)
    assert result.is_valid
    assert any("age" in w for w in result.warnings)


def test_invalid_binary_field_rejected():
    bad = dict(GOOD_READING)
    bad["currentSmoker"] = 2  # only 0 or 1 allowed
    result = validate_reading(bad)
    assert not result.is_valid


def test_invalid_education_value_rejected():
    bad = dict(GOOD_READING)
    bad["education"] = 9  # only 1-4 allowed
    result = validate_reading(bad)
    assert not result.is_valid


def test_spo2_in_plausible_range_accepted():
    good = dict(GOOD_READING)
    good["spo2"] = 97
    result = validate_reading(good)
    assert result.is_valid


def test_spo2_out_of_range_rejected():
    bad = dict(GOOD_READING)
    bad["spo2"] = 40  # below plausible [70, 100]
    result = validate_reading(bad)
    assert not result.is_valid


def test_spo2_none_is_fine():
    # spo2 is optional -- absence should never cause a failure.
    result = validate_reading(GOOD_READING)
    assert result.is_valid
