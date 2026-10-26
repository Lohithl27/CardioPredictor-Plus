"""
validate.py
Plausibility checks for incoming readings, whether typed in manually
or (later) sent by hardware. The base CardioPredictor paper has no
such safeguard -- it will predict on any number it is given, sensor
noise and all. This module gives every prediction a validation report
before it reaches the model.
"""

from dataclasses import dataclass, field

from preprocess import PLAUSIBLE_RANGES

# Fields required for every prediction request.
REQUIRED_FIELDS = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]

# Binary/categorical fields and their allowed values.
BINARY_FIELDS = {
    "male": {0, 1},
    "currentSmoker": {0, 1},
    "BPMeds": {0, 1},
    "prevalentStroke": {0, 1},
    "prevalentHyp": {0, 1},
    "diabetes": {0, 1},
}
EDUCATION_VALUES = {1, 2, 3, 4}

# Optional field added as a modification beyond the base paper. Not
# part of the trained Framingham model (Framingham has no SpO2 column)
# -- it is accepted, validated, and stored for trend tracking / future
# model versions, but does not currently influence the risk score.
SPO2_RANGE = (70, 100)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def validate_reading(data: dict) -> ValidationResult:
    errors = []
    warnings = []

    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] is None]
    if missing:
        errors.append(f"Missing required fields: {missing}")
        return ValidationResult(is_valid=False, errors=errors)

    for field_name, allowed in BINARY_FIELDS.items():
        if data[field_name] not in allowed:
            errors.append(f"{field_name} must be one of {allowed}, got {data[field_name]}")

    if data["education"] not in EDUCATION_VALUES:
        errors.append(f"education must be one of {EDUCATION_VALUES}, got {data['education']}")

    for field_name, (lo, hi) in PLAUSIBLE_RANGES.items():
        if field_name not in data:
            continue
        value = data[field_name]
        if value < lo or value > hi:
            errors.append(
                f"{field_name}={value} is outside the plausible range [{lo}, {hi}]"
            )
        elif value < lo * 1.1 or value > hi * 0.9:
            # Inside range but near the edge -- flag, don't block.
            warnings.append(
                f"{field_name}={value} is near the edge of the plausible range "
                f"[{lo}, {hi}] -- double-check the reading"
            )

    spo2 = data.get("spo2")
    if spo2 is not None:
        lo, hi = SPO2_RANGE
        if spo2 < lo or spo2 > hi:
            errors.append(f"spo2={spo2} is outside the plausible range [{lo}, {hi}]")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


if __name__ == "__main__":
    sample_good = {
        "male": 1, "age": 45, "education": 2, "currentSmoker": 0,
        "cigsPerDay": 0, "BPMeds": 0, "prevalentStroke": 0,
        "prevalentHyp": 0, "diabetes": 0, "totChol": 210, "sysBP": 120,
        "diaBP": 80, "BMI": 24.5, "heartRate": 72, "glucose": 85,
    }
    sample_bad = dict(sample_good)
    sample_bad["sysBP"] = 400  # implausible

    print("Good reading:", validate_reading(sample_good))
    print("Bad reading:", validate_reading(sample_bad))
