"""Server-side validation of the assessment form.

Browser ``min``/``max`` attributes are advisory only; every value is
re-validated here before it reaches the model. Two kinds of bounds exist:

* ``limits``   - hard, physiologically plausible bounds. Outside them the
                 input is rejected.
* ``training`` - the range seen in the Cleveland training data. Inside the
                 hard limits but outside this range the prediction is still
                 produced, but flagged as an extrapolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NumericField:
    name: str
    label: str
    limits: tuple[float, float]
    training: tuple[float, float]
    integer: bool = False


@dataclass(frozen=True)
class ChoiceField:
    name: str
    label: str
    choices: dict[str, int] = field(default_factory=dict)


FIELDS = [
    NumericField("age", "Age", (18, 110), (29, 77), integer=True),
    ChoiceField("sex", "Sex", {"1": 1, "0": 0}),
    NumericField("trestbps", "Resting blood pressure", (60, 260), (94, 200)),
    NumericField("chol", "Serum cholesterol", (80, 700), (126, 564)),
    ChoiceField("fbs", "Fasting blood sugar > 120 mg/dl", {"0": 0, "1": 1}),
    NumericField("thalach", "Max heart rate achieved", (40, 250), (71, 202)),
    ChoiceField("exang", "Exercise induced angina", {"0": 0, "1": 1}),
    NumericField("oldpeak", "ST depression", (0, 10), (0, 6.2)),
    NumericField("ca", "Major vessels coloured", (0, 3), (0, 3), integer=True),
    ChoiceField("cp", "Chest pain type",
                {"typical": 0, "atypical": 1, "non": 2, "asymptomatic": 3}),
    ChoiceField("restecg", "Resting ECG", {"normal": 0, "stt": 1, "lv": 2}),
    ChoiceField("slope", "ST segment slope", {"up": 0, "flat": 1, "down": 2}),
    ChoiceField("thal", "Thallium stress test", {"normal": 0, "fixed": 1, "reversible": 2}),
]


def validate(form) -> tuple[dict, dict[str, str], list[str]]:
    """Return ``(values, errors, warnings)`` for a submitted mapping."""
    values, errors, warnings = {}, {}, []
    for spec in FIELDS:
        raw = (form.get(spec.name) or "").strip()
        if not raw:
            errors[spec.name] = f"{spec.label} is required."
            continue
        if isinstance(spec, ChoiceField):
            if raw not in spec.choices:
                errors[spec.name] = f"{spec.label}: invalid option."
            else:
                values[spec.name] = spec.choices[raw]
            continue
        try:
            number = float(raw)
        except ValueError:
            errors[spec.name] = f"{spec.label} must be a number."
            continue
        low, high = spec.limits
        if not math.isfinite(number) or not low <= number <= high:
            errors[spec.name] = f"{spec.label} must be between {low:g} and {high:g}."
            continue
        if spec.integer and not number.is_integer():
            errors[spec.name] = f"{spec.label} must be a whole number."
            continue
        t_low, t_high = spec.training
        if not t_low <= number <= t_high:
            warnings.append(
                f"{spec.label} ({number:g}) is outside the training range "
                f"{t_low:g}–{t_high:g}; the estimate is an extrapolation."
            )
        values[spec.name] = number
    return values, errors, warnings
