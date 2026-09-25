"""Guideline pre-test probability (PTP) of obstructive coronary artery disease.

Source: 2019 ESC Guidelines for the diagnosis and management of chronic
coronary syndromes (Knuuti et al., Eur Heart J 2020;41:407-477), Table 5,
derived from pooled contemporary cohorts (Juarez-Orozco et al., 2019).

The table covers symptomatic patients aged 30 and over. It does not apply to
asymptomatic patients, and the dyspnoea column is not used because the intake
form does not record dyspnoea as the leading symptom.

NOTE: the 2024 ESC chronic coronary syndrome guideline replaced this table
with a risk-factor-weighted clinical likelihood model. Clinical governance
must confirm which one the deploying clinic follows. Table values must be
verified against the published source before clinical use (see
tests/test_scores.py, which pins them).
"""
from __future__ import annotations

from dataclasses import dataclass

# (lower age bound) -> {chest pain type: (male %, female %)}
ESC_2019_PTP = {
    30: {"typical": (3, 5), "atypical": (4, 3), "non": (1, 1)},
    40: {"typical": (22, 10), "atypical": (10, 6), "non": (3, 2)},
    50: {"typical": (32, 13), "atypical": (17, 6), "non": (11, 3)},
    60: {"typical": (44, 16), "atypical": (26, 11), "non": (22, 6)},
    70: {"typical": (52, 27), "atypical": (34, 19), "non": (24, 10)},
}

CP_CODES = {0: "typical", 1: "atypical", 2: "non", 3: "asymptomatic"}


@dataclass(frozen=True)
class PretestProbability:
    percent: float | None
    note: str


def esc_2019_ptp(age: float, sex: int, cp: int) -> PretestProbability:
    """``sex`` is 1 for male, 0 for female; ``cp`` uses the model's 0-3 coding."""
    symptom = CP_CODES[cp]
    if symptom == "asymptomatic":
        return PretestProbability(None, "Not applicable: the ESC table covers symptomatic patients only.")
    if age < 30:
        return PretestProbability(None, "Not applicable: the ESC table starts at age 30.")
    band = max(b for b in ESC_2019_PTP if b <= age)
    male, female = ESC_2019_PTP[band][symptom]
    percent = male if sex == 1 else female
    if percent < 5:
        note = "<5%: testing can usually be deferred (ESC 2019)."
    elif percent <= 15:
        note = "5-15%: consider testing based on clinical likelihood (ESC 2019)."
    else:
        note = ">15%: non-invasive testing is most beneficial (ESC 2019)."
    return PretestProbability(float(percent), note)


def discordant(ptp: float | None, ml_probability: float) -> bool:
    """Heuristic flag (not validated): the two estimates point opposite ways.

    The scores are not on the same scale - the ML model was trained on a
    1980s referral cohort with 46% prevalence, the ESC table on contemporary
    cohorts with far lower prevalence - so only gross disagreement is flagged:
    the guideline says testing can be deferred while the model is in its High
    band, or the guideline favours testing while the model is below 10%.
    """
    if ptp is None:
        return False
    return (ptp < 5 and ml_probability >= 0.65) or (ptp > 15 and ml_probability < 0.10)
