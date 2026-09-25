import pytest

from cardiorisk.scores import ESC_2019_PTP, discordant, esc_2019_ptp

MALE, FEMALE = 1, 0
TYPICAL, ATYPICAL, NON_ANGINAL, ASYMPTOMATIC = 0, 1, 2, 3


def test_table_values_are_pinned():
    # Pinned so any edit is deliberate; verify against ESC 2019 Table 5 before clinical use.
    assert ESC_2019_PTP == {
        30: {"typical": (3, 5), "atypical": (4, 3), "non": (1, 1)},
        40: {"typical": (22, 10), "atypical": (10, 6), "non": (3, 2)},
        50: {"typical": (32, 13), "atypical": (17, 6), "non": (11, 3)},
        60: {"typical": (44, 16), "atypical": (26, 11), "non": (22, 6)},
        70: {"typical": (52, 27), "atypical": (34, 19), "non": (24, 10)},
    }


@pytest.mark.parametrize("age,sex,cp,expected", [
    (54, MALE, ATYPICAL, 17), (54, FEMALE, ATYPICAL, 6), (49, MALE, TYPICAL, 22),
    (50, MALE, TYPICAL, 32), (85, FEMALE, NON_ANGINAL, 10), (30, FEMALE, TYPICAL, 5),
])
def test_lookup(age, sex, cp, expected):
    assert esc_2019_ptp(age, sex, cp).percent == expected


def test_not_applicable_cases():
    assert esc_2019_ptp(54, MALE, ASYMPTOMATIC).percent is None
    assert esc_2019_ptp(25, MALE, TYPICAL).percent is None


@pytest.mark.parametrize("pct,band_text", [(3, "<5%"), (10, "5-15%"), (15, "5-15%"), (16, ">15%")])
def test_guideline_bands(pct, band_text, monkeypatch):
    monkeypatch.setitem(ESC_2019_PTP, 40, {"typical": (pct, pct), "atypical": (1, 1), "non": (1, 1)})
    assert esc_2019_ptp(45, MALE, TYPICAL).note.startswith(band_text)


def test_discordance_flag():
    assert discordant(3, 0.80) and discordant(30, 0.05)
    # Similar numbers on different scales are not "opposite directions".
    assert not discordant(27, 0.28) and not discordant(3, 0.40) and not discordant(10, 0.02)
    assert not discordant(None, 0.99)
