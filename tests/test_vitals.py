"""Heart rate and SpO2 estimation on synthetic PPG signals."""
import numpy as np
import pytest

from vitals import estimate_vitals, estimate_hr, estimate_spo2


def _signal(hr_bpm, seconds=20, fs=100, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fs)
    f = hr_bpm / 60
    ir = 90000 + 4000 * np.sin(2 * np.pi * f * t) + rng.normal(0, 150, t.size)
    red = 70000 + 1620 * np.sin(2 * np.pi * f * t) + rng.normal(0, 100, t.size)
    return np.column_stack([ir, red])


@pytest.mark.parametrize("bpm", [60, 72, 95])
def test_heart_rate_recovered_within_2_bpm(bpm):
    v = estimate_vitals(_signal(bpm))
    assert v["finger"] is True
    assert abs(v["hr"] - bpm) <= 2
    assert v["quality"] == "good"


def test_spo2_in_plausible_range_for_synthetic_ratio():
    v = estimate_vitals(_signal(72))
    assert 95 <= v["spo2"] <= 99


def test_no_finger_is_detected():
    flat = np.column_stack([np.full(2000, 5000.0), np.full(2000, 4000.0)])
    v = estimate_vitals(flat)
    assert v["finger"] is False and v["hr"] is None


def test_short_window_returns_no_data():
    v = estimate_vitals(_signal(72, seconds=1))
    assert v["quality"] == "no data"


def test_flat_pulse_has_no_heart_rate():
    fs = 100
    ir = np.full(20 * fs, 90000.0)
    hr, var = estimate_hr(ir, fs)
    assert hr is None


def test_hw827_single_channel_heart_rate_and_no_finger():
    import numpy as np
    from vitals import estimate_vitals_single
    t = np.arange(0, 12, 0.01)
    for bpm in (62, 78, 100):
        wave = 13200 + 260 * np.sin(2 * np.pi * bpm / 60 * t) + np.random.default_rng(1).normal(0, 6, t.size)
        v = estimate_vitals_single(np.column_stack([wave, wave]))
        assert v["finger"] and abs(v["hr"] - bpm) <= 3
        assert v["spo2"] is None                       # one wavelength: no SpO2
    flat = 13200 + np.random.default_rng(2).normal(0, 4, t.size)
    assert estimate_vitals_single(np.column_stack([flat, flat]))["finger"] is False
