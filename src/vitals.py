"""
vitals.py
Heart rate and approximate SpO2 from raw MAX30102 samples (red + IR PPG).

Pure numpy, no hardware needed, so it is unit-tested on synthetic signals.

  * Finger present:  mean IR above FINGER_IR_MIN (no finger gives a very low IR).
  * Heart rate:      remove slow baseline, smooth, count peaks, median interval.
  * SpO2:            "ratio of ratios" R = (AC_red/DC_red)/(AC_ir/DC_ir),
                     SpO2 ~= 110 - 25*R (a standard empirical approximation).
                     Hobby-grade sensor, uncalibrated: treat as indicative only.
"""

from __future__ import annotations

import numpy as np

FS = 100                 # samples per second after FIFO averaging (see max30102.py)
FINGER_IR_MIN = 30000    # below this the sensor is looking at air
MIN_BEATS = 4            # need at least this many intervals for a heart rate
# HW-827 (analog pulse sensor through an ADS1115): a finger is "present" when the
# pulse swing is at least this many ADC counts (0.125 mV each). Tune with HW827_MIN_AC.
import os as _os
HW827_MIN_AC = float(_os.environ.get("HW827_MIN_AC", "40"))


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average. Edges are padded by repeating the end values,
    so the start and end of the window do not get pulled toward zero."""
    window = max(1, int(window))
    x = np.asarray(x, dtype=float)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(x, (left, right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def estimate_hr(ir: np.ndarray, fs: int = FS) -> tuple[float | None, float | None]:
    """Return (bpm, interval_variability) or (None, None) if no clear pulse.

    interval_variability = std/mean of beat intervals. Low (<0.15) means a
    steady pulse, high means noise or motion.
    """
    ir = np.asarray(ir, dtype=float)
    if ir.size < fs * 3:
        return None, None
    y = ir - _moving_average(ir, fs)          # drop baseline (1 s window)
    y = _moving_average(y, 5)                  # light smoothing
    thr = 0.5 * np.percentile(np.abs(y), 99)
    if thr <= 0:
        return None, None
    min_gap = int(0.33 * fs)                   # max ~180 bpm
    peaks = []
    last = -min_gap
    for i in range(1, y.size - 1):
        if y[i] > thr and y[i] >= y[i - 1] and y[i] > y[i + 1] and i - last >= min_gap:
            peaks.append(i)
            last = i
    if len(peaks) < MIN_BEATS:
        return None, None
    intervals = np.diff(peaks) / fs
    intervals = intervals[(intervals >= 0.33) & (intervals <= 2.0)]
    if intervals.size < MIN_BEATS - 1:
        return None, None
    bpm = 60.0 / float(np.median(intervals))
    if not (30 <= bpm <= 220):
        return None, None
    variability = float(np.std(intervals) / np.mean(intervals))
    return round(bpm, 1), round(variability, 3)


def _ac_dc(x: np.ndarray, fs: int) -> tuple[float, float]:
    dc = float(np.mean(x))
    ac = float(np.std(x - _moving_average(x, fs)))
    return ac, dc


def estimate_spo2(red: np.ndarray, ir: np.ndarray, fs: int = FS) -> float | None:
    red = np.asarray(red, dtype=float)
    ir = np.asarray(ir, dtype=float)
    if red.size < fs * 3:
        return None
    ac_r, dc_r = _ac_dc(red, fs)
    ac_i, dc_i = _ac_dc(ir, fs)
    if dc_r <= 0 or dc_i <= 0 or ac_i <= 0:
        return None
    R = (ac_r / dc_r) / (ac_i / dc_i)
    spo2 = 110.0 - 25.0 * R
    if not (70 <= spo2 <= 100):
        return None
    return round(spo2, 1)


def estimate_vitals_single(samples: np.ndarray, fs: int = FS) -> dict:
    """Heart rate from ONE waveform (HW-827 analog pulse sensor). SpO2 is not available."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < fs * 3:
        n = samples.shape[0] if samples.ndim == 2 else 0
        return {"finger": False, "hr": None, "spo2": None, "quality": "no data",
                "variability": None, "window_seconds": round(n / fs, 1)}
    x = samples[:, 0]
    window_s = round(x.size / fs, 1)
    ac = float(np.std(x - _moving_average(x, fs)))
    if ac < HW827_MIN_AC:
        return {"finger": False, "hr": None, "spo2": None, "quality": "no finger",
                "variability": None, "window_seconds": window_s}
    hr, var = estimate_hr(x, fs)
    if hr is None:
        quality = "no clear pulse"
    elif var is not None and var < 0.15:
        quality = "good"
    elif var is not None and var < 0.30:
        quality = "fair"
    else:
        quality = "poor"
    return {"finger": True, "hr": hr, "spo2": None, "quality": quality,
            "variability": var, "window_seconds": window_s}


def estimate_vitals(samples: np.ndarray, fs: int = FS) -> dict:
    """samples: array shape (N, 2) with columns [ir, red]."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < fs * 3:
        return {"finger": False, "hr": None, "spo2": None, "quality": "no data",
                "variability": None, "window_seconds": round(samples.shape[0] / fs, 1)}
    ir, red = samples[:, 0], samples[:, 1]
    window_s = round(samples.shape[0] / fs, 1)
    if float(np.mean(ir)) < FINGER_IR_MIN:
        return {"finger": False, "hr": None, "spo2": None, "quality": "no finger",
                "variability": None, "window_seconds": window_s}
    hr, var = estimate_hr(ir, fs)
    spo2 = estimate_spo2(red, ir, fs)
    if hr is None:
        quality = "no clear pulse"
    elif var is not None and var < 0.15:
        quality = "good"
    elif var is not None and var < 0.30:
        quality = "fair"
    else:
        quality = "poor"
    return {"finger": True, "hr": hr, "spo2": spo2, "quality": quality,
            "variability": var, "window_seconds": window_s}
