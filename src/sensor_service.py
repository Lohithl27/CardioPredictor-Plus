"""
sensor_service.py
Keeps the MAX30102 running in the background, all day, while the server is up.

  * A thread reads the sensor continuously into a rolling buffer (2 minutes).
  * Every second it estimates heart rate / SpO2 from the last 10 seconds
    (this is what the live panel shows).
  * Every few seconds it can save a reading via a callback (the database).
  * Measurements (e.g. 20 s for a patient) are taken from the same buffer,
    so the live panel and the measurement never fight over the sensor.

If the sensor is not found, it falls back to a simulated pulse and says so
in every response (source = "simulated"), so nobody mistakes it for real data.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from collections import deque

import numpy as np

from vitals import FS, estimate_vitals, estimate_vitals_single

log = logging.getLogger("sensor")


class SimulatedSource:
    """Synthetic pulse at a slowly drifting heart rate, for demos and tests."""

    def __init__(self, base_hr: float = 72.0):
        self.base_hr = base_hr
        self.t = 0.0
        self.rng = random.Random(1)

    def read_samples(self):
        time.sleep(0.1)                      # 10 samples per call = 100 Hz
        out = []
        for _ in range(FS // 10):
            hr = self.base_hr + 3 * math.sin(self.t / 15)
            ph = 2 * math.pi * (hr / 60) * self.t
            ir = 90000 + 4000 * math.sin(ph) + self.rng.gauss(0, 150)
            red = 70000 + 1620 * math.sin(ph) + self.rng.gauss(0, 100)
            out.append((int(ir), int(red)))
            self.t += 1 / FS
        return out


class NoSensorSource:
    """No sensor at all (SENSOR_MODE=off). Produces no samples, so the app
    never shows a heart rate that did not come from a person or a sensor.
    Heart rate is then typed in by hand."""

    def read_samples(self):
        time.sleep(0.2)
        return []


class SensorService:
    BUFFER_SECONDS = 120
    LIVE_WINDOW_SECONDS = 10
    TRACE_LENGTH = 120                       # one live HR value per second

    def __init__(self, source, source_name: str, on_reading=None, store_every: float = 5.0):
        self.source = source
        self.source_name = source_name
        self.single_channel = bool(getattr(source, "single_channel", False))
        self.on_reading = on_reading
        self.store_every = store_every
        self.buffer: deque = deque(maxlen=self.BUFFER_SECONDS * FS)
        self.trace: deque = deque(maxlen=self.TRACE_LENGTH)
        self.lock = threading.Lock()
        self.live: dict = {"finger": False, "hr": None, "spo2": None,
                           "quality": "starting", "source": source_name,
                           "updated_at": None}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sensor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---- worker ----------------------------------------------------------
    def _run(self) -> None:
        last_live = 0.0
        last_store = 0.0
        while not self._stop.is_set():
            try:
                samples = self.source.read_samples()
            except Exception as exc:                      # I2C glitch: keep going
                self.error = str(exc)
                log.warning("sensor read failed: %s", exc)
                time.sleep(0.5)
                continue
            if not samples:
                time.sleep(0.02)          # FIFO empty: do not spin on the I2C bus
            with self.lock:
                self.buffer.extend(samples)
            now = time.monotonic()
            if now - last_live >= 1.0:
                last_live = now
                self._update_live()
                with self.lock:
                    self.trace.append(self.live.get("hr"))
                if (self.on_reading and self.live.get("hr")
                        and now - last_store >= self.store_every):
                    last_store = now
                    try:
                        self.on_reading(dict(self.live))
                    except Exception as exc:
                        log.warning("store failed: %s", exc)

    def _update_live(self) -> None:
        window = self.recent_samples(self.LIVE_WINDOW_SECONDS * FS)
        v = self.estimate(window) if window.shape[0] else {
            "finger": False, "hr": None, "spo2": None, "quality": "waiting"}
        v.update(source=self.source_name, updated_at=time.time())
        with self.lock:
            self.live = v

    def estimate(self, window: np.ndarray) -> dict:
        """Vitals for a window of samples, using the right method for this sensor."""
        return estimate_vitals_single(window) if self.single_channel else estimate_vitals(window)

    # ---- access ----------------------------------------------------------
    def recent_samples(self, n: int) -> np.ndarray:
        with self.lock:
            data = list(self.buffer)[-n:]
        return np.asarray(data, dtype=float) if data else np.empty((0, 2))

    def get_live(self) -> dict:
        with self.lock:
            out = dict(self.live)
            out["trace"] = list(self.trace)
        out["error"] = self.error
        return out


def make_sensor_service(on_reading=None) -> SensorService:
    """Pick the sensor from SENSOR_MODE: off | simulate | max30102 | hw827 | auto (default)."""
    mode = os.environ.get("SENSOR_MODE", "auto").lower()
    if mode == "off":
        log.info("SENSOR_MODE=off: no sensor, heart rate is entered manually")
        return SensorService(NoSensorSource(), "none", on_reading=None)
    has_i2c = os.path.exists("/dev/i2c-1")

    if mode in ("auto", "max30102") and has_i2c:
        try:
            from hardware.max30102 import MAX30102
            log.info("MAX30102 sensor found")
            return SensorService(MAX30102(), "max30102", on_reading=on_reading)
        except Exception as exc:
            log.warning("MAX30102 unavailable (%s)", exc)

    if mode in ("auto", "hw827") and has_i2c:
        try:
            from hardware.hw827 import HW827ADS1115
            log.info("HW-827 pulse sensor (ADS1115) found")
            return SensorService(HW827ADS1115(), "hw827", on_reading=on_reading)
        except Exception as exc:
            log.warning("HW-827 / ADS1115 unavailable (%s)", exc)
            if mode == "hw827":
                svc = SensorService(NoSensorSource(), "none", on_reading=None)
                svc.error = f"HW-827 not found: {exc}"
                return svc
    elif mode in ("max30102", "hw827"):
        log.warning("SENSOR_MODE=%s but /dev/i2c-1 is missing; no sensor", mode)
        svc = SensorService(NoSensorSource(), "none", on_reading=None)
        svc.error = "/dev/i2c-1 is missing: enable I2C (sudo raspi-config nonint do_i2c 0) and reboot"
        return svc

    return SensorService(SimulatedSource(), "simulated", on_reading=on_reading)
