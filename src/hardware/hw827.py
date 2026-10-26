"""
hw827.py
HW-827 analog pulse sensor (PulseSensor-style green LED + photodiode) read
through an ADS1115 16-bit ADC, because the Raspberry Pi has NO analog inputs.

Wiring (everything on 3.3 V, never 5 V, so the signal can never exceed 3.3 V):
    HW-827  (+) / VCC  ->  Pi pin 1  (3.3 V)
    HW-827  (-) / GND  ->  Pi pin 6  (GND)
    HW-827  S  / signal ->  ADS1115 A0
    ADS1115 VDD -> Pi pin 1 (3.3 V)      ADS1115 GND  -> Pi pin 6 (GND)
    ADS1115 SDA -> Pi pin 3 (GPIO2)      ADS1115 SCL  -> Pi pin 5 (GPIO3)
    ADS1115 ADDR -> GND  (I2C address 0x48; it does not clash with the MAX30102 at 0x57)

The sensor gives ONE waveform (no red channel), so heart rate works but SpO2
is not available from it. Samples are produced at 100 per second to match vitals.FS.
"""

from __future__ import annotations

import os
import time

REG_CONVERSION = 0x00
REG_CONFIG = 0x01


class HW827ADS1115:
    single_channel = True          # tells the sensor service: one waveform, no SpO2

    def __init__(self, bus_no: int = 1, address: int | None = None, channel: int | None = None):
        from smbus2 import SMBus
        self.address = address if address is not None else int(os.environ.get("HW827_ADS_ADDR", "0x48"), 0)
        self.channel = channel if channel is not None else int(os.environ.get("HW827_CHANNEL", "0"))
        if not 0 <= self.channel <= 3:
            raise ValueError("HW827_CHANNEL must be 0 to 3")
        self.bus = SMBus(bus_no)
        # MUX = single-ended AINx, PGA = +/-4.096 V, continuous mode, 475 samples/s, comparator off.
        config = ((0b100 | self.channel) << 12) | 0x0200 | 0x00C0 | 0x0003
        try:
            self.bus.write_i2c_block_data(self.address, REG_CONFIG, [(config >> 8) & 0xFF, config & 0xFF])
            time.sleep(0.01)
            self._read_raw()                 # raises OSError if the ADS1115 is not there
        except OSError as exc:
            raise RuntimeError(f"ADS1115 not found at 0x{self.address:02X} on I2C bus {bus_no}: {exc}") from exc

    def _read_raw(self) -> int:
        d = self.bus.read_i2c_block_data(self.address, REG_CONVERSION, 2)
        v = (d[0] << 8) | d[1]
        return v - 65536 if v & 0x8000 else v

    def read_samples(self) -> list[tuple[int, int]]:
        """About 0.1 s of data: ten samples spaced 10 ms apart, as (value, value)."""
        out = []
        for _ in range(10):
            t0 = time.monotonic()
            v = self._read_raw()
            out.append((v, v))
            wait = 0.01 - (time.monotonic() - t0)
            if wait > 0:
                time.sleep(wait)
        return out
