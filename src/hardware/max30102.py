"""
max30102.py
Register-level driver for the MAX30102 pulse oximeter on a Raspberry Pi
(I2C bus 1, address 0x57). Wiring: VIN->3V3, GND->GND, SDA->GPIO2 (pin 3),
SCL->GPIO3 (pin 5). Leave INT unconnected.

FIFO averaging of 4 with a 400 Hz ADC gives 100 samples/s out (matches vitals.FS).
"""

from __future__ import annotations

import time

ADDR = 0x57
PART_ID_REG = 0xFF
PART_ID_VALUE = 0x15

REG_FIFO_WR = 0x04
REG_FIFO_RD = 0x06
REG_FIFO_DATA = 0x07
REG_FIFO_CFG = 0x08
REG_MODE = 0x09
REG_SPO2_CFG = 0x0A
REG_LED_RED = 0x0C
REG_LED_IR = 0x0D


class MAX30102:
    def __init__(self, bus_no: int = 1, led_current: int = 0x3F):
        from smbus2 import SMBus  # imported here so simulated runs need no smbus
        self.bus = SMBus(bus_no)
        part = self._read(PART_ID_REG)
        if part != PART_ID_VALUE:
            raise RuntimeError(f"MAX30102 not found at 0x{ADDR:02X} (part id 0x{part:02X})")
        self._write(REG_MODE, 0x40)          # reset
        time.sleep(0.1)
        for reg in (REG_FIFO_WR, 0x05, REG_FIFO_RD):
            self._write(reg, 0x00)           # clear FIFO pointers
        self._write(REG_FIFO_CFG, 0x5F)      # average 4, rollover on
        self._write(REG_MODE, 0x03)          # SpO2 mode: red + IR
        self._write(REG_SPO2_CFG, 0x2F)      # ADC range 4096 nA, 400 Hz, 411 us pulse
        self._write(REG_LED_RED, led_current)
        self._write(REG_LED_IR, led_current)

    def _read(self, reg: int) -> int:
        return self.bus.read_byte_data(ADDR, reg)

    def _write(self, reg: int, value: int) -> None:
        self.bus.write_byte_data(ADDR, reg, value)

    def read_samples(self) -> list[tuple[int, int]]:
        """Return all new (ir, red) samples waiting in the FIFO."""
        wr = self._read(REG_FIFO_WR)
        rd = self._read(REG_FIFO_RD)
        count = (wr - rd) & 0x1F
        out = []
        for _ in range(count):
            d = self.bus.read_i2c_block_data(ADDR, REG_FIFO_DATA, 6)
            red = ((d[0] << 16) | (d[1] << 8) | d[2]) & 0x3FFFF
            ir = ((d[3] << 16) | (d[4] << 8) | d[5]) & 0x3FFFF
            out.append((ir, red))
        return out

    def close(self) -> None:
        self.bus.close()
