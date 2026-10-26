"""
storage.py
Minimal SQLite-backed store for prediction history, so the API can
show a risk trend over time per user/patient -- something the base
CardioPredictor paper does not do (it shows one prediction per visit,
with no history).

This is intentionally simple (a single table, no ORM) so it is easy
to read, defend in a viva, and swap for PostgreSQL later if needed --
that migration is already on the original project roadmap.
"""

from typing import Optional
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import os

DB_PATH = Path(os.environ.get("CARDIO_DATA_DIR", Path(__file__).resolve().parent.parent / "data")) / "predictions.db"


def _get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            input_data TEXT NOT NULL,
            calibrated_probability REAL NOT NULL,
            risk_category TEXT NOT NULL,
            validation_warnings TEXT
        )
        """
    )
    # Raw readings posted by the ESP8266 (one row per POST).
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS device_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            patient_id TEXT,
            timestamp TEXT NOT NULL,
            heart_rate REAL,
            spo2 REAL,
            signal_ok INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    return conn


def save_device_reading(device_id: str, patient_id: Optional[str], heart_rate: Optional[float],
                        spo2: Optional[float], signal_ok: bool) -> dict:
    """Store one sensor reading and return it with its server timestamp."""
    conn = _get_connection()
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO device_readings
            (device_id, patient_id, timestamp, heart_rate, spo2, signal_ok)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (device_id, patient_id, timestamp, heart_rate, spo2, 1 if signal_ok else 0),
    )
    conn.commit()
    conn.close()
    return {"device_id": device_id, "patient_id": patient_id, "timestamp": timestamp,
            "heart_rate": heart_rate, "spo2": spo2, "signal_ok": signal_ok}


def get_latest_device_reading(device_id: str) -> Optional[dict]:
    conn = _get_connection()
    row = conn.execute(
        """
        SELECT device_id, patient_id, timestamp, heart_rate, spo2, signal_ok
        FROM device_readings
        WHERE device_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (device_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {"device_id": row[0], "patient_id": row[1], "timestamp": row[2],
            "heart_rate": row[3], "spo2": row[4], "signal_ok": bool(row[5])}


def get_recent_device_readings(device_id: str, limit: int = 60) -> list[dict]:
    """Last `limit` readings for one device, oldest -> newest (for the live chart)."""
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT timestamp, heart_rate, spo2, signal_ok
        FROM device_readings
        WHERE device_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (device_id, limit),
    ).fetchall()
    conn.close()
    return [
        {"timestamp": r[0], "heart_rate": r[1], "spo2": r[2], "signal_ok": bool(r[3])}
        for r in reversed(rows)
    ]


def save_prediction(
    patient_id: str,
    input_data: dict,
    calibrated_probability: float,
    risk_category: str,
    validation_warnings: list,
) -> int:
    conn = _get_connection()
    cursor = conn.execute(
        """
        INSERT INTO predictions
            (patient_id, timestamp, input_data, calibrated_probability,
             risk_category, validation_warnings)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(input_data),
            calibrated_probability,
            risk_category,
            json.dumps(validation_warnings),
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_history(patient_id: str, limit: int = 50) -> list:
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT timestamp, calibrated_probability, risk_category
        FROM predictions
        WHERE patient_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (patient_id, limit),
    ).fetchall()
    conn.close()
    return [
        {"timestamp": r[0], "calibrated_probability": r[1], "risk_category": r[2]}
        for r in rows
    ]
