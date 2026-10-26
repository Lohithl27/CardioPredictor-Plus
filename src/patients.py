"""
patients.py
One folder per patient, so old data can be found by patient ID:

    data/patients/<patient_id>/
        patient.json          name, age, sex, notes, created_at, share link
        photos/               photo_first.jpg (taken at registration) + later photos
        measurements/         one JSON per finger measurement (time-stamped)
        assessments/          one JSON per risk assessment (inputs + result)
        reports/              generated PNG / TXT / DOCX reports

SQLite (data/predictions.db) still holds the trend history used by the
dashboard; these folders are the readable copy you can open by hand.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CARDIO_DATA_DIR", ROOT_DIR / "data"))
PATIENTS_DIR = DATA_DIR / "patients"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,40}$")
SEX_VALUES = {"male", "female", "other"}


class PatientError(ValueError):
    """Bad input or a conflict (invalid ID, duplicate, missing data)."""


class PatientNotFound(LookupError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def validate_id(patient_id: str) -> str:
    if not ID_RE.match(patient_id or ""):
        raise PatientError("Patient ID must be 3-40 characters: letters, digits, '-' or '_'.")
    return patient_id


def folder(patient_id: str) -> Path:
    return PATIENTS_DIR / validate_id(patient_id)


def exists(patient_id: str) -> bool:
    try:
        return (folder(patient_id) / "patient.json").exists()
    except PatientError:
        return False


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create(patient_id: str, name: str, age: float | None, sex: str | None,
           notes: str = "") -> dict:
    validate_id(patient_id)
    if not name or not name.strip():
        raise PatientError("Patient name is required.")
    if sex is not None and sex not in SEX_VALUES:
        raise PatientError("Sex must be male, female or other.")
    if exists(patient_id):
        raise PatientError(f"Patient '{patient_id}' is already registered.")
    base = folder(patient_id)
    for sub in ("photos", "measurements", "assessments", "reports"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    record = {"patient_id": patient_id, "name": name.strip(), "age": age, "sex": sex,
              "notes": (notes or "").strip(), "created_at": _now(),
              "first_photo": None, "share": None}
    _write_json(base / "patient.json", record)
    return record


def update(patient_id: str, name: str | None = None, age: float | None = None,
           sex: str | None = None, notes: str | None = None) -> dict:
    """Edit the details of a registered patient (the ID and folder never change)."""
    rec = load(patient_id)
    if name is not None:
        if not name.strip():
            raise PatientError("Patient name is required.")
        rec["name"] = name.strip()
    if age is not None:
        rec["age"] = age
    if sex is not None:
        if sex and sex not in SEX_VALUES:
            raise PatientError("Sex must be male, female or other.")
        rec["sex"] = sex or None
    if notes is not None:
        rec["notes"] = notes.strip()
    rec["updated_at"] = _now()
    save(rec)
    return rec


def check_writable() -> str | None:
    """Return None if the data folder can be written, else a readable explanation."""
    try:
        PATIENTS_DIR.mkdir(parents=True, exist_ok=True)
        probe = PATIENTS_DIR / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return None
    except OSError as exc:
        return (f"Cannot write to {PATIENTS_DIR} ({exc.strerror or exc}). "
                f"Fix on the Pi with:  sudo chown -R $USER:$USER {DATA_DIR}")


def load(patient_id: str) -> dict:
    path = folder(patient_id) / "patient.json"
    if not path.exists():
        raise PatientNotFound(patient_id)
    return _read_json(path)


def save(record: dict) -> None:
    _write_json(folder(record["patient_id"]) / "patient.json", record)


def list_patients() -> list[dict]:
    if not PATIENTS_DIR.exists():
        return []
    out = []
    for p in sorted(PATIENTS_DIR.iterdir()):
        if (p / "patient.json").exists():
            rec = _read_json(p / "patient.json")
            out.append({"patient_id": rec["patient_id"], "name": rec["name"],
                        "age": rec.get("age"), "sex": rec.get("sex"),
                        "created_at": rec.get("created_at"),
                        "assessments": len(list_files(rec["patient_id"], "assessments")),
                        "has_photo": bool(rec.get("first_photo"))})
    return out


# ---- photos -----------------------------------------------------------------

def _to_jpeg(data: bytes) -> bytes:
    """Accept any common image (JPEG, PNG, WebP, BMP, GIF), return a JPEG of at most 1600 px."""
    import io
    try:
        from PIL import Image, ImageOps
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)          # phone photos: keep them upright
        img = img.convert("RGB")
    except Exception as exc:
        raise PatientError("That file is not a readable image (use JPEG, PNG or WebP).") from exc
    img.thumbnail((1600, 1600))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue()


def save_photo(patient_id: str, data: bytes) -> str:
    """Save an uploaded image as JPEG. The first photo is the registration photo."""
    rec = load(patient_id)
    if len(data) > 20 * 1024 * 1024:
        raise PatientError("Photo is larger than 20 MB.")
    data = _to_jpeg(data)
    photos = folder(patient_id) / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    if not rec.get("first_photo"):
        name = "photo_first.jpg"
        rec["first_photo"] = name
    else:
        name = f"photo_{_stamp()}.jpg"
    (photos / name).write_bytes(data)
    save(rec)
    return name


def save_photo_base64(patient_id: str, b64: str) -> str:
    try:
        data = base64.b64decode(b64.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PatientError("Photo data is not valid base64.") from exc
    return save_photo(patient_id, data)


def photo_path(patient_id: str, name: str) -> Path:
    if "/" in name or "\\" in name or name.startswith("."):
        raise PatientError("Invalid file name.")
    path = folder(patient_id) / "photos" / name
    if not path.exists():
        raise PatientNotFound(name)
    return path


def first_photo_path(patient_id: str) -> Path | None:
    rec = load(patient_id)
    if not rec.get("first_photo"):
        return None
    path = folder(patient_id) / "photos" / rec["first_photo"]
    return path if path.exists() else None


def new_photo_path(patient_id: str) -> Path:
    return folder(patient_id) / "photos" / f"photo_{_stamp()}.jpg"


def register_photo_path(patient_id: str, path: Path) -> str:
    """Record a photo file that was just captured straight into the folder."""
    rec = load(patient_id)
    name = path.name
    if not rec.get("first_photo"):
        first = folder(patient_id) / "photos" / "photo_first.jpg"
        path.replace(first)
        rec["first_photo"] = "photo_first.jpg"
        name = "photo_first.jpg"
        save(rec)
    return name


# ---- records ----------------------------------------------------------------

def save_measurement(patient_id: str, record: dict) -> str:
    name = f"measurement_{_stamp()}.json"
    _write_json(folder(patient_id) / "measurements" / name, {**record, "saved_at": _now()})
    return name


def save_assessment(patient_id: str, record: dict) -> str:
    name = f"assessment_{_stamp()}.json"
    _write_json(folder(patient_id) / "assessments" / name, {**record, "saved_at": _now()})
    return name


def list_records(patient_id: str, kind: str) -> list[dict]:
    d = folder(patient_id) / kind
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        out.append({"file": p.name, **_read_json(p)})
    return out


def list_files(patient_id: str, kind: str) -> list[str]:
    d = folder(patient_id) / kind
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file())


def report_path(patient_id: str, fmt: str) -> Path:
    ext = {"png": "png", "txt": "txt", "docx": "docx"}[fmt]
    return folder(patient_id) / "reports" / f"report_{_stamp()}.{ext}"


# ---- share links ------------------------------------------------------------

def create_share(patient_id: str, days: int = 7) -> dict:
    days = max(1, min(int(days), 30))
    rec = load(patient_id)
    token = secrets.token_urlsafe(24)
    expires = datetime.now(timezone.utc) + timedelta(days=days)
    rec["share"] = {"token": token, "expires_at": expires.isoformat(timespec="seconds"),
                    "created_at": _now()}
    save(rec)
    return rec["share"]


def revoke_share(patient_id: str) -> None:
    rec = load(patient_id)
    rec["share"] = None
    save(rec)


def patient_for_token(token: str) -> str | None:
    """Return the patient ID for a valid, unexpired share token, else None."""
    if not token or len(token) < 20 or not PATIENTS_DIR.exists():
        return None
    for p in PATIENTS_DIR.iterdir():
        f = p / "patient.json"
        if not f.exists():
            continue
        share = _read_json(f).get("share") or {}
        if share.get("token") and secrets.compare_digest(share["token"], token):
            expires = datetime.fromisoformat(share["expires_at"])
            if expires > datetime.now(timezone.utc):
                return p.name
            return None
    return None
