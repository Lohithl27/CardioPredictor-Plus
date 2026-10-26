"""
main.py
CardioPredictor+ server. Runs continuously on a Raspberry Pi (or a laptop
for development). It:

  * keeps the MAX30102 sensor running in the background (or a simulated pulse);
  * registers patients and stores one folder per patient under data/patients/;
  * takes a photo at registration with the camera (Pi camera or USB webcam);
  * runs a timed finger measurement and shows the reading on the dashboard;
  * runs the 12-year risk estimate and saves every assessment to the folder;
  * builds PNG / TXT / DOCX reports and read-only share links for patients;
  * still accepts readings from an ESP8266 (optional, /device/* endpoints).

Start (development):   uvicorn api.main:app --host 0.0.0.0 --port 8000
Start (Raspberry Pi):  see deploy/RASPBERRY_PI_GUIDE.md (runs as a service)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from validate import validate_reading          # noqa: E402
from storage import (                          # noqa: E402
    save_prediction, get_history, save_device_reading, get_latest_device_reading,
    get_recent_device_readings,
)
from explain import Explainer                  # noqa: E402
from vitals import FS                          # noqa: E402
from sensor_service import make_sensor_service  # noqa: E402
from hardware.camera import capture_photo, CameraError  # noqa: E402
import patients                                # noqa: E402
import reports                                 # noqa: E402

log = logging.getLogger("cardio")
MODELS_DIR = ROOT_DIR / "models"
STATIC_DIR = ROOT_DIR / "api" / "static"

# Optional: set CARDIO_DASHBOARD_PASSWORD to require a login for the dashboard
# and patient data (HTTP Basic auth; any username). Share links stay open
# because they carry their own secret token.
DASHBOARD_PASSWORD = os.environ.get("CARDIO_DASHBOARD_PASSWORD") or None
# Shared secret for the optional ESP8266 path (/device/* endpoints).
DEVICE_API_KEY = os.environ.get("CARDIO_DEVICE_KEY") or None
DEVICE_STALE_SECONDS = 30
HORIZON_YEARS = 12
MODEL_HORIZON_YEARS = 10
PUBLIC_PREFIXES = ("/share/", "/device/", "/health")

SENSOR = None
DATA_ERROR: str | None = None          # set at start-up if the data folder cannot be written
_SENSOR_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Sensor (background, started once)
# ---------------------------------------------------------------------------

def _store_sensor_reading(live: dict) -> None:
    save_device_reading(device_id="rpi-sensor", patient_id=None,
                        heart_rate=live.get("hr"), spo2=live.get("spo2"), signal_ok=True)


def get_sensor():
    global SENSOR
    with _SENSOR_LOCK:
        if SENSOR is None:
            SENSOR = make_sensor_service(on_reading=_store_sensor_reading)
            SENSOR.start()
    return SENSOR


@asynccontextmanager
async def lifespan(_app):
    global DATA_ERROR
    DATA_ERROR = patients.check_writable()
    if DATA_ERROR:
        log.error(DATA_ERROR)
    get_sensor()
    yield


app = FastAPI(
    title="CardioPredictor+ API",
    description=(
        "Research prototype for 12-year cardiovascular risk estimation. "
        "Not a medical device. Outputs are model estimates, not diagnoses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Any unexpected error: log the traceback and tell the browser WHAT failed,
# instead of a bare "Request failed (500)".
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    if isinstance(exc, OSError):
        hint = DATA_ERROR or patients.check_writable()
        detail = hint or f"File error: {exc.strerror or exc}"
    else:
        detail = f"Server error: {type(exc).__name__}: {exc}"
    return JSONResponse(status_code=500, content={"detail": detail})


# ---------------------------------------------------------------------------
# Optional dashboard login
# ---------------------------------------------------------------------------

def _basic_ok(header: str | None) -> bool:
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        raw = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    _, _, password = raw.partition(":")
    return secrets.compare_digest(password, DASHBOARD_PASSWORD or "")


@app.middleware("http")
async def dashboard_login(request: Request, call_next):
    path = request.url.path
    if DASHBOARD_PASSWORD and not path.startswith(PUBLIC_PREFIXES):
        if not _basic_ok(request.headers.get("authorization")):
            return Response(status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="CardioPredictor+"'},
                            content="Login required")
    return await call_next(request)


# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------

def _load_artifacts():
    model_path = MODELS_DIR / "best_model.joblib"
    columns_path = MODELS_DIR / "feature_columns.joblib"
    threshold_path = MODELS_DIR / "decision_threshold.json"
    importance_path = MODELS_DIR / "feature_importance.json"
    if not model_path.exists():
        raise RuntimeError(
            f"No trained model found at {model_path}. Run `python src/train.py` first."
        )
    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
    threshold = 0.5
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold = json.load(f)["threshold"]
    importance = None
    if importance_path.exists():
        with open(importance_path) as f:
            importance = json.load(f)
    return model, feature_columns, threshold, importance


try:
    MODEL, FEATURE_COLUMNS, DECISION_THRESHOLD, FEATURE_IMPORTANCE = _load_artifacts()
    _load_error = None
except RuntimeError as exc:
    MODEL, FEATURE_COLUMNS, DECISION_THRESHOLD, FEATURE_IMPORTANCE = None, None, 0.5, None
    _load_error = str(exc)

EXPLAINER = None
if MODEL is not None and (MODELS_DIR / "background.joblib").exists():
    EXPLAINER = Explainer(MODEL, joblib.load(MODELS_DIR / "background.joblib"))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PatientCreate(BaseModel):
    patient_id: str = Field(..., min_length=3, max_length=40)
    name: str = Field(..., min_length=1, max_length=120)
    age: float | None = Field(None, ge=0, le=120)
    sex: str | None = Field(None, pattern="^(male|female|other)$")
    notes: str = Field("", max_length=1000)
    take_photo: bool = False


class PatientUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    age: float | None = Field(None, ge=0, le=120)
    sex: str | None = Field(None, pattern="^(male|female|other|)$")
    notes: str | None = Field(None, max_length=1000)


class PhotoUpload(BaseModel):
    image_base64: str = Field(..., min_length=10)


class MeasureRequest(BaseModel):
    patient_id: str
    seconds: int = Field(20, ge=10, le=60)


class ShareRequest(BaseModel):
    days: int = Field(7, ge=1, le=30)


class PredictionRequest(BaseModel):
    patient_id: str = Field(..., description="Registered patient ID")
    male: int = Field(..., ge=0, le=1)
    age: float = Field(..., ge=18, le=100)
    education: int = Field(..., ge=1, le=4)
    currentSmoker: int = Field(..., ge=0, le=1)
    cigsPerDay: float = Field(..., ge=0, le=80)
    BPMeds: int = Field(..., ge=0, le=1)
    prevalentStroke: int = Field(..., ge=0, le=1)
    prevalentHyp: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    totChol: float = Field(..., ge=100, le=700)
    sysBP: float = Field(..., ge=70, le=260)
    diaBP: float = Field(..., ge=40, le=160)
    BMI: float = Field(..., ge=12, le=70)
    heartRate: float = Field(..., ge=30, le=220)
    glucose: float = Field(..., ge=40, le=500)
    spo2: float | None = Field(None, ge=70, le=100,
                               description="Stored for tracking; not a model input.")
    hr_source: str = Field("manual", pattern="^(sensor|manual)$")


class DeviceReading(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=64)
    patient_id: str | None = Field(None, max_length=64)
    heart_rate: float | None = Field(None, ge=30, le=220)
    spo2: float | None = Field(None, ge=70, le=100)
    signal_ok: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_12_year(p10: float) -> float:
    """Extend a 10-year risk to 12 years assuming a constant yearly hazard.

    The Framingham data only has a 10-year outcome, so no 12-year model was
    trained. This is an approximation and is labelled as one in the output.
    """
    return 1.0 - (1.0 - p10) ** (HORIZON_YEARS / MODEL_HORIZON_YEARS)


def categorize_risk(probability: float) -> str:
    """Simple, transparent bands (a design choice, not clinical guidance)."""
    if probability < 0.10:
        return "Low"
    if probability < 0.20:
        return "Moderate"
    return "High"


def _require_device_key(provided: str | None) -> None:
    if DEVICE_API_KEY is None:
        raise HTTPException(status_code=503,
                            detail="Device endpoints disabled: CARDIO_DEVICE_KEY is not set.")
    if provided != DEVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Device-Key.")


def _patient_or_404(patient_id: str) -> dict:
    try:
        return patients.load(patient_id)
    except patients.PatientNotFound:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' is not registered.")
    except patients.PatientError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _patient_error(exc: patients.PatientError) -> HTTPException:
    code = 409 if "already" in str(exc) else 400
    return HTTPException(status_code=code, detail=str(exc))


def _folder_str(patient_id: str) -> str:
    try:
        return str(patients.folder(patient_id).relative_to(ROOT_DIR))
    except ValueError:
        return str(patients.folder(patient_id))


def _file_url(patient_id: str, kind: str, name: str) -> str:
    return f"/patients/{patient_id}/{kind}/{name}"


# ---------------------------------------------------------------------------
# Health, pages
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    sensor = get_sensor()
    return {
        "status": "ok" if MODEL is not None else "model_not_loaded",
        "detail": _load_error,
        "sensor": sensor.source_name,
        "sensor_error": sensor.error,
        "data_folder": str(patients.DATA_DIR),
        "data_folder_error": DATA_ERROR,
        "camera_backend": os.environ.get("CAMERA_BACKEND", "auto"),
        "horizon_years": HORIZON_YEARS,
        "dashboard_login": DASHBOARD_PASSWORD is not None,
    }


@app.get("/")
def root():
    return RedirectResponse("/dashboard/")


# ---------------------------------------------------------------------------
# Patients (register, folder, photo)
# ---------------------------------------------------------------------------

@app.get("/patients")
def patients_list():
    return {"patients": patients.list_patients()}


@app.post("/patients", status_code=201)
def patients_create(body: PatientCreate):
    try:
        rec = patients.create(body.patient_id, body.name, body.age, body.sex, body.notes)
    except patients.PatientError as exc:
        raise _patient_error(exc)
    photo_note = None
    if body.take_photo:
        try:
            _capture_into_patient(body.patient_id)
        except Exception as exc:          # a camera problem must never block registration
            photo_note = f"Patient registered, but the photo failed: {exc}"
    rec = patients.load(body.patient_id)
    return {"patient": rec, "folder": _folder_str(body.patient_id), "photo_warning": photo_note}


@app.put("/patients/{patient_id}")
def patients_update(patient_id: str, body: PatientUpdate):
    _patient_or_404(patient_id)
    try:
        rec = patients.update(patient_id, body.name, body.age, body.sex, body.notes)
    except patients.PatientError as exc:
        raise _patient_error(exc)
    return {"patient": rec}


@app.get("/patients/{patient_id}")
def patient_detail(patient_id: str):
    rec = _patient_or_404(patient_id)
    measurements = patients.list_records(patient_id, "measurements")
    assessments = patients.list_records(patient_id, "assessments")
    return {
        "patient": rec,
        "folder": _folder_str(patient_id),
        "photo_url": _file_url(patient_id, "photos", rec["first_photo"]) if rec.get("first_photo") else None,
        "photos": [_file_url(patient_id, "photos", n) for n in patients.list_files(patient_id, "photos")],
        "measurements": measurements[-50:],
        "assessments": assessments[-50:],
        "reports": [_file_url(patient_id, "reports", n) for n in patients.list_files(patient_id, "reports")],
    }


def _capture_into_patient(patient_id: str) -> str:
    _patient_or_404(patient_id)
    path = patients.new_photo_path(patient_id)
    backend = capture_photo(path)
    name = patients.register_photo_path(patient_id, path)
    return f"{name} ({backend})"


@app.post("/patients/{patient_id}/photo/capture")
def photo_capture(patient_id: str):
    try:
        saved = _capture_into_patient(patient_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Camera error: {exc}")
    return {"saved": saved, "photo_url": _file_url(patient_id, "photos",
                                                   patients.load(patient_id)["first_photo"])}


@app.post("/patients/{patient_id}/photo")
def photo_upload(patient_id: str, body: PhotoUpload):
    _patient_or_404(patient_id)
    try:
        name = patients.save_photo_base64(patient_id, body.image_base64)
    except patients.PatientError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"saved": name, "url": _file_url(patient_id, "photos", name)}


@app.get("/patients/{patient_id}/photos/{name}")
def photo_file(patient_id: str, name: str):
    try:
        return FileResponse(patients.photo_path(patient_id, name))
    except (patients.PatientNotFound, patients.PatientError):
        raise HTTPException(status_code=404, detail="Photo not found.")


# ---------------------------------------------------------------------------
# Sensor: live panel and timed measurement
# ---------------------------------------------------------------------------

@app.get("/sensor/live")
def sensor_live():
    live = get_sensor().get_live()
    live["finger_message"] = ("Finger detected" if live.get("finger")
                              else "Place your finger on the sensor")
    return live


def _run_measurement(job_id: str, patient_id: str, seconds: int) -> None:
    sensor = get_sensor()

    def update(**kw):
        with _JOBS_LOCK:
            JOBS[job_id].update(**kw)

    try:
        # Phase 1: wait (up to 30 s) for a finger on the sensor.
        waited = 0.0
        while not sensor.get_live().get("finger"):
            if waited >= 30:
                update(status="failed", phase="done",
                       message="No finger detected. Place your finger on the sensor and try again.")
                return
            update(phase="waiting_for_finger", progress=0.0)
            time.sleep(0.5)
            waited += 0.5

        # Phase 2: record for `seconds`.
        start = time.time()
        update(phase="measuring")
        while time.time() - start < seconds:
            update(progress=round((time.time() - start) / seconds, 3))
            time.sleep(0.5)

        v = sensor.estimate(sensor.recent_samples(seconds * FS))
        record = {"patient_id": patient_id, "hr": v.get("hr"), "spo2": v.get("spo2"),
                  "quality": v.get("quality"), "finger": v.get("finger"),
                  "variability": v.get("variability"),
                  "window_seconds": v.get("window_seconds"), "source": sensor.source_name}
        if v.get("hr") is None:
            file = patients.save_measurement(patient_id, record)
            update(status="failed", phase="done", progress=1.0, result=record, file=file,
                   message=f"Could not read a steady pulse ({v.get('quality')}). Keep still and try again.")
            return
        file = patients.save_measurement(patient_id, record)
        update(status="done", phase="done", progress=1.0, result=record, file=file,
               message="Measurement saved to the patient folder.")
    except Exception as exc:                         # never leave a job stuck
        log.exception("measurement failed")
        update(status="failed", phase="done", message=f"Measurement error: {exc}")


@app.post("/measure")
def measure_start(body: MeasureRequest):
    _patient_or_404(body.patient_id)
    if get_sensor().source_name == "none":
        raise HTTPException(status_code=409,
                            detail="No sensor is connected. Type the heart rate in the assessment form instead.")
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        JOBS[job_id] = {"job_id": job_id, "patient_id": body.patient_id, "seconds": body.seconds,
                        "status": "running", "phase": "starting", "progress": 0.0,
                        "message": "Place your finger on the sensor.", "result": None,
                        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    threading.Thread(target=_run_measurement, args=(job_id, body.patient_id, body.seconds),
                     daemon=True).start()
    return JOBS[job_id]


@app.get("/measure/{job_id}")
def measure_status(job_id: str):
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown measurement job.")
    return job


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_load_error}")
    _patient_or_404(request.patient_id)

    data = request.model_dump()
    patient_id = data.pop("patient_id")
    spo2 = data.pop("spo2", None)
    hr_source = data.pop("hr_source", "manual")

    validation = validate_reading({**data, "spo2": spo2})
    if not validation.is_valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    row = pd.DataFrame([{col: data[col] for col in FEATURE_COLUMNS}])
    p10 = float(MODEL.predict_proba(row)[0, 1])
    p12 = to_12_year(p10)
    category = categorize_risk(p12)

    top_factors = EXPLAINER.explain(row, top_k=5) if EXPLAINER is not None else None
    inputs = {**data, "spo2": spo2, "hr_source": hr_source}

    save_prediction(patient_id=patient_id, input_data=inputs,
                    calibrated_probability=p10, risk_category=categorize_risk(p10),
                    validation_warnings=validation.warnings)
    saved_file = patients.save_assessment(patient_id, {
        "patient_id": patient_id, "inputs": inputs,
        "risk_12_year_probability": round(p12, 4),
        "risk_10_year_probability": round(p10, 4),
        "risk_category": category, "horizon_years": HORIZON_YEARS,
        "top_contributing_factors": top_factors,
        "validation_warnings": validation.warnings,
    })

    return {
        "patient_id": patient_id,
        "horizon_years": HORIZON_YEARS,
        "risk_12_year_probability": round(p12, 4),
        "risk_10_year_probability": round(p10, 4),
        "model_estimated_probability": round(p12, 4),   # headline value shown on the dashboard
        "risk_category": category,
        "decision_threshold_used": round(DECISION_THRESHOLD, 4),
        "validation_warnings": validation.warnings,
        "top_contributing_factors": top_factors,
        "explanation_method": EXPLAINER.method if EXPLAINER is not None else None,
        "spo2_used_in_model": False,
        "hr_source": hr_source,
        "saved_to": f"{_folder_str(patient_id)}/assessments/{saved_file}",
        "disclaimer": (
            f"Research prototype. The {HORIZON_YEARS}-year figure is extended from a "
            f"{MODEL_HORIZON_YEARS}-year model using a constant-risk assumption. It is an "
            "estimate, not a diagnosis. Consult a qualified clinician."
        ),
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@app.get("/patients/{patient_id}/report")
def report_generate(patient_id: str, format: str = "png"):
    _patient_or_404(patient_id)
    if format not in ("png", "txt", "docx"):
        raise HTTPException(status_code=400, detail="format must be png, txt or docx")
    path = reports.generate(patient_id, format)
    return FileResponse(path, filename=f"{patient_id}_report.{format}")


@app.get("/patients/{patient_id}/reports/{name}")
def report_file(patient_id: str, name: str):
    _patient_or_404(patient_id)
    if "/" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid file name.")
    path = patients.folder(patient_id) / "reports" / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, filename=name)


# ---------------------------------------------------------------------------
# Share links (read-only, expiring)
# ---------------------------------------------------------------------------

@app.post("/patients/{patient_id}/share")
def share_create(patient_id: str, body: ShareRequest, request: Request):
    _patient_or_404(patient_id)
    share = patients.create_share(patient_id, body.days)
    base = str(request.base_url).rstrip("/")
    return {"url": f"{base}/share/{share['token']}", "expires_at": share["expires_at"]}


@app.delete("/patients/{patient_id}/share")
def share_revoke(patient_id: str):
    _patient_or_404(patient_id)
    patients.revoke_share(patient_id)
    return {"revoked": True}


def _share_patient(token: str) -> str:
    pid = patients.patient_for_token(token)
    if pid is None:
        raise HTTPException(status_code=404, detail="This link has expired or was revoked.")
    return pid


@app.get("/share/{token}", response_class=HTMLResponse)
def share_page(token: str):
    _share_patient(token)
    return HTMLResponse((STATIC_DIR / "share.html").read_text(encoding="utf-8"))


@app.get("/share/{token}/data")
def share_data(token: str):
    pid = _share_patient(token)
    rec = patients.load(pid)
    assessments = patients.list_records(pid, "assessments")
    measurements = patients.list_records(pid, "measurements")
    return {
        "patient": {"name": rec["name"], "patient_id": pid, "age": rec.get("age"),
                    "sex": rec.get("sex")},
        "expires_at": (rec.get("share") or {}).get("expires_at"),
        "photo_url": f"/share/{token}/photo" if rec.get("first_photo") else None,
        "assessments": [{k: a.get(k) for k in ("saved_at", "risk_12_year_probability",
                         "risk_category", "inputs")} for a in assessments],
        "measurements": [{k: m.get(k) for k in ("saved_at", "hr", "spo2", "quality")}
                         for m in measurements],
    }


@app.get("/share/{token}/photo")
def share_photo(token: str):
    pid = _share_patient(token)
    path = patients.first_photo_path(pid)
    if path is None:
        raise HTTPException(status_code=404, detail="No photo.")
    return FileResponse(path)


@app.get("/share/{token}/report")
def share_report(token: str, format: str = "png"):
    pid = _share_patient(token)
    if format not in ("png", "txt", "docx"):
        raise HTTPException(status_code=400, detail="format must be png, txt or docx")
    return FileResponse(reports.generate(pid, format), filename=f"{pid}_report.{format}")


# ---------------------------------------------------------------------------
# History and trend (SQLite)
# ---------------------------------------------------------------------------

@app.get("/history/{patient_id}")
def history(patient_id: str, limit: int = 50):
    rows = get_history(patient_id, limit)
    for r in rows:
        r["risk_12_year_probability"] = round(to_12_year(r["calibrated_probability"]), 4)
    return {"patient_id": patient_id, "history": rows}


@app.get("/trend/{patient_id}")
def trend(patient_id: str, limit: int = 50):
    rows = list(reversed(get_history(patient_id, limit)))
    points = [{"timestamp": r["timestamp"],
               "probability": round(to_12_year(r["calibrated_probability"]), 4),
               "risk_category": r["risk_category"]} for r in rows]
    summary = {"count": len(points), "direction": "insufficient data",
               "previous": None, "current": None, "change": None, "horizon_years": HORIZON_YEARS}
    if points:
        summary["current"] = points[-1]["probability"]
    if len(points) >= 2:
        prev, cur = points[-2]["probability"], points[-1]["probability"]
        change = cur - prev
        summary.update(previous=prev, change=round(change, 4))
        summary["direction"] = ("rising" if change > 0.01 else
                                "falling" if change < -0.01 else "stable")
    return {"patient_id": patient_id, "points": points, "summary": summary}


# ---------------------------------------------------------------------------
# Optional ESP8266 path (kept for the earlier design)
# ---------------------------------------------------------------------------

@app.post("/device/reading")
def device_reading(reading: DeviceReading, x_device_key: str | None = Header(None)):
    _require_device_key(x_device_key)
    if reading.signal_ok and reading.heart_rate is None:
        raise HTTPException(status_code=422, detail="heart_rate is required when signal_ok is true.")
    saved = save_device_reading(device_id=reading.device_id, patient_id=reading.patient_id,
                                heart_rate=reading.heart_rate, spo2=reading.spo2,
                                signal_ok=reading.signal_ok)
    return {"stored": True, **saved}


@app.get("/device/latest/{device_id}")
def device_latest(device_id: str, x_device_key: str | None = Header(None)):
    _require_device_key(x_device_key)
    latest = get_latest_device_reading(device_id)
    if latest is None:
        return {"device_id": device_id, "status": "no_data", "reading": None}
    received = datetime.fromisoformat(latest["timestamp"])
    age = (datetime.now(timezone.utc) - received).total_seconds()
    status = "stale" if age > DEVICE_STALE_SECONDS else "live"
    if status == "live" and not latest["signal_ok"]:
        status = "no_signal"
    return {"device_id": device_id, "status": status, "age_seconds": round(age, 1), "reading": latest}


@app.get("/device/recent/{device_id}")
def device_recent(device_id: str, limit: int = 60, x_device_key: str | None = Header(None)):
    _require_device_key(x_device_key)
    limit = max(1, min(limit, 300))
    return {"device_id": device_id, "readings": get_recent_device_readings(device_id, limit)}


# Dashboard (login-protected when CARDIO_DASHBOARD_PASSWORD is set).
app.mount("/dashboard", StaticFiles(directory=str(STATIC_DIR), html=True), name="dashboard")
