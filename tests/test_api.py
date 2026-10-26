"""
Tests for api/main.py -- schema validation, prediction response shape,
and the honesty of the spo2_used_in_model flag.

These use FastAPI's TestClient, which calls the app directly in-process
(no need for a running uvicorn server).

Run: pytest tests/test_api.py -v

Requires the model to already be trained (python src/train.py), since
the API loads models/best_model.joblib at import time.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

TEST_PATIENTS = ["pytest_demo", "pytest_history_check", "pytest_high", "pytest_trend_user",
                 "pytest_hr_source", "pytest_factors"]


@pytest.fixture(autouse=True)
def _registered_patients():
    """/predict requires a registered patient, so register the test IDs first."""
    import patients
    for pid in TEST_PATIENTS:
        if not patients.exists(pid):
            patients.create(pid, "Test Patient", 50, "male")
    yield

VALID_PAYLOAD = {
    "patient_id": "pytest_demo",
    "male": 1, "age": 45, "education": 2, "currentSmoker": 0,
    "cigsPerDay": 0, "BPMeds": 0, "prevalentStroke": 0,
    "prevalentHyp": 0, "diabetes": 0, "totChol": 210, "sysBP": 120,
    "diaBP": 80, "BMI": 24.5, "heartRate": 72, "glucose": 85,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_payload_returns_200():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_expected_shape():
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    for field in [
        "patient_id", "model_estimated_probability", "risk_category",
        "decision_threshold_used", "validation_warnings",
        "top_contributing_factors", "spo2_used_in_model", "disclaimer",
    ]:
        assert field in body


def test_predict_probability_is_valid_range():
    response = client.post("/predict", json=VALID_PAYLOAD)
    proba = response.json()["model_estimated_probability"]
    assert 0.0 <= proba <= 1.0


def test_predict_risk_category_is_one_of_expected_values():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["risk_category"] in {"Low", "Moderate", "High"}


def test_spo2_field_is_never_marked_as_used():
    """This is the honesty check for the Tier-2 SpO2 claim: the flag
    must always say False right now, since the model was never trained
    on SpO2 data. If this ever flips to True without the model actually
    being retrained on SpO2, that's a bug, not a feature."""
    payload = dict(VALID_PAYLOAD)
    payload["spo2"] = 97
    response = client.post("/predict", json=payload)
    assert response.json()["spo2_used_in_model"] is False


def test_spo2_is_optional():
    response = client.post("/predict", json=VALID_PAYLOAD)  # no spo2 key at all
    assert response.status_code == 200


def test_out_of_range_field_rejected_by_schema():
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["sysBP"] = 400  # outside Pydantic's le=260 bound
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_missing_required_field_rejected():
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["age"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_prediction_is_persisted_and_retrievable_via_history():
    patient_id = "pytest_history_check"
    payload = dict(VALID_PAYLOAD)
    payload["patient_id"] = patient_id

    client.post("/predict", json=payload)
    response = client.get(f"/history/{patient_id}")

    assert response.status_code == 200
    history = response.json()["history"]
    assert len(history) >= 1
    assert "risk_category" in history[0]


def test_history_for_unknown_patient_returns_empty_list():
    response = client.get("/history/no_such_patient_xyz")
    assert response.status_code == 200
    assert response.json()["history"] == []


# ---------------------------------------------------------------------------
# Per-patient explanation + trend (added after the "same factors every time" bug)
# ---------------------------------------------------------------------------

HIGH_RISK_PAYLOAD = {
    **VALID_PAYLOAD,
    "patient_id": "pytest_high",
    "male": 0, "age": 63, "currentSmoker": 1, "cigsPerDay": 30,
    "prevalentHyp": 1, "diabetes": 1, "totChol": 290, "sysBP": 185,
    "diaBP": 110, "glucose": 210,
}


def test_top_factors_change_with_input():
    low = client.post("/predict", json=VALID_PAYLOAD).json()
    high = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
    low_c = {f["feature"]: f["contribution"] for f in low["top_contributing_factors"]}
    high_c = {f["feature"]: f["contribution"] for f in high["top_contributing_factors"]}
    assert low_c != high_c
    assert high["model_estimated_probability"] > low["model_estimated_probability"]
    # the high-risk patient's top factors should push risk UP
    assert all(f["contribution"] > 0 for f in high["top_contributing_factors"][:3])


def test_factor_entries_have_value_and_direction():
    body = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
    for f in body["top_contributing_factors"]:
        assert set(f) >= {"feature", "value", "contribution", "direction"}


def test_trend_endpoint_reports_direction():
    pid = "pytest_trend_user"
    client.post("/predict", json={**VALID_PAYLOAD, "patient_id": pid})
    client.post("/predict", json={**HIGH_RISK_PAYLOAD, "patient_id": pid})
    t = client.get(f"/trend/{pid}").json()
    assert t["summary"]["count"] >= 2
    assert t["summary"]["direction"] == "rising"
    assert t["summary"]["change"] > 0
    probs = [p["probability"] for p in t["points"]]
    assert len(probs) == t["summary"]["count"]


# ---------------------------------------------------------------------------
# Device (ESP8266 + MAX30102) endpoints
# ---------------------------------------------------------------------------

DEVICE_KEY = "pytest-device-key"


@pytest.fixture
def device_key(monkeypatch):
    import api.main as main
    monkeypatch.setattr(main, "DEVICE_API_KEY", DEVICE_KEY)
    return DEVICE_KEY


def test_device_endpoints_refuse_without_key(monkeypatch):
    import api.main as main
    monkeypatch.setattr(main, "DEVICE_API_KEY", None)
    r = client.post("/device/reading", json={"device_id": "d1", "heart_rate": 70})
    assert r.status_code == 503


def test_device_rejects_wrong_key(device_key):
    r = client.post("/device/reading", json={"device_id": "d1", "heart_rate": 70},
                    headers={"X-Device-Key": "wrong"})
    assert r.status_code == 401


def test_device_reading_stored_and_latest_returned(device_key):
    dev = "pytest-dev-latest"
    r = client.post("/device/reading", json={"device_id": dev, "heart_rate": 81, "spo2": 97},
                    headers={"X-Device-Key": device_key})
    assert r.status_code == 200
    latest = client.get(f"/device/latest/{dev}", headers={"X-Device-Key": device_key}).json()
    assert latest["status"] == "live"
    assert latest["reading"]["heart_rate"] == 81


def test_device_no_finger_reading_needs_no_heart_rate(device_key):
    dev = "pytest-dev-nofinger"
    ok = client.post("/device/reading",
                     json={"device_id": dev, "heart_rate": None, "signal_ok": False},
                     headers={"X-Device-Key": device_key})
    assert ok.status_code == 200
    bad = client.post("/device/reading",
                      json={"device_id": dev, "heart_rate": None, "signal_ok": True},
                      headers={"X-Device-Key": device_key})
    assert bad.status_code == 422


def test_device_recent_returns_oldest_to_newest(device_key):
    dev = "pytest-dev-recent"
    for hr in (60, 70, 80):
        client.post("/device/reading", json={"device_id": dev, "heart_rate": hr},
                    headers={"X-Device-Key": device_key})
    rec = client.get(f"/device/recent/{dev}?limit=3",
                     headers={"X-Device-Key": device_key}).json()["readings"]
    assert [r["heart_rate"] for r in rec] == [60, 70, 80]


def test_prediction_records_hr_source(device_key):
    pid = "pytest_hr_source"
    r = client.post("/predict", json={**VALID_PAYLOAD, "patient_id": pid, "hr_source": "sensor"})
    assert r.status_code == 200
    hist = client.get(f"/history/{pid}").json()["history"]
    assert len(hist) >= 1
    assert client.post("/predict", json={**VALID_PAYLOAD, "patient_id": pid,
                                         "hr_source": "bogus"}).status_code == 422


# ---------------------------------------------------------------------------
# 12-year horizon, patient register, reports, sharing, login
# ---------------------------------------------------------------------------

def _tiny_jpeg_b64() -> str:
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (20, 120, 110)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_prediction_reports_12_year_risk_and_keeps_10_year_output():
    body = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
    assert body["horizon_years"] == 12
    p10, p12 = body["risk_10_year_probability"], body["risk_12_year_probability"]
    assert p12 > p10
    assert abs(p12 - (1 - (1 - p10) ** 1.2)) < 1e-3
    assert body["model_estimated_probability"] == p12


def test_prediction_requires_registered_patient():
    r = client.post("/predict", json={**VALID_PAYLOAD, "patient_id": "pytest_not_registered"})
    assert r.status_code == 404


def test_patient_register_and_duplicate_rejected():
    pid = "pytest_register_me"
    body = {"patient_id": pid, "name": "Asha Rao", "age": 58, "sex": "female", "notes": ""}
    assert client.post("/patients", json=body).status_code == 201
    dup = client.post("/patients", json=body)
    assert dup.status_code == 409
    detail = client.get(f"/patients/{pid}").json()
    assert detail["patient"]["name"] == "Asha Rao"
    assert pid in detail["folder"]


def test_bad_patient_id_rejected():
    r = client.post("/patients", json={"patient_id": "a b", "name": "X"})
    assert r.status_code in (400, 422)


def test_photo_upload_is_saved_and_served():
    pid = "pytest_photo_user"
    if client.get(f"/patients/{pid}").status_code == 404:
        client.post("/patients", json={"patient_id": pid, "name": "Photo Test"})
    r = client.post(f"/patients/{pid}/photo", json={"image_base64": _tiny_jpeg_b64()})
    assert r.status_code == 200
    url = r.json()["url"]
    assert client.get(url).status_code == 200


def test_camera_placeholder_when_no_camera_configured():
    pid = "pytest_camera_user"
    if client.get(f"/patients/{pid}").status_code == 404:
        client.post("/patients", json={"patient_id": pid, "name": "Camera Test"})
    r = client.post(f"/patients/{pid}/photo/capture")
    assert r.status_code == 200
    assert r.json()["saved"].startswith("photo_")


def test_reports_generated_in_all_three_formats():
    pid = "pytest_history_check"
    client.post("/predict", json={**VALID_PAYLOAD, "patient_id": pid})
    png = client.get(f"/patients/{pid}/report?format=png")
    txt = client.get(f"/patients/{pid}/report?format=txt")
    docx = client.get(f"/patients/{pid}/report?format=docx")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert txt.status_code == 200 and b"PATIENT REPORT" in txt.content
    assert docx.status_code == 200 and docx.content[:2] == b"PK"   # docx is a zip
    assert client.get(f"/patients/{pid}/report?format=pdf").status_code == 400
    listing = client.get(f"/patients/{pid}").json()["reports"]
    assert len(listing) >= 3


def test_share_link_read_only_and_revocable():
    pid = "pytest_share_user"
    if client.get(f"/patients/{pid}").status_code == 404:
        client.post("/patients", json={"patient_id": pid, "name": "Share Test"})
    client.post("/predict", json={**VALID_PAYLOAD, "patient_id": pid})
    link = client.post(f"/patients/{pid}/share", json={"days": 7}).json()["url"]
    token = link.rsplit("/", 1)[-1]
    assert client.get(f"/share/{token}").status_code == 200
    data = client.get(f"/share/{token}/data").json()
    assert data["patient"]["patient_id"] == pid
    assert "notes" not in data["patient"]
    assert client.get(f"/share/{token}/report?format=txt").status_code == 200
    assert client.delete(f"/patients/{pid}/share").status_code == 200
    assert client.get(f"/share/{token}/data").status_code == 404
    assert client.get("/share/not-a-real-token-but-long-enough-1234").status_code == 404


def test_dashboard_login_when_password_set(monkeypatch):
    import base64
    import api.main as main
    monkeypatch.setattr(main, "DASHBOARD_PASSWORD", "clinic-pass")
    assert client.get("/dashboard/").status_code == 401
    good = base64.b64encode(b"doctor:clinic-pass").decode()
    assert client.get("/dashboard/", headers={"Authorization": f"Basic {good}"}).status_code == 200
    bad = base64.b64encode(b"doctor:wrong").decode()
    assert client.get("/patients", headers={"Authorization": f"Basic {bad}"}).status_code == 401


def test_sensor_live_reports_source_and_trace():
    r = client.get("/sensor/live").json()
    assert r["source"] in ("simulated", "max30102")
    assert "trace" in r and "finger_message" in r


# ---- fixes from the Raspberry Pi test run ---------------------------------

def _png_b64():
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 30, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_edit_patient_details():
    c = client
    c.post("/patients", json={"patient_id": "edit-01", "name": "Old Name", "age": 30, "sex": "male"})
    r = c.put("/patients/edit-01", json={"name": "New Name", "age": 31, "notes": "updated"})
    assert r.status_code == 200
    assert r.json()["patient"]["name"] == "New Name"
    assert c.get("/patients/edit-01").json()["patient"]["notes"] == "updated"
    assert c.put("/patients/nobody-99", json={"name": "x"}).status_code == 404


def test_upload_any_image_is_stored_as_jpeg():
    c = client
    c.post("/patients", json={"patient_id": "photo-01", "name": "Photo Test"})
    r = c.post("/patients/photo-01/photo", json={"image_base64": "data:image/png;base64," + _png_b64()})
    assert r.status_code == 200
    assert r.json()["saved"] == "photo_first.jpg"
    bad = c.post("/patients/photo-01/photo", json={"image_base64": "aGVsbG8gd29ybGQ="})
    assert bad.status_code == 400


def test_register_without_camera_still_works(monkeypatch):
    import hardware.camera as cam
    monkeypatch.setenv("CAMERA_BACKEND", "auto")
    monkeypatch.setattr(cam, "_capture_picamera2", lambda d: (_ for _ in ()).throw(cam.CameraError("no pi camera")))
    monkeypatch.setattr(cam, "_capture_usb", lambda d: (_ for _ in ()).throw(cam.CameraError("no usb camera")))
    c = client
    r = c.post("/patients", json={"patient_id": "nocam-01", "name": "No Camera", "take_photo": True})
    assert r.status_code == 201
    assert "No camera detected" in r.json()["photo_warning"]


def test_unexpected_error_names_the_cause(monkeypatch):
    import api.main as m
    from fastapi.testclient import TestClient
    def boom(*a, **k):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(m.patients, "create", boom)
    r = TestClient(m.app, raise_server_exceptions=False).post(
        "/patients", json={"patient_id": "perm-01", "name": "x"})
    assert r.status_code == 500
    assert "detail" in r.json() and r.json()["detail"] != "Internal Server Error"
