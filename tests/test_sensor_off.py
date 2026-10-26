import os
from sensor_service import make_sensor_service


def test_sensor_off_reports_no_sensor(monkeypatch):
    """SENSOR_MODE=off (the Mac setting) must never invent a pulse."""
    monkeypatch.setenv("SENSOR_MODE", "off")
    svc = make_sensor_service()
    assert svc.source_name == "none"
    samples = svc.source.read_samples()
    assert samples == []
    svc._update_live()
    live = svc.get_live()
    assert live["source"] == "none"
    assert live["finger"] is False
    assert live["hr"] is None
