import time

import pytest

from backend.app.api import routes
from backend.tests.helpers import CARD, app_of


def test_health_ok_shape(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "model_loaded", "model_version", "uptime_seconds"}
    assert body["status"] == "ok" and body["model_loaded"] is True and body["model_version"] == CARD["version"]


def test_inference_threads_setting_reaches_the_session(make_client):
    detector = app_of(make_client(inference_threads=2)).state.runtime.detector
    assert detector.session.get_session_options().intra_op_num_threads == 2


def test_uptime_counts_from_startup(client, monkeypatch):
    started = app_of(client).state.runtime.started_at
    monkeypatch.setattr(routes.time, "monotonic", lambda: started + 12.3456)
    assert client.get("/api/health").json()["uptime_seconds"] == pytest.approx(12.346)


def test_uptime_grows_in_real_time(client):
    first = client.get("/api/health").json()["uptime_seconds"]
    time.sleep(0.05)
    second = client.get("/api/health").json()["uptime_seconds"]
    assert first >= 0 and second - first >= 0.04
