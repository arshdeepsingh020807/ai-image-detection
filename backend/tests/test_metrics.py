import json
import logging

import pytest

from backend.app.services.metrics import read_metrics
from backend.tests.helpers import assert_error

SUMMARY = {"model": "efficientnet_b0", "test": {"roc_auc": 0.91, "confusion_matrix": [[5, 1], [2, 7]]}}
ROC = {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0], "auc": 0.8}


@pytest.fixture
def metrics_dir(tmp_path):
    directory = tmp_path / "metrics"
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps(SUMMARY))
    (directory / "roc.json").write_text(json.dumps(ROC))
    return directory


def test_reads_every_json_keyed_by_stem_sorted(metrics_dir):
    (metrics_dir / "notes.txt").write_text("ignored")
    (metrics_dir / "list.json").write_text("[1, 2]")
    assert read_metrics(metrics_dir) == {
        "available": ["list", "roc", "summary"],
        "metrics": {"list": [1, 2], "roc": ROC, "summary": SUMMARY},
    }


def test_missing_directory_yields_nothing(tmp_path):
    assert read_metrics(tmp_path / "absent") == {"available": [], "metrics": {}}


@pytest.mark.parametrize(
    "content",
    ['{"truncated": ', '{"auc": NaN}', '{"x": Infinity}', b"\xff\xfe\x00bad utf-8"],
)
def test_malformed_files_are_skipped_and_logged(metrics_dir, caplog, content):
    bad = metrics_dir / "broken.json"
    bad.write_bytes(content if isinstance(content, bytes) else content.encode())
    with caplog.at_level(logging.WARNING, logger="backend"):
        result = read_metrics(metrics_dir)
    assert result["available"] == ["roc", "summary"]
    assert [(r.getMessage(), r.fields["file"]) for r in caplog.records] == [("skipping metrics file", "broken.json")]


def test_directory_named_like_json_is_skipped(metrics_dir, caplog):
    (metrics_dir / "folder.json").mkdir()
    with caplog.at_level(logging.WARNING, logger="backend"):
        assert read_metrics(metrics_dir)["available"] == ["roc", "summary"]
    assert len(caplog.records) == 1


# ---------- GET /api/metrics ----------


def test_endpoint_returns_files_from_metrics_dir(make_client, metrics_dir):
    (metrics_dir / "bad.json").write_text("nope")
    response = make_client(metrics_dir=metrics_dir).get("/api/metrics")
    assert response.status_code == 200
    assert response.json() == {"available": ["roc", "summary"], "metrics": {"roc": ROC, "summary": SUMMARY}}


def test_endpoint_sees_files_added_after_startup(make_client, metrics_dir):
    client = make_client(metrics_dir=metrics_dir)
    (metrics_dir / "history.json").write_text('{"epochs": []}')
    assert client.get("/api/metrics").json()["available"] == ["history", "roc", "summary"]


def test_endpoint_with_absent_dir(make_client, tmp_path):
    response = make_client(metrics_dir=tmp_path / "missing").get("/api/metrics")
    assert response.status_code == 200 and response.json() == {"available": [], "metrics": {}}


def test_endpoint_rejects_post(client):
    assert_error(client.post("/api/metrics"), 405, "method_not_allowed")
