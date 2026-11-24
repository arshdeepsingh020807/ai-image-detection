import time

import pytest

from ml.evaluation import latency
from ml.scripts.export_onnx import export
from ml.training.model import Detector


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("lat")
    return export(Detector("mobilenetv3_small_050", pretrained=False), out, {})


def test_latency_report_shape_and_plausible_values(onnx_path):
    started = time.perf_counter()
    report = latency.measure(onnx_path, runs=10)
    wall_ms = (time.perf_counter() - started) * 1000
    cpu = report["cpu"]
    assert 0 < cpu["p50_ms"] <= cpu["p95_ms"] < wall_ms  # each run is shorter than the whole measurement
    assert report["runs"] == 10 and cpu["threads"] == 1
    assert report["onnx_size_mb"] == pytest.approx(onnx_path.stat().st_size / 2**20, abs=1e-3)
    assert 20 < cpu["rss_mb"] < 8000 and report["machine"]


def test_more_runs_are_counted(onnx_path):
    assert latency.measure(onnx_path, runs=3, warmup=0)["runs"] == 3
