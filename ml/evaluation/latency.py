"""CPU inference latency of an exported ONNX model, measured the way the service runs it."""

from __future__ import annotations

import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KiB on Linux


def measure(model_path: Path, runs: int = 50, threads: int = 1, warmup: int = 5) -> dict:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(1, 3, 224, 224)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"image": x})
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, {"image": x})
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "cpu": {
            "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)),
            "rss_mb": round(_rss_mb(), 1),
            "threads": threads,
        },
        "onnx_size_mb": round(model_path.stat().st_size / 2**20, 3),
        "runs": runs,
        "machine": f"{platform.system()} {platform.machine()} ({platform.processor() or 'unknown cpu'})",
    }
