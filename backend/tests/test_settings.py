from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.settings import REPO_ROOT, Settings

ENV_NAMES = [
    "MAX_UPLOAD_BYTES",
    "MAX_IMAGE_PIXELS",
    "RATE_LIMIT_PER_MINUTE",
    "ALLOWED_ORIGINS",
    "MODEL_DIR",
    "METRICS_DIR",
    "FRONTEND_DIST",
    "TRUST_PROXY_HEADERS",
    "MAX_CONCURRENT_INFERENCES",
    "INFERENCE_THREADS",
    "LOG_LEVEL",
]


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_match_env_example(clean_env):
    settings = Settings()
    assert settings.max_upload_bytes == 10485760 and settings.max_image_pixels == 40000000
    assert settings.rate_limit_per_minute == 30 and settings.origins == ["http://localhost:5173"]
    assert (
        settings.model_dir == REPO_ROOT / "artifacts/models" and settings.metrics_dir == REPO_ROOT / "artifacts/metrics"
    )
    assert settings.frontend_dist == REPO_ROOT / "frontend/dist"
    assert settings.trust_proxy_headers is False and settings.max_concurrent_inferences == 1
    assert settings.inference_threads == 1 and settings.log_level == "INFO"
    assert (REPO_ROOT / "backend" / "app" / "core" / "settings.py").is_file()


def test_environment_overrides(clean_env):
    clean_env.setenv("MAX_UPLOAD_BYTES", "123")
    clean_env.setenv("RATE_LIMIT_PER_MINUTE", "7")
    clean_env.setenv("ALLOWED_ORIGINS", " https://a.example , ,https://b.example,")
    clean_env.setenv("MODEL_DIR", "/srv/models")
    clean_env.setenv("TRUST_PROXY_HEADERS", "true")
    settings = Settings()
    assert settings.max_upload_bytes == 123 and settings.rate_limit_per_minute == 7
    assert settings.origins == ["https://a.example", "https://b.example"]
    assert settings.model_dir == Path("/srv/models") and settings.trust_proxy_headers is True


@pytest.mark.parametrize(
    "name, value",
    [
        ("MAX_UPLOAD_BYTES", "0"),
        ("MAX_IMAGE_PIXELS", "0"),
        ("MAX_IMAGE_PIXELS", "x"),
        ("RATE_LIMIT_PER_MINUTE", "-1"),
        ("MAX_CONCURRENT_INFERENCES", "0"),
        ("INFERENCE_THREADS", "0"),
        ("TRUST_PROXY_HEADERS", "maybe"),
    ],
)
def test_invalid_values_fail_fast(clean_env, name, value):
    clean_env.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()


def test_empty_origins_allow_nobody(clean_env):
    clean_env.setenv("ALLOWED_ORIGINS", "")
    assert Settings().origins == []
