from pathlib import Path

from pydantic import PositiveInt
from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration read from environment variables (names are the upper-case field names)."""

    max_upload_bytes: PositiveInt = 10 * 1024 * 1024
    max_image_pixels: PositiveInt = 40_000_000
    rate_limit_per_minute: PositiveInt = 30
    allowed_origins: str = "http://localhost:5173"
    model_dir: Path = REPO_ROOT / "artifacts" / "models"
    metrics_dir: Path = REPO_ROOT / "artifacts" / "metrics"
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"
    trust_proxy_headers: bool = False
    client_ip_header: str = ""
    # A 40 MP upload peaks at a few hundred MB while decoding, so one image at a time is the
    # only setting that is safe on a 512 MB instance; raise it on bigger machines.
    max_concurrent_inferences: PositiveInt = 1
    inference_threads: PositiveInt = 1
    log_level: str = "INFO"

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
