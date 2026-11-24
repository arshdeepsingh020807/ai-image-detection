from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Label = Literal["real", "ai_generated", "inconclusive"]
ImageFormat = Literal["JPEG", "PNG", "WEBP"]


class Band(BaseModel):
    low: Probability
    high: Probability

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError("band.low must not exceed band.high")
        return self


class Probabilities(BaseModel):
    real: Probability
    ai_generated: Probability


class ImageInfo(BaseModel):
    width: int
    height: int
    format: ImageFormat
    size_bytes: int


class Region(BaseModel):
    x: Probability
    y: Probability
    width: Probability
    height: Probability


class Explanation(BaseModel):
    heatmap_png_base64: str
    region: Region
    method: Literal["grad-cam"] = "grad-cam"
    note: str


class Timing(BaseModel):
    decode: float
    preprocess: float
    inference: float
    explain: float
    total: float


class ModelInfo(BaseModel):
    architecture: str
    version: str
    trained: bool


class PredictResponse(BaseModel):
    label: Label
    probabilities: Probabilities
    confidence: Probability
    band: Band
    image: ImageInfo
    explanation: Explanation | None
    timing_ms: Timing
    model: ModelInfo


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float


class MetricsResponse(BaseModel):
    available: list[str]
    metrics: dict[str, Any]
