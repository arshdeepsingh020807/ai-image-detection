import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, PositiveFloat

from backend.app.core.logging import get_logger
from backend.app.schemas.responses import Band
from detector_core.preprocess import CROP

logger = get_logger("model")

MODEL_FILE = "detector.onnx"
CARD_FILE = "model_card.json"
PRIVATE_CARD_FIELDS = ("cam_weights", "cam_bias")
FEATURE_GRID = 7


class ModelCard(BaseModel):
    """The model_card.json fields serving relies on; the rest only reaches clients via `public_card`."""

    architecture: str
    version: str
    trained: bool
    temperature: PositiveFloat
    band: Band
    cam_weights: list[float]
    cam_bias: float


@dataclass(frozen=True)
class Detector:
    session: ort.InferenceSession
    card: ModelCard
    cam_weights: np.ndarray
    public_card: dict[str, Any]

    def run(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        """(3, 224, 224) normalised image -> (AI-generated logit, (C, 7, 7) feature map)."""
        logit, features = self.session.run(["logit", "features"], {"image": image[None]})
        return float(logit[0, 0]), features[0]


def load_detector(model_dir: Path, threads: int) -> Detector:
    card_raw = json.loads((model_dir / CARD_FILE).read_text("utf-8"))
    card = ModelCard.model_validate(card_raw)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    model_path = model_dir / MODEL_FILE
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    detector = Detector(
        session=session,
        card=card,
        cam_weights=np.asarray(card.cam_weights, dtype=np.float32),
        public_card={
            **{key: value for key, value in card_raw.items() if key not in PRIVATE_CARD_FIELDS},
            "onnx_size_mb": round(model_path.stat().st_size / 2**20, 3),
        },
    )
    # A warm-up pass proves the graph runs and that the card belongs to this model before any request arrives.
    _, features = detector.run(np.zeros((3, CROP, CROP), dtype=np.float32))
    expected = (len(card.cam_weights), FEATURE_GRID, FEATURE_GRID)
    if features.shape != expected:
        raise ValueError(f"feature map {features.shape} does not match the card's {expected}")
    return detector


def try_load_detector(model_dir: Path, threads: int) -> Detector | None:
    """Degraded mode instead of a crash loop: the service stays up and reports why it cannot predict."""
    try:
        detector = load_detector(model_dir, threads)
    except Exception:  # any failure (missing file, corrupt graph, bad card) means the same thing: no model
        logger.exception("model unavailable", extra={"fields": {"model_dir": str(model_dir)}})
        return None
    logger.info("model loaded", extra={"fields": {"version": detector.card.version, "trained": detector.card.trained}})
    return detector
