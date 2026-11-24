import json
from pathlib import Path
from typing import Any

from backend.app.core.logging import get_logger

logger = get_logger("metrics")


def _reject_non_finite(token: str) -> Any:
    raise ValueError(f"non-finite number {token}")


def read_metrics(directory: Path) -> dict[str, Any]:
    """Every parseable `*.json` in `directory`, keyed by file stem; unreadable files are skipped and logged."""
    metrics: dict[str, Any] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            # NaN/Infinity are not JSON and would break the response, so they count as malformed.
            metrics[path.stem] = json.loads(path.read_text("utf-8"), parse_constant=_reject_non_finite)
        except (OSError, ValueError) as exc:
            logger.warning("skipping metrics file", extra={"fields": {"file": path.name, "error": str(exc)}})
    return {"available": list(metrics), "metrics": metrics}
