"""Fill the generated blocks of README.md from artifacts/metrics/*.json (never hand-typed numbers).

Blocks look like `<!-- generated:NAME -->` ... `<!-- /generated:NAME -->`. Run after evaluation;
`--check` exits non-zero when README.md is out of date (used by the test suite).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

NOT_YET = "_Not available yet: this section is generated after the evaluation script has run._"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def dataset_block(m: dict) -> str | None:
    report = m.get("dataset_report")
    if not report:
        return None
    sources = [
        [name, f"{s['real']:,}", f"{s['ai_generated']:,}", s["domain"], s["role"].replace("_", " ")]
        for name, s in report["by_source"].items()
    ]
    splits = [[split, f"{v['real']:,}", f"{v['ai_generated']:,}"] for split, v in report["splits"].items()]
    removed = ", ".join(
        f"{reason.replace('_', ' ')} {count:,}" for reason, count in report["removed_by_reason"].items()
    )
    leaks = report["leakage_check"]
    return "\n\n".join(
        [
            f"{report['original_images']:,} images audited, {report['cleaned_images']:,} kept, "
            f"{report['removed_images']:,} removed ({removed}). Leakage check across splits: "
            f"{leaks['exact_duplicate_leaks']} exact and {leaks['near_duplicate_leaks']} near-duplicate leaks.",
            _table(["Source", "Real", "AI-generated", "Domain", "Role"], sources),
            _table(["Split", "Real", "AI-generated"], splits),
        ]
    )


def results_block(m: dict) -> str | None:
    summary = m.get("summary")
    if not summary:
        return None
    test, band = summary["test"], summary["test_with_band"]
    rows = [
        [k.replace("_", " "), _pct(test[k])]
        for k in ("accuracy", "precision", "recall", "f1", "specificity", "false_positive_rate", "false_negative_rate")
    ]
    rows += [
        ["ROC-AUC", _num(test["roc_auc"])],
        ["PR-AUC", _num(test["pr_auc"])],
        ["log loss", _num(test["log_loss"])],
        ["ECE (calibration error)", _num(test["ece"])],
    ]
    tn, fp = test["confusion_matrix"][0]
    fn, tp = test["confusion_matrix"][1]
    return "\n\n".join(
        [
            f"Model `{summary['model']}`, held-out test split (n = {test['n']:,}), calibrated with temperature "
            f"{summary['temperature']:.3f} fitted on validation.",
            _table(["Metric", "Value"], rows),
            _table(
                ["", "Predicted real", "Predicted AI"],
                [["Actually real", f"{tn:,}", f"{fp:,}"], ["Actually AI", f"{fn:,}", f"{tp:,}"]],
            ),
            f"With the inconclusive band ({_pct(summary['band']['low'])} to {_pct(summary['band']['high'])} "
            f"probability of AI), the model gives a verdict on {_pct(band['coverage'])} of test images, "
            f"{_pct(band['accuracy_on_decided'])} of which are correct; {band['inconclusive']:,} are inconclusive.",
        ]
    )


def external_block(m: dict) -> str | None:
    external = m.get("external")
    if not external:
        return None
    rows = [
        [s["name"], s["protocol"], f"{s['n']:,}", _pct(s["metrics"]["accuracy"]), _num(s["metrics"]["roc_auc"])]
        for s in external["sets"]
    ]
    notes = "\n".join(f"- {note}" for note in external["notes"])
    return _table(["Set", "Protocol", "n", "Accuracy", "ROC-AUC"], rows) + (f"\n\n{notes}" if notes else "")


def robustness_block(m: dict) -> str | None:
    robustness = m.get("robustness")
    if not robustness:
        return None
    rows = [
        [c["description"], _pct(c["accuracy"]), _num(c["roc_auc"]), _pct(c["false_positive_rate"])]
        for c in robustness["conditions"]
    ]
    return _table(["Condition", "Accuracy", "ROC-AUC", "False-positive rate"], rows)


def comparison_block(m: dict) -> str | None:
    comparison = m.get("model_comparison")
    if not comparison:
        return None
    rows = [
        [
            c["name"],
            f"{c['parameters_millions']:.1f} M",
            f"{c['onnx_size_mb']:.1f} MB",
            _num(c["validation_auc"]),
            _num(c["external_auc"]),
            f"{c['latency_ms']:.0f} ms",
        ]
        for c in comparison["models"]
    ]
    return _table(["Model", "Parameters", "ONNX size", "Validation AUC", "Unseen-generator AUC", "CPU latency"], rows)


def latency_block(m: dict) -> str | None:
    latency = m.get("latency")
    if not latency:
        return None
    cpu = latency["cpu"]
    return (
        f"ONNX model {latency['onnx_size_mb']:.1f} MB; single-image CPU inference p50 {cpu['p50_ms']:.1f} ms, "
        f"p95 {cpu['p95_ms']:.1f} ms with {cpu['threads']} thread ({latency['machine']}, {latency['runs']} runs)."
    )


BLOCKS: dict[str, Callable[[dict], str | None]] = {
    "dataset": dataset_block,
    "results": results_block,
    "external": external_block,
    "robustness": robustness_block,
    "comparison": comparison_block,
    "latency": latency_block,
}


def load_metrics(metrics_dir: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(metrics_dir.glob("*.json"))}


def render(readme: str, metrics: dict) -> str:
    for name, build in BLOCKS.items():
        pattern = re.compile(rf"(<!-- generated:{name} -->\n).*?(\n<!-- /generated:{name} -->)", re.S)
        if not pattern.search(readme):
            raise ValueError(f"README is missing the generated:{name} block")
        body = build(metrics) or NOT_YET
        readme = pattern.sub(lambda match, b=body: match.group(1) + b + match.group(2), readme)
    return readme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/metrics"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = args.readme.read_text()
    updated = render(current, load_metrics(args.metrics))
    if args.check:
        sys.exit(0 if updated == current else "README.md is out of date: run python -m ml.scripts.update_readme")
    args.readme.write_text(updated)


if __name__ == "__main__":
    main()
