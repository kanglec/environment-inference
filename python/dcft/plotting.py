"""Deterministic plotting of analyzed scaling and comparison curves."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import canonical_json, discover_artifacts, read_table
from .config import CampaignConfig
from .planning import read_plan, read_state


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_campaign(config: CampaignConfig) -> dict[str, object]:
    plan = read_plan(config)
    state = read_state(config)
    analysis_task = next(task for task in plan.tasks if task.kind == "analysis")
    current_ids = {
        str(value) for value in state["tasks"][analysis_task.task_id]["artifacts"]
    }
    artifacts = [
        artifact
        for artifact in discover_artifacts(
            config.campaign.output_root, "analysis-curves"
        )
        if artifact.artifact_id in current_ids
    ]
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        rows.extend(cast(list[dict[str, Any]], read_table(artifact).to_pylist()))
    if not rows:
        return {"plots": 0, "files": []}
    output = config.campaign.output_root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["noise"]), str(row["family"]))].append(row)
    files: list[Path] = []
    for (noise, family), values in sorted(grouped.items()):
        figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        series: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            series[(int(row["lx"]), float(row["p"]), str(row["protocol_id"]))].append(row)
        for (lx, p, protocol), points in sorted(series.items(), key=lambda item: repr(item[0])):
            points.sort(key=lambda row: int(row["separation"]))
            axis.plot(
                [row["separation_over_lx"] for row in points],
                [row["witness"] for row in points],
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label=f"L={lx}, p={p:g}, {protocol}",
            )
        axis.set_xlabel("separation / L")
        axis.set_ylabel(f"{family} witness")
        axis.set_title(f"{noise.upper()} noise")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6, ncol=2)
        path = output / f"{noise}-{family}-witness.png"
        figure.savefig(path, dpi=180, metadata={"Software": "dcft-simulation"})
        plt.close(figure)
        files.append(path)
    manifest = {
        "kind": "derived-plots",
        "source_artifacts": [artifact.artifact_id for artifact in artifacts],
        "files": {path.name: _checksum(path) for path in files},
        "regenerable": True,
    }
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    return {"plots": len(files), "files": [str(path) for path in files]}
