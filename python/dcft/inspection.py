"""Human- and machine-readable inspection of campaigns and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .artifacts import discover_artifacts, load_artifact, verify_artifact


def inspect_path(path: Path) -> dict[str, Any]:
    target = path.expanduser().resolve()
    if (target / "manifest.json").is_file():
        artifact = load_artifact(target)
        verification = verify_artifact(target)
        return {"type": "artifact", "path": str(target), "verification": verification, **artifact.manifest}
    if (target / "plan.json").is_file():
        plan = cast(dict[str, Any], json.loads((target / "plan.json").read_text()))
        state = cast(dict[str, Any], json.loads((target / "state.json").read_text()))
        statuses: dict[str, int] = {}
        for task in state["tasks"].values():
            status = str(task["status"])
            statuses[status] = statuses.get(status, 0) + 1
        current_ids = {
            str(artifact_id)
            for task in state["tasks"].values()
            for artifact_id in task["artifacts"]
        }
        discovered = discover_artifacts(target)
        artifacts = [
            artifact for artifact in discovered if artifact.artifact_id in current_ids
        ]
        return {
            "type": "campaign",
            "path": str(target),
            "campaign": plan["campaign"],
            "config_digest": plan["config_digest"],
            "tasks": statuses,
            "unreferenced_immutable_artifacts": len(discovered) - len(artifacts),
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.manifest["kind"],
                    "rows": artifact.manifest["row_count"],
                    "source_digest": artifact.manifest["source_digest"],
                }
                for artifact in artifacts
            ],
        }
    raise FileNotFoundError(f"not a DCFT artifact or campaign directory: {target}")
