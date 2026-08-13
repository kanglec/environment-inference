"""Deterministic campaign DAG construction and resumable task state."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .artifacts import canonical_json
from .config import CampaignConfig, scientific_config
from .registries import resolve_measurements


class PlanError(RuntimeError):
    """The on-disk plan is incompatible with the requested campaign."""


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    dependencies: tuple[str, ...]
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Plan:
    plan_version: int
    campaign: str
    config_digest: str
    tasks: tuple[Task, ...]


def _task(kind: str, dependencies: Iterable[str] = (), **parameters: Any) -> Task:
    payload = {"kind": kind, "parameters": parameters}
    identifier = f"{kind}-{hashlib.sha256(canonical_json(payload)).hexdigest()[:16]}"
    return Task(identifier, kind, tuple(sorted(dependencies)), parameters)


def build_plan(config: CampaignConfig) -> Plan:
    tasks: list[Task] = []
    clean_by_size: dict[int, Task] = {}
    if config.mc.enabled:
        for lx in config.lattice.sizes:
            clean = _task(
                "clean",
                lx=lx,
                lt=config.lattice.lt(lx),
                global_id_start=0,
                global_id_stop=config.mc.outer_records,
            )
            tasks.append(clean)
            clean_by_size[lx] = clean
            for noise in config.protocols.noises:
                for p in config.protocols.p_values:
                    for point in resolve_measurements(
                        config.protocols.measurements,
                        p,
                        config.protocols.gaussian_fractions,
                    ):
                        for update in config.mc.updates:
                            tasks.append(
                                _task(
                                    "mc",
                                    [clean.task_id],
                                    lx=lx,
                                    lt=config.lattice.lt(lx),
                                    noise=noise,
                                    p=p,
                                    measurement=point.name,
                                    protocol_id=point.identifier,
                                    gamma=point.gamma,
                                    update=update,
                                    tnmc_bond_dimension=(
                                        config.mc.tnmc_bond_dimension if update == "tnmc" else None
                                    ),
                                    global_id_start=0,
                                    global_id_stop=config.mc.outer_records,
                                )
                            )
    if config.ed.enabled:
        for lx in config.lattice.sizes:
            if lx <= config.ed.max_sites:
                tasks.append(
                    _task(
                        "exact",
                        lx=lx,
                        lt=config.lattice.lt(lx),
                        noises=list(config.protocols.noises),
                        p_values=list(config.protocols.p_values),
                        priors=list(config.ed.priors),
                    )
                )
    compute = [task.task_id for task in tasks]
    analysis = _task("analysis", compute)
    validation = _task("validation", [*compute, analysis.task_id])
    tasks.extend((analysis, validation))
    config_payload = scientific_config(config)
    digest = hashlib.sha256(canonical_json(config_payload)).hexdigest()
    return Plan(2, config.campaign.name, digest, tuple(tasks))


def plan_path(config: CampaignConfig) -> Path:
    return config.campaign.output_root / "plan.json"


def state_path(config: CampaignConfig) -> Path:
    return config.campaign.output_root / "state.json"


def _plan_dict(plan: Plan) -> dict[str, Any]:
    return {
        "plan_version": plan.plan_version,
        "campaign": plan.campaign,
        "config_digest": plan.config_digest,
        "tasks": [
            {
                "task_id": task.task_id,
                "kind": task.kind,
                "dependencies": list(task.dependencies),
                "parameters": task.parameters,
            }
            for task in plan.tasks
        ],
    }


def write_plan(config: CampaignConfig) -> Plan:
    plan = build_plan(config)
    root = config.campaign.output_root
    root.mkdir(parents=True, exist_ok=True)
    destination = plan_path(config)
    if destination.exists():
        existing = cast(dict[str, Any], json.loads(destination.read_text()))
        if existing != _plan_dict(plan):
            raise PlanError(
                f"existing plan at {destination} was produced by a different configuration"
            )
    else:
        destination.write_bytes(canonical_json(_plan_dict(plan)))
        (root / "config.snapshot.json").write_bytes(canonical_json(scientific_config(config)))
    state = state_path(config)
    if not state.exists():
        initial = {
            "plan_version": plan.plan_version,
            "config_digest": plan.config_digest,
            "tasks": {
                task.task_id: {"status": "pending", "artifacts": [], "error": None}
                for task in plan.tasks
            },
        }
        state.write_bytes(canonical_json(initial))
    return plan


def read_plan(config: CampaignConfig) -> Plan:
    expected = build_plan(config)
    source = plan_path(config)
    if not source.is_file():
        return write_plan(config)
    raw = cast(dict[str, Any], json.loads(source.read_text()))
    tasks = tuple(
        Task(
            task_id=str(item["task_id"]),
            kind=str(item["kind"]),
            dependencies=tuple(item["dependencies"]),
            parameters=cast(dict[str, Any], item["parameters"]),
        )
        for item in raw["tasks"]
    )
    actual = Plan(
        plan_version=int(raw["plan_version"]),
        campaign=str(raw["campaign"]),
        config_digest=str(raw["config_digest"]),
        tasks=tasks,
    )
    if _plan_dict(actual) != _plan_dict(expected):
        raise PlanError("on-disk plan differs from the deterministic plan for this configuration")
    return actual


def read_state(config: CampaignConfig) -> dict[str, Any]:
    write_plan(config)
    return cast(dict[str, Any], json.loads(state_path(config).read_text()))


def update_task_state(
    config: CampaignConfig,
    task_id: str,
    status: str,
    *,
    artifacts: Iterable[str] = (),
    error: str | None = None,
) -> None:
    if status not in {"pending", "running", "complete", "failed"}:
        raise ValueError(f"invalid task status {status!r}")
    write_plan(config)
    destination = state_path(config)
    lock_path = destination.with_suffix(".lock")
    # Slurm array workers update one shared operational state file.  Hold an
    # advisory lock across the complete read/modify/replace transaction so a
    # worker can never erase a sibling's completion record.
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            state = cast(dict[str, Any], json.loads(destination.read_text()))
            if state.get("config_digest") != build_plan(config).config_digest:
                raise PlanError("state file belongs to a different campaign configuration")
            tasks = cast(dict[str, dict[str, Any]], state["tasks"])
            if task_id not in tasks:
                raise PlanError(f"unknown task id {task_id}")
            tasks[task_id] = {
                "status": status,
                "artifacts": list(artifacts),
                "error": error,
            }
            temporary = destination.with_suffix(f".tmp-{os.getpid()}-{threading.get_ident()}")
            with temporary.open("wb") as handle:
                handle.write(canonical_json(state))
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def task_summary(config: CampaignConfig) -> dict[str, int]:
    state = read_state(config)
    counts = {"pending": 0, "running": 0, "complete": 0, "failed": 0}
    for item in cast(dict[str, dict[str, Any]], state["tasks"]).values():
        counts[str(item["status"])] += 1
    return counts
