"""The single ``dcft`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import CLUSTER_QUALIFICATION_STATUS
from .analysis import analyze_campaign
from .campaign import run_campaign
from .cluster import cluster_status, doctor, render_cluster, resume, submit
from .config import CampaignConfig, load_config
from .inspection import inspect_path
from .planning import task_summary, write_plan
from .validation import ValidationError, validate_campaign


def _default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "comparison.toml"


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=_default_config())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dcft", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    campaign = subcommands.add_parser("campaign", help="plan, execute, validate, and analyze")
    campaign_commands = campaign.add_subparsers(dest="campaign_command", required=True)
    for name in ("plan", "validate", "analyze"):
        child = campaign_commands.add_parser(name)
        _add_config(child)
    run = campaign_commands.add_parser("run")
    _add_config(run)
    run.add_argument("--executor", choices=("local",), default="local")
    run.add_argument("--task-id", action="append", default=None)

    cluster = subcommands.add_parser("cluster", help="render and operate Bouchet Slurm jobs")
    cluster_commands = cluster.add_subparsers(dest="cluster_command", required=True)
    render = cluster_commands.add_parser("render")
    _add_config(render)
    render.add_argument("--output", type=Path)
    cluster_doctor = cluster_commands.add_parser("doctor")
    _add_config(cluster_doctor)
    cluster_submit = cluster_commands.add_parser("submit")
    _add_config(cluster_submit)
    status = cluster_commands.add_parser("status")
    status.add_argument("job_id")
    cluster_resume = cluster_commands.add_parser("resume")
    _add_config(cluster_resume)

    inspect = subcommands.add_parser("inspect", help="inspect a campaign or immutable artifact")
    inspect.add_argument("path", type=Path)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _config(arguments: argparse.Namespace) -> CampaignConfig:
    return load_config(arguments.config)


def dispatch(arguments: argparse.Namespace) -> int:
    if arguments.command == "campaign":
        config = _config(arguments)
        if arguments.campaign_command == "plan":
            plan = write_plan(config)
            _print(
                {
                    "campaign": plan.campaign,
                    "config_digest": plan.config_digest,
                    "tasks": len(plan.tasks),
                    "kinds": {
                        kind: sum(task.kind == kind for task in plan.tasks)
                        for kind in sorted({task.kind for task in plan.tasks})
                    },
                    "state": task_summary(config),
                    "plan": str(config.campaign.output_root / "plan.json"),
                }
            )
            return 0
        if arguments.campaign_command == "run":
            _print(
                run_campaign(
                    config,
                    executor=arguments.executor,
                    task_ids=arguments.task_id,
                )
            )
            return 0
        if arguments.campaign_command == "analyze":
            from .plotting import plot_campaign

            result = analyze_campaign(config)
            result["plotting"] = plot_campaign(config)
            _print(result)
            return 0
        if arguments.campaign_command == "validate":
            try:
                result = validate_campaign(config)
            except ValidationError as error:
                _print(
                    {
                        "status": "failed",
                        "error": str(error),
                        "cluster_qualification": CLUSTER_QUALIFICATION_STATUS,
                    }
                )
                return 2
            _print(result)
            return 0
    if arguments.command == "cluster":
        if arguments.cluster_command == "status":
            _print(cluster_status(arguments.job_id))
            return 0
        config = _config(arguments)
        if arguments.cluster_command == "render":
            rendered = render_cluster(config, arguments.output)
            _print(
                {
                    "root": str(rendered.root),
                    "script": str(rendered.script),
                    "task_file": str(rendered.task_file),
                    "doctor": str(rendered.doctor_script),
                    "tasks": rendered.task_count,
                    "cluster_qualification": CLUSTER_QUALIFICATION_STATUS,
                }
            )
            return 0
        if arguments.cluster_command == "doctor":
            result = doctor(config)
            _print(result)
            return 0 if result["status"] != "failed" else 2
        if arguments.cluster_command == "submit":
            _print(submit(config))
            return 0
        if arguments.cluster_command == "resume":
            _print(resume(config))
            return 0
    if arguments.command == "inspect":
        _print(inspect_path(arguments.path))
        return 0
    raise RuntimeError("unreachable command dispatch")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    try:
        status = dispatch(parser.parse_args(argv))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"dcft: error: {error}", file=sys.stderr)
        status = 2
    raise SystemExit(status)


if __name__ == "__main__":
    main()
