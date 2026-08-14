from __future__ import annotations

import json
from pathlib import Path

import pytest

from dcft.artifacts import write_artifact
from dcft.cli import dispatch


def test_cli_plan_and_render(smoke_config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from dcft.cli import build_parser

    parser = build_parser()
    assert (
        dispatch(parser.parse_args(["campaign", "plan", "--config", str(smoke_config_path)])) == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["kinds"]["clean"] == 1
    assert plan["kinds"]["merge"] > 0
    assert (
        dispatch(parser.parse_args(["cluster", "render", "--config", str(smoke_config_path)])) == 0
    )
    render = json.loads(capsys.readouterr().out)
    assert render["tasks"] > plan["kinds"]["mc"]


def test_cli_update_benchmark_covers_speed_mixing_and_thermalization(
    smoke_config_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dcft.cli import build_parser

    destination = tmp_path / "benchmark.json"
    arguments = [
        "benchmark",
        "updates",
        "--config",
        str(smoke_config_path),
        "--update",
        "metropolis",
        "--warmup-sweeps",
        "1",
        "--speed-sweeps",
        "2",
        "--probes",
        "4",
        "--thermalization-sweeps",
        "1",
        "--thermalization-measurements",
        "4",
        "--chains",
        "2",
        "--workers",
        "2",
        "--output",
        str(destination),
    ]
    assert dispatch(build_parser().parse_args(arguments)) == 0
    report = json.loads(capsys.readouterr().out)
    method = report["methods"][0]
    assert method["sweep_speed"]["sweeps_per_second"] > 0.0
    assert method["autocorrelation"]["planted_overlap"]["tau_sweeps"] >= 0.5
    assert method["thermalization"]["chains"] == 2
    assert destination.is_file()


def test_inspect_campaign_reports_only_state_referenced_artifacts(
    smoke_config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import pyarrow as pa

    from dcft.cli import build_parser
    from dcft.config import load_config

    config = load_config(smoke_config_path)
    dispatch(build_parser().parse_args(["campaign", "plan", "--config", str(config.source)]))
    capsys.readouterr()
    write_artifact(
        config.campaign.output_root,
        "orphan",
        pa.Table.from_pylist([{"value": 1}]),
        metadata={"purpose": "inspection-test"},
        project_root=config.campaign.project_root,
    )
    assert dispatch(build_parser().parse_args(["inspect", str(config.campaign.output_root)])) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["artifacts"] == []
    assert result["unreferenced_immutable_artifacts"] == 1
