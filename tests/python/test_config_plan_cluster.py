from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import dcft.cluster as cluster_module
from dcft.cluster import doctor, render_cluster
from dcft.config import ConfigError, load_config
from dcft.planning import build_plan, chunk_ranges, read_state, update_task_state, write_plan


def test_presets_use_parallel_execution_without_module_configuration(project_root: Path) -> None:
    for name in (
        "comparison.toml",
        "scaling.toml",
        "acceptance.toml",
        "local-smoke.toml",
        "mc-only-smoke.toml",
    ):
        config = load_config(project_root / "configs" / name)
        assert config.schema_version == 2
        assert config.execution.mc_chunks > 0
        assert config.execution.local_workers > 1
        assert config.cluster.cpus_per_task > 1
        assert not hasattr(config.cluster, "max_array_concurrency")
        assert not hasattr(config.cluster, "use_modules")
        assert config.campaign.project_root != config.campaign.scratch_root


def test_project_and_scratch_separation_is_enforced(
    smoke_config_path: Path, tmp_path: Path
) -> None:
    text = smoke_config_path.read_text()
    project_line = next(line for line in text.splitlines() if line.startswith("project_root"))
    text = text.replace(
        next(line for line in text.splitlines() if line.startswith("scratch_root")),
        project_line.replace("project_root", "scratch_root"),
    )
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(text)
    with pytest.raises(ConfigError, match="project_root and scratch_root"):
        load_config(invalid)


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (
            'updates = ["metropolis", "corrected-wolff", "tnmc", "tnmc-global"]',
            'updates = ["metropolis", "metropolis"]',
            "duplicates",
        ),
        (
            'measurements = ["heterodyne", "homodyne", "local-x"]',
            'measurements = ["heterodyne", "gaussian", "local-x"]',
            "gaussian_fractions",
        ),
        ("seed = 42", "seed = -1", "unsigned 64-bit"),
        ("diagnostic_outer_records = 4", "diagnostic_outer_records = 1", "at least two"),
        ("tnmc_bond_dimension = 8", "tnmc_bond_dimension = 0", "positive"),
        ("mc_chunks = 2", "mc_chunks = 17", "cannot exceed"),
        ("local_workers = 4", "local_workers = 0", "positive"),
        ('separations = "all"', "separations = [0, 1]", "maximum physical distance"),
    ],
)
def test_strict_configuration_rejects_ambiguous_campaigns(
    smoke_config_path: Path,
    tmp_path: Path,
    before: str,
    after: str,
    message: str,
) -> None:
    invalid = tmp_path / f"invalid-{message.replace(' ', '-')}.toml"
    invalid.write_text(smoke_config_path.read_text().replace(before, after))
    with pytest.raises(ConfigError, match=message):
        load_config(invalid)


def test_chunk_ranges_are_balanced_and_complete() -> None:
    assert chunk_ranges(10, 3) == ((0, 4), (4, 7), (7, 10))
    with pytest.raises(ValueError, match="chunks"):
        chunk_ranges(2, 3)


def test_scaling_separations_are_canonical_and_include_half_system(
    project_root: Path,
) -> None:
    config = load_config(project_root / "configs" / "scaling.toml")
    expected = {
        16: (0, 1, 2, 4, 8),
        24: (0, 1, 2, 4, 8, 12),
        32: (0, 1, 2, 4, 8, 12, 16),
        48: (0, 1, 2, 4, 8, 12, 16, 24),
        64: (0, 1, 2, 4, 8, 12, 16, 24, 32),
    }
    assert {lx: config.separations_for(lx) for lx in config.lattice.sizes} == expected


def test_plan_chunks_mc_and_merges_each_parameter_point(smoke_config_path: Path) -> None:
    config = load_config(smoke_config_path)
    plan = write_plan(config)
    clean = next(task for task in plan.tasks if task.kind == "clean")
    merge = next(task for task in plan.tasks if task.kind == "merge")
    chunks = [task for task in plan.tasks if task.task_id in merge.dependencies]
    analysis = next(task for task in plan.tasks if task.kind == "analysis")
    validation = next(task for task in plan.tasks if task.kind == "validation")
    assert len(chunks) == config.execution.mc_chunks
    assert all(task.kind == "mc" and task.dependencies == (clean.task_id,) for task in chunks)
    assert [
        (task.parameters["global_id_start"], task.parameters["global_id_stop"]) for task in chunks
    ] == [
        (0, 8),
        (8, 16),
    ]
    assert merge.task_id in analysis.dependencies
    assert clean.task_id in analysis.dependencies
    assert all(task.task_id not in analysis.dependencies for task in chunks)
    assert validation.dependencies == (analysis.task_id,)
    tnmc_tasks = [
        task
        for task in plan.tasks
        if task.kind in {"mc", "merge"} and task.parameters["update"].startswith("tnmc")
    ]
    assert {task.parameters["update"] for task in tnmc_tasks} == {"tnmc", "tnmc-global"}
    assert all(
        task.parameters["tnmc_bond_dimension"] == config.mc.tnmc_bond_dimension
        for task in tnmc_tasks
    )
    update_task_state(config, clean.task_id, "complete", artifacts=("abc",))
    assert read_state(config)["tasks"][clean.task_id]["artifacts"] == ["abc"]
    write_plan(config)
    assert read_state(config)["tasks"][clean.task_id]["status"] == "complete"
    snapshot = json.loads((config.campaign.output_root / "config.snapshot.json").read_text())
    assert "cluster" not in snapshot
    assert "execution" not in snapshot


def test_scientific_identity_ignores_paths_and_execution_shape(
    smoke_config_path: Path, tmp_path: Path
) -> None:
    baseline = load_config(smoke_config_path)
    changed_path = tmp_path / "operational-shape.toml"
    changed_path.write_text(
        smoke_config_path.read_text()
        .replace(
            f'output_root = "{baseline.campaign.output_root}"',
            f'output_root = "{tmp_path / "alternate"}"',
        )
        .replace(
            f'scratch_root = "{baseline.campaign.scratch_root}"',
            f'scratch_root = "{tmp_path / "scratch-alternate"}"',
        )
        .replace("mc_chunks = 2", "mc_chunks = 1")
        .replace("local_workers = 4", "local_workers = 2")
        .replace("cpus_per_task = 4", "cpus_per_task = 16")
    )
    changed = load_config(changed_path)
    assert build_plan(changed).config_digest == build_plan(baseline).config_digest
    assert build_plan(changed).tasks != build_plan(baseline).tasks

    scientific_path = tmp_path / "scientific-change.toml"
    scientific_path.write_text(
        smoke_config_path.read_text()
        .replace(
            f'output_root = "{baseline.campaign.output_root}"',
            f'output_root = "{tmp_path / "scientific"}"',
        )
        .replace("p_values = [0.1]", "p_values = [0.2]")
    )
    scientific = load_config(scientific_path)
    assert build_plan(scientific).config_digest != build_plan(baseline).config_digest


def test_parallel_state_updates_do_not_lose_sibling_completions(
    smoke_config_path: Path,
) -> None:
    config = load_config(smoke_config_path)
    plan = write_plan(config)
    task_ids = [task.task_id for task in plan.tasks[:12]]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                update_task_state,
                config,
                task_id,
                "complete",
                artifacts=(f"artifact-{index}",),
            )
            for index, task_id in enumerate(task_ids)
        ]
        for future in futures:
            future.result()
    state = read_state(config)
    for index, task_id in enumerate(task_ids):
        assert state["tasks"][task_id] == {
            "status": "complete",
            "artifacts": [f"artifact-{index}"],
            "error": None,
        }


def test_cluster_render_runs_prepared_checkout_in_place(
    smoke_config_path: Path, tmp_path: Path
) -> None:
    config = load_config(smoke_config_path)
    rendered = render_cluster(config, tmp_path / "cluster")
    script = rendered.script.read_text()
    assert "#SBATCH --array=" in script
    assert "%" not in next(line for line in script.splitlines() if "--array=" in line)
    assert f"#SBATCH --cpus-per-task={config.cluster.cpus_per_task}" in script
    assert "#SBATCH --mem=2G" in script
    assert "--mem-per-cpu" not in script
    assert 'DCFT="$PROJECT_ROOT/.venv/bin/dcft"' in script
    assert '--workers "$SLURM_CPUS_PER_TASK"' in script
    assert "module load" not in script
    assert "uv sync" not in script
    assert "cargo fetch" not in script
    assert "rsync" not in script
    assert "flock" not in script
    assert "environment" not in rendered.stage_scripts
    assert set(rendered.stage_scripts) == {
        "clean",
        "exact",
        "mc",
        "merge",
        "analysis",
        "validation",
    }
    dag = json.loads((rendered.root / "dag.json").read_text())
    assert dag["execution"]["checkout_mode"] == "in-place"
    assert dag["stage_dependencies"]["mc"] == ["clean"]
    assert dag["stage_dependencies"]["merge"] == ["mc"]
    assert dag["stage_dependencies"]["validation"] == ["analysis"]
    for stage_script in rendered.stage_scripts.values():
        subprocess.run(["bash", "-n", str(stage_script)], check=True)
    assert doctor(config)["status"] == "ready"


def test_submission_uses_chunk_merge_dag(
    smoke_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(smoke_config_path)
    submissions: list[tuple[str, list[int] | None, list[str]]] = []

    def fake_submit(
        script: Path,
        *,
        indices: list[int] | None = None,
        dependencies: list[str] | None = None,
    ) -> str:
        submissions.append((script.stem, indices, dependencies or []))
        return f"job-{script.stem}"

    monkeypatch.setattr(cluster_module, "_submit_script", fake_submit)
    result = cluster_module._submit_dag(config)
    by_stage = {stage: dependencies for stage, _indices, dependencies in submissions}
    assert list(result["jobs"]) == ["clean", "exact", "mc", "merge", "analysis", "validation"]
    assert by_stage["clean"] == []
    assert by_stage["exact"] == []
    assert by_stage["mc"] == ["job-clean"]
    assert by_stage["merge"] == ["job-mc"]
    assert by_stage["analysis"] == ["job-clean", "job-exact", "job-mc", "job-merge"]
    assert by_stage["validation"] == ["job-analysis"]
    assert result["resource_shape"]["mc_chunks_per_point"] == 2


def test_subset_submission_has_no_program_imposed_array_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    submitted: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        submitted.extend(command)

        class Result:
            stdout = "12345\n"

        return Result()

    monkeypatch.setattr(cluster_module.subprocess, "run", fake_run)
    script = tmp_path / "mc.sbatch"
    script.write_text("#!/usr/bin/env bash\n")
    job_id = cluster_module._submit_script(script, indices=[0, 3, 7])
    assert job_id == "12345"
    assert "--array=0,3,7" in submitted
    assert all("%" not in argument for argument in submitted)


def test_configuration_has_no_aggregate_cpu_ceiling(
    smoke_config_path: Path, tmp_path: Path
) -> None:
    large = tmp_path / "large.toml"
    large.write_text(
        smoke_config_path.read_text().replace("cpus_per_task = 4", "cpus_per_task = 64")
    )
    config = load_config(large)
    assert config.cluster.cpus_per_task == 64
    assert not hasattr(config.cluster, "max_array_concurrency")
