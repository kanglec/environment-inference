from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import dcft.cluster as cluster_module
from dcft.cluster import doctor, render_cluster
from dcft.config import ConfigError, load_config
from dcft.planning import build_plan, read_state, update_task_state, write_plan


def test_presets_have_static_cluster_contract(project_root: Path) -> None:
    for name in (
        "comparison.toml",
        "scaling.toml",
        "acceptance.toml",
        "local-smoke.toml",
        "mc-only-smoke.toml",
    ):
        config = load_config(project_root / "configs" / name)
        assert config.cluster.partition == "day"
        assert "/" in config.cluster.python_module
        assert "/" in config.cluster.rust_module
        assert doctor(config)["checks"]["python_module_matches_lock"]
        assert doctor(config)["checks"]["rust_module_matches_lock"]
        assert config.campaign.project_root != config.campaign.scratch_root
        assert config.mc.inner_budget_multipliers == (1, 2, 4)


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
            'updates = ["metropolis", "corrected-wolff", "tnmc"]',
            'updates = ["metropolis", "metropolis"]',
            "duplicates",
        ),
        (
            'measurements = ["heterodyne", "homodyne", "local-x"]',
            'measurements = ["heterodyne", "gaussian", "local-x"]',
            "gaussian_fractions",
        ),
        ("seed = 42", "seed = -1", "unsigned 64-bit"),
        (
            "diagnostic_outer_records = 4",
            "diagnostic_outer_records = 1",
            "at least two outer records",
        ),
        ("tnmc_bond_dimension = 8", "tnmc_bond_dimension = 0", "positive"),
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


def test_plan_dag_and_resumability(smoke_config_path: Path) -> None:
    config = load_config(smoke_config_path)
    plan = write_plan(config)
    assert plan == build_plan(config)
    clean = next(task for task in plan.tasks if task.kind == "clean")
    mc = next(task for task in plan.tasks if task.kind == "mc")
    analysis = next(task for task in plan.tasks if task.kind == "analysis")
    validation = next(task for task in plan.tasks if task.kind == "validation")
    assert mc.dependencies == (clean.task_id,)
    tnmc = next(
        task for task in plan.tasks if task.kind == "mc" and task.parameters["update"] == "tnmc"
    )
    assert tnmc.parameters["tnmc_bond_dimension"] == config.mc.tnmc_bond_dimension
    assert mc.task_id in analysis.dependencies
    assert analysis.task_id in validation.dependencies
    update_task_state(config, clean.task_id, "complete", artifacts=("abc",))
    assert read_state(config)["tasks"][clean.task_id]["artifacts"] == ["abc"]
    # Planning again preserves resumable state.
    write_plan(config)
    assert read_state(config)["tasks"][clean.task_id]["status"] == "complete"


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


def test_cluster_render_is_static_and_versioned(smoke_config_path: Path, tmp_path: Path) -> None:
    config = load_config(smoke_config_path)
    rendered = render_cluster(config, tmp_path / "cluster")
    script = rendered.script.read_text()
    assert "#SBATCH --partition=day" in script
    assert config.cluster.python_module in script
    assert config.cluster.rust_module in script
    assert "uv sync" not in script
    assert "cargo fetch" not in script
    assert "rsync" not in script
    assert 'RUNTIME_DCFT="$RUNTIME_PROJECT/.venv/bin/dcft"' in script
    assert '"$RUNTIME_DCFT" campaign run' in script
    assert "PROJECT_ROOT" in script and "SCRATCH_ROOT" in script
    environment = rendered.stage_scripts["environment"].read_text()
    assert "uv sync --frozen --no-dev" in environment
    assert "cargo fetch --locked" in environment
    assert "flock 9" in environment
    assert 'chmod -R a-w "$RUNTIME_ROOT"' in environment
    dag = json.loads((rendered.root / "dag.json").read_text())
    compute = [task for task in dag["tasks"] if task["kind"] in {"clean", "mc", "exact"}]
    assert len(compute) == rendered.task_count
    assert dag["stage_dependencies"]["environment"] == []
    assert dag["stage_dependencies"]["clean"] == ["environment"]
    assert dag["stage_dependencies"]["mc"] == ["environment", "clean"]
    assert dag["stage_dependencies"]["analysis"] == ["environment", "mc", "exact"]
    assert dag["source_digest"] in dag["runtime_root"]
    assert set(rendered.stage_scripts) == {
        "environment",
        "clean",
        "exact",
        "mc",
        "analysis",
        "validation",
    }
    for stage, stage_script in rendered.stage_scripts.items():
        subprocess.run(["bash", "-n", str(stage_script)], check=True)
        if stage == "environment":
            continue
        text = stage_script.read_text()
        assert "uv sync" not in text
        assert "cargo fetch" not in text
        assert "rsync" not in text
    result = doctor(config)
    assert result["cluster_qualification"] == "complete"


def test_submission_prepares_one_environment_before_all_tasks(
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
        stage = script.stem
        submissions.append((stage, indices, dependencies or []))
        return f"job-{stage}"

    monkeypatch.setattr(cluster_module, "_submit_script", fake_submit)
    result = cluster_module._submit_dag(config)

    by_stage = {stage: dependencies for stage, _indices, dependencies in submissions}
    assert next(iter(result["jobs"])) == "environment"
    assert by_stage["environment"] == []
    assert by_stage["clean"] == ["job-environment"]
    assert by_stage["exact"] == ["job-environment"]
    assert by_stage["mc"] == ["job-environment", "job-clean"]
    assert by_stage["analysis"] == [
        "job-environment",
        "job-clean",
        "job-exact",
        "job-mc",
    ]
    assert by_stage["validation"] == ["job-environment", "job-analysis"]


def test_cluster_render_supports_user_installed_toolchains(
    smoke_config_path: Path, tmp_path: Path
) -> None:
    config_path = tmp_path / "direct-tools.toml"
    config_path.write_text(
        smoke_config_path.read_text().replace(
            'memory = "2G"', 'memory = "5120"\nuse_modules = false'
        )
    )
    config = load_config(config_path)

    rendered = render_cluster(config, tmp_path / "direct-tools")
    script = rendered.script.read_text()
    doctor_script = rendered.doctor_script.read_text()
    assert not config.cluster.use_modules
    assert "module load" not in script
    assert "module load" not in doctor_script
    assert "${HOME}/.local/bin:${HOME}/.cargo/bin" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --mem-per-cpu=5120" in script
    assert (config.campaign.output_root / "cluster" / "logs").is_dir()
    result = doctor(config)
    assert result["checks"]["direct_tool_mode"]
    assert result["status"] == "local-static-ok"


def test_bouchet_smoke_uses_skill_resource_defaults(project_root: Path) -> None:
    config = load_config(project_root / "configs" / "bouchet-smoke.toml")
    assert config.cluster.account == "pi_mc2832"
    assert config.cluster.partition == "day"
    assert config.cluster.time_limit == "01:00:00"
    assert config.cluster.cpus_per_task == 1
    assert config.cluster.memory == "5120"
    assert not config.cluster.use_modules
    assert config.mc.updates == ("tnmc",)
