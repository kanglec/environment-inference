"""Bouchet Slurm rendering and qualification helpers.

Rendering is local and side-effect free beyond generated files. Submission is
performed only by the explicit ``dcft cluster submit`` or ``resume`` command.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from . import CLUSTER_QUALIFICATION_STATUS, QUALIFICATION_STATUS
from .artifacts import canonical_json, discover_artifacts, source_digest, verify_artifact
from .config import CampaignConfig, as_serializable
from .planning import Plan, Task, read_plan, read_state, write_plan


class ClusterError(RuntimeError):
    """Cluster scripts or local prerequisites are invalid."""


@dataclass(frozen=True)
class RenderedCluster:
    root: Path
    script: Path
    task_file: Path
    doctor_script: Path
    task_count: int
    stage_scripts: dict[str, Path]
    stage_task_files: dict[str, Path]


def _shell(value: str | Path) -> str:
    return shlex.quote(str(value))


def _locked_toolchain_versions(config: CampaignConfig) -> tuple[str, str]:
    python_version = (config.campaign.project_root / ".python-version").read_text().strip()
    rust_document = tomllib.loads(
        (config.campaign.project_root / "rust-toolchain.toml").read_text()
    )
    rust_version = str(cast(dict[str, Any], rust_document["toolchain"])["channel"])
    return python_version, rust_version


def _module_matches(module: str, version: str, *, allow_patch: bool = False) -> bool:
    advertised = module.partition("/")[2]
    if not advertised:
        return False
    if advertised == version or advertised.startswith(f"{version}-"):
        return True
    return allow_patch and advertised.startswith(f"{version}.")


def _toolchain_setup(config: CampaignConfig) -> str:
    if config.cluster.use_modules:
        return f"""module purge
module load {_shell(config.cluster.python_module)}
module load {_shell(config.cluster.rust_module)}
"""
    return 'export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"\n'


def _runtime_root(config: CampaignConfig, digest: str) -> Path:
    return config.campaign.scratch_root / "environments" / digest


def _task_bootstrap(config: CampaignConfig, *, array: bool, digest: str) -> str:
    task_suffix = "${SLURM_ARRAY_TASK_ID}" if array else "singleton"
    digest_check = (
        "from pathlib import Path; from dcft.artifacts import source_digest; "
        f"print(source_digest(Path({str(config.campaign.project_root)!r})))"
    )
    return f"""set -euo pipefail
{_toolchain_setup(config)}

PROJECT_ROOT={_shell(config.campaign.project_root)}
SCRATCH_ROOT={_shell(config.campaign.scratch_root)}
CONFIG_PATH={_shell(config.source)}
RUNTIME_ROOT={_shell(_runtime_root(config, digest))}
RUNTIME_PROJECT="$RUNTIME_ROOT/project"
RUNTIME_PYTHON="$RUNTIME_PROJECT/.venv/bin/python"
RUNTIME_DCFT="$RUNTIME_PROJECT/.venv/bin/dcft"
TASK_SCRATCH="${{SCRATCH_ROOT}}/${{SLURM_JOB_ID}}/{task_suffix}"

test "$PROJECT_ROOT" != "$SCRATCH_ROOT"
mkdir -p "$TASK_SCRATCH" {_shell(config.campaign.output_root / "cluster" / "logs")}
test -x "$RUNTIME_PYTHON"
test -x "$RUNTIME_DCFT"
grep -Fx {_shell(digest)} "$RUNTIME_ROOT/.dcft-runtime-ready" >/dev/null
ACTUAL_SOURCE_DIGEST=$("$RUNTIME_PYTHON" -c {_shell(digest_check)})
test "$ACTUAL_SOURCE_DIGEST" = {_shell(digest)}
export VIRTUAL_ENV="$RUNTIME_PROJECT/.venv"
export PATH="$VIRTUAL_ENV/bin:${{PATH}}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TASK_SCRATCH/matplotlib"
export RAYON_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export DCFT_TASK_SCRATCH="$TASK_SCRATCH"
cd "$TASK_SCRATCH"
"""


def _write_environment_stage(config: CampaignConfig, root: Path, digest: str) -> Path:
    runtime_root = _runtime_root(config, digest)
    runtime_check = (
        "from pathlib import Path; from dcft.artifacts import source_digest; "
        "from dcft import _core; "
        f"assert source_digest(Path({str(runtime_root / 'project')!r})) == {digest!r}; "
        "assert 'tnmc' in _core.update_registry()"
    )
    script = root / "environment.sbatch"
    script.write_text(
        _header(config, "environment", array_size=None)
        + f"""set -euo pipefail
{_toolchain_setup(config)}

PROJECT_ROOT={_shell(config.campaign.project_root)}
RUNTIME_ROOT={_shell(runtime_root)}
RUNTIME_PROJECT="$RUNTIME_ROOT/project"
RUNTIME_ENV="$RUNTIME_PROJECT/.venv"
READY_MARKER="$RUNTIME_ROOT/.dcft-runtime-ready"
LOCK_FILE="${{RUNTIME_ROOT}}.lock"

mkdir -p "$(dirname "$RUNTIME_ROOT")" {_shell(config.campaign.output_root / "cluster" / "logs")}
exec 9>"$LOCK_FILE"
flock 9
if test -x "$RUNTIME_ENV/bin/dcft" && grep -Fx {_shell(digest)} "$READY_MARKER" >/dev/null 2>&1; then
  "$RUNTIME_ENV/bin/python" -c {_shell(runtime_check)}
  echo "Reusing prepared DCFT environment $RUNTIME_ROOT"
  exit 0
fi

mkdir -p "$RUNTIME_PROJECT"
rsync -a --delete \
  --exclude archive --exclude artifacts --exclude scratch --exclude target --exclude .venv \
  "$PROJECT_ROOT/" "$RUNTIME_PROJECT/"
chmod -R u+w "$RUNTIME_PROJECT"
cd "$RUNTIME_PROJECT"
unset VIRTUAL_ENV CONDA_PREFIX PYO3_PYTHON
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ENV"
export CARGO_TARGET_DIR="$RUNTIME_PROJECT/target"
export UV_LINK_MODE=copy
cargo fetch --locked
uv sync --frozen --no-dev
"$RUNTIME_ENV/bin/python" -c {_shell(runtime_check)}
printf '%s\n' {_shell(digest)} > "$READY_MARKER"
chmod -R a-w "$RUNTIME_ROOT"
echo "Prepared immutable DCFT environment $RUNTIME_ROOT"
"""
    )
    script.chmod(0o755)
    return script


def _header(config: CampaignConfig, stage: str, *, array_size: int | None) -> str:
    array_line = (
        f"#SBATCH --array=0-{array_size - 1}%{config.cluster.max_array_concurrency}\n"
        if array_size is not None
        else ""
    )
    log_suffix = "%A_%a" if array_size is not None else "%j"
    return f"""#!/usr/bin/env bash
# Generated by dcft; qualification status: {QUALIFICATION_STATUS}
# Stage dependencies are submitted with Slurm afterok constraints.
#SBATCH --job-name=dcft-{config.campaign.name}-{stage}
#SBATCH --partition={config.cluster.partition}
#SBATCH --account={config.cluster.account}
#SBATCH --time={config.cluster.time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.cluster.cpus_per_task}
#SBATCH --mem-per-cpu={config.cluster.memory}
{array_line}#SBATCH --output={config.campaign.output_root}/cluster/logs/{stage}_{log_suffix}.out
#SBATCH --error={config.campaign.output_root}/cluster/logs/{stage}_{log_suffix}.err

"""


def _write_array_stage(
    config: CampaignConfig,
    root: Path,
    stage: str,
    tasks: list[Task],
    digest: str,
) -> tuple[Path, Path]:
    task_file = root / f"{stage}-tasks.txt"
    task_file.write_text("".join(f"{task.task_id}\n" for task in tasks))
    script = root / f"{stage}.sbatch"
    script.write_text(
        _header(config, stage, array_size=len(tasks))
        + _task_bootstrap(config, array=True, digest=digest)
        + f"""TASK_FILE={_shell(task_file)}
TASK_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
test -n "$TASK_ID"
"$RUNTIME_DCFT" campaign run --config "$CONFIG_PATH" --executor local --task-id "$TASK_ID"
"""
    )
    script.chmod(0o755)
    return script, task_file


def _write_singleton_stage(
    config: CampaignConfig,
    root: Path,
    stage: str,
    command: str,
    digest: str,
) -> Path:
    script = root / f"{stage}.sbatch"
    script.write_text(
        _header(config, stage, array_size=None)
        + _task_bootstrap(config, array=False, digest=digest)
        + f'"$RUNTIME_DCFT" campaign {command} --config "$CONFIG_PATH"\n'
    )
    script.chmod(0o755)
    return script


def render_cluster(config: CampaignConfig, output: Path | None = None) -> RenderedCluster:
    plan = write_plan(config)
    digest = source_digest(config.campaign.project_root)
    python_version, rust_version = _locked_toolchain_versions(config)
    root = (
        output.expanduser().resolve()
        if output is not None
        else (config.campaign.output_root / "cluster" / "rendered").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    (config.campaign.output_root / "cluster" / "logs").mkdir(parents=True, exist_ok=True)
    stages = {
        kind: [task for task in plan.tasks if task.kind == kind]
        for kind in ("clean", "exact", "mc")
    }
    stage_scripts: dict[str, Path] = {}
    stage_task_files: dict[str, Path] = {}
    stage_scripts["environment"] = _write_environment_stage(config, root, digest)
    for stage, tasks in stages.items():
        if tasks:
            script, task_file = _write_array_stage(config, root, stage, tasks, digest)
            stage_scripts[stage] = script
            stage_task_files[stage] = task_file
    stage_scripts["analysis"] = _write_singleton_stage(config, root, "analysis", "analyze", digest)
    stage_scripts["validation"] = _write_singleton_stage(
        config, root, "validation", "validate", digest
    )

    compute = [task for task in plan.tasks if task.kind in {"clean", "mc", "exact"}]
    (root / "dag.json").write_bytes(
        canonical_json(
            {
                "plan_version": plan.plan_version,
                "config_digest": plan.config_digest,
                "source_digest": digest,
                "runtime_root": str(_runtime_root(config, digest)),
                "execution": {
                    "config_path": str(config.source),
                    "output_root": str(config.campaign.output_root),
                    "project_root": str(config.campaign.project_root),
                    "scratch_root": str(config.campaign.scratch_root),
                    "cluster": as_serializable(config)["cluster"],
                },
                "stage_dependencies": {
                    "environment": [],
                    "clean": ["environment"],
                    "exact": ["environment"],
                    "mc": ["environment", "clean"],
                    "analysis": ["environment", "mc", "exact"],
                    "validation": ["environment", "analysis"],
                },
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "kind": task.kind,
                        "dependencies": list(task.dependencies),
                    }
                    for task in plan.tasks
                ],
            }
        )
    )
    doctor_script = root / "doctor.sh"
    doctor_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
{_toolchain_setup(config)}
test -f {_shell(config.campaign.project_root / "uv.lock")}
test -f {_shell(config.campaign.project_root / "Cargo.lock")}
test {_shell(config.campaign.project_root)} != {_shell(config.campaign.scratch_root)}
command -v uv
command -v cargo
command -v sbatch
command -v rsync
command -v flock
uv --version
RUSTUP_TOOLCHAIN=stable cargo --version
grep -Fx {_shell(python_version)} {_shell(config.campaign.project_root / ".python-version")} >/dev/null
grep -F {_shell(f'channel = "{rust_version}"')} {_shell(config.campaign.project_root / "rust-toolchain.toml")} >/dev/null
echo "Bouchet static doctor checks passed; locked toolchains will be provisioned by Slurm."
"""
    )
    doctor_script.chmod(0o755)
    legacy_stage = "mc" if "mc" in stage_scripts else next(iter(stage_task_files))
    return RenderedCluster(
        root=root,
        script=stage_scripts[legacy_stage],
        task_file=stage_task_files[legacy_stage],
        doctor_script=doctor_script,
        task_count=len(compute),
        stage_scripts=stage_scripts,
        stage_task_files=stage_task_files,
    )


def doctor(config: CampaignConfig) -> dict[str, Any]:
    """Run local/static checks only; this does not contact Bouchet."""
    python_version, rust_version = _locked_toolchain_versions(config)
    checks = {
        "uv_available": shutil.which("uv") is not None,
        "cargo_available": shutil.which("cargo") is not None,
        "slurm_client_available": shutil.which("sbatch") is not None,
        "rsync_available": shutil.which("rsync") is not None,
        "flock_available": shutil.which("flock") is not None,
        "uv_lock_committed": (config.campaign.project_root / "uv.lock").is_file(),
        "cargo_lock_committed": (config.campaign.project_root / "Cargo.lock").is_file(),
        "project_scratch_separate": (
            config.campaign.project_root.resolve() != config.campaign.scratch_root.resolve()
        ),
        "partition_is_day": config.cluster.partition == "day",
        "module_mode": config.cluster.use_modules,
        "direct_tool_mode": not config.cluster.use_modules,
        "python_module_versioned": (
            not config.cluster.use_modules or "/" in config.cluster.python_module
        ),
        "rust_module_versioned": (
            not config.cluster.use_modules or "/" in config.cluster.rust_module
        ),
        "python_module_matches_lock": (
            not config.cluster.use_modules
            or _module_matches(config.cluster.python_module, python_version, allow_patch=True)
        ),
        "rust_module_matches_lock": (
            not config.cluster.use_modules
            or _module_matches(config.cluster.rust_module, rust_version)
        ),
        "account_configured": not config.cluster.account.startswith("SET_"),
    }
    local_required = (
        "uv_available",
        "cargo_available",
        "uv_lock_committed",
        "cargo_lock_committed",
        "project_scratch_separate",
        "partition_is_day",
        "python_module_versioned",
        "rust_module_versioned",
        "python_module_matches_lock",
        "rust_module_matches_lock",
    )
    return {
        "status": "local-static-ok" if all(checks[name] for name in local_required) else "failed",
        "cluster_qualification": CLUSTER_QUALIFICATION_STATUS,
        "locked_toolchains": {"python": python_version, "rust": rust_version},
        "checks": checks,
        "note": "Bouchet CPU-only day qualification completed on 2026-08-13; checks describe the current configuration",
    }


def _require_submit_ready(config: CampaignConfig) -> None:
    result = doctor(config)
    checks = cast(dict[str, bool], result["checks"])
    required_names = (
        "uv_available",
        "cargo_available",
        "slurm_client_available",
        "rsync_available",
        "flock_available",
        "uv_lock_committed",
        "cargo_lock_committed",
        "project_scratch_separate",
        "partition_is_day",
        "python_module_versioned",
        "rust_module_versioned",
        "python_module_matches_lock",
        "rust_module_matches_lock",
        "account_configured",
    )
    failed = [name for name in required_names if not checks[name]]
    if failed:
        raise ClusterError(f"cluster submission prerequisites failed: {failed}")


def _submit_script(
    script: Path,
    *,
    indices: list[int] | None = None,
    dependencies: list[str] | None = None,
    max_concurrency: int | None = None,
) -> str:
    command = ["sbatch", "--parsable"]
    if indices is not None:
        array = ",".join(str(index) for index in indices)
        if max_concurrency is not None:
            array = f"{array}%{max_concurrency}"
        command.append(f"--array={array}")
    if dependencies:
        command.append(f"--dependency=afterok:{':'.join(dependencies)}")
    command.append(str(script))
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    return process.stdout.strip().split(";")[0]


def _current_task_ids(config: CampaignConfig, state: dict[str, Any]) -> set[str]:
    digest = source_digest(config.campaign.project_root)
    artifacts = {
        artifact.artifact_id: artifact
        for artifact in discover_artifacts(config.campaign.output_root)
    }
    current: set[str] = set()
    for task_id, task_state in cast(dict[str, dict[str, Any]], state["tasks"]).items():
        artifact_ids = [str(value) for value in task_state["artifacts"]]
        if task_state["status"] != "complete" or not artifact_ids:
            continue
        try:
            valid = all(
                artifact_id in artifacts
                and artifacts[artifact_id].manifest["source_digest"] == digest
                and verify_artifact(artifacts[artifact_id].path)["status"] == "valid"
                for artifact_id in artifact_ids
            )
        except Exception:
            valid = False
        if valid:
            current.add(task_id)
    dependencies = {task.task_id: set(task.dependencies) for task in read_plan(config).tasks}
    changed = True
    while changed:
        changed = False
        for task_id in tuple(current):
            if not dependencies[task_id] <= current:
                current.remove(task_id)
                changed = True
    return current


def _pending_indices(plan: Plan, current_task_ids: set[str], kind: str) -> list[int]:
    tasks = [task for task in plan.tasks if task.kind == kind]
    return [index for index, task in enumerate(tasks) if task.task_id not in current_task_ids]


def _submit_dag(config: CampaignConfig) -> dict[str, Any]:
    rendered = render_cluster(config)
    plan = read_plan(config)
    state = read_state(config)
    current_task_ids = _current_task_ids(config, state)
    jobs: dict[str, str] = {}
    clean_indices = _pending_indices(plan, current_task_ids, "clean")
    exact_indices = _pending_indices(plan, current_task_ids, "exact")
    mc_indices = _pending_indices(plan, current_task_ids, "mc")
    analysis_task = next(task for task in plan.tasks if task.kind == "analysis")
    validation_task = next(task for task in plan.tasks if task.kind == "validation")
    analysis_complete = analysis_task.task_id in current_task_ids
    validation_complete = validation_task.task_id in current_task_ids
    needs_work = bool(
        clean_indices
        or exact_indices
        or mc_indices
        or not analysis_complete
        or not validation_complete
    )
    if needs_work:
        jobs["environment"] = _submit_script(rendered.stage_scripts["environment"])
    environment_dependencies = [jobs["environment"]] if "environment" in jobs else []
    if clean_indices:
        jobs["clean"] = _submit_script(
            rendered.stage_scripts["clean"],
            indices=clean_indices,
            dependencies=environment_dependencies,
            max_concurrency=config.cluster.max_array_concurrency,
        )
    if exact_indices:
        jobs["exact"] = _submit_script(
            rendered.stage_scripts["exact"],
            indices=exact_indices,
            dependencies=environment_dependencies,
            max_concurrency=config.cluster.max_array_concurrency,
        )
    if mc_indices:
        dependencies = [
            *environment_dependencies,
            *([jobs["clean"]] if "clean" in jobs else []),
        ]
        jobs["mc"] = _submit_script(
            rendered.stage_scripts["mc"],
            indices=mc_indices,
            dependencies=dependencies,
            max_concurrency=config.cluster.max_array_concurrency,
        )

    compute_jobs = [
        *environment_dependencies,
        *(jobs[name] for name in ("clean", "exact", "mc") if name in jobs),
    ]
    if not analysis_complete:
        jobs["analysis"] = _submit_script(
            rendered.stage_scripts["analysis"], dependencies=compute_jobs
        )
    if not validation_complete:
        dependencies = [
            *environment_dependencies,
            *([jobs["analysis"]] if "analysis" in jobs else compute_jobs),
        ]
        dependencies = list(dict.fromkeys(dependencies))
        jobs["validation"] = _submit_script(
            rendered.stage_scripts["validation"], dependencies=dependencies
        )
    return {
        "jobs": jobs,
        "root_job_id": jobs.get("validation", jobs.get("analysis", next(iter(jobs.values()), ""))),
        "rendered_root": str(rendered.root),
        "resource_shape": {
            "cpus_per_task": config.cluster.cpus_per_task,
            "max_array_concurrency": config.cluster.max_array_concurrency,
            "memory_per_cpu": config.cluster.memory,
            "time_limit": config.cluster.time_limit,
        },
        "cluster_qualification": CLUSTER_QUALIFICATION_STATUS,
    }


def submit(config: CampaignConfig) -> dict[str, Any]:
    _require_submit_ready(config)
    return _submit_dag(config)


def cluster_status(job_id: str) -> dict[str, Any]:
    if not job_id or any(character not in "0123456789_" for character in job_id):
        raise ClusterError("job id must contain only digits and underscore")
    command = ["squeue", "--noheader", "--jobs", job_id, "--format", "%i|%T|%M|%R"]
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        raise ClusterError(process.stderr.strip() or "squeue failed")
    rows = [line.split("|") for line in process.stdout.splitlines() if line]
    return {
        "job_id": job_id,
        "tasks": [
            {"id": row[0], "state": row[1], "elapsed": row[2], "reason_or_node": row[3]}
            for row in rows
        ],
    }


def resume(config: CampaignConfig) -> dict[str, Any]:
    _require_submit_ready(config)
    result = _submit_dag(config)
    result["status"] = "submitted" if result["jobs"] else "nothing-to-resume"
    return result
