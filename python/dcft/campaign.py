"""Campaign planning and deterministic local execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa

from . import _core
from .artifacts import (
    Artifact,
    ArtifactError,
    configuration_hash,
    discover_artifacts,
    load_artifact,
    read_table,
    source_digest,
    verify_artifact,
    write_artifact,
)
from .config import CampaignConfig
from .exact import (
    build_priors,
    default_observables,
    density_reference,
    evaluate_exact_protocol,
    linear_expectation,
    physical_fidelity_from_reference,
    prior_diagnostic_rows,
    translated_observables,
)
from .planning import PlanError, Task, read_state, update_task_state, write_plan
from .registries import MeasurementPoint, resolve_measurements, verify_rust_registries
from .schemas import table_from_rows


class CampaignError(RuntimeError):
    """A campaign task cannot be completed safely."""


def _couplings(config: CampaignConfig) -> tuple[float, float]:
    return _core.lattice_couplings(config.lattice.regularization, config.lattice.delta_tau)


def _artifact_by_id(root: Path, artifact_id: str) -> Artifact:
    candidates = list(root.rglob(f"{artifact_id}/manifest.json"))
    if len(candidates) != 1:
        raise CampaignError(
            f"expected exactly one artifact {artifact_id} beneath {root}; found {len(candidates)}"
        )
    return load_artifact(candidates[0].parent)


def _task_state(config: CampaignConfig, task_id: str) -> dict[str, Any]:
    state = read_state(config)
    return cast(dict[str, Any], state["tasks"][task_id])


def _execute_clean(config: CampaignConfig, task: Task) -> tuple[str, ...]:
    lx = int(task.parameters["lx"])
    lt = int(task.parameters["lt"])
    count = int(task.parameters["global_id_stop"]) - int(task.parameters["global_id_start"])
    kx, kt = _couplings(config)
    packed_configurations = _core.clean_configurations(
        lx,
        lt,
        kx,
        kt,
        config.campaign.seed,
        config.mc.clean_thermalization_sweeps,
        config.mc.clean_saving_interval,
        count,
    )
    rows: list[dict[str, object]] = []
    for offset, packed_values in enumerate(packed_configurations):
        packed = bytes(packed_values)
        global_id = int(task.parameters["global_id_start"]) + offset
        rows.append(
            {
                "global_id": global_id,
                "lx": lx,
                "lt": lt,
                "configuration": packed,
                "configuration_hash": configuration_hash(packed),
                "regularization": config.lattice.regularization,
                "kx": kx,
                "kt": kt,
                "delta_tau": config.lattice.delta_tau,
                "seed": config.campaign.seed,
                "clean_thermalization_sweeps": config.mc.clean_thermalization_sweeps,
                "clean_saving_interval": config.mc.clean_saving_interval,
            }
        )
    table = table_from_rows("clean", rows)
    artifact = write_artifact(
        config.campaign.output_root,
        "clean",
        table,
        metadata={
            "task_id": task.task_id,
            "lattice": {"lx": lx, "lt": lt},
            "regularization": config.lattice.regularization,
            "couplings": {"kx": kx, "kt": kt, "delta_tau": config.lattice.delta_tau},
            "sweep_semantics": "one clean sweep is one Wolff cluster construction and flip",
            "global_id_range": [
                task.parameters["global_id_start"],
                task.parameters["global_id_stop"],
            ],
        },
        project_root=config.campaign.project_root,
        partition_by=("lx",),
    )
    return (artifact.artifact_id,)


@dataclass(frozen=True)
class _ChainSpec:
    multiplier: int
    replica: int
    role: str
    initialization: str


def _chain_schedule(config: CampaignConfig, global_id: int) -> tuple[_ChainSpec, ...]:
    """Separate estimator, finite-budget, and overdispersed convergence chains."""
    production = _ChainSpec(1, 0, "production", "planted")
    if global_id >= config.mc.diagnostic_outer_records:
        return (production,)
    finite_inner = tuple(
        _ChainSpec(multiplier, replica, "finite-inner", "planted")
        for multiplier in config.mc.inner_budget_multipliers
        for replica in range(1 if multiplier == 1 else 0, config.mc.replicated_chains)
    )
    convergence = tuple(
        _ChainSpec(
            1,
            replica,
            "convergence",
            "all-plus" if replica == 0 else "all-minus" if replica == 1 else "random",
        )
        for replica in range(config.mc.replicated_chains)
    )
    return (production, *finite_inner, *convergence)


def _packed_initial_configuration(
    lx: int,
    lt: int,
    initialization: str,
    *,
    seed: int,
    global_id: int,
    stream_label: str,
    planted: bytes,
) -> bytes:
    count = lx * lt
    byte_count = (count + 7) // 8
    if initialization == "planted":
        return planted
    if initialization == "all-plus":
        return bytes(byte_count)
    if initialization == "all-minus":
        packed = bytearray([0xFF] * byte_count)
        if count % 8:
            packed[-1] &= (1 << (count % 8)) - 1
        return bytes(packed)
    if initialization != "random":
        raise CampaignError(f"unknown posterior initialization {initialization!r}")
    uniforms = _core.stream_uniforms(
        seed,
        f"posterior-initial/{stream_label}",
        global_id,
        count,
    )
    packed = bytearray(byte_count)
    for index, uniform in enumerate(uniforms):
        if float(uniform) < 0.5:
            packed[index // 8] |= 1 << (index % 8)
    return bytes(packed)


def _execute_mc(config: CampaignConfig, task: Task, *, workers: int) -> tuple[str, ...]:
    dependency = _task_state(config, task.dependencies[0])
    parent_ids = tuple(str(value) for value in dependency["artifacts"])
    if dependency["status"] != "complete" or len(parent_ids) != 1:
        raise CampaignError(f"clean dependency {task.dependencies[0]} is incomplete")
    clean_artifact = _artifact_by_id(config.campaign.output_root, parent_ids[0])
    clean = read_table(clean_artifact)

    lx = int(task.parameters["lx"])
    lt = int(task.parameters["lt"])
    noise = str(task.parameters["noise"])
    measurement = str(task.parameters["measurement"])
    gamma = cast(float | None, task.parameters["gamma"])
    p = float(task.parameters["p"])
    protocol_id = str(task.parameters["protocol_id"])
    update = str(task.parameters["update"])
    tnmc_bond_dimension = config.mc.tnmc_bond_dimension
    point = MeasurementPoint(measurement, gamma)
    if point.identifier != protocol_id:
        raise CampaignError("planned protocol identifier does not match resolved measurement")
    parameters = _core.protocol_parameters(measurement, p, gamma)
    kx, kt = _couplings(config)
    separations = config.separations_for(lx)
    prepared: list[dict[str, Any]] = []

    for clean_row in clean.to_pylist():
        global_id = int(clean_row["global_id"])
        if (
            not int(task.parameters["global_id_start"])
            <= global_id
            < int(task.parameters["global_id_stop"])
        ):
            continue
        packed = bytes(clean_row["configuration"])
        expected_hash = str(clean_row["configuration_hash"])
        if configuration_hash(packed) != expected_hash:
            raise ArtifactError(f"configuration hash mismatch for global id {global_id}")
        boundary = _core.boundary_from_packed(lx, lt, list(packed))
        generated = _core.generate_record(
            boundary,
            noise,
            measurement,
            p,
            config.campaign.seed,
            global_id,
            gamma,
        )
        planted_variables = [int(value) for value in generated["variables"]]
        planted_bond = [boundary[x] * boundary[(x + 1) % lx] for x in range(lx)]
        for chain in _chain_schedule(config, global_id):
            measurements = config.mc.inner_measurements * chain.multiplier
            stream_label = (
                f"{noise}/{protocol_id}/p={p:.17g}/{update}/chi={tnmc_bond_dimension}/"
                f"role={chain.role}/budget={chain.multiplier}/replica={chain.replica}/"
                f"initialization={chain.initialization}"
            )
            initial = _packed_initial_configuration(
                lx,
                lt,
                chain.initialization,
                seed=config.campaign.seed,
                global_id=global_id,
                stream_label=stream_label,
                planted=packed,
            )
            prepared.append(
                {
                    "global_id": global_id,
                    "packed": list(packed),
                    "expected_hash": expected_hash,
                    "boundary": boundary,
                    "planted_bond": planted_bond,
                    "generated": generated,
                    "planted_variables": planted_variables,
                    "initial": list(initial),
                    "initial_hash": configuration_hash(initial),
                    "multiplier": chain.multiplier,
                    "replica": chain.replica,
                    "chain_role": chain.role,
                    "initialization": chain.initialization,
                    "measurements": measurements,
                    "stream_label": stream_label,
                    "retain_trace": chain.role == "convergence",
                }
            )
    if not prepared:
        raise CampaignError(f"MC task {task.task_id} selected no clean configurations")

    results = _core.posterior_observables_batch(
        lx,
        lt,
        kx,
        kt,
        noise,
        [list(item["generated"]["record_couplings"]) for item in prepared],
        [cast(list[int], item["packed"]) for item in prepared],
        [cast(list[int], item["initial"]) for item in prepared],
        update,
        config.campaign.seed,
        [int(item["global_id"]) for item in prepared],
        [str(item["stream_label"]) for item in prepared],
        config.mc.posterior_decorrelation_gap,
        [int(item["measurements"]) for item in prepared],
        config.mc.inner_saving_interval,
        list(separations),
        [bool(item["retain_trace"]) for item in prepared],
        tnmc_bond_dimension,
        workers,
    )
    rows: list[dict[str, object]] = []
    for item, result in zip(prepared, results, strict=True):
        global_id = int(item["global_id"])
        generated = cast(dict[str, Any], item["generated"])
        boundary = cast(list[int], item["boundary"])
        planted_bond = cast(list[int], item["planted_bond"])
        spin_profile = [float(value) for value in result["spin_profile"]]
        bond_profile = [float(value) for value in result["bond_profile"]]
        spin_overlap = float(np.mean(np.asarray(boundary) * np.asarray(spin_profile)))
        bond_overlap = float(np.mean(np.asarray(planted_bond) * np.asarray(bond_profile)))
        planted_spin_correlator = [
            float(np.mean(np.asarray(boundary) * np.roll(np.asarray(spin_profile), -separation)))
            for separation in separations
        ]
        planted_bond_correlator = [
            float(
                np.mean(np.asarray(planted_bond) * np.roll(np.asarray(bond_profile), -separation))
            )
            for separation in separations
        ]
        rows.append(
            {
                "global_id": global_id,
                "lx": lx,
                "lt": lt,
                "noise": noise,
                "measurement": measurement,
                "protocol_id": protocol_id,
                "update": update,
                "tnmc_bond_dimension": tnmc_bond_dimension,
                "p": p,
                "lambda": float(parameters["lambda"]),
                "gamma": parameters["gamma"],
                "kappa": parameters["kappa"],
                "protocol_coupling": parameters["coupling"],
                "planted_configuration_hash": str(item["expected_hash"]),
                "initial_configuration_hash": str(item["initial_hash"]),
                "chain_role": str(item["chain_role"]),
                "initialization": str(item["initialization"]),
                "planted_variables": cast(list[int], item["planted_variables"]),
                "raw_record": [float(value) for value in generated["raw_record"]],
                "record_couplings": [float(value) for value in generated["record_couplings"]],
                "standard_variates": [float(value) for value in generated["standard_variates"]],
                "inner_budget_multiplier": int(item["multiplier"]),
                "replica": int(item["replica"]),
                "posterior_decorrelation_gap": config.mc.posterior_decorrelation_gap,
                "inner_measurements": int(item["measurements"]),
                "inner_saving_interval": config.mc.inner_saving_interval,
                "energy": float(result["energy"]),
                "magnetization": float(result["magnetization"]),
                "boundary_magnetization": float(result["boundary_magnetization"]),
                "spin_profile": spin_profile,
                "bond_profile": bond_profile,
                "separations": list(separations),
                "spin_correlator_profile": [
                    [float(value) for value in profile]
                    for profile in result["spin_correlator_profile"]
                ],
                "bond_correlator_profile": [
                    [float(value) for value in profile]
                    for profile in result["bond_correlator_profile"]
                ],
                "planted_spin_overlap": spin_overlap,
                "planted_bond_overlap": bond_overlap,
                "planted_spin_correlator": planted_spin_correlator,
                "planted_bond_correlator": planted_bond_correlator,
                "energy_trace": [float(value) for value in result["energy_trace"]],
                "magnetization_trace": [float(value) for value in result["magnetization_trace"]],
                "boundary_magnetization_trace": [
                    float(value) for value in result["boundary_magnetization_trace"]
                ],
                "planted_spin_overlap_trace": [
                    float(value) for value in result["planted_spin_overlap_trace"]
                ],
                "planted_bond_overlap_trace": [
                    float(value) for value in result["planted_bond_overlap_trace"]
                ],
                "sweeps": int(result["sweeps"]),
                "local_proposed": int(result["local_proposed"]),
                "local_accepted": int(result["local_accepted"]),
                "cluster_proposed": int(result["cluster_proposed"]),
                "cluster_accepted": int(result["cluster_accepted"]),
                "cluster_sites_proposed": int(result["cluster_sites_proposed"]),
                "global_proposed": int(result["global_proposed"]),
                "global_attempted": int(result["global_attempted"]),
                "global_accepted": int(result["global_accepted"]),
                "tnmc_proposed": int(result["tnmc_proposed"]),
                "tnmc_accepted": int(result["tnmc_accepted"]),
                "tnmc_sites_proposed": int(result["tnmc_sites_proposed"]),
                "tnmc_conditionals_regularized": int(result["tnmc_conditionals_regularized"]),
            }
        )
    artifact = write_artifact(
        config.campaign.output_root,
        "mc-chunk",
        table_from_rows("mc-records", rows),
        metadata={
            "task_id": task.task_id,
            "chunk_id": task.parameters["chunk_id"],
            "chunk_count": task.parameters["chunk_count"],
            "global_id_range": [
                task.parameters["global_id_start"],
                task.parameters["global_id_stop"],
            ],
            "workers": workers,
            "parent_clean_artifact": clean_artifact.artifact_id,
            "lattice": {"lx": lx, "lt": lt, "kx": kx, "kt": kt},
            "noise": noise,
            "measurement": measurement,
            "protocol_id": protocol_id,
            "protocol_parameters": parameters,
            "update": update,
            "sweep_semantics": {
                "metropolis": "lx*lt uniformly random single-spin proposals",
                "sequential-metropolis": "one row-major single-spin pass",
                "metropolis-global": "random Metropolis sweep plus lazy global proposal",
                "corrected-wolff": "one clean-cluster proposal including rejection",
                "tnmc": "one random frozen-row/frozen-column conditional TNMC proposal",
                "tnmc-global": (
                    "one conditional TNMC proposal followed by one lazy global proposal"
                ),
            }[update],
            "tnmc_bond_dimension": tnmc_bond_dimension,
            "posterior_initialization": {
                "production": "full planted clean state (exact posterior draw)",
                "finite-inner": "full planted clean state with independent streams",
                "convergence": "overdispersed all-plus, all-minus, then random states",
            },
            "ordinary_burn_in": 0,
            "decorrelation_gap": config.mc.posterior_decorrelation_gap,
            "inner_budget_multipliers": list(config.mc.inner_budget_multipliers),
            "diagnostic_outer_records": config.mc.diagnostic_outer_records,
            "replicated_chains": config.mc.replicated_chains,
            "separations": list(separations),
        },
        project_root=config.campaign.project_root,
        parents=(clean_artifact.artifact_id,),
        partition_by=("noise", "measurement", "update"),
    )
    return (artifact.artifact_id,)


def _execute_merge(config: CampaignConfig, task: Task) -> tuple[str, ...]:
    artifacts: list[Artifact] = []
    for dependency_id in task.dependencies:
        dependency = _task_state(config, dependency_id)
        artifact_ids = tuple(str(value) for value in dependency["artifacts"])
        if dependency["status"] != "complete" or len(artifact_ids) != 1:
            raise CampaignError(f"MC chunk dependency {dependency_id} is incomplete")
        artifact = _artifact_by_id(config.campaign.output_root, artifact_ids[0])
        if artifact.manifest["kind"] != "mc-chunk":
            raise CampaignError(f"merge dependency {dependency_id} is not an MC chunk")
        artifacts.append(artifact)
    if len(artifacts) != int(task.parameters["chunk_count"]):
        raise CampaignError("merge received the wrong number of MC chunks")
    table = pa.concat_tables([read_table(artifact) for artifact in artifacts])
    table = table.sort_by(
        [
            ("global_id", "ascending"),
            ("chain_role", "ascending"),
            ("inner_budget_multiplier", "ascending"),
            ("replica", "ascending"),
        ]
    )
    production_ids = [
        int(row["global_id"])
        for row in table.to_pylist()
        if row["chain_role"] == "production"
    ]
    expected_ids = list(
        range(
            int(task.parameters["global_id_start"]),
            int(task.parameters["global_id_stop"]),
        )
    )
    if production_ids != expected_ids:
        raise CampaignError("MC chunks are overlapping, incomplete, or out of order")
    artifact = write_artifact(
        config.campaign.output_root,
        "mc-records",
        table,
        metadata={
            "task_id": task.task_id,
            "merged_chunks": len(artifacts),
            "global_id_range": [
                task.parameters["global_id_start"],
                task.parameters["global_id_stop"],
            ],
            "lattice": {"lx": task.parameters["lx"], "lt": task.parameters["lt"]},
            "noise": task.parameters["noise"],
            "measurement": task.parameters["measurement"],
            "protocol_id": task.parameters["protocol_id"],
            "p": task.parameters["p"],
            "update": task.parameters["update"],
            "tnmc_bond_dimension": task.parameters["tnmc_bond_dimension"],
        },
        project_root=config.campaign.project_root,
        parents=tuple(artifact.artifact_id for artifact in artifacts),
        partition_by=("noise", "measurement", "update"),
    )
    return (artifact.artifact_id,)


def _execute_exact(config: CampaignConfig, task: Task) -> tuple[str, ...]:
    lx = int(task.parameters["lx"])
    lt = int(task.parameters["lt"])
    kx, kt = _couplings(config)
    ground, priors = build_priors(lx, lt, kx, kt)
    separations = config.separations_for(lx)
    observables = default_observables(separations)
    prior_rows = prior_diagnostic_rows(
        lx,
        lt,
        config.lattice.regularization,
        config.lattice.delta_tau,
        priors,
    )
    result_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    for noise in cast(list[str], task.parameters["noises"]):
        for p_value in cast(list[float], task.parameters["p_values"]):
            p = float(p_value)
            points = resolve_measurements(
                config.protocols.measurements,
                p,
                config.protocols.gaussian_fractions,
            )
            for prior_name in cast(list[str], task.parameters["priors"]):
                prior = priors.named(prior_name)
                state = ground.state if prior_name == "quantum" else np.sqrt(prior)
                reference = density_reference(
                    state,
                    p,
                    noise,
                    clip_tolerance=config.ed.positivity_tolerance,
                )
                observable_references = {}
                for observable in observables:
                    translated = translated_observables(lx, observable)
                    observable_references[observable] = (
                        float(np.mean([linear_expectation(prior, value) for value in translated])),
                        float(
                            np.mean(
                                [
                                    physical_fidelity_from_reference(reference, value)
                                    for value in translated
                                ]
                            )
                        ),
                    )
                for point in points:
                    rows, raw = evaluate_exact_protocol(
                        sites=lx,
                        noise=noise,
                        point=point,
                        p=p,
                        prior_name=prior_name,
                        prior=prior,
                        ground=ground,
                        observables=observables,
                        seed=config.campaign.seed,
                        gaussian_records=config.ed.gaussian_outer_records,
                        local_x_enumeration_limit=config.ed.local_x_enumeration_limit,
                        sampled_binary_records=config.ed.sampled_binary_records,
                        positivity_tolerance=config.ed.positivity_tolerance,
                        reference=reference,
                        observable_references=observable_references,
                    )
                    result_rows.extend(rows)
                    record_rows.extend(raw)
    common_metadata = {
        "task_id": task.task_id,
        "lattice": {"lx": lx, "lt": lt, "kx": kx, "kt": kt},
        "ground_energy": ground.energy,
        "ground_residual": ground.residual,
        "ground_normalization_error": ground.normalization_error,
        "local_x_enumeration_limit": config.ed.local_x_enumeration_limit,
        "sampled_fallback_label": "sampled-binary-fallback",
        "separations": list(separations),
    }
    prior_artifact = write_artifact(
        config.campaign.output_root,
        "prior-diagnostics",
        table_from_rows("prior-diagnostics", prior_rows),
        metadata=common_metadata,
        project_root=config.campaign.project_root,
        partition_by=("lx",),
    )
    results_artifact = write_artifact(
        config.campaign.output_root,
        "ed-results",
        table_from_rows("ed-results", result_rows),
        metadata=common_metadata,
        project_root=config.campaign.project_root,
        parents=(prior_artifact.artifact_id,),
        partition_by=("noise", "measurement", "prior"),
    )
    records_artifact = write_artifact(
        config.campaign.output_root,
        "ed-records",
        table_from_rows("ed-records", record_rows),
        metadata=common_metadata,
        project_root=config.campaign.project_root,
        parents=(prior_artifact.artifact_id, results_artifact.artifact_id),
        partition_by=("noise", "measurement", "prior"),
    )
    return (
        prior_artifact.artifact_id,
        results_artifact.artifact_id,
        records_artifact.artifact_id,
    )


def run_campaign(
    config: CampaignConfig,
    *,
    executor: str = "local",
    task_ids: Iterable[str] | None = None,
    workers: int | None = None,
) -> dict[str, int]:
    if executor != "local":
        raise CampaignError("development execution supports only --executor local")
    effective_workers = config.execution.local_workers if workers is None else workers
    if effective_workers < 1:
        raise CampaignError("workers must be positive")
    verify_rust_registries()
    plan = write_plan(config)
    selected = set(task_ids) if task_ids is not None else None
    known = {task.task_id for task in plan.tasks}
    current_source_digest = source_digest(config.campaign.project_root)
    if selected is not None and not selected <= known:
        raise PlanError(f"unknown requested task ids: {sorted(selected - known)}")
    completed = 0
    skipped = 0
    for task in plan.tasks:
        if task.kind in {"analysis", "validation"}:
            continue
        if selected is not None and task.task_id not in selected:
            continue
        item = _task_state(config, task.task_id)
        if item["status"] == "complete":
            current = True
            for artifact_id in item["artifacts"]:
                artifact = _artifact_by_id(config.campaign.output_root, artifact_id)
                verify_artifact(artifact.path)
                current = current and artifact.manifest["source_digest"] == current_source_digest
            if current:
                skipped += 1
                continue
        for dependency_id in task.dependencies:
            dependency = _task_state(config, dependency_id)
            if dependency["status"] != "complete":
                raise CampaignError(
                    f"task {task.task_id} requires incomplete dependency {dependency_id}"
                )
            stale_dependencies = [
                artifact_id
                for artifact_id in dependency["artifacts"]
                if _artifact_by_id(config.campaign.output_root, str(artifact_id)).manifest[
                    "source_digest"
                ]
                != current_source_digest
            ]
            if stale_dependencies:
                raise CampaignError(
                    f"task {task.task_id} requires source-current dependency "
                    f"{dependency_id}; rerun its upstream task first"
                )
        update_task_state(config, task.task_id, "running")
        try:
            if task.kind == "clean":
                artifacts = _execute_clean(config, task)
            elif task.kind == "mc":
                artifacts = _execute_mc(config, task, workers=effective_workers)
            elif task.kind == "merge":
                artifacts = _execute_merge(config, task)
            elif task.kind == "exact":
                artifacts = _execute_exact(config, task)
            else:
                raise CampaignError(f"unsupported task kind {task.kind}")
        except Exception as error:
            update_task_state(config, task.task_id, "failed", error=str(error))
            raise
        update_task_state(config, task.task_id, "complete", artifacts=artifacts)
        completed += 1
    return {"completed": completed, "skipped": skipped, "workers": effective_workers}


def campaign_artifacts(config: CampaignConfig) -> list[Artifact]:
    return discover_artifacts(config.campaign.output_root)
