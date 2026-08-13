"""Campaign planning and deterministic local execution."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np

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
    return _core.lattice_couplings(
        config.lattice.regularization, config.lattice.delta_tau
    )


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


def _diagnostic_schedule(config: CampaignConfig, global_id: int) -> tuple[tuple[int, int], ...]:
    if global_id >= config.mc.diagnostic_outer_records:
        return ((1, 0),)
    return tuple(
        (multiplier, replica)
        for multiplier in config.mc.inner_budget_multipliers
        for replica in range(config.mc.replicated_chains)
    )


def _execute_mc(config: CampaignConfig, task: Task) -> tuple[str, ...]:
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
    rows: list[dict[str, object]] = []

    for clean_row in clean.to_pylist():
        global_id = int(clean_row["global_id"])
        if not int(task.parameters["global_id_start"]) <= global_id < int(
            task.parameters["global_id_stop"]
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
        for multiplier, replica in _diagnostic_schedule(config, global_id):
            measurements = config.mc.inner_measurements * multiplier
            stream_label = (
                f"{noise}/{protocol_id}/p={p:.17g}/{update}/chi={tnmc_bond_dimension}/"
                f"budget={multiplier}/replica={replica}"
            )
            result = _core.posterior_observables(
                lx,
                lt,
                kx,
                kt,
                noise,
                generated["record_couplings"],
                list(packed),
                update,
                config.campaign.seed,
                global_id,
                stream_label,
                config.mc.posterior_decorrelation_gap,
                measurements,
                config.mc.inner_saving_interval,
                list(separations),
                global_id < config.mc.diagnostic_outer_records,
                tnmc_bond_dimension,
            )
            spin_profile = [float(value) for value in result["spin_profile"]]
            bond_profile = [float(value) for value in result["bond_profile"]]
            planted_spin = boundary
            planted_bond = [
                boundary[x] * boundary[(x + 1) % lx]
                for x in range(lx)
            ]
            spin_overlap = float(np.mean(np.asarray(planted_spin) * np.asarray(spin_profile)))
            bond_overlap = float(np.mean(np.asarray(planted_bond) * np.asarray(bond_profile)))
            planted_spin_correlator = [
                float(
                    np.mean(
                        np.asarray(planted_spin)
                        * np.roll(np.asarray(spin_profile), -separation)
                    )
                )
                for separation in separations
            ]
            planted_bond_correlator = [
                float(
                    np.mean(
                        np.asarray(planted_bond)
                        * np.roll(np.asarray(bond_profile), -separation)
                    )
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
                    "planted_configuration_hash": expected_hash,
                    "planted_variables": planted_variables,
                    "raw_record": [float(value) for value in generated["raw_record"]],
                    "record_couplings": [
                        float(value) for value in generated["record_couplings"]
                    ],
                    "standard_variates": [
                        float(value) for value in generated["standard_variates"]
                    ],
                    "inner_budget_multiplier": multiplier,
                    "replica": replica,
                    "posterior_decorrelation_gap": config.mc.posterior_decorrelation_gap,
                    "inner_measurements": measurements,
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
                    "magnetization_trace": [
                        float(value) for value in result["magnetization_trace"]
                    ],
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
                    "tnmc_conditionals_regularized": int(
                        result["tnmc_conditionals_regularized"]
                    ),
                }
            )
    if not rows:
        raise CampaignError(f"MC task {task.task_id} selected no clean configurations")
    artifact = write_artifact(
        config.campaign.output_root,
        "mc-records",
        table_from_rows("mc-records", rows),
        metadata={
            "task_id": task.task_id,
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
            }[update],
            "tnmc_bond_dimension": tnmc_bond_dimension,
            "posterior_initialization": "full planted clean state (exact posterior draw)",
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
) -> dict[str, int]:
    if executor != "local":
        raise CampaignError("development execution supports only --executor local")
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
                if _artifact_by_id(
                    config.campaign.output_root, str(artifact_id)
                ).manifest["source_digest"]
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
                artifacts = _execute_mc(config, task)
            elif task.kind == "exact":
                artifacts = _execute_exact(config, task)
            else:
                raise CampaignError(f"unsupported task kind {task.kind}")
        except Exception as error:
            update_task_state(config, task.task_id, "failed", error=str(error))
            raise
        update_task_state(config, task.task_id, "complete", artifacts=artifacts)
        completed += 1
    return {"completed": completed, "skipped": skipped}


def campaign_artifacts(config: CampaignConfig) -> list[Artifact]:
    return discover_artifacts(config.campaign.output_root)
