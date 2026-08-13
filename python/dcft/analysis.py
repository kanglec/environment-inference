"""Curve-preserving analysis of immutable per-record campaign outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, cast

import numpy as np
import pyarrow as pa

from . import _core
from .artifacts import Artifact, discover_artifacts, read_table, source_digest, write_artifact
from .config import CampaignConfig
from .planning import read_plan, read_state, update_task_state
from .statistics import (
    common_resample_immse,
    geyer_autocorrelation,
    moving_block_bootstrap,
    moving_block_indices,
    nested_variance_components,
    simultaneous_curve_interval,
    split_rhat,
)


class AnalysisError(RuntimeError):
    """Campaign data are incomplete or inconsistent for analysis."""


def _production_rows(artifacts: Iterable[Artifact]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        for row in read_table(artifact).to_pylist():
            if int(row["inner_budget_multiplier"]) == 1 and int(row["replica"]) == 0:
                rows.append(cast(dict[str, Any], row))
    return rows


def _group_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["lx"]),
            int(row["lt"]),
            str(row["noise"]),
            float(row["p"]),
            str(row["measurement"]),
            str(row["protocol_id"]),
            str(row["update"]),
            None if row["gamma"] is None else float(row["gamma"]),
        )
        grouped[key].append(row)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["global_id"]))
        ids = [int(item["global_id"]) for item in values]
        if ids != list(range(ids[0], ids[0] + len(ids))):
            raise AnalysisError(f"outer ids are not a contiguous range: {ids[:3]}...{ids[-3:]}")
    return grouped


def _clean_reference(
    config: CampaignConfig, artifacts: Iterable[Artifact]
) -> dict[int, dict[str, Any]]:
    references: dict[int, dict[str, Any]] = {}
    for artifact in artifacts:
        table = read_table(artifact)
        rows = table.to_pylist()
        lx = int(rows[0]["lx"])
        lt = int(rows[0]["lt"])
        kx = float(rows[0]["kx"])
        kt = float(rows[0]["kt"])
        separations = config.separations_for(lx)
        observations_by_id: dict[int, dict[str, Any]] = {}
        for row in rows:
            observation = _core.configuration_observables(
                lx,
                lt,
                kx,
                kt,
                list(bytes(row["configuration"])),
                list(separations),
            )
            observations_by_id[int(row["global_id"])] = {
                **observation,
                "spin_correlator": np.mean(
                    observation["spin_correlator_profile"], axis=1
                ),
                "bond_correlator": np.mean(
                    observation["bond_correlator_profile"], axis=1
                ),
            }
        observations = list(observations_by_id.values())
        references[lx] = {
            "energy": float(np.mean([item["energy"] for item in observations])),
            "magnetization": float(np.mean([item["magnetization"] for item in observations])),
            "boundary_magnetization": float(
                np.mean([item["boundary_magnetization"] for item in observations])
            ),
            "spin_correlator": np.mean(
                [np.mean(item["spin_correlator_profile"], axis=1) for item in observations],
                axis=0,
            ),
            "bond_correlator": np.mean(
                [np.mean(item["bond_correlator_profile"], axis=1) for item in observations],
                axis=0,
            ),
            "separations": separations,
            "by_global_id": observations_by_id,
        }
    return references


def _protocol_shift(row: dict[str, Any], lx: int) -> float:
    if row["gamma"] is not None:
        return lx * float(row["gamma"])
    kappa = float(row["kappa"])
    coupling = float(row["protocol_coupling"])
    return lx * kappa * coupling


def _curve_and_scalar_rows(
    config: CampaignConfig,
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]],
    clean: dict[int, dict[str, Any]],
    bias_envelopes: dict[tuple[Any, ...], float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    curve_rows: list[dict[str, object]] = []
    scalar_rows: list[dict[str, object]] = []
    for key, records in sorted(grouped.items(), key=lambda item: repr(item[0])):
        lx, lt, noise, p, measurement, protocol_id, update, gamma = key
        count = len(records)
        block_length = min(config.statistics.block_length, count)
        energies = np.asarray([row["energy"] for row in records], dtype=np.float64)
        magnetization = np.asarray([row["magnetization"] for row in records], dtype=np.float64)
        boundary_magnetization = np.asarray(
            [row["boundary_magnetization"] for row in records], dtype=np.float64
        )
        scalar_values = np.column_stack((energies, magnetization, boundary_magnetization))
        scalar_interval = moving_block_bootstrap(
            scalar_values,
            block_length=block_length,
            resamples=config.statistics.bootstrap_resamples,
            confidence=config.statistics.confidence,
            seed=config.campaign.seed ^ lx,
        )
        autocorrelation = geyer_autocorrelation(
            energies, saving_interval_sweeps=config.mc.clean_saving_interval
        )
        local_proposed = sum(int(row["local_proposed"]) for row in records)
        local_accepted = sum(int(row["local_accepted"]) for row in records)
        cluster_proposed = sum(int(row["cluster_proposed"]) for row in records)
        cluster_accepted = sum(int(row["cluster_accepted"]) for row in records)
        global_attempted = sum(int(row["global_attempted"]) for row in records)
        global_accepted = sum(int(row["global_accepted"]) for row in records)
        tnmc_proposed = sum(int(row.get("tnmc_proposed", 0)) for row in records)
        tnmc_accepted = sum(int(row.get("tnmc_accepted", 0)) for row in records)
        tnmc_regularized = sum(
            int(row.get("tnmc_conditionals_regularized", 0)) for row in records
        )
        overlap_field = "planted_spin_overlap" if noise == "z" else "planted_bond_overlap"
        planted_q = np.asarray([row[overlap_field] for row in records], dtype=np.float64)
        profile_field = "spin_profile" if noise == "z" else "bond_profile"
        direct_q = np.asarray(
            [np.mean(np.square(np.asarray(row[profile_field], dtype=np.float64))) for row in records]
        )
        q_interval = moving_block_bootstrap(
            np.column_stack((planted_q, direct_q)),
            block_length=block_length,
            resamples=config.statistics.bootstrap_resamples,
            confidence=config.statistics.confidence,
            seed=config.campaign.seed ^ (lx << 8),
        )
        expected_energy = clean[lx]["energy"] - _protocol_shift(records[0], lx)
        clean_records = [
            clean[lx]["by_global_id"][int(row["global_id"])] for row in records
        ]
        energy_residual_interval = moving_block_bootstrap(
            energies
            - np.asarray([row["energy"] for row in clean_records], dtype=np.float64)
            + _protocol_shift(records[0], lx),
            block_length=block_length,
            resamples=config.statistics.bootstrap_resamples,
            confidence=config.statistics.confidence,
            seed=config.campaign.seed ^ (lx << 12),
        )
        scalar_rows.append(
            {
                "lx": lx,
                "lt": lt,
                "noise": noise,
                "p": p,
                "measurement": measurement,
                "protocol_id": protocol_id,
                "gamma": gamma,
                "update": update,
                "outer_records": count,
                "energy": float(scalar_interval.estimate[0]),
                "energy_standard_error": float(scalar_interval.standard_error[0]),
                "energy_lower": float(scalar_interval.lower[0]),
                "energy_upper": float(scalar_interval.upper[0]),
                "expected_annealed_energy": expected_energy,
                "energy_shift_residual": float(energy_residual_interval.estimate[0]),
                "energy_shift_residual_standard_error": float(
                    energy_residual_interval.standard_error[0]
                ),
                "energy_shift_residual_lower": float(energy_residual_interval.lower[0]),
                "energy_shift_residual_upper": float(energy_residual_interval.upper[0]),
                "magnetization": float(scalar_interval.estimate[1]),
                "clean_magnetization": clean[lx]["magnetization"],
                "boundary_magnetization": float(scalar_interval.estimate[2]),
                "clean_boundary_magnetization": clean[lx]["boundary_magnetization"],
                "q_ea_planted": float(q_interval.estimate[0]),
                "q_ea_planted_standard_error": float(q_interval.standard_error[0]),
                "q_ea_direct_diagnostic": float(q_interval.estimate[1]),
                "q_ea_direct_standard_error": float(q_interval.standard_error[1]),
                "tau_int_saved": autocorrelation.tau_saved,
                "tau_int_sweeps": autocorrelation.tau_sweeps,
                "outer_effective_samples": autocorrelation.effective_samples,
                "autocorrelation_window": autocorrelation.window_lag,
                "local_acceptance": (
                    local_accepted / local_proposed if local_proposed else None
                ),
                "cluster_acceptance": (
                    cluster_accepted / cluster_proposed if cluster_proposed else None
                ),
                "global_acceptance": (
                    global_accepted / global_attempted if global_attempted else None
                ),
                "tnmc_acceptance": (
                    tnmc_accepted / tnmc_proposed if tnmc_proposed else None
                ),
                "tnmc_conditionals_regularized": tnmc_regularized,
                "tnmc_regularized_per_proposal": (
                    tnmc_regularized / tnmc_proposed if tnmc_proposed else None
                ),
            }
        )

        separations = tuple(int(value) for value in records[0]["separations"])
        for family in ("spin", "bond"):
            profiles = np.asarray(
                [row[f"{family}_correlator_profile"] for row in records], dtype=np.float64
            )
            linear_values = np.mean(profiles, axis=2)
            witness_values = np.mean(np.abs(profiles), axis=2)
            planted_values = np.asarray(
                [row[f"planted_{family}_correlator"] for row in records], dtype=np.float64
            )
            posterior_profiles = np.asarray(
                [row[f"{family}_profile"] for row in records], dtype=np.float64
            )
            direct_posterior_values = np.column_stack(
                [
                    np.mean(
                        posterior_profiles
                        * np.roll(posterior_profiles, -separation, axis=1),
                        axis=1,
                    )
                    for separation in separations
                ]
            )
            linear_interval = moving_block_bootstrap(
                linear_values,
                block_length=block_length,
                resamples=config.statistics.bootstrap_resamples,
                confidence=config.statistics.confidence,
                seed=config.campaign.seed ^ (lx << 16) ^ (0 if family == "spin" else 1),
            )
            witness_interval = simultaneous_curve_interval(
                witness_values,
                block_length=block_length,
                resamples=config.statistics.bootstrap_resamples,
                confidence=config.statistics.confidence,
                seed=config.campaign.seed ^ (lx << 20) ^ (0 if family == "spin" else 1),
            )
            posterior_interval = moving_block_bootstrap(
                planted_values,
                block_length=block_length,
                resamples=config.statistics.bootstrap_resamples,
                confidence=config.statistics.confidence,
                seed=config.campaign.seed ^ (lx << 21) ^ (0 if family == "spin" else 1),
            )
            direct_posterior_interval = moving_block_bootstrap(
                direct_posterior_values,
                block_length=block_length,
                resamples=config.statistics.bootstrap_resamples,
                confidence=config.statistics.confidence,
                seed=config.campaign.seed ^ (lx << 22) ^ (0 if family == "spin" else 1),
            )
            clean_values = cast(np.ndarray[Any, Any], clean[lx][f"{family}_correlator"])
            clean_record_values = np.asarray(
                [row[f"{family}_correlator"] for row in clean_records],
                dtype=np.float64,
            )
            annealed_residual_interval = moving_block_bootstrap(
                linear_values - clean_record_values,
                block_length=block_length,
                resamples=config.statistics.bootstrap_resamples,
                confidence=config.statistics.confidence,
                seed=config.campaign.seed
                ^ (lx << 23)
                ^ (0 if family == "spin" else 1),
            )
            for index, separation in enumerate(separations):
                bias_key = (
                    lx,
                    noise,
                    p,
                    protocol_id,
                    update,
                    family,
                    separation,
                )
                curve_rows.append(
                    {
                        "lx": lx,
                        "lt": lt,
                        "noise": noise,
                        "p": p,
                        "measurement": measurement,
                        "protocol_id": protocol_id,
                        "gamma": gamma,
                        "update": update,
                        "family": family,
                        "separation": separation,
                        "separation_over_lx": separation / lx,
                        "outer_records": count,
                        "linear": float(linear_interval.estimate[index]),
                        "linear_standard_error": float(linear_interval.standard_error[index]),
                        "linear_lower": float(linear_interval.lower[index]),
                        "linear_upper": float(linear_interval.upper[index]),
                        "clean_linear": float(clean_values[index]),
                        "annealed_residual": float(
                            annealed_residual_interval.estimate[index]
                        ),
                        "annealed_residual_standard_error": float(
                            annealed_residual_interval.standard_error[index]
                        ),
                        "annealed_residual_lower": float(
                            annealed_residual_interval.lower[index]
                        ),
                        "annealed_residual_upper": float(
                            annealed_residual_interval.upper[index]
                        ),
                        "witness": float(witness_interval.estimate[index]),
                        "witness_standard_error": float(witness_interval.standard_error[index]),
                        "witness_simultaneous_lower": float(witness_interval.lower[index]),
                        "witness_simultaneous_upper": float(witness_interval.upper[index]),
                        "witness_inner_bias_envelope": bias_envelopes.get(bias_key),
                        "posterior_correlator_planted": float(
                            posterior_interval.estimate[index]
                        ),
                        "posterior_correlator_planted_standard_error": float(
                            posterior_interval.standard_error[index]
                        ),
                        "posterior_correlator_direct_diagnostic": float(
                            direct_posterior_interval.estimate[index]
                        ),
                        "posterior_correlator_direct_standard_error": float(
                            direct_posterior_interval.standard_error[index]
                        ),
                    }
                )
    return curve_rows, scalar_rows


def _immse_rows(
    config: CampaignConfig,
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]],
) -> list[dict[str, object]]:
    curves: dict[tuple[int, int, str, float, str], dict[float, list[dict[str, Any]]]] = defaultdict(dict)
    for key, records in grouped.items():
        lx, lt, noise, p, measurement, _protocol_id, update, gamma = key
        if gamma is None:
            continue
        bucket = curves[(lx, lt, noise, p, update)]
        existing = bucket.get(float(gamma))
        # Explicit gaussian grid wins over a named duplicate for integration.
        if existing is None or measurement == "gaussian":
            bucket[float(gamma)] = records
    output: list[dict[str, object]] = []
    for (lx, lt, noise, p, update), by_gamma in sorted(curves.items(), key=lambda item: repr(item[0])):
        if 0.0 not in by_gamma or len(by_gamma) < 2:
            continue
        grid = np.asarray(sorted(by_gamma), dtype=np.float64)
        id_sets = [set(int(row["global_id"]) for row in by_gamma[value]) for value in grid]
        common_ids = sorted(set.intersection(*id_sets))
        if len(common_ids) < 2:
            continue
        field = "planted_spin_overlap" if noise == "z" else "planted_bond_overlap"
        matrix = np.empty((len(common_ids), len(grid)), dtype=np.float64)
        for column, gamma in enumerate(grid):
            indexed = {int(row["global_id"]): float(row[field]) for row in by_gamma[float(gamma)]}
            matrix[:, column] = [indexed[global_id] for global_id in common_ids]
        estimate = common_resample_immse(
            grid,
            matrix,
            block_length=min(config.statistics.block_length, len(common_ids)),
            resamples=config.statistics.bootstrap_resamples,
            confidence=config.statistics.confidence,
            seed=config.campaign.seed ^ (lx << 24),
        )
        for index, gamma in enumerate(grid):
            output.append(
                {
                    "lx": lx,
                    "lt": lt,
                    "noise": noise,
                    "p": p,
                    "update": update,
                    "gamma": float(gamma),
                    "outer_records": len(common_ids),
                    "q_ea": float(estimate.q_ea[index]),
                    "information_per_noise_site": float(estimate.information_per_site[index]),
                    "information_lower": float(estimate.lower[index]),
                    "information_upper": float(estimate.upper[index]),
                    "quadrature_systematic": float(estimate.quadrature_systematic[index]),
                }
            )
    return output


def _curve_bias_envelopes(
    config: CampaignConfig,
    rows: list[dict[str, Any]],
) -> dict[tuple[Any, ...], float]:
    """Bootstrap an upper envelope for finite-inner absolute-value bias."""
    grouped: dict[
        tuple[Any, ...], dict[int, dict[int, list[np.ndarray[Any, Any]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if int(row["global_id"]) >= config.mc.diagnostic_outer_records:
            continue
        base = (
            int(row["lx"]),
            str(row["noise"]),
            float(row["p"]),
            str(row["protocol_id"]),
            str(row["update"]),
        )
        multiplier = int(row["inner_budget_multiplier"])
        for family in ("spin", "bond"):
            profiles = np.asarray(
                row[f"{family}_correlator_profile"], dtype=np.float64
            )
            values = np.mean(np.abs(profiles), axis=1)
            grouped[(*base, family)][multiplier][int(row["global_id"])].append(
                values
            )
    output: dict[tuple[Any, ...], float] = {}
    for key, budgets in grouped.items():
        if any(multiplier not in budgets for multiplier in (1, 2, 4)):
            continue
        common_ids = sorted(
            set.intersection(
                *(set(budgets[multiplier]) for multiplier in (1, 2, 4))
            )
        )
        if len(common_ids) < 2:
            continue
        matrices = {
            multiplier: np.asarray(
                [
                    np.mean(budgets[multiplier][global_id], axis=0)
                    for global_id in common_ids
                ],
                dtype=np.float64,
            )
            for multiplier in (1, 2, 4)
        }
        indices = moving_block_indices(
            len(common_ids),
            min(config.statistics.block_length, len(common_ids)),
            config.statistics.bootstrap_resamples,
            seed=config.campaign.seed ^ (int(key[0]) << 25),
        )
        bootstrap_differences = [
            np.mean((matrices[multiplier] - matrices[4])[indices], axis=1)
            for multiplier in (1, 2)
        ]
        envelope = np.maximum.reduce(
            [
                np.quantile(
                    np.abs(difference), config.statistics.confidence, axis=0
                )
                for difference in bootstrap_differences
            ]
        )
        output_separations = config.separations_for(int(key[0]))
        for separation, value in zip(output_separations, envelope, strict=True):
            output[(*key, separation)] = float(value)
    return output


def _diagnostic_summary_rows(
    config: CampaignConfig,
    rows: list[dict[str, Any]],
) -> list[dict[str, object]]:
    """Summarize replicated traces and the 1x/2x/4x inner-budget ladder."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("energy_trace"):
            continue
        key = (
            int(row["lx"]),
            int(row["lt"]),
            str(row["noise"]),
            float(row["p"]),
            str(row["measurement"]),
            str(row["protocol_id"]),
            str(row["update"]),
        )
        grouped[key].append(row)

    aggregate_extractors: dict[str, Callable[[dict[str, Any]], float]] = {
        "energy": lambda row: float(row["energy"]),
        "magnetization": lambda row: float(row["magnetization"]),
        "boundary_magnetization": lambda row: float(row["boundary_magnetization"]),
        "planted_spin_overlap": lambda row: float(row["planted_spin_overlap"]),
        "planted_bond_overlap": lambda row: float(row["planted_bond_overlap"]),
        "spin_absolute_profile": lambda row: float(np.mean(np.abs(row["spin_profile"]))),
        "bond_absolute_profile": lambda row: float(np.mean(np.abs(row["bond_profile"]))),
    }
    trace_fields = {
        "energy": "energy_trace",
        "magnetization": "magnetization_trace",
        "boundary_magnetization": "boundary_magnetization_trace",
        "planted_spin_overlap": "planted_spin_overlap_trace",
        "planted_bond_overlap": "planted_bond_overlap_trace",
    }
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: repr(item[0])):
        lx, lt, noise, p, measurement, protocol_id, update = key
        outer_ids = sorted({int(row["global_id"]) for row in group})
        for metric, extractor in aggregate_extractors.items():
            budget_means: dict[int, float | None] = {}
            for multiplier in config.mc.inner_budget_multipliers:
                values = [
                    extractor(row)
                    for row in group
                    if int(row["inner_budget_multiplier"]) == multiplier
                ]
                budget_means[multiplier] = float(np.mean(values)) if values else None

            replicate_rows: list[list[float]] = []
            rhat_values: list[float] = []
            for global_id in outer_ids:
                selected = sorted(
                    (
                        row
                        for row in group
                        if int(row["global_id"]) == global_id
                        and int(row["inner_budget_multiplier"]) == 1
                    ),
                    key=lambda row: int(row["replica"]),
                )
                if len(selected) != config.mc.replicated_chains:
                    continue
                replicate_rows.append([extractor(row) for row in selected])
                trace_field = trace_fields.get(metric)
                if trace_field is not None:
                    traces = np.asarray([row[trace_field] for row in selected], dtype=np.float64)
                    if traces.ndim == 2 and traces.shape[1] >= 4:
                        rhat_values.append(split_rhat(traces))

            if len(replicate_rows) >= 2:
                between, within = nested_variance_components(np.asarray(replicate_rows))
            else:
                between, within = None, None
            reference = budget_means.get(4)
            differences = [
                abs(value - reference)
                for multiplier in (1, 2)
                if (value := budget_means.get(multiplier)) is not None
                and reference is not None
            ]
            finite_rhat = [value for value in rhat_values if math.isfinite(value)]
            infinite_rhat_count = sum(math.isinf(value) for value in rhat_values)
            maximum_rhat = max(finite_rhat) if finite_rhat else None
            output.append(
                {
                    "lx": lx,
                    "lt": lt,
                    "noise": noise,
                    "p": p,
                    "measurement": measurement,
                    "protocol_id": protocol_id,
                    "update": update,
                    "metric": metric,
                    "diagnostic_outer_records": len(outer_ids),
                    "replicated_chains": config.mc.replicated_chains,
                    "budget_1x": budget_means.get(1),
                    "budget_2x": budget_means.get(2),
                    "budget_4x": budget_means.get(4),
                    "budget_bias_envelope": max(differences) if differences else None,
                    "between_record_variance": between,
                    "within_record_inner_variance": within,
                    "maximum_split_rhat": maximum_rhat,
                    "median_split_rhat": (
                        float(np.median(finite_rhat)) if finite_rhat else None
                    ),
                    "split_rhat_evaluations": len(rhat_values),
                    "infinite_split_rhat_count": infinite_rhat_count,
                    "rhat_target": 1.01,
                    "rhat_converged": bool(rhat_values)
                    and infinite_rhat_count == 0
                    and maximum_rhat is not None
                    and maximum_rhat <= 1.01,
                }
            )
    return output


def _comparison_rows(
    config: CampaignConfig,
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]],
    exact_results: list[dict[str, Any]],
    exact_records: list[dict[str, Any]],
    bias_envelopes: dict[tuple[Any, ...], float],
) -> list[dict[str, object]]:
    """Bootstrap complete MC-minus-exact curves with simultaneous bands."""
    exact_index = {
        (
            int(row["lx"]),
            str(row["noise"]),
            float(row["p"]),
            str(row["protocol_id"]),
            str(row["observable"]),
            int(row["separation"]),
        ): row
        for row in exact_results
        if row["prior"] == "finite-transfer"
        and row["observable"] in {"spin-pair", "bond-pair"}
        and row["separation"] is not None
    }
    raw_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in exact_records:
        if (
            row["prior"] == "finite-transfer"
            and row["observable"] in {"spin-pair", "bond-pair"}
            and row["separation"] is not None
        ):
            raw_key = (
                int(row["lx"]),
                str(row["noise"]),
                float(row["p"]),
                str(row["protocol_id"]),
                str(row["observable"]),
            )
            raw_groups[raw_key].append(row)

    output: list[dict[str, object]] = []
    for mc_key, records in sorted(grouped.items(), key=lambda item: repr(item[0])):
        lx, lt, noise, p, measurement, protocol_id, update, gamma = mc_key
        separations = tuple(int(value) for value in records[0]["separations"])
        block_length = min(config.statistics.block_length, len(records))
        mc_indices = moving_block_indices(
            len(records),
            block_length,
            config.statistics.bootstrap_resamples,
            seed=config.campaign.seed ^ (lx << 26),
        )
        for family in ("spin", "bond"):
            observable = f"{family}-pair"
            references = [
                exact_index.get((lx, noise, p, protocol_id, observable, separation))
                for separation in separations
            ]
            if any(reference is None for reference in references):
                continue
            exact_curve = np.asarray(
                [float(cast(dict[str, Any], reference)["measurement_witness"]) for reference in references]
            )
            physical_curve = np.asarray(
                [float(cast(dict[str, Any], reference)["physical_fidelity"]) for reference in references]
            )
            mc_profiles = np.asarray(
                [row[f"{family}_correlator_profile"] for row in records], dtype=np.float64
            )
            mc_values = np.mean(np.abs(mc_profiles), axis=2)
            mc_bootstrap = np.mean(mc_values[mc_indices], axis=1)

            raw = raw_groups.get((lx, noise, p, protocol_id, observable), [])
            record_mode = str(cast(dict[str, Any], references[0])["record_mode"])
            if raw and record_mode.startswith("sampled"):
                by_separation: dict[int, dict[int, float]] = defaultdict(dict)
                for row in raw:
                    by_separation[int(row["separation"])][int(row["global_id"])] = float(
                        row["absolute_contribution"]
                    )
                common_ids = sorted(
                    set.intersection(
                        *(set(by_separation[separation]) for separation in separations)
                    )
                )
                exact_values = np.column_stack(
                    [
                        [by_separation[separation][global_id] for global_id in common_ids]
                        for separation in separations
                    ]
                )
                exact_indices = moving_block_indices(
                    len(common_ids),
                    1,
                    config.statistics.bootstrap_resamples,
                    seed=config.campaign.seed ^ (lx << 27),
                )
                exact_bootstrap = np.mean(exact_values[exact_indices], axis=1)
            else:
                exact_bootstrap = np.broadcast_to(
                    exact_curve, (config.statistics.bootstrap_resamples, len(separations))
                )
            difference_bootstrap = mc_bootstrap - exact_bootstrap
            center = np.mean(mc_values, axis=0) - exact_curve
            standard_error = np.std(difference_bootstrap, axis=0, ddof=1)
            safe = np.where(standard_error > 0.0, standard_error, 1.0)
            maximum = np.max(
                np.abs((difference_bootstrap - center) / safe), axis=1
            )
            critical = float(np.quantile(maximum, config.statistics.confidence))
            lower = center - critical * standard_error
            upper = center + critical * standard_error
            for index, separation in enumerate(separations):
                bias_envelope = bias_envelopes.get(
                    (lx, noise, p, protocol_id, update, family, separation), 0.0
                )
                output.append(
                    {
                        "lx": lx,
                        "lt": lt,
                        "noise": noise,
                        "p": p,
                        "measurement": measurement,
                        "protocol_id": protocol_id,
                        "gamma": gamma,
                        "update": update,
                        "family": family,
                        "separation": separation,
                        "mc_witness": float(np.mean(mc_values, axis=0)[index]),
                        "exact_finite_transfer_witness": float(exact_curve[index]),
                        "exact_physical_fidelity": float(physical_curve[index]),
                        "difference": float(center[index]),
                        "difference_standard_error": float(standard_error[index]),
                        "difference_simultaneous_lower": float(
                            lower[index] - bias_envelope
                        ),
                        "difference_simultaneous_upper": float(
                            upper[index] + bias_envelope
                        ),
                        "mc_inner_bias_envelope": bias_envelope,
                        "simultaneous_confidence": config.statistics.confidence,
                        "mc_outer_records": len(records),
                        "exact_outer_records": int(
                            cast(dict[str, Any], references[index])["outer_records"]
                        ),
                        "exact_record_mode": record_mode,
                    }
                )
    return output


def _finite_size_rows(curves: list[dict[str, Any]]) -> list[dict[str, object]]:
    long_distance: dict[tuple[Any, ...], dict[int, tuple[int, float, float]]] = defaultdict(dict)
    for row in curves:
        key = (
            row["noise"],
            row["p"],
            row["protocol_id"],
            row["update"],
            row["family"],
        )
        lx = int(row["lx"])
        separation = int(row["separation"])
        current = long_distance[key].get(lx)
        if current is None or separation > current[0]:
            long_distance[key][lx] = (
                separation,
                float(row["witness"]),
                float(row["witness_standard_error"]),
            )
    output: list[dict[str, object]] = []
    for key, by_size in sorted(long_distance.items(), key=lambda item: repr(item[0])):
        sizes = np.asarray(sorted(by_size), dtype=np.float64)
        values = np.asarray([by_size[int(size)][1] for size in sizes])
        positive = values > 0.0
        exponent: float | None = None
        amplitude: float | None = None
        if np.count_nonzero(positive) >= 2:
            slope, intercept = np.polyfit(np.log(sizes[positive]), np.log(values[positive]), 1)
            exponent = float(-slope)
            amplitude = float(np.exp(intercept))
        noise, p, protocol_id, update, family = key
        for size in sizes:
            separation, value, standard_error = by_size[int(size)]
            output.append(
                {
                    "noise": noise,
                    "p": p,
                    "protocol_id": protocol_id,
                    "update": update,
                    "family": family,
                    "lx": int(size),
                    "separation": separation,
                    "witness": value,
                    "witness_standard_error": standard_error,
                    "fit_amplitude_exploratory": amplitude,
                    "decay_exponent_exploratory": exponent,
                    "fit_status": "exploratory-unweighted" if exponent is not None else "insufficient_sizes",
                }
            )
    return output


def analyze_campaign(config: CampaignConfig) -> dict[str, object]:
    plan = read_plan(config)
    state = read_state(config)
    analysis_task = next(task for task in plan.tasks if task.kind == "analysis")
    incomplete = [
        dependency
        for dependency in analysis_task.dependencies
        if state["tasks"][dependency]["status"] != "complete"
    ]
    if incomplete:
        raise AnalysisError(f"analysis dependencies are incomplete: {incomplete[:5]}")
    code_digest = source_digest(config.campaign.project_root)
    discovered = {
        artifact.artifact_id: artifact
        for artifact in discover_artifacts(config.campaign.output_root)
    }
    stale_dependencies = [
        str(artifact_id)
        for dependency in analysis_task.dependencies
        for artifact_id in state["tasks"][dependency]["artifacts"]
        if str(artifact_id) not in discovered
        or discovered[str(artifact_id)].manifest["source_digest"] != code_digest
    ]
    if stale_dependencies:
        raise AnalysisError(
            "analysis dependencies were generated by another source digest; "
            "rerun campaign compute tasks"
        )
    current = state["tasks"][analysis_task.task_id]
    if current["status"] == "complete":
        existing = discovered
        dependency_artifacts = {
            str(artifact_id)
            for dependency in analysis_task.dependencies
            for artifact_id in state["tasks"][dependency]["artifacts"]
        }
        current_parents = {
            str(parent)
            for artifact_id in current["artifacts"]
            if artifact_id in existing
            for parent in existing[artifact_id].manifest["parents"]
        }
        if dependency_artifacts <= current_parents and all(
            artifact_id in existing
            and existing[artifact_id].manifest["source_digest"] == code_digest
            for artifact_id in current["artifacts"]
        ):
            return {"status": "already-complete", "artifacts": current["artifacts"]}
    referenced_ids = {
        str(artifact_id)
        for task_state in state["tasks"].values()
        for artifact_id in task_state["artifacts"]
    }
    referenced_artifacts = [
        artifact
        for artifact in discover_artifacts(config.campaign.output_root)
        if artifact.artifact_id in referenced_ids
    ]
    mc_artifacts = [
        artifact for artifact in referenced_artifacts if artifact.manifest["kind"] == "mc-records"
    ]
    clean_artifacts = [
        artifact for artifact in referenced_artifacts if artifact.manifest["kind"] == "clean"
    ]
    if not mc_artifacts:
        raise AnalysisError("no MC per-record artifacts were found")
    update_task_state(config, analysis_task.task_id, "running")
    try:
        all_rows: list[dict[str, Any]] = []
        for artifact in mc_artifacts:
            all_rows.extend(cast(list[dict[str, Any]], read_table(artifact).to_pylist()))
        rows = [
            row
            for row in all_rows
            if int(row["inner_budget_multiplier"]) == 1 and int(row["replica"]) == 0
        ]
        grouped = _group_rows(rows)
        clean = _clean_reference(config, clean_artifacts)
        bias_envelopes = _curve_bias_envelopes(config, all_rows)
        curve_rows, scalar_rows = _curve_and_scalar_rows(
            config, grouped, clean, bias_envelopes
        )
        immse_rows = _immse_rows(config, grouped)
        diagnostic_rows = _diagnostic_summary_rows(config, all_rows)
        exact_result_artifacts = [
            artifact
            for artifact in referenced_artifacts
            if artifact.manifest["kind"] == "ed-results"
        ]
        exact_record_artifacts = [
            artifact
            for artifact in referenced_artifacts
            if artifact.manifest["kind"] == "ed-records"
        ]
        exact_result_rows: list[dict[str, Any]] = []
        for artifact in exact_result_artifacts:
            exact_result_rows.extend(
                cast(list[dict[str, Any]], read_table(artifact).to_pylist())
            )
        exact_record_rows: list[dict[str, Any]] = []
        for artifact in exact_record_artifacts:
            exact_record_rows.extend(
                cast(list[dict[str, Any]], read_table(artifact).to_pylist())
            )
        comparison_rows = _comparison_rows(
            config,
            grouped,
            exact_result_rows,
            exact_record_rows,
            bias_envelopes,
        )
        finite_rows = _finite_size_rows(curve_rows)
        raw_parent_kinds = {
            "clean",
            "mc-records",
            "prior-diagnostics",
            "ed-results",
            "ed-records",
        }
        parents = tuple(
            artifact.artifact_id
            for artifact in referenced_artifacts
            if artifact.manifest["kind"] in raw_parent_kinds
        )
        common = {
            "analysis_task": analysis_task.task_id,
            "estimator": "moving-block bootstrap over contiguous outer ids",
            "block_length": config.statistics.block_length,
            "bootstrap_resamples": config.statistics.bootstrap_resamples,
            "confidence": config.statistics.confidence,
            "paired_protocols": True,
            "inner_budget_diagnostics": [1, 2, 4],
        }
        artifacts = [
            write_artifact(
                config.campaign.output_root,
                "analysis-scalars",
                pa.Table.from_pylist(scalar_rows),
                metadata=common,
                project_root=config.campaign.project_root,
                parents=parents,
                partition_by=("noise", "measurement", "update"),
            ),
            write_artifact(
                config.campaign.output_root,
                "analysis-curves",
                pa.Table.from_pylist(curve_rows),
                metadata={**common, "interval": "pointwise linear; simultaneous max-t witness"},
                project_root=config.campaign.project_root,
                parents=parents,
                partition_by=("noise", "measurement", "update"),
            ),
            write_artifact(
                config.campaign.output_root,
                "finite-size-tables",
                pa.Table.from_pylist(finite_rows),
                metadata={
                    **common,
                    "fit_warning": "unweighted log-log fits are exploratory, never final exponents",
                },
                project_root=config.campaign.project_root,
                parents=parents,
                partition_by=("noise", "family"),
            ),
        ]
        if immse_rows:
            artifacts.append(
                write_artifact(
                    config.campaign.output_root,
                    "analysis-immse",
                    pa.Table.from_pylist(immse_rows),
                    metadata={
                        **common,
                        "integration": "common-resample trapezoid",
                        "quadrature_systematic": "absolute Simpson-minus-trapezoid difference",
                    },
                    project_root=config.campaign.project_root,
                    parents=parents,
                    partition_by=("noise", "update"),
                )
            )
        if diagnostic_rows:
            artifacts.append(
                write_artifact(
                    config.campaign.output_root,
                    "analysis-diagnostics",
                    pa.Table.from_pylist(diagnostic_rows),
                    metadata={
                        **common,
                        "rhat": "split R-hat over replicated retained traces",
                        "nested_variance": "method-of-moments between/within record decomposition",
                        "absolute_bias": (
                            "contiguous-outer-id bootstrap upper envelope relative to "
                            "the 4x inner budget"
                        ),
                    },
                    project_root=config.campaign.project_root,
                    parents=parents,
                    partition_by=("noise", "measurement", "update"),
                )
            )
        if comparison_rows:
            artifacts.append(
                write_artifact(
                    config.campaign.output_root,
                    "comparison-curves",
                    pa.Table.from_pylist(comparison_rows),
                    metadata={
                        **common,
                        "comparison": "MC minus matched finite-transfer exact posterior",
                        "interval": (
                            "simultaneous max-t bootstrap over the complete separation curve "
                            "plus the bootstrapped 1x/2x/4x inner-bias envelope"
                        ),
                        "ed_resampling": "iid for sampled Gaussian records; deterministic for enumerated local-X",
                    },
                    project_root=config.campaign.project_root,
                    parents=parents
                    + tuple(
                        artifact.artifact_id
                        for artifact in (*exact_result_artifacts, *exact_record_artifacts)
                    ),
                    partition_by=("noise", "measurement", "update"),
                )
            )
    except Exception as error:
        update_task_state(config, analysis_task.task_id, "failed", error=str(error))
        raise
    artifact_ids = [artifact.artifact_id for artifact in artifacts]
    update_task_state(config, analysis_task.task_id, "complete", artifacts=artifact_ids)
    return {
        "status": "complete",
        "artifacts": artifact_ids,
        "scalar_rows": len(scalar_rows),
        "curve_rows": len(curve_rows),
        "immse_rows": len(immse_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "comparison_rows": len(comparison_rows),
        "finite_size_rows": len(finite_rows),
    }
