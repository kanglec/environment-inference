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
    paired_curve_difference,
    simultaneous_curve_interval,
    split_rhat,
)


class AnalysisError(RuntimeError):
    """Campaign data are incomplete or inconsistent for analysis."""


def _is_production_row(row: dict[str, Any]) -> bool:
    role = row.get("chain_role")
    if role is not None:
        return str(role) == "production"
    return int(row["inner_budget_multiplier"]) == 1 and int(row["replica"]) == 0


def _is_finite_inner_row(row: dict[str, Any]) -> bool:
    role = row.get("chain_role")
    return role in {None, "production", "finite-inner"}


def _production_rows(artifacts: Iterable[Artifact]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        for row in read_table(artifact).to_pylist():
            if _is_production_row(cast(dict[str, Any], row)):
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
        ordered_ids = sorted(observations_by_id)
        observations = [observations_by_id[global_id] for global_id in ordered_ids]
        autocorrelations: dict[str, dict[str, object]] = {}

        def add_autocorrelation(
            metric: str,
            values: np.ndarray[Any, Any],
            *,
            output: dict[str, dict[str, object]] = autocorrelations,
            saved_count: int = len(observations),
        ) -> None:
            estimate = geyer_autocorrelation(
                values,
                saving_interval_sweeps=config.mc.clean_saving_interval,
            )
            output[metric] = {
                "metric": metric,
                "tau_saved": estimate.tau_saved,
                "tau_sweeps": estimate.tau_sweeps,
                "effective_samples": estimate.effective_samples,
                "window_lag": estimate.window_lag,
                "saved_configurations": saved_count,
            }

        add_autocorrelation(
            "energy",
            np.asarray([item["energy"] for item in observations], dtype=np.float64),
        )
        add_autocorrelation(
            "boundary-magnetization",
            np.asarray(
                [item["boundary_magnetization"] for item in observations],
                dtype=np.float64,
            ),
        )
        add_autocorrelation(
            "boundary-bond-magnetization",
            np.asarray(
                [np.mean(item["bond_profile"]) for item in observations],
                dtype=np.float64,
            ),
        )
        for index, separation in enumerate(separations):
            for family in ("spin", "bond"):
                add_autocorrelation(
                    f"{family}-correlator/r={separation}",
                    np.asarray(
                        [item[f"{family}_correlator"][index] for item in observations],
                        dtype=np.float64,
                    ),
                )
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
            "outer_autocorrelations": autocorrelations,
        }
    return references


def _outer_autocorrelation_rows(clean: dict[int, dict[str, Any]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for lx, reference in sorted(clean.items()):
        for metric, values in sorted(reference["outer_autocorrelations"].items()):
            output.append({"lx": lx, "metric": metric, **values})
    return output


def _block_stability(
    values: np.ndarray[Any, Any],
    *,
    block_length: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[list[int], float | None, bool]:
    count = values.shape[0]
    lengths = sorted({max(1, block_length // 2), block_length, min(count, 2 * block_length)})
    if len(lengths) < 3:
        return lengths, None, False
    standard_errors = [
        moving_block_bootstrap(
            values,
            block_length=length,
            resamples=min(resamples, 500),
            confidence=confidence,
            seed=seed ^ length,
        ).standard_error
        for length in lengths
    ]
    matrix = np.asarray(standard_errors, dtype=np.float64)
    positive = np.min(matrix, axis=0) > 0.0
    ratio = (
        float(np.max(np.max(matrix[:, positive], axis=0) / np.min(matrix[:, positive], axis=0)))
        if np.any(positive)
        else 1.0
    )
    return lengths, ratio, ratio <= 2.0 and count >= 4 * block_length


def _finite_inner_absolute_tolerance(metric: str, lx: int, lt: int) -> float:
    """Return a scale-aware floor for finite-inner diagnostic comparisons."""
    if metric == "energy":
        return 0.005 * lx * lt
    return 0.01


def _is_identical_constant_saturation(traces: np.ndarray[Any, Any]) -> bool:
    """Return whether every overdispersed trace is the same exact constant."""
    return bool(traces.size and np.all(traces == traces[0, 0]))


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
        posterior_energy_autocorrelation = geyer_autocorrelation(
            energies, saving_interval_sweeps=config.mc.clean_saving_interval
        )
        relevant_prefix = "spin-correlator/" if noise == "z" else "bond-correlator/"
        relevant_named = (
            "boundary-magnetization" if noise == "z" else "boundary-bond-magnetization"
        )
        clean_autocorrelation = max(
            (
                values
                for metric, values in clean[lx]["outer_autocorrelations"].items()
                if metric in {"energy", relevant_named} or metric.startswith(relevant_prefix)
            ),
            key=lambda values: float(values["tau_saved"]),
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
        stability_values = np.column_stack(
            (
                scalar_values,
                planted_q,
                direct_q,
                np.mean(
                    np.abs(
                        np.asarray(
                            [row["spin_correlator_profile"] for row in records],
                            dtype=np.float64,
                        )
                    ),
                    axis=2,
                ),
                np.mean(
                    np.abs(
                        np.asarray(
                            [row["bond_correlator_profile"] for row in records],
                            dtype=np.float64,
                        )
                    ),
                    axis=2,
                ),
            )
        )
        stability_lengths, stability_ratio, block_stable = _block_stability(
            stability_values,
            block_length=block_length,
            resamples=config.statistics.bootstrap_resamples,
            confidence=config.statistics.confidence,
            seed=config.campaign.seed ^ (lx << 10),
        )
        minimum_correlation_block = max(
            1, math.ceil(2.0 * float(clean_autocorrelation["tau_saved"]))
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
                "outer_autocorrelation_source": "clean-chain boundary observables",
                "outer_autocorrelation_metric": str(clean_autocorrelation["metric"]),
                "tau_int_saved": float(clean_autocorrelation["tau_saved"]),
                "tau_int_sweeps": float(clean_autocorrelation["tau_sweeps"]),
                "outer_effective_samples": float(clean_autocorrelation["effective_samples"]),
                "autocorrelation_window": int(clean_autocorrelation["window_lag"]),
                "posterior_energy_tau_int_saved_secondary": (
                    posterior_energy_autocorrelation.tau_saved
                ),
                "posterior_energy_tau_int_sweeps_secondary": (
                    posterior_energy_autocorrelation.tau_sweeps
                ),
                "outer_block_length_used": block_length,
                "outer_minimum_correlation_block": minimum_correlation_block,
                "outer_block_correlation_adequate": block_length
                >= minimum_correlation_block,
                "outer_block_stability_lengths": stability_lengths,
                "outer_block_standard_error_ratio": stability_ratio,
                "outer_block_stable": block_stable,
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
        if not _is_finite_inner_row(row):
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
    """Summarize overdispersed traces and the planted 1x/2x/4x budget ladder."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["global_id"]) >= config.mc.diagnostic_outer_records:
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
            finite_inner = [row for row in group if _is_finite_inner_row(row)]
            budget_means: dict[int, float | None] = {}
            for multiplier in config.mc.inner_budget_multipliers:
                values = [
                    extractor(row)
                    for row in finite_inner
                    if int(row["inner_budget_multiplier"]) == multiplier
                ]
                budget_means[multiplier] = float(np.mean(values)) if values else None

            replicate_rows: list[list[float]] = []
            budget_by_id: dict[int, dict[int, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for row in finite_inner:
                budget_by_id[int(row["inner_budget_multiplier"])][
                    int(row["global_id"])
                ].append(extractor(row))
            common_budget_ids = sorted(
                set.intersection(
                    *(
                        set(budget_by_id[multiplier])
                        for multiplier in config.mc.inner_budget_multipliers
                    )
                )
            )
            for global_id in common_budget_ids:
                values = budget_by_id[1][global_id]
                if len(values) == config.mc.replicated_chains:
                    replicate_rows.append(values)

            budget_differences: dict[int, float | None] = {1: None, 2: None}
            budget_standard_errors: dict[int, float | None] = {1: None, 2: None}
            budget_tolerances: dict[int, float | None] = {1: None, 2: None}
            if len(common_budget_ids) >= 2:
                record_budget_means = {
                    multiplier: np.asarray(
                        [
                            np.mean(budget_by_id[multiplier][global_id])
                            for global_id in common_budget_ids
                        ],
                        dtype=np.float64,
                    )
                    for multiplier in config.mc.inner_budget_multipliers
                }
                paired_differences = np.column_stack(
                    tuple(
                        record_budget_means[multiplier] - record_budget_means[4]
                        for multiplier in (1, 2)
                    )
                )
                budget_interval = moving_block_bootstrap(
                    paired_differences,
                    block_length=min(config.statistics.block_length, len(common_budget_ids)),
                    resamples=config.statistics.bootstrap_resamples,
                    confidence=config.statistics.confidence,
                    seed=config.campaign.seed ^ (lx << 28) ^ len(metric),
                )
                for index, multiplier in enumerate((1, 2)):
                    difference = float(budget_interval.estimate[index])
                    standard_error = float(budget_interval.standard_error[index])
                    budget_differences[multiplier] = difference
                    budget_standard_errors[multiplier] = standard_error
                    budget_tolerances[multiplier] = max(
                        5.0 * standard_error,
                        _finite_inner_absolute_tolerance(metric, lx, lt),
                        64.0 * math.ulp(max(1.0, abs(difference))),
                    )

            if len(replicate_rows) >= 2:
                between, within = nested_variance_components(np.asarray(replicate_rows))
            else:
                between, within = None, None

            convergence_rows = [
                row for row in group if row.get("chain_role") == "convergence"
            ]
            rhat_values: list[float] = []
            identical_saturation_count = 0
            trace_lengths: list[int] = []
            trace_transitions: list[int] = []
            initialization_by_replica: dict[int, str] = {}
            initializations_consistent = True
            for global_id in outer_ids:
                selected = sorted(
                    (
                        row
                        for row in convergence_rows
                        if int(row["global_id"]) == global_id
                        and int(row["inner_budget_multiplier"]) == 1
                    ),
                    key=lambda row: int(row["replica"]),
                )
                if len(selected) != config.mc.replicated_chains:
                    continue
                trace_field = trace_fields.get(metric)
                if trace_field is not None:
                    traces = np.asarray([row[trace_field] for row in selected], dtype=np.float64)
                    if traces.ndim == 2 and traces.shape[1] >= 4:
                        rhat_value = split_rhat(traces)
                        rhat_values.append(rhat_value)
                        if math.isinf(rhat_value) and _is_identical_constant_saturation(
                            traces
                        ):
                            identical_saturation_count += 1
                        for trace in traces:
                            trace_lengths.append(int(trace.size))
                            trace_transitions.append(int(np.count_nonzero(np.diff(trace))))
                    for row in selected:
                        replica = int(row["replica"])
                        initialization = str(row.get("initialization", "unknown"))
                        previous = initialization_by_replica.setdefault(replica, initialization)
                        initializations_consistent &= previous == initialization

            finite_rhat = [value for value in rhat_values if math.isfinite(value)]
            infinite_rhat_count = sum(math.isinf(value) for value in rhat_values)
            unresolved_infinite_rhat_count = (
                infinite_rhat_count - identical_saturation_count
            )
            maximum_rhat = max(finite_rhat) if finite_rhat else None
            expected_initializations = {0: "all-plus", 1: "all-minus"}
            expected_initializations.update(
                {replica: "random" for replica in range(2, config.mc.replicated_chains)}
            )
            expected_trace_evaluations = len(outer_ids) * config.mc.replicated_chains
            trace_complete = (
                bool(trace_fields.get(metric))
                and len(trace_lengths) == expected_trace_evaluations
                and len(rhat_values) == len(outer_ids)
            )
            movement_detected = (
                trace_complete
                and bool(trace_transitions)
                and min(trace_transitions) > 0
            )
            budget_consistent = all(
                budget_differences[multiplier] is not None
                and budget_tolerances[multiplier] is not None
                and abs(cast(float, budget_differences[multiplier]))
                <= cast(float, budget_tolerances[multiplier])
                for multiplier in (1, 2)
            )
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
                    "budget_1x_minus_4x": budget_differences[1],
                    "budget_2x_minus_4x": budget_differences[2],
                    "budget_1x_minus_4x_standard_error": budget_standard_errors[1],
                    "budget_2x_minus_4x_standard_error": budget_standard_errors[2],
                    "budget_1x_minus_4x_tolerance": budget_tolerances[1],
                    "budget_2x_minus_4x_tolerance": budget_tolerances[2],
                    "budget_bias_envelope": max(
                        (
                            abs(value)
                            for value in budget_differences.values()
                            if value is not None
                        ),
                        default=None,
                    ),
                    "budget_outer_records": len(common_budget_ids),
                    "budget_complete": len(common_budget_ids) == len(outer_ids),
                    "budget_consistent": budget_consistent,
                    "between_record_variance": between,
                    "within_record_inner_variance": within,
                    "maximum_split_rhat": maximum_rhat,
                    "median_split_rhat": (
                        float(np.median(finite_rhat)) if finite_rhat else None
                    ),
                    "split_rhat_evaluations": len(rhat_values),
                    "infinite_split_rhat_count": infinite_rhat_count,
                    "identical_saturation_count": identical_saturation_count,
                    "unresolved_infinite_split_rhat_count": (
                        unresolved_infinite_rhat_count
                    ),
                    "trace_evaluations": len(trace_lengths),
                    "expected_trace_evaluations": expected_trace_evaluations,
                    "minimum_trace_length": min(trace_lengths) if trace_lengths else None,
                    "minimum_trace_transitions": (
                        min(trace_transitions) if trace_transitions else None
                    ),
                    "constant_trace_count": sum(value == 0 for value in trace_transitions),
                    "trace_complete": trace_complete,
                    "movement_detected": movement_detected,
                    "initializations": [
                        initialization_by_replica[replica]
                        for replica in sorted(initialization_by_replica)
                    ],
                    "overdispersed_initializations_complete": (
                        initializations_consistent
                        and
                        initialization_by_replica == expected_initializations
                    ),
                    "rhat_target": 1.01,
                    "rhat_converged": bool(rhat_values)
                    and trace_complete
                    and initializations_consistent
                    and initialization_by_replica == expected_initializations
                    and infinite_rhat_count == 0
                    and maximum_rhat is not None
                    and maximum_rhat <= 1.01,
                    "rhat_or_saturation_converged": bool(rhat_values)
                    and trace_complete
                    and initializations_consistent
                    and initialization_by_replica == expected_initializations
                    and unresolved_infinite_rhat_count == 0
                    and (maximum_rhat is None or maximum_rhat <= 1.01),
                }
            )
    return output


def _protocol_difference_bias_envelopes(
    config: CampaignConfig,
    rows: list[dict[str, Any]],
) -> dict[tuple[Any, ...], float]:
    grouped: dict[
        tuple[Any, ...], dict[int, dict[int, list[np.ndarray[Any, Any]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if (
            int(row["global_id"]) >= config.mc.diagnostic_outer_records
            or row["measurement"] not in {"heterodyne", "homodyne"}
            or not _is_finite_inner_row(row)
        ):
            continue
        multiplier = int(row["inner_budget_multiplier"])
        for family in ("spin", "bond"):
            values = np.mean(
                np.abs(np.asarray(row[f"{family}_correlator_profile"], dtype=np.float64)),
                axis=1,
            )
            key = (
                int(row["lx"]),
                str(row["noise"]),
                float(row["p"]),
                str(row["update"]),
                str(row["measurement"]),
                family,
            )
            grouped[key][multiplier][int(row["global_id"])].append(values)

    output: dict[tuple[Any, ...], float] = {}
    bases = {(*key[:4], key[5]) for key in grouped}
    for lx, noise, p, update, family in sorted(bases, key=repr):
        protocols = {
            measurement: grouped.get((lx, noise, p, update, measurement, family))
            for measurement in ("heterodyne", "homodyne")
        }
        if any(value is None for value in protocols.values()):
            continue
        required_sets = [
            set(cast(dict[int, dict[int, list[np.ndarray[Any, Any]]]], protocols[measurement])[multiplier])
            for measurement in ("heterodyne", "homodyne")
            for multiplier in config.mc.inner_budget_multipliers
        ]
        common_ids = sorted(set.intersection(*required_sets))
        if len(common_ids) < 2:
            continue
        budget_deltas: dict[int, np.ndarray[Any, Any]] = {}
        for multiplier in config.mc.inner_budget_multipliers:
            protocol_means = {}
            for measurement in ("heterodyne", "homodyne"):
                bucket = cast(
                    dict[int, dict[int, list[np.ndarray[Any, Any]]]],
                    protocols[measurement],
                )[multiplier]
                protocol_means[measurement] = np.asarray(
                    [np.mean(bucket[global_id], axis=0) for global_id in common_ids],
                    dtype=np.float64,
                )
            budget_deltas[multiplier] = paired_curve_difference(
                protocol_means["homodyne"], protocol_means["heterodyne"]
            )
        indices = moving_block_indices(
            len(common_ids),
            min(config.statistics.block_length, len(common_ids)),
            config.statistics.bootstrap_resamples,
            seed=config.campaign.seed ^ (lx << 29) ^ (0 if family == "spin" else 1),
        )
        bootstrap_differences = [
            np.mean((budget_deltas[multiplier] - budget_deltas[4])[indices], axis=1)
            for multiplier in (1, 2)
        ]
        envelope = np.maximum.reduce(
            [
                np.quantile(np.abs(values), config.statistics.confidence, axis=0)
                for values in bootstrap_differences
            ]
        )
        for separation, value in zip(config.separations_for(lx), envelope, strict=True):
            output[(lx, noise, p, update, family, separation)] = float(value)
    return output


def _protocol_difference_rows(
    config: CampaignConfig,
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]],
    exact_records: list[dict[str, Any]],
    bias_envelopes: dict[tuple[Any, ...], float],
) -> list[dict[str, object]]:
    """Build paired homodyne-minus-heterodyne curves for MC and sampled ED."""
    output: list[dict[str, object]] = []
    mc_index: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for key, records in grouped.items():
        lx, lt, noise, p, measurement, _protocol_id, update, _gamma = key
        if measurement in {"heterodyne", "homodyne"}:
            mc_index[(lx, lt, noise, p, update)][measurement] = records

    for base, protocols in sorted(mc_index.items(), key=lambda item: repr(item[0])):
        if set(protocols) != {"heterodyne", "homodyne"}:
            continue
        lx, lt, noise, p, update = base
        heterodyne = {int(row["global_id"]): row for row in protocols["heterodyne"]}
        homodyne = {int(row["global_id"]): row for row in protocols["homodyne"]}
        if heterodyne.keys() != homodyne.keys():
            raise AnalysisError(f"paired MC protocols have different outer ids for {base!r}")
        common_ids = sorted(heterodyne)
        for global_id in common_ids:
            left = heterodyne[global_id]
            right = homodyne[global_id]
            if "planted_configuration_hash" in left and (
                left["planted_configuration_hash"] != right["planted_configuration_hash"]
                or left["standard_variates"] != right["standard_variates"]
            ):
                raise AnalysisError(
                    f"paired MC protocols do not share planted data for {base!r}, id={global_id}"
                )
        separations = tuple(int(value) for value in protocols["heterodyne"][0]["separations"])
        if separations != tuple(int(value) for value in protocols["homodyne"][0]["separations"]):
            raise AnalysisError(f"paired MC protocols have different separation grids for {base!r}")
        for family in ("spin", "bond"):
            values = {}
            for measurement, by_id in (
                ("heterodyne", heterodyne),
                ("homodyne", homodyne),
            ):
                profiles = np.asarray(
                    [by_id[global_id][f"{family}_correlator_profile"] for global_id in common_ids],
                    dtype=np.float64,
                )
                values[measurement] = np.mean(np.abs(profiles), axis=2)
            difference = paired_curve_difference(values["homodyne"], values["heterodyne"])
            interval = simultaneous_curve_interval(
                difference,
                block_length=min(config.statistics.block_length, len(common_ids)),
                resamples=config.statistics.bootstrap_resamples,
                confidence=config.statistics.confidence,
                seed=config.campaign.seed ^ (int(lx) << 30) ^ (0 if family == "spin" else 1),
            )
            for index, separation in enumerate(separations):
                inner_bias = bias_envelopes.get(
                    (lx, noise, p, update, family, separation), 0.0
                )
                output.append(
                    {
                        "source": "mc",
                        "lx": lx,
                        "lt": lt,
                        "noise": noise,
                        "p": p,
                        "prior": None,
                        "update": update,
                        "family": family,
                        "separation": separation,
                        "difference": float(interval.estimate[index]),
                        "difference_standard_error": float(interval.standard_error[index]),
                        "difference_simultaneous_lower": float(interval.lower[index] - inner_bias),
                        "difference_simultaneous_upper": float(interval.upper[index] + inner_bias),
                        "inner_bias_envelope": inner_bias,
                        "outer_records": len(common_ids),
                        "record_mode": "paired-mc",
                    }
                )

    exact_grouped: dict[
        tuple[Any, ...], dict[str, dict[int, dict[int, float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    exact_modes: dict[tuple[Any, ...], str] = {}
    exact_pairing: dict[tuple[Any, ...], tuple[Any, tuple[float, ...]]] = {}
    for row in exact_records:
        if (
            row["measurement"] not in {"heterodyne", "homodyne"}
            or row["observable"] not in {"spin-pair", "bond-pair"}
            or row["separation"] is None
            or not str(row["record_mode"]).startswith("sampled")
        ):
            continue
        family = str(row["observable"]).removesuffix("-pair")
        key = (
            int(row["lx"]),
            str(row["noise"]),
            float(row["p"]),
            str(row["prior"]),
            family,
        )
        exact_grouped[key][str(row["measurement"])][int(row["separation"])][
            int(row["global_id"])
        ] = float(row["absolute_contribution"])
        exact_modes[key] = str(row["record_mode"])
        exact_pairing[(*key, str(row["measurement"]), int(row["global_id"]))] = (
            row.get("planted_state"),
            tuple(float(value) for value in row.get("standard_variates", ())),
        )

    for key, exact_protocols in sorted(exact_grouped.items(), key=lambda item: repr(item[0])):
        if set(exact_protocols) != {"heterodyne", "homodyne"}:
            continue
        lx, noise, p, prior, family = key
        exact_separations = sorted(
            set(exact_protocols["heterodyne"]) & set(exact_protocols["homodyne"])
        )
        if not exact_separations:
            continue
        id_sets = [
            set(exact_protocols[measurement][separation])
            for measurement in ("heterodyne", "homodyne")
            for separation in exact_separations
        ]
        common_ids = sorted(set.intersection(*id_sets))
        if len(common_ids) < 2:
            continue
        for global_id in common_ids:
            if exact_pairing[(*key, "heterodyne", global_id)] != exact_pairing[
                (*key, "homodyne", global_id)
            ]:
                raise AnalysisError(
                    f"paired ED protocols do not share random numbers for {key!r}, id={global_id}"
                )
        values = {}
        for measurement in ("heterodyne", "homodyne"):
            values[measurement] = np.column_stack(
                [
                    [
                        exact_protocols[measurement][separation][global_id]
                        for global_id in common_ids
                    ]
                    for separation in exact_separations
                ]
            )
        difference = paired_curve_difference(values["homodyne"], values["heterodyne"])
        interval = simultaneous_curve_interval(
            difference,
            block_length=1,
            resamples=config.statistics.bootstrap_resamples,
            confidence=config.statistics.confidence,
            seed=config.campaign.seed ^ (int(lx) << 31) ^ (0 if family == "spin" else 1),
        )
        for index, separation in enumerate(exact_separations):
            output.append(
                {
                    "source": "ed",
                    "lx": lx,
                    "lt": config.lattice.lt(int(lx)),
                    "noise": noise,
                    "p": p,
                    "prior": prior,
                    "update": None,
                    "family": family,
                    "separation": separation,
                    "difference": float(interval.estimate[index]),
                    "difference_standard_error": float(interval.standard_error[index]),
                    "difference_simultaneous_lower": float(interval.lower[index]),
                    "difference_simultaneous_upper": float(interval.upper[index]),
                    "inner_bias_envelope": 0.0,
                    "outer_records": len(common_ids),
                    "record_mode": exact_modes[key],
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
        if separation == lx // 2:
            long_distance[key][lx] = (
                separation,
                float(row["witness"]),
                float(row["witness_standard_error"]),
            )
    output: list[dict[str, object]] = []
    for key, by_size in sorted(long_distance.items(), key=lambda item: repr(item[0])):
        expected_sizes = {int(row["lx"]) for row in curves if (
            row["noise"], row["p"], row["protocol_id"], row["update"], row["family"]
        ) == key}
        missing = sorted(expected_sizes - set(by_size))
        if missing:
            raise AnalysisError(
                f"finite-size grid is missing r=floor(L/2) for sizes {missing} and key {key!r}"
            )
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
            if _is_production_row(row)
        ]
        grouped = _group_rows(rows)
        clean = _clean_reference(config, clean_artifacts)
        bias_envelopes = _curve_bias_envelopes(config, all_rows)
        protocol_bias_envelopes = _protocol_difference_bias_envelopes(config, all_rows)
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
        protocol_difference_rows = _protocol_difference_rows(
            config,
            grouped,
            exact_record_rows,
            protocol_bias_envelopes,
        )
        comparison_rows = _comparison_rows(
            config,
            grouped,
            exact_result_rows,
            exact_record_rows,
            bias_envelopes,
        )
        finite_rows = _finite_size_rows(curve_rows)
        outer_autocorrelation_rows = _outer_autocorrelation_rows(clean)
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
            write_artifact(
                config.campaign.output_root,
                "analysis-outer-autocorrelation",
                pa.Table.from_pylist(outer_autocorrelation_rows),
                metadata={
                    **common,
                    "source": "ordered saved clean configurations",
                    "selection": (
                        "slowest clean energy and noise-relevant boundary observable "
                        "sets each analysis block scale"
                    ),
                    "units": "saved clean configurations and Wolff sweeps",
                },
                project_root=config.campaign.project_root,
                parents=parents,
                partition_by=("lx",),
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
        if protocol_difference_rows:
            artifacts.append(
                write_artifact(
                    config.campaign.output_root,
                    "analysis-protocol-differences",
                    pa.Table.from_pylist(protocol_difference_rows),
                    metadata={
                        **common,
                        "difference": "homodyne witness minus heterodyne witness",
                        "pairing": (
                            "common planted outer ids and common Gaussian variates; one "
                            "moving-block resample for each complete separation curve"
                        ),
                        "interval": "simultaneous max-t band plus paired finite-inner bias",
                    },
                    project_root=config.campaign.project_root,
                    parents=parents,
                    partition_by=("source", "noise", "family"),
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
        "protocol_difference_rows": len(protocol_difference_rows),
        "outer_autocorrelation_rows": len(outer_autocorrelation_rows),
        "comparison_rows": len(comparison_rows),
        "finite_size_rows": len(finite_rows),
    }
