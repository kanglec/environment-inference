"""Dense exact diagonalization and exact posterior summation."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _core
from .registries import MeasurementPoint

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GroundState:
    energy: float
    state: FloatArray
    residual: float
    normalization_error: float


@dataclass(frozen=True)
class DensityDiagnostics:
    trace_error: float
    hermiticity_error: float
    minimum_eigenvalue: float


@dataclass(frozen=True)
class DensityReference:
    rho: FloatArray
    square_root: FloatArray
    diagonal: FloatArray
    diagnostics: DensityDiagnostics
    entropy: float
    purity: float


@dataclass(frozen=True)
class PriorSet:
    quantum: FloatArray
    finite_transfer: FloatArray
    transfer_ground: FloatArray

    def named(self, name: str) -> FloatArray:
        return {
            "quantum": self.quantum,
            "finite-transfer": self.finite_transfer,
            "transfer-ground": self.transfer_ground,
        }[name]


@dataclass(frozen=True)
class ObservableSpec:
    family: str
    separation: int | None

    @property
    def name(self) -> str:
        if self.family in {"spin", "bond"}:
            return self.family
        return self.family


def tfim_hamiltonian(sites: int, *, j: float = 1.0, h: float = 1.0) -> FloatArray:
    if sites < 2:
        raise ValueError("sites must be at least two")
    dimension = 1 << sites
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    diagonal = np.zeros(dimension, dtype=np.float64)
    for origin in range(sites):
        diagonal -= j * np.asarray(
            _core.observable_eigenvalues(sites, "bond", origin), dtype=np.float64
        )
    np.fill_diagonal(matrix, diagonal)
    states = np.arange(dimension, dtype=np.int64)
    for site in range(sites):
        matrix[states ^ (1 << site), states] -= h
    return matrix


def ground_state(sites: int, *, j: float = 1.0, h: float = 1.0) -> GroundState:
    hamiltonian = tfim_hamiltonian(sites, j=j, h=h)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    state = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    if float(np.sum(state)) < 0.0:
        state = -state
    state /= np.linalg.norm(state)
    energy = float(eigenvalues[0])
    residual = float(np.linalg.norm(hamiltonian @ state - energy * state))
    return GroundState(
        energy=energy,
        state=state,
        residual=residual,
        normalization_error=abs(float(state @ state) - 1.0),
    )


def noise_eigenvalues(sites: int, noise: str) -> npt.NDArray[np.int8]:
    values = np.asarray(_core.noise_eigenvalues_all(sites, noise), dtype=np.int8)
    expected = (1 << sites, sites)
    if values.shape != expected:
        raise RuntimeError(f"Rust noise eigenvalue shape {values.shape} differs from {expected}")
    return values


def decohered_density_matrix(state: FloatArray, p: float, noise: str) -> FloatArray:
    dimension = state.size
    sites = dimension.bit_length() - 1
    if 1 << sites != dimension:
        raise ValueError("state length must be a power of two")
    if not 0.0 <= p < 0.5:
        raise ValueError("p must satisfy 0 <= p < 1/2")
    variables = noise_eigenvalues(sites, noise).astype(np.float64)
    factor = np.ones((dimension, dimension), dtype=np.float64)
    for site in range(variables.shape[1]):
        column = variables[:, site]
        factor *= (1.0 - p) + p * np.multiply.outer(column, column)
    return np.multiply.outer(state, state) * factor


def density_diagnostics(rho: FloatArray) -> DensityDiagnostics:
    hermitian = (rho + rho.T.conj()) / 2.0
    return DensityDiagnostics(
        trace_error=abs(float(np.trace(rho).real) - 1.0),
        hermiticity_error=float(np.linalg.norm(rho - rho.T.conj())),
        minimum_eigenvalue=float(np.linalg.eigvalsh(hermitian)[0]),
    )


def entropy_and_purity(rho: FloatArray, *, clip_tolerance: float = 1e-12) -> tuple[float, float]:
    values = np.linalg.eigvalsh((rho + rho.T.conj()) / 2.0)
    if float(values[0]) < -clip_tolerance:
        raise ValueError(f"density matrix has eigenvalue {values[0]} below tolerance")
    values = np.clip(values, 0.0, None)
    positive = values[values > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    purity = float(np.sum(values * values))
    return entropy, purity


def density_reference(
    state: FloatArray,
    p: float,
    noise: str,
    *,
    clip_tolerance: float = 1e-12,
) -> DensityReference:
    rho = decohered_density_matrix(state, p, noise)
    diagnostics = density_diagnostics(rho)
    hermitian = (rho + rho.T.conj()) / 2.0
    values, vectors = np.linalg.eigh(hermitian)
    if float(values[0]) < -clip_tolerance:
        raise ValueError(f"density matrix is not positive: lambda_min={values[0]}")
    clipped = np.clip(values, 0.0, None)
    square_root = (vectors * np.sqrt(clipped)) @ vectors.T.conj()
    positive = clipped[clipped > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    purity = float(np.sum(clipped * clipped))
    return DensityReference(
        rho=rho,
        square_root=np.asarray(square_root, dtype=np.float64),
        diagonal=np.asarray(np.diag(rho).real, dtype=np.float64),
        diagnostics=diagnostics,
        entropy=entropy,
        purity=purity,
    )


def diagonal_observable(
    sites: int,
    family: str,
    *,
    origin: int = 0,
    separation: int | None = None,
) -> FloatArray:
    return np.asarray(
        _core.observable_eigenvalues(sites, family, origin, separation), dtype=np.float64
    )


def linear_expectation(prior: FloatArray, observable: FloatArray) -> float:
    return float(prior @ observable)


def physical_fidelity(
    rho: FloatArray,
    observable: FloatArray,
    *,
    clip_tolerance: float = 1e-12,
) -> float:
    hermitian = (rho + rho.T.conj()) / 2.0
    values, vectors = np.linalg.eigh(hermitian)
    if float(values[0]) < -clip_tolerance:
        raise ValueError(f"density matrix is not positive: lambda_min={values[0]}")
    root = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T.conj()
    inserted = root @ (observable[:, None] * root)
    inserted = (inserted + inserted.T.conj()) / 2.0
    numerator = float(np.sum(np.abs(np.linalg.eigvalsh(inserted))))
    denominator_sq = float(np.sum(np.diag(rho).real * observable * observable))
    if denominator_sq <= 0.0:
        raise ValueError("fidelity denominator is zero")
    return numerator / math.sqrt(denominator_sq)


def physical_fidelity_from_reference(
    reference: DensityReference,
    observable: FloatArray,
) -> float:
    inserted = reference.square_root @ (observable[:, None] * reference.square_root)
    inserted = (inserted + inserted.T.conj()) / 2.0
    numerator = float(np.sum(np.abs(np.linalg.eigvalsh(inserted))))
    denominator_sq = float(np.sum(reference.diagonal * observable * observable))
    if denominator_sq <= 0.0:
        raise ValueError("fidelity denominator is zero")
    return numerator / math.sqrt(denominator_sq)


def transfer_matrix(sites: int, kx: float, kt: float) -> FloatArray:
    dimension = 1 << sites
    spins = np.empty((dimension, sites), dtype=np.float64)
    for state in range(dimension):
        spins[state] = [1.0 if state & (1 << site) == 0 else -1.0 for site in range(sites)]
    spatial = np.sum(spins * np.roll(spins, -1, axis=1), axis=1)
    temporal = spins @ spins.T
    matrix = np.exp(0.5 * kx * (spatial[:, None] + spatial[None, :]) + kt * temporal)
    return np.asarray((matrix + matrix.T) / 2.0, dtype=np.float64)


def transfer_priors(sites: int, lt: int, kx: float, kt: float) -> tuple[FloatArray, FloatArray]:
    matrix = transfer_matrix(sites, kx, kt)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    dominant = vectors[:, 0]
    if float(np.sum(dominant)) < 0.0:
        dominant = -dominant
    transfer_ground = dominant * dominant
    transfer_ground /= np.sum(transfer_ground)

    ratio = values / values[0]
    powered = np.power(ratio, lt)
    diagonal = (vectors * vectors) @ powered
    finite = np.asarray(diagonal / np.sum(diagonal), dtype=np.float64)
    if float(np.min(finite)) < -1e-13:
        raise ValueError("finite-transfer prior has a negative component")
    finite = np.clip(finite, 0.0, None)
    finite /= np.sum(finite)
    return finite, transfer_ground


def build_priors(sites: int, lt: int, kx: float, kt: float) -> tuple[GroundState, PriorSet]:
    ground = ground_state(sites)
    finite, transfer_ground = transfer_priors(sites, lt, kx, kt)
    quantum = ground.state * ground.state
    return ground, PriorSet(quantum=quantum, finite_transfer=finite, transfer_ground=transfer_ground)


def total_variation(left: FloatArray, right: FloatArray) -> float:
    return 0.5 * float(np.sum(np.abs(left - right)))


def prior_diagnostic_rows(
    sites: int,
    lt: int,
    regularization: str,
    delta_tau: float | None,
    priors: PriorSet,
) -> list[dict[str, object]]:
    named = {
        "quantum": priors.quantum,
        "finite-transfer": priors.finite_transfer,
        "transfer-ground": priors.transfer_ground,
    }
    return [
        {
            "lx": sites,
            "lt": lt,
            "regularization": regularization,
            "delta_tau": delta_tau,
            "left_prior": left,
            "right_prior": right,
            "total_variation": total_variation(named[left], named[right]),
        }
        for left, right in itertools.combinations(named, 2)
    ]


def posterior_probabilities(prior: FloatArray, variables: FloatArray, record: FloatArray) -> FloatArray:
    if variables.shape[0] != prior.size or variables.shape[1] != record.size:
        raise ValueError("posterior input shapes differ")
    with np.errstate(divide="ignore"):
        logarithm = np.log(prior) + variables @ record
    maximum = float(np.max(logarithm))
    weights = np.exp(logarithm - maximum)
    weights /= np.sum(weights)
    return np.asarray(weights, dtype=np.float64)


def _basis_spins(state: int, sites: int) -> list[int]:
    return [1 if state & (1 << site) == 0 else -1 for site in range(sites)]


def _sample_discrete(probabilities: FloatArray, uniform: float) -> int:
    return min(int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")), len(probabilities) - 1)


def translated_observables(sites: int, spec: ObservableSpec) -> list[FloatArray]:
    return [
        diagonal_observable(
            sites,
            spec.family,
            origin=origin,
            separation=spec.separation,
        )
        for origin in range(sites)
    ]


def default_observables(separations: Sequence[int]) -> tuple[ObservableSpec, ...]:
    output = [ObservableSpec("spin", None), ObservableSpec("bond", None)]
    output.extend(ObservableSpec("spin-pair", separation) for separation in separations)
    output.extend(ObservableSpec("bond-pair", separation) for separation in separations)
    return tuple(output)


def _aggregate_posterior(
    posterior: FloatArray,
    observables: Sequence[FloatArray],
    planted_state: int | None,
) -> tuple[float, float, float]:
    means = np.asarray([posterior @ observable for observable in observables])
    planted = (
        means * np.asarray([observable[planted_state] for observable in observables])
        if planted_state is not None
        else means * means
    )
    return float(np.mean(means)), float(np.mean(np.abs(means))), float(np.mean(planted))


def _gaussian_records(
    *,
    sites: int,
    noise: str,
    point: MeasurementPoint,
    p: float,
    prior_name: str,
    prior: FloatArray,
    variables: FloatArray,
    observables: Sequence[FloatArray],
    seed: int,
    count: int,
    spec: ObservableSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plant_domain = f"ed-planted/{sites}/{noise}/{prior_name}"
    for global_id in range(count):
        uniform = float(_core.stream_uniforms(seed, plant_domain, global_id, 1)[0])
        planted_state = _sample_discrete(prior, uniform)
        generated = _core.generate_record(
            _basis_spins(planted_state, sites),
            noise,
            point.name,
            p,
            seed,
            global_id,
            point.gamma,
        )
        record = np.asarray(generated["record_couplings"], dtype=np.float64)
        posterior = posterior_probabilities(prior, variables, record)
        mean, absolute, planted = _aggregate_posterior(posterior, observables, planted_state)
        rows.append(
            {
                "global_id": global_id,
                "lx": sites,
                "noise": noise,
                "measurement": point.name,
                "protocol_id": point.identifier,
                "prior": prior_name,
                "p": p,
                "gamma": generated["gamma"],
                "planted_state": planted_state,
                "raw_record": [float(value) for value in generated["raw_record"]],
                "standard_variates": [float(value) for value in generated["standard_variates"]],
                "observable": spec.name,
                "separation": spec.separation,
                "posterior_mean": mean,
                "absolute_contribution": absolute,
                "planted_contribution": planted,
                "record_weight": 1.0 / count,
                "record_mode": "sampled-gaussian",
            }
        )
    return rows


def _local_x_enumerated_records(
    *,
    sites: int,
    noise: str,
    point: MeasurementPoint,
    p: float,
    prior_name: str,
    prior: FloatArray,
    variables: FloatArray,
    observables: Sequence[FloatArray],
    spec: ObservableSpec,
) -> list[dict[str, Any]]:
    parameters = _core.protocol_parameters("local-x", p)
    kappa = float(parameters["kappa"])
    coupling = float(parameters["coupling"])
    rows: list[dict[str, Any]] = []
    for outcome_bits in range(1 << sites):
        outcome = np.asarray(_basis_spins(outcome_bits, sites), dtype=np.float64)
        likelihood = np.prod((1.0 + kappa * variables * outcome[None, :]) / 2.0, axis=1)
        evidence = float(prior @ likelihood)
        if evidence <= 0.0:
            continue
        posterior = prior * likelihood / evidence
        mean, absolute, planted = _aggregate_posterior(posterior, observables, None)
        rows.append(
            {
                "global_id": outcome_bits,
                "lx": sites,
                "noise": noise,
                "measurement": point.name,
                "protocol_id": point.identifier,
                "prior": prior_name,
                "p": p,
                "gamma": None,
                "planted_state": None,
                "raw_record": outcome.tolist(),
                "standard_variates": [],
                "observable": spec.name,
                "separation": spec.separation,
                "posterior_mean": mean,
                "absolute_contribution": absolute,
                "planted_contribution": planted,
                "record_weight": evidence,
                "record_mode": "enumerated-binary",
                "_coupling": coupling,
            }
        )
    total = sum(float(row["record_weight"]) for row in rows)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=5e-12):
        raise RuntimeError(f"local-X evidence does not normalize: {total}")
    for row in rows:
        row.pop("_coupling")
    return rows


def _local_x_sampled_records(
    *,
    sites: int,
    noise: str,
    point: MeasurementPoint,
    p: float,
    prior_name: str,
    prior: FloatArray,
    variables: FloatArray,
    observables: Sequence[FloatArray],
    seed: int,
    count: int,
    spec: ObservableSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plant_domain = f"ed-planted/{sites}/{noise}/{prior_name}"
    for global_id in range(count):
        uniform = float(_core.stream_uniforms(seed, plant_domain, global_id, 1)[0])
        planted_state = _sample_discrete(prior, uniform)
        generated = _core.generate_record(
            _basis_spins(planted_state, sites),
            noise,
            "local-x",
            p,
            seed,
            global_id,
        )
        record = np.asarray(generated["record_couplings"], dtype=np.float64)
        posterior = posterior_probabilities(prior, variables, record)
        mean, absolute, planted = _aggregate_posterior(posterior, observables, planted_state)
        rows.append(
            {
                "global_id": global_id,
                "lx": sites,
                "noise": noise,
                "measurement": point.name,
                "protocol_id": point.identifier,
                "prior": prior_name,
                "p": p,
                "gamma": None,
                "planted_state": planted_state,
                "raw_record": [float(value) for value in generated["raw_record"]],
                "standard_variates": [float(value) for value in generated["standard_variates"]],
                "observable": spec.name,
                "separation": spec.separation,
                "posterior_mean": mean,
                "absolute_contribution": absolute,
                "planted_contribution": planted,
                "record_weight": 1.0 / count,
                "record_mode": "sampled-binary-fallback",
            }
        )
    return rows


def evaluate_exact_protocol(
    *,
    sites: int,
    noise: str,
    point: MeasurementPoint,
    p: float,
    prior_name: str,
    prior: FloatArray,
    ground: GroundState,
    observables: Iterable[ObservableSpec],
    seed: int,
    gaussian_records: int,
    local_x_enumeration_limit: int,
    sampled_binary_records: int,
    positivity_tolerance: float,
    reference: DensityReference | None = None,
    observable_references: dict[ObservableSpec, tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state = ground.state if prior_name == "quantum" else np.sqrt(prior)
    if reference is None:
        reference = density_reference(
            state, p, noise, clip_tolerance=positivity_tolerance
        )
    diagnostics = reference.diagnostics
    entropy = reference.entropy
    purity = reference.purity
    variables = noise_eigenvalues(sites, noise).astype(np.float64)
    parameters = _core.protocol_parameters(point.name, p, point.gamma)
    result_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []

    for spec in observables:
        translated = translated_observables(sites, spec)
        if point.name == "local-x" and sites <= local_x_enumeration_limit:
            rows = _local_x_enumerated_records(
                sites=sites,
                noise=noise,
                point=point,
                p=p,
                prior_name=prior_name,
                prior=prior,
                variables=variables,
                observables=translated,
                spec=spec,
            )
        elif point.name == "local-x":
            rows = _local_x_sampled_records(
                sites=sites,
                noise=noise,
                point=point,
                p=p,
                prior_name=prior_name,
                prior=prior,
                variables=variables,
                observables=translated,
                seed=seed,
                count=sampled_binary_records,
                spec=spec,
            )
        else:
            rows = _gaussian_records(
                sites=sites,
                noise=noise,
                point=point,
                p=p,
                prior_name=prior_name,
                prior=prior,
                variables=variables,
                observables=translated,
                seed=seed,
                count=gaussian_records,
                spec=spec,
            )
        record_rows.extend(rows)
        weights = np.asarray([float(row["record_weight"]) for row in rows])
        contributions = np.asarray([float(row["absolute_contribution"]) for row in rows])
        witness = float(weights @ contributions)
        if rows[0]["record_mode"].startswith("sampled"):
            standard_error = float(np.std(contributions, ddof=1) / math.sqrt(len(contributions)))
        else:
            standard_error = 0.0
        if observable_references is not None and spec in observable_references:
            linear, fidelity = observable_references[spec]
        else:
            linear = float(np.mean([linear_expectation(prior, value) for value in translated]))
            fidelity = float(
                np.mean(
                    [physical_fidelity_from_reference(reference, value) for value in translated]
                )
            )
        result_rows.append(
            {
                "lx": sites,
                "noise": noise,
                "measurement": point.name,
                "protocol_id": point.identifier,
                "prior": prior_name,
                "observable": spec.name,
                "separation": spec.separation,
                "p": p,
                "lambda": float(parameters["lambda"]),
                "gamma": parameters["gamma"],
                "kappa": parameters["kappa"],
                "linear_exact": linear,
                "physical_fidelity": fidelity,
                "measurement_witness": witness,
                "witness_standard_error": standard_error,
                "fidelity_gap": fidelity - witness,
                "entropy": entropy,
                "purity": purity,
                "ground_energy": ground.energy if prior_name == "quantum" else None,
                "ground_residual": ground.residual if prior_name == "quantum" else None,
                "trace_error": diagnostics.trace_error,
                "hermiticity_error": diagnostics.hermiticity_error,
                "minimum_eigenvalue": diagnostics.minimum_eigenvalue,
                "record_mode": rows[0]["record_mode"],
                "outer_records": len(rows),
            }
        )
    return result_rows, record_rows


def heterodyne_purity_from_kernel(prior: FloatArray, p: float, noise: str) -> float:
    sites = prior.size.bit_length() - 1
    variables = noise_eigenvalues(sites, noise).astype(np.float64)
    gamma = float(_core.protocol_parameters("heterodyne", p)["gamma"])
    overlap = variables @ variables.T
    kernel = np.exp(gamma * (overlap - sites))
    return float(prior @ kernel @ prior)
