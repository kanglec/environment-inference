"""NumPy reference calculations for the finite decohered TFIM.

The original modules in this package deliberately remain dependency-light and
use Python lists.  This module is the production backend for the numerical
comparison: dense linear algebra is appropriate for the requested small
systems and makes the trace-norm fidelity calculation explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ObservableFamily:
    """A translation family of diagonal involutive observables."""

    name: str
    separation: int | None
    eigenvalues: FloatArray  # shape: (translations, 2**n_sites)


@dataclass(frozen=True)
class GaussianWitnessEstimate:
    """Gaussian-measurement witness and its outer-sample standard error."""

    value: float
    standard_error: float
    linear_sample_mean: float
    linear_standard_error: float


def spin_configurations(n_sites: int) -> FloatArray:
    """Return all computational-basis spin configurations as +/-1."""

    if n_sites < 2:
        raise ValueError("n_sites must be at least 2")
    states = np.arange(1 << n_sites, dtype=np.uint64)[:, None]
    sites = np.arange(n_sites, dtype=np.uint64)[None, :]
    return (1 - 2 * ((states >> sites) & 1).astype(np.int8)).astype(np.float64)


def tfim_hamiltonian(n_sites: int, *, j: float = 1.0, h: float = 1.0) -> FloatArray:
    """Build the periodic dense TFIM Hamiltonian."""

    spins = spin_configurations(n_sites)
    dim = spins.shape[0]
    spatial = np.sum(spins * np.roll(spins, -1, axis=1), axis=1)
    matrix = np.diag(-j * spatial)
    for state in range(dim):
        for site in range(n_sites):
            matrix[state ^ (1 << site), state] -= h
    return matrix


def tfim_ground_state(n_sites: int, *, j: float = 1.0, h: float = 1.0) -> tuple[float, FloatArray]:
    """Return the finite periodic TFIM ground energy and positive ground state."""

    values, vectors = np.linalg.eigh(tfim_hamiltonian(n_sites, j=j, h=h))
    state = vectors[:, 0]
    if state[np.argmax(np.abs(state))] < 0.0:
        state = -state
    return float(values[0]), state


def z_noise_density(state: FloatArray, n_sites: int, p: float) -> FloatArray:
    """Apply independent Z noise to a pure real state."""

    if not 0.0 <= p < 0.5:
        raise ValueError("p must satisfy 0 <= p < 1/2")
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (1 << n_sites,):
        raise ValueError("state shape does not match n_sites")
    spins = spin_configurations(n_sites)
    mismatches = np.rint((n_sites - spins @ spins.T) / 2.0).astype(np.int16)
    gram = np.power(1.0 - 2.0 * p, mismatches, dtype=np.float64)
    return np.outer(state, state) * gram


def observable_families(n_sites: int) -> list[ObservableFamily]:
    """Return the local and two-point observable families used in the notes."""

    spins = spin_configurations(n_sites)
    bonds = spins * np.roll(spins, -1, axis=1)
    families = [
        ObservableFamily("local_spin", None, spins.T.copy()),
        ObservableFamily("local_bond", None, bonds.T.copy()),
    ]
    for separation in range(n_sites // 2 + 1):
        spin_pairs = spins * np.roll(spins, -separation, axis=1)
        bond_pairs = bonds * np.roll(bonds, -separation, axis=1)
        families.extend(
            [
                ObservableFamily("spin_pair", separation, spin_pairs.T.copy()),
                ObservableFamily("bond_pair", separation, bond_pairs.T.copy()),
            ]
        )
    return families


def linear_expectation(prior: FloatArray, family: ObservableFamily) -> float:
    """Translation-averaged clean/annealed expectation for one family."""

    return float(np.mean(family.eigenvalues @ prior))


def density_square_root(rho: FloatArray) -> FloatArray:
    """Return the positive square root of a real symmetric density matrix."""

    values, vectors = np.linalg.eigh((rho + rho.T) * 0.5)
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) < -1e-10 * scale:
        raise ValueError("density matrix is not positive semidefinite")
    values = np.clip(values, 0.0, None)
    values[values < 1e-14 * scale] = 0.0
    return (vectors * np.sqrt(values)) @ vectors.T


def fidelity_average_from_sqrt(
    rho: FloatArray,
    sqrt_rho: FloatArray,
    observable_eigenvalues: FloatArray,
) -> float:
    """Evaluate the root fidelity average for a diagonal Hermitian observable."""

    a = np.asarray(observable_eigenvalues, dtype=np.float64)
    if a.shape != (rho.shape[0],):
        raise ValueError("observable length does not match density matrix")
    insertion = (sqrt_rho * a[None, :]) @ sqrt_rho
    numerator = float(np.sum(np.abs(np.linalg.eigvalsh((insertion + insertion.T) * 0.5))))
    denominator = float(np.sqrt(np.dot(np.diag(rho), a * a)))
    if denominator == 0.0:
        raise ValueError("A rho A is zero")
    return numerator / denominator


def family_fidelity(rho: FloatArray, family: ObservableFamily) -> float:
    """Average the exact fidelity over translations in a family."""

    sqrt_rho = density_square_root(rho)
    values = [fidelity_average_from_sqrt(rho, sqrt_rho, a) for a in family.eigenvalues]
    return float(np.mean(values))


def transfer_boundary_prior(
    n_sites: int,
    *,
    l_tau: int,
    delta_tau: float,
) -> tuple[FloatArray, FloatArray]:
    """Return finite-torus and infinite-time boundary priors of the Trotter model.

    The symmetric row transfer matrix is used, so the infinite-time boundary
    prior is the squared dominant eigenvector.  The finite-torus marginal is
    ``diag(T**l_tau) / Tr(T**l_tau)``.
    """

    if l_tau < 2:
        raise ValueError("l_tau must be at least 2")
    if not np.isfinite(delta_tau) or delta_tau <= 0.0:
        raise ValueError("delta_tau must be positive and finite")
    spins = spin_configurations(n_sites)
    spatial = np.sum(spins * np.roll(spins, -1, axis=1), axis=1)
    kx = delta_tau
    kt = -0.5 * np.log(np.tanh(delta_tau))
    transfer = np.exp(0.5 * kx * (spatial[:, None] + spatial[None, :]) + kt * (spins @ spins.T))
    values, vectors = np.linalg.eigh(transfer)
    tolerance = 1e-12 * float(values[-1])
    if float(values[0]) < -tolerance:
        raise ValueError("transfer matrix has a materially negative eigenvalue")
    values = np.clip(values, 0.0, None)
    ratios = values / values[-1]
    weights = np.power(ratios, l_tau, dtype=np.float64)
    finite = (vectors * vectors) @ weights
    finite /= np.sum(finite)
    infinite = vectors[:, -1] ** 2
    infinite /= np.sum(infinite)
    return finite, infinite


def gaussian_measurement_witness(
    prior: FloatArray,
    families: list[ObservableFamily],
    *,
    n_sites: int,
    p: float,
    samples: int,
    seed: int,
    batch_size: int = 2048,
    measurement: str = "heterodyne",
) -> dict[tuple[str, int | None], GaussianWitnessEstimate]:
    """Estimate Gaussian-record witnesses using exact posterior sums.

    Only the outer Gaussian integral is sampled.  For every record, posterior
    expectations are evaluated by summing all ``2**n_sites`` boundary states.
    """

    if samples < 1:
        raise ValueError("samples must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= p < 0.5:
        raise ValueError("p must satisfy 0 <= p < 1/2")
    if measurement not in {"heterodyne", "homodyne"}:
        raise ValueError("measurement must be 'heterodyne' or 'homodyne'")
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != (1 << n_sites,) or np.any(prior < 0.0):
        raise ValueError("prior is invalid")
    prior = prior / np.sum(prior)
    spins = spin_configurations(n_sites)
    operators = np.concatenate([family.eigenvalues.T for family in families], axis=1)
    slices: list[slice] = []
    offset = 0
    for family in families:
        width = family.eigenvalues.shape[0]
        slices.append(slice(offset, offset + width))
        offset += width

    rng = np.random.default_rng(seed)
    mu = -np.log(1.0 - 2.0 * p)
    if measurement == "homodyne":
        mu *= 2.0
    log_prior = np.full_like(prior, -np.inf)
    positive = prior > 0.0
    log_prior[positive] = np.log(prior[positive])
    sum_abs = np.zeros(len(families), dtype=np.float64)
    sum_abs_sq = np.zeros(len(families), dtype=np.float64)
    sum_signed = np.zeros(len(families), dtype=np.float64)
    sum_signed_sq = np.zeros(len(families), dtype=np.float64)

    completed = 0
    while completed < samples:
        count = min(batch_size, samples - completed)
        labels = rng.choice(prior.size, size=count, p=prior)
        records = mu * spins[labels]
        if mu > 0.0:
            records = records + np.sqrt(mu) * rng.standard_normal((count, n_sites))
        log_weights = log_prior[None, :] + records @ spins.T
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        weights /= np.sum(weights, axis=1, keepdims=True)
        posterior_means = weights @ operators
        for index, family_slice in enumerate(slices):
            translated = posterior_means[:, family_slice]
            per_record_abs = np.mean(np.abs(translated), axis=1)
            per_record_signed = np.mean(translated, axis=1)
            sum_abs[index] += float(np.sum(per_record_abs))
            sum_abs_sq[index] += float(np.dot(per_record_abs, per_record_abs))
            sum_signed[index] += float(np.sum(per_record_signed))
            sum_signed_sq[index] += float(np.dot(per_record_signed, per_record_signed))
        completed += count

    def estimate(total: float, total_sq: float) -> tuple[float, float]:
        mean = total / samples
        if samples == 1:
            return mean, 0.0
        variance = max(0.0, (total_sq - samples * mean * mean) / (samples - 1))
        return mean, float(np.sqrt(variance / samples))

    output: dict[tuple[str, int | None], GaussianWitnessEstimate] = {}
    for index, family in enumerate(families):
        absolute, absolute_se = estimate(sum_abs[index], sum_abs_sq[index])
        signed, signed_se = estimate(sum_signed[index], sum_signed_sq[index])
        output[(family.name, family.separation)] = GaussianWitnessEstimate(
            value=absolute,
            standard_error=absolute_se,
            linear_sample_mean=signed,
            linear_standard_error=signed_se,
        )
    return output


def local_x_measurement_witness(
    prior: FloatArray,
    families: list[ObservableFamily],
    *,
    n_sites: int,
    p: float,
) -> dict[tuple[str, int | None], GaussianWitnessEstimate]:
    """Evaluate the local environment-qubit X-measurement witness exactly."""

    if not 0.0 <= p < 0.5:
        raise ValueError("p must satisfy 0 <= p < 1/2")
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != (1 << n_sites,) or np.any(prior < 0.0):
        raise ValueError("prior is invalid")
    prior = prior / np.sum(prior)
    spins = spin_configurations(n_sites)
    outcomes = spins
    kappa = 2.0 * np.sqrt(p * (1.0 - p))
    factors = 0.5 * (1.0 + kappa * outcomes[:, None, :] * spins[None, :, :])
    conditional = np.prod(factors, axis=2)
    joint = conditional * prior[None, :]
    evidence = np.sum(joint, axis=1)
    if np.any(evidence <= 0.0):
        raise ValueError("local-X outcome has nonpositive evidence")

    output: dict[tuple[str, int | None], GaussianWitnessEstimate] = {}
    for family in families:
        posterior_means = (joint @ family.eigenvalues.T) / evidence[:, None]
        per_outcome_abs = np.mean(np.abs(posterior_means), axis=1)
        per_outcome_signed = np.mean(posterior_means, axis=1)
        output[(family.name, family.separation)] = GaussianWitnessEstimate(
            value=float(evidence @ per_outcome_abs),
            standard_error=0.0,
            linear_sample_mean=float(evidence @ per_outcome_signed),
            linear_standard_error=0.0,
        )
    return output


def measurement_witness(
    prior: FloatArray,
    families: list[ObservableFamily],
    *,
    n_sites: int,
    p: float,
    measurement: str,
    samples: int,
    seed: int,
    batch_size: int = 2048,
) -> dict[tuple[str, int | None], GaussianWitnessEstimate]:
    """Evaluate any supported environment-measurement witness."""

    if measurement == "local_x":
        return local_x_measurement_witness(prior, families, n_sites=n_sites, p=p)
    return gaussian_measurement_witness(
        prior,
        families,
        n_sites=n_sites,
        p=p,
        samples=samples,
        seed=seed,
        batch_size=batch_size,
        measurement=measurement,
    )
