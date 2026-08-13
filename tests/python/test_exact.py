from __future__ import annotations

import itertools

import numpy as np
import pytest

from dcft.exact import (
    ObservableSpec,
    build_priors,
    decohered_density_matrix,
    density_diagnostics,
    diagonal_observable,
    entropy_and_purity,
    evaluate_exact_protocol,
    ground_state,
    heterodyne_purity_from_kernel,
    physical_fidelity,
    posterior_probabilities,
    tfim_hamiltonian,
    total_variation,
)
from dcft.registries import MeasurementPoint


@pytest.mark.parametrize("sites", [2, 3, 4, 5, 6])
def test_tfim_ground_state_residual(sites: int) -> None:
    ground = ground_state(sites)
    assert ground.residual < 1e-11
    assert ground.normalization_error < 1e-13
    np.testing.assert_allclose(tfim_hamiltonian(sites), tfim_hamiltonian(sites).T)


@pytest.mark.parametrize("noise", ["z", "zz"])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.3, 0.49])
def test_decohered_density_matrix_diagnostics(noise: str, p: float) -> None:
    ground = ground_state(4)
    rho = decohered_density_matrix(ground.state, p, noise)
    diagnostics = density_diagnostics(rho)
    assert diagnostics.trace_error < 1e-12
    assert diagnostics.hermiticity_error < 1e-12
    assert diagnostics.minimum_eigenvalue > -1e-12
    entropy, purity = entropy_and_purity(rho)
    assert entropy >= -1e-12
    assert 1.0 / rho.shape[0] <= purity <= 1.0 + 1e-12


@pytest.mark.parametrize("noise", ["z", "zz"])
def test_decoherence_matches_explicit_channel_sum(noise: str) -> None:
    sites = 3
    p = 0.2
    ground = ground_state(sites)
    expected = np.zeros((1 << sites, 1 << sites))
    for errors in itertools.product((0, 1), repeat=sites):
        probability = np.prod([p if value else 1.0 - p for value in errors])
        diagonal = np.ones(1 << sites)
        for site, enabled in enumerate(errors):
            if not enabled:
                continue
            observable = diagonal_observable(sites, "spin" if noise == "z" else "bond", origin=site)
            diagonal *= observable
        state = diagonal * ground.state
        expected += probability * np.outer(state, state)
    actual = decohered_density_matrix(ground.state, p, noise)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_priors_normalize_and_tv_is_metric() -> None:
    _, priors = build_priors(4, 8, 0.3, 0.7)
    for prior in (priors.quantum, priors.finite_transfer, priors.transfer_ground):
        assert np.sum(prior) == pytest.approx(1.0)
        assert np.min(prior) >= 0.0
    assert total_variation(priors.quantum, priors.quantum) == pytest.approx(0.0)
    assert total_variation(priors.quantum, priors.finite_transfer) <= 1.0


@pytest.mark.parametrize("noise", ["z", "zz"])
def test_heterodyne_purity_calibration(noise: str) -> None:
    p = 0.27
    ground = ground_state(5)
    prior = ground.state**2
    rho = decohered_density_matrix(ground.state, p, noise)
    _, purity = entropy_and_purity(rho)
    assert heterodyne_purity_from_kernel(prior, p, noise) == pytest.approx(purity, abs=2e-13)


@pytest.mark.parametrize("noise", ["z", "zz"])
def test_physical_fidelity_bounds_linear_magnitude(noise: str) -> None:
    ground = ground_state(4)
    rho = decohered_density_matrix(ground.state, 0.2, noise)
    observable = diagonal_observable(4, "spin-pair", separation=2)
    fidelity = physical_fidelity(rho, observable)
    linear = float(np.diag(rho) @ observable)
    assert abs(linear) <= fidelity + 1e-12
    assert fidelity <= 1.0 + 1e-12


def test_exact_posterior_sampling_normalization() -> None:
    prior = np.asarray([0.1, 0.2, 0.3, 0.4])
    variables = np.asarray([[1, 1], [-1, 1], [1, -1], [-1, -1]], dtype=np.float64)
    posterior = posterior_probabilities(prior, variables, np.asarray([0.3, -0.2]))
    assert np.sum(posterior) == pytest.approx(1.0)
    assert np.min(posterior) >= 0.0


def test_planted_local_x_conditional_law_equals_posterior() -> None:
    from dcft import _core

    prior = np.asarray([0.1, 0.2, 0.3, 0.4])
    variables = np.asarray([[1, 1], [-1, 1], [1, -1], [-1, -1]], dtype=np.float64)
    parameters = _core.protocol_parameters("local-x", 0.2)
    kappa = float(parameters["kappa"])
    coupling = float(parameters["coupling"])
    for outcome in itertools.product((-1.0, 1.0), repeat=2):
        z = np.asarray(outcome)
        likelihood = np.prod((1.0 + kappa * variables * z[None, :]) / 2.0, axis=1)
        conditional_from_joint = prior * likelihood / float(prior @ likelihood)
        posterior = posterior_probabilities(prior, variables, coupling * z)
        np.testing.assert_allclose(conditional_from_joint, posterior, atol=2e-15)


@pytest.mark.parametrize(
    ("enumeration_limit", "expected_mode", "expected_records"),
    [(14, "enumerated-binary", 8), (2, "sampled-binary-fallback", 11)],
)
def test_local_x_exact_enumeration_and_labeled_fallback(
    enumeration_limit: int, expected_mode: str, expected_records: int
) -> None:
    sites = 3
    ground = ground_state(sites)
    prior = ground.state**2
    result, records = evaluate_exact_protocol(
        sites=sites,
        noise="z",
        point=MeasurementPoint("local-x"),
        p=0.2,
        prior_name="quantum",
        prior=prior,
        ground=ground,
        observables=(ObservableSpec("spin-pair", 1),),
        seed=17,
        gaussian_records=13,
        local_x_enumeration_limit=enumeration_limit,
        sampled_binary_records=expected_records,
        positivity_tolerance=1e-12,
    )
    assert len(result) == 1
    assert len(records) == expected_records
    assert result[0]["record_mode"] == expected_mode
    assert {row["record_mode"] for row in records} == {expected_mode}
    assert sum(float(row["record_weight"]) for row in records) == pytest.approx(1.0)
