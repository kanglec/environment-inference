from __future__ import annotations

import math

import numpy as np
import pytest

from dcft import _core
from dcft.exact import diagonal_observable, noise_eigenvalues
from dcft.registries import verify_rust_registries


def _spins(state: int, sites: int) -> list[int]:
    return [1 if state & (1 << site) == 0 else -1 for site in range(sites)]


def test_explicit_registries_match_rust() -> None:
    verify_rust_registries()


@pytest.mark.parametrize("noise", ["z", "zz"])
@pytest.mark.parametrize("sites", range(2, 8))
def test_cross_language_noise_eigenvalues(noise: str, sites: int) -> None:
    rust = noise_eigenvalues(sites, noise)
    expected = []
    for state in range(1 << sites):
        spins = _spins(state, sites)
        if noise == "z":
            expected.append(spins)
        else:
            expected.append([spins[x] * spins[(x + 1) % sites] for x in range(sites)])
    np.testing.assert_array_equal(rust, expected)


@pytest.mark.parametrize("family", ["spin", "bond", "spin-pair", "bond-pair"])
def test_cross_language_observable_eigenvalues(family: str) -> None:
    sites = 5
    origin = 2
    separation = 3
    actual = diagonal_observable(sites, family, origin=origin, separation=separation)
    expected: list[int] = []
    for state in range(1 << sites):
        spins = _spins(state, sites)
        if family == "spin":
            value = spins[origin]
        elif family == "bond":
            value = spins[origin] * spins[(origin + 1) % sites]
        elif family == "spin-pair":
            value = spins[origin] * spins[(origin + separation) % sites]
        else:
            displaced = (origin + separation) % sites
            value = (
                spins[origin]
                * spins[(origin + 1) % sites]
                * spins[displaced]
                * spins[(displaced + 1) % sites]
            )
        expected.append(value)
    np.testing.assert_array_equal(actual, expected)


def test_protocol_parameters_and_range() -> None:
    p = 0.25
    expected_lambda = -0.5 * math.log(1.0 - 2.0 * p)
    heterodyne = _core.protocol_parameters("heterodyne", p)
    homodyne = _core.protocol_parameters("homodyne", p)
    assert heterodyne["lambda"] == pytest.approx(expected_lambda)
    assert heterodyne["gamma"] == pytest.approx(2.0 * expected_lambda)
    assert homodyne["gamma"] == pytest.approx(4.0 * expected_lambda)
    _core.protocol_parameters("gaussian", p, 4.0 * expected_lambda)
    with pytest.raises(ValueError):
        _core.protocol_parameters("gaussian", p, 4.0 * expected_lambda + 1e-12)


def test_gaussian_strengths_share_normal_variates() -> None:
    boundary = [1, -1, 1, -1]
    p = 0.2
    weak = _core.generate_record(boundary, "z", "heterodyne", p, 99, 5)
    strong = _core.generate_record(boundary, "z", "homodyne", p, 99, 5)
    arbitrary = _core.generate_record(
        boundary, "z", "gaussian", p, 99, 5, weak["gamma"] * 0.5
    )
    assert weak["standard_variates"] == strong["standard_variates"]
    assert weak["standard_variates"] == arbitrary["standard_variates"]
    local_x = _core.generate_record(boundary, "z", "local-x", p, 99, 5)
    assert local_x["standard_variates"] != weak["standard_variates"]


def test_local_x_channel_normalizes() -> None:
    p = 0.3
    parameters = _core.protocol_parameters("local-x", p)
    kappa = parameters["kappa"]
    for variable in (-1, 1):
        probabilities = [(1.0 + kappa * outcome * variable) / 2.0 for outcome in (-1, 1)]
        assert sum(probabilities) == pytest.approx(1.0)
        assert all(probability >= 0.0 for probability in probabilities)


def test_outer_id_results_are_order_and_thread_key_independent() -> None:
    boundary = [1, -1, -1, 1]
    ids = list(range(12))
    forward = {
        global_id: _core.generate_record(boundary, "zz", "heterodyne", 0.17, 123, global_id)
        for global_id in ids
    }
    reverse = {
        global_id: _core.generate_record(boundary, "zz", "heterodyne", 0.17, 123, global_id)
        for global_id in reversed(ids)
    }
    assert forward == reverse


def test_mc_bytes_do_not_depend_on_thread_count_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kx, kt = _core.lattice_couplings("isotropic", None)

    def run(thread_count: int) -> tuple[list[list[int]], dict[str, object]]:
        monkeypatch.setenv("RAYON_NUM_THREADS", str(thread_count))
        clean = _core.clean_configurations(3, 4, kx, kt, 71, 30, 5, 3)
        boundary = _core.boundary_from_packed(3, 4, clean[1])
        record = _core.generate_record(boundary, "z", "heterodyne", 0.2, 71, 1)
        posterior = _core.posterior_observables(
            3,
            4,
            kx,
            kt,
            "z",
            record["record_couplings"],
            clean[1],
            "metropolis",
            71,
            1,
            "thread-count-contract",
            3,
            12,
            2,
            [0, 1],
            True,
            16,
        )
        return clean, posterior

    assert run(1) == run(4)


def test_parallel_posterior_batch_preserves_order_and_results() -> None:
    lx, lt = 4, 8
    kx, kt = _core.lattice_couplings("isotropic", None)
    planted = _core.clean_configurations(lx, lt, kx, kt, 91, 8, 2, 4)
    records = []
    for global_id, packed in enumerate(planted):
        boundary = _core.boundary_from_packed(lx, lt, packed)
        records.append(
            _core.generate_record(boundary, "z", "heterodyne", 0.2, 91, global_id)[
                "record_couplings"
            ]
        )
    arguments = (
        lx,
        lt,
        kx,
        kt,
        "z",
        records,
        planted,
        "metropolis",
        91,
        list(range(4)),
        [f"batch/{global_id}" for global_id in range(4)],
        4,
        [8] * 4,
        1,
        [0, 1, 2],
        [False] * 4,
        8,
    )
    serial = _core.posterior_observables_batch(*arguments, 1)
    parallel = _core.posterior_observables_batch(*arguments, 4)
    assert parallel == serial


@pytest.mark.parametrize("noise", ["z", "zz"])
def test_tnmc_python_entry_point_reports_block_statistics(noise: str) -> None:
    kx, kt = _core.lattice_couplings("isotropic", None)
    clean = _core.clean_configurations(3, 3, kx, kt, 913, 20, 3, 1)[0]
    boundary = _core.boundary_from_packed(3, 3, clean)
    record = _core.generate_record(boundary, noise, "heterodyne", 0.2, 913, 0)
    result = _core.posterior_observables(
        3,
        3,
        kx,
        kt,
        noise,
        record["record_couplings"],
        clean,
        "tnmc",
        913,
        0,
        f"tnmc-python-smoke/{noise}",
        2,
        3,
        1,
        [0, 1],
        True,
        2,
    )
    assert result["sweeps"] == 5
    assert result["tnmc_proposed"] == 5
    assert 0 <= result["tnmc_accepted"] <= 5
    assert result["tnmc_sites_proposed"] == 20
    assert result["tnmc_conditionals_regularized"] == 0
    assert result["local_proposed"] == result["cluster_proposed"] == 0
