from __future__ import annotations

import numpy as np
import pytest

from dcft.statistics import (
    common_resample_immse,
    geyer_autocorrelation,
    moving_block_bootstrap,
    moving_block_indices,
    nested_variance_components,
    paired_curve_difference,
    simultaneous_curve_interval,
    split_rhat,
)


def test_geyer_iid_and_correlated_series() -> None:
    generator = np.random.default_rng(7)
    iid = generator.normal(size=50_000)
    iid_estimate = geyer_autocorrelation(iid, saving_interval_sweeps=4)
    assert 0.45 <= iid_estimate.tau_saved <= 0.65
    correlated = np.empty_like(iid)
    correlated[0] = iid[0]
    for index in range(1, iid.size):
        correlated[index] = 0.8 * correlated[index - 1] + iid[index]
    assert geyer_autocorrelation(correlated).tau_saved > 2.0
    assert iid_estimate.tau_sweeps == pytest.approx(4 * iid_estimate.tau_saved)


def test_moving_blocks_are_contiguous() -> None:
    indices = moving_block_indices(12, 4, 10, seed=3)
    assert indices.shape == (10, 12)
    for row in indices:
        for start in range(0, 12, 4):
            np.testing.assert_array_equal(np.diff(row[start : start + 4]), 1)


def test_curve_bootstrap_and_simultaneous_band() -> None:
    generator = np.random.default_rng(2)
    values = generator.normal(size=(64, 5))
    pointwise = moving_block_bootstrap(values, block_length=4, resamples=200, seed=1)
    simultaneous = simultaneous_curve_interval(values, block_length=4, resamples=200, seed=1)
    assert pointwise.estimate.shape == (5,)
    assert np.all(simultaneous.lower <= simultaneous.estimate)
    assert np.all(simultaneous.upper >= simultaneous.estimate)


def test_pairing_and_nested_variance() -> None:
    left = np.arange(12, dtype=np.float64).reshape(4, 3)
    right = left - 2.0
    np.testing.assert_array_equal(paired_curve_difference(left, right), 2.0)
    replicas = np.asarray([[1.0, 1.2, 0.8], [3.0, 3.2, 2.8], [5.0, 5.2, 4.8]])
    between, within = nested_variance_components(replicas)
    assert between > within


def test_split_rhat() -> None:
    generator = np.random.default_rng(10)
    chains = generator.normal(size=(4, 2000))
    assert split_rhat(chains) < 1.01


def test_common_resample_immse_constant_q() -> None:
    gamma = np.linspace(0.0, 2.0, 9)
    overlaps = np.full((32, gamma.size), 0.25)
    estimate = common_resample_immse(
        gamma, overlaps, block_length=4, resamples=100, seed=8
    )
    np.testing.assert_allclose(estimate.information_per_site, 0.5 * gamma * 0.75)
    np.testing.assert_allclose(estimate.quadrature_systematic, 0.0, atol=1e-14)

