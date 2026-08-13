"""Autocorrelation, nested-sampling diagnostics, and curve-aware resampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class AutocorrelationEstimate:
    tau_saved: float
    tau_sweeps: float
    effective_samples: float
    window_lag: int
    rule: str = "Geyer initial-positive monotone sequence"


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: FloatArray
    lower: FloatArray
    upper: FloatArray
    standard_error: FloatArray
    confidence: float
    block_length: int
    resamples: int


@dataclass(frozen=True)
class ImmseEstimate:
    gamma: FloatArray
    q_ea: FloatArray
    information_per_site: FloatArray
    lower: FloatArray
    upper: FloatArray
    quadrature_systematic: FloatArray


def _autocovariance(values: FloatArray) -> FloatArray:
    centered = values - np.mean(values)
    count = centered.size
    if count < 2:
        raise ValueError("autocorrelation requires at least two samples")
    variance = float(centered @ centered)
    if variance == 0.0:
        return np.asarray([0.0], dtype=np.float64)
    length = 1 << (2 * count - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=length)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=length)[:count]
    covariance /= np.arange(count, 0, -1)
    return np.asarray(covariance, dtype=np.float64)


def geyer_autocorrelation(
    values: npt.ArrayLike,
    *,
    saving_interval_sweeps: int = 1,
) -> AutocorrelationEstimate:
    """Estimate tau_int using Geyer's initial-positive monotone sequence."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("autocorrelation input must be one-dimensional")
    if saving_interval_sweeps < 1:
        raise ValueError("saving interval must be positive")
    covariance = _autocovariance(series)
    if covariance.size == 1 or covariance[0] <= 0.0:
        return AutocorrelationEstimate(0.5, 0.5 * saving_interval_sweeps, float(series.size), 0)
    pair_sums = np.asarray(
        [
            covariance[lag] + covariance[lag + 1]
            for lag in range(0, covariance.size - 1, 2)
        ],
        dtype=np.float64,
    )
    positive_count = 0
    for value in pair_sums:
        if value <= 0.0:
            break
        positive_count += 1
    if positive_count == 0:
        tau = 0.5
        window = 0
    else:
        monotone = np.minimum.accumulate(pair_sums[:positive_count])
        tau = max(0.5, float(-0.5 + np.sum(monotone) / covariance[0]))
        window = 2 * positive_count - 1
    effective = min(float(series.size), series.size / (2.0 * tau))
    return AutocorrelationEstimate(
        tau_saved=tau,
        tau_sweeps=tau * saving_interval_sweeps,
        effective_samples=effective,
        window_lag=window,
    )


def moving_block_indices(
    count: int,
    block_length: int,
    resamples: int,
    *,
    seed: int,
) -> IntArray:
    """Draw ordinary moving-block resamples without crossing curve boundaries."""
    if count < 2 or not 1 <= block_length <= count or resamples < 1:
        raise ValueError("invalid moving-block bootstrap dimensions")
    generator = np.random.Generator(np.random.PCG64(seed))
    blocks = math.ceil(count / block_length)
    starts = generator.integers(0, count - block_length + 1, size=(resamples, blocks))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets).reshape(resamples, -1)
    return np.asarray(indices[:, :count], dtype=np.int64)


def moving_block_bootstrap(
    values: npt.ArrayLike,
    *,
    block_length: int,
    resamples: int,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim < 2:
        raise ValueError("bootstrap values need an outer-id axis")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    indices = moving_block_indices(array.shape[0], block_length, resamples, seed=seed)
    bootstrap = np.mean(array[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=np.mean(array, axis=0),
        lower=np.quantile(bootstrap, tail, axis=0),
        upper=np.quantile(bootstrap, 1.0 - tail, axis=0),
        standard_error=np.std(bootstrap, axis=0, ddof=1),
        confidence=confidence,
        block_length=block_length,
        resamples=resamples,
    )


def simultaneous_curve_interval(
    values: npt.ArrayLike,
    *,
    block_length: int,
    resamples: int,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a max-t simultaneous interval for an entire correlated curve."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("simultaneous intervals require outer_ids x curve_points")
    indices = moving_block_indices(array.shape[0], block_length, resamples, seed=seed)
    bootstrap = np.mean(array[indices], axis=1)
    center = np.mean(array, axis=0)
    standard_error = np.std(bootstrap, axis=0, ddof=1)
    safe = np.where(standard_error > 0.0, standard_error, 1.0)
    maximum = np.max(np.abs((bootstrap - center) / safe), axis=1)
    critical = float(np.quantile(maximum, confidence))
    return BootstrapInterval(
        estimate=center,
        lower=center - critical * standard_error,
        upper=center + critical * standard_error,
        standard_error=standard_error,
        confidence=confidence,
        block_length=block_length,
        resamples=resamples,
    )


def paired_curve_difference(left: npt.ArrayLike, right: npt.ArrayLike) -> FloatArray:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("paired protocol arrays must have identical outer-id and curve shapes")
    return np.asarray(left_values - right_values, dtype=np.float64)


def split_rhat(chains: npt.ArrayLike) -> float:
    """Compute split R-hat from replicated diagnostic chains."""
    values = np.asarray(chains, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("split R-hat requires at least two chains of length four")
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    length = split.shape[1]
    chain_means = np.mean(split, axis=1)
    between = length * float(np.var(chain_means, ddof=1))
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    if within == 0.0:
        return 1.0 if between == 0.0 else math.inf
    marginal = (length - 1.0) / length * within + between / length
    return math.sqrt(marginal / within)


def nested_variance_components(replicates: npt.ArrayLike) -> tuple[float, float]:
    """Estimate between-record and within-record variance.

    ``replicates`` has shape ``(outer_records, independent_inner_chains)``.
    """
    values = np.asarray(replicates, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("nested variance needs at least two records and two replicas")
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    means = np.mean(values, axis=1)
    between_observed = float(np.var(means, ddof=1))
    between = max(0.0, between_observed - within / values.shape[1])
    return between, within


def common_resample_immse(
    gamma: npt.ArrayLike,
    planted_overlaps: npt.ArrayLike,
    *,
    block_length: int,
    resamples: int,
    confidence: float = 0.95,
    seed: int = 0,
) -> ImmseEstimate:
    """Integrate I-MMSE with one common outer-id resample across gamma."""
    grid = np.asarray(gamma, dtype=np.float64)
    overlaps = np.asarray(planted_overlaps, dtype=np.float64)
    if grid.ndim != 1 or overlaps.ndim != 2 or overlaps.shape[1] != grid.size:
        raise ValueError("I-MMSE inputs must have shapes gamma and outer_ids x gamma")
    if grid[0] != 0.0 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("gamma grid must begin at zero and increase strictly")
    indices = moving_block_indices(overlaps.shape[0], block_length, resamples, seed=seed)
    bootstrap_q = np.mean(overlaps[indices], axis=1)
    q_mean = np.mean(overlaps, axis=0)

    def cumulative_trapezoid(curve: FloatArray) -> FloatArray:
        output = np.zeros((*curve.shape[:-1], grid.size), dtype=np.float64)
        increments = 0.25 * np.diff(grid) * (2.0 - curve[..., :-1] - curve[..., 1:])
        output[..., 1:] = np.cumsum(increments, axis=-1)
        return output

    integrated = cumulative_trapezoid(bootstrap_q)
    estimate = cumulative_trapezoid(q_mean)
    tail = (1.0 - confidence) / 2.0
    lower = np.quantile(integrated, tail, axis=0)
    upper = np.quantile(integrated, 1.0 - tail, axis=0)
    systematic = np.zeros(grid.size, dtype=np.float64)
    for stop in range(2, grid.size + 1):
        trap = estimate[stop - 1]
        higher = 0.5 * float(simpson(1.0 - q_mean[:stop], x=grid[:stop]))
        systematic[stop - 1] = abs(higher - trap)
    return ImmseEstimate(
        gamma=grid,
        q_ea=q_mean,
        information_per_site=estimate,
        lower=lower,
        upper=upper,
        quadrature_systematic=systematic,
    )
