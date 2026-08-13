"""IID decoherence channels for diagonal Z-basis noise operators."""

from __future__ import annotations

import math
from typing import Literal

from .basis import bond_sites, bond_zz_eigenvalue, dimension, validate_n_sites, z_eigenvalue

NoiseKind = Literal["z", "zz"]


def lambda_from_noise_probability(p: float) -> float:
    """Return ``lambda = -1/2 log(1 - 2p)`` from ``main.tex``."""

    _validate_probability(p)
    return -0.5 * math.log(1.0 - 2.0 * p)


def decoherence_factor(
    left_state: int,
    right_state: int,
    *,
    n_sites: int,
    p: float,
    noise: NoiseKind,
    periodic: bool = True,
) -> float:
    """Return the channel factor multiplying ``rho[left_state][right_state]``."""

    validate_n_sites(n_sites)
    _validate_probability(p)

    factor = 1.0
    if noise == "z":
        for site in range(n_sites):
            factor *= (1.0 - p) + p * z_eigenvalue(left_state, site) * z_eigenvalue(
                right_state,
                site,
            )
        return factor

    if noise == "zz":
        for site in bond_sites(n_sites, periodic=periodic):
            left = bond_zz_eigenvalue(left_state, site, n_sites, periodic=periodic)
            right = bond_zz_eigenvalue(right_state, site, n_sites, periodic=periodic)
            factor *= (1.0 - p) + p * left * right
        return factor

    raise ValueError("noise must be either 'z' or 'zz'")


def apply_decoherence_channel(
    rho: list[list[float]],
    *,
    n_sites: int,
    p: float,
    noise: NoiseKind,
    periodic: bool = True,
) -> list[list[float]]:
    """Apply the iid ``Z`` or ``ZZ`` decoherence channel.

    For these diagonal noise operators,

    ``rho[a,b] -> rho[a,b] prod_x [(1-p) + p o_x(a) o_x(b)]``.
    """

    dim = dimension(n_sites)
    _validate_density_shape(rho, dim)
    return [
        [
            rho[left][right]
            * decoherence_factor(
                left,
                right,
                n_sites=n_sites,
                p=p,
                noise=noise,
                periodic=periodic,
            )
            for right in range(dim)
        ]
        for left in range(dim)
    ]


def _validate_probability(p: float) -> None:
    if not isinstance(p, int | float):
        raise TypeError("p must be a real number")
    if p < 0.0 or p >= 0.5:
        raise ValueError("p must satisfy 0 <= p < 1/2")


def _validate_density_shape(rho: list[list[float]], dim: int) -> None:
    if len(rho) != dim or any(len(row) != dim for row in rho):
        raise ValueError("rho shape does not match n_sites")

