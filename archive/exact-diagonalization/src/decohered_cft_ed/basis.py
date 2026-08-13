"""Computational-basis utilities.

Basis states are integers in ``0..2**n_sites``. Bit value 0 represents the
``Z=+1`` eigenstate, and bit value 1 represents the ``Z=-1`` eigenstate.
"""

from __future__ import annotations


def dimension(n_sites: int) -> int:
    """Return the Hilbert-space dimension for ``n_sites`` qubits."""

    validate_n_sites(n_sites)
    return 1 << n_sites


def validate_n_sites(n_sites: int) -> None:
    if not isinstance(n_sites, int):
        raise TypeError("n_sites must be an integer")
    if n_sites < 2:
        raise ValueError("n_sites must be at least 2")


def validate_site(n_sites: int, site: int) -> None:
    validate_n_sites(n_sites)
    if not isinstance(site, int):
        raise TypeError("site must be an integer")
    if site < 0 or site >= n_sites:
        raise ValueError("site index out of range")


def z_eigenvalue(state: int, site: int) -> int:
    """Return the ``Z_site`` eigenvalue of a computational-basis state."""

    if (state >> site) & 1:
        return -1
    return 1


def bond_zz_eigenvalue(
    state: int,
    site: int,
    n_sites: int,
    *,
    periodic: bool = True,
) -> int:
    """Return the ``Z_site Z_{site+1}`` eigenvalue."""

    validate_site(n_sites, site)
    right = site + 1
    if right == n_sites:
        if not periodic:
            raise ValueError("open-boundary bond starts must satisfy site < n_sites - 1")
        right = 0
    return z_eigenvalue(state, site) * z_eigenvalue(state, right)


def bond_sites(n_sites: int, *, periodic: bool = True) -> range:
    """Return the sites ``x`` labeling bonds ``(x, x+1)``."""

    validate_n_sites(n_sites)
    if periodic:
        return range(n_sites)
    return range(n_sites - 1)

