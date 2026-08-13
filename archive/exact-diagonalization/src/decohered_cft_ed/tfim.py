"""Transverse-field Ising Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass

from .basis import bond_sites, bond_zz_eigenvalue, dimension, validate_n_sites


@dataclass(frozen=True)
class TFIMHamiltonian:
    """Finite-size TFIM Hamiltonian.

    The Hamiltonian is

    ``H = -j sum_x Z_x Z_{x+1} - h sum_x X_x``.
    """

    n_sites: int
    j: float = 1.0
    h: float = 1.0
    periodic: bool = True

    def __post_init__(self) -> None:
        validate_n_sites(self.n_sites)
        if not isinstance(self.j, int | float) or not isinstance(self.h, int | float):
            raise TypeError("j and h must be real numbers")

    @property
    def dim(self) -> int:
        return dimension(self.n_sites)

    def diagonal_energy(self, state: int) -> float:
        """Return the diagonal ``-j sum Z_x Z_{x+1}`` energy."""

        total = 0.0
        for site in bond_sites(self.n_sites, periodic=self.periodic):
            total -= self.j * bond_zz_eigenvalue(
                state,
                site,
                self.n_sites,
                periodic=self.periodic,
            )
        return total

    def apply(self, vector: list[float]) -> list[float]:
        """Return ``H @ vector`` without first materializing the matrix."""

        if len(vector) != self.dim:
            raise ValueError("vector length does not match Hamiltonian dimension")

        out = [0.0 for _ in range(self.dim)]
        for state, amplitude in enumerate(vector):
            out[state] += self.diagonal_energy(state) * amplitude
            for site in range(self.n_sites):
                flipped = state ^ (1 << site)
                out[flipped] -= self.h * amplitude
        return out

    def to_dense(self) -> list[list[float]]:
        """Return the dense Hamiltonian matrix as nested Python lists."""

        matrix = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
        for state in range(self.dim):
            matrix[state][state] = self.diagonal_energy(state)
            for site in range(self.n_sites):
                flipped = state ^ (1 << site)
                matrix[flipped][state] -= self.h
        return matrix

