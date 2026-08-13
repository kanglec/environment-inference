"""Density-matrix helpers."""

from __future__ import annotations


def pure_density_matrix(state: list[float]) -> list[list[float]]:
    """Return ``|state><state|`` for a real normalized state vector."""

    return [[left * right for right in state] for left in state]


def trace(matrix: list[list[float]]) -> float:
    """Return the trace of a square matrix."""

    _validate_square(matrix)
    return sum(matrix[i][i] for i in range(len(matrix)))


def expectation_value(operator: list[list[float]], rho: list[list[float]]) -> float:
    """Return ``Tr(operator @ rho)`` for real dense matrices."""

    _validate_square(operator)
    _validate_square(rho)
    if len(operator) != len(rho):
        raise ValueError("operator and density matrix dimensions differ")

    total = 0.0
    n = len(rho)
    for row in range(n):
        for col in range(n):
            total += operator[row][col] * rho[col][row]
    return total


def _validate_square(matrix: list[list[float]]) -> None:
    if not matrix:
        raise ValueError("matrix must be non-empty")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")

