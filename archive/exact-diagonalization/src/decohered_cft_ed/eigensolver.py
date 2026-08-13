"""Small dense eigensolver utilities."""

from __future__ import annotations

import math

from .tfim import TFIMHamiltonian


def ground_state(
    hamiltonian: TFIMHamiltonian,
    *,
    tolerance: float = 1e-12,
    max_sweeps: int = 100,
) -> tuple[float, list[float]]:
    """Return the lowest eigenvalue and normalized eigenvector.

    This dependency-free implementation uses Jacobi rotations for real
    symmetric dense matrices. It is meant for small exact-diagonalization
    checks, not production-scale diagonalization.
    """

    eigenvalues, eigenvectors = symmetric_eigendecomposition(
        hamiltonian.to_dense(),
        tolerance=tolerance,
        max_sweeps=max_sweeps,
    )
    index = min(range(len(eigenvalues)), key=eigenvalues.__getitem__)
    vector = [row[index] for row in eigenvectors]
    vector = _normalized(vector)
    pivot = max(range(len(vector)), key=lambda i: abs(vector[i]))
    if vector[pivot] < 0.0:
        vector = [-value for value in vector]
    return eigenvalues[index], vector


def symmetric_eigendecomposition(
    matrix: list[list[float]],
    *,
    tolerance: float = 1e-12,
    max_sweeps: int = 100,
) -> tuple[list[float], list[list[float]]]:
    """Diagonalize a real symmetric matrix by Jacobi rotations."""

    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be non-empty")
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")

    a = [list(row) for row in matrix]
    _validate_symmetric(a, tolerance=max(tolerance, 1e-14))
    v = [[1.0 if row == col else 0.0 for col in range(n)] for row in range(n)]

    for _ in range(max_sweeps * n * n):
        p, q, offdiag = _largest_offdiag(a)
        if offdiag < tolerance:
            break
        _rotate(a, v, p, q)
    else:
        raise RuntimeError("Jacobi eigensolver did not converge")

    return [a[i][i] for i in range(n)], v


def _validate_symmetric(matrix: list[list[float]], *, tolerance: float) -> None:
    n = len(matrix)
    for row in range(n):
        for col in range(row + 1, n):
            if abs(matrix[row][col] - matrix[col][row]) > tolerance:
                raise ValueError("matrix must be symmetric")


def _largest_offdiag(matrix: list[list[float]]) -> tuple[int, int, float]:
    n = len(matrix)
    best_p = 0
    best_q = 1 if n > 1 else 0
    best = 0.0
    for row in range(n):
        for col in range(row + 1, n):
            value = abs(matrix[row][col])
            if value > best:
                best = value
                best_p = row
                best_q = col
    return best_p, best_q, best


def _rotate(matrix: list[list[float]], vectors: list[list[float]], p: int, q: int) -> None:
    if matrix[p][q] == 0.0:
        return

    app = matrix[p][p]
    aqq = matrix[q][q]
    apq = matrix[p][q]
    tau = (aqq - app) / (2.0 * apq)
    if tau >= 0.0:
        t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
    else:
        t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
    c = 1.0 / math.sqrt(1.0 + t * t)
    s = t * c

    n = len(matrix)
    for k in range(n):
        if k != p and k != q:
            akp = matrix[k][p]
            akq = matrix[k][q]
            matrix[k][p] = c * akp - s * akq
            matrix[p][k] = matrix[k][p]
            matrix[k][q] = s * akp + c * akq
            matrix[q][k] = matrix[k][q]

    matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
    matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
    matrix[p][q] = 0.0
    matrix[q][p] = 0.0

    for k in range(n):
        vkp = vectors[k][p]
        vkq = vectors[k][q]
        vectors[k][p] = c * vkp - s * vkq
        vectors[k][q] = s * vkp + c * vkq


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("cannot normalize zero vector")
    return [value / norm for value in vector]

