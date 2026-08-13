"""Explicit, closed registries shared by configuration and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from . import _core

MEASUREMENTS: Final[tuple[str, ...]] = ("heterodyne", "homodyne", "gaussian", "local-x")
NOISES: Final[tuple[str, ...]] = ("z", "zz")
UPDATES: Final[tuple[str, ...]] = (
    "metropolis",
    "sequential-metropolis",
    "metropolis-global",
    "corrected-wolff",
    "tnmc",
)
OBSERVABLES: Final[tuple[str, ...]] = (
    "energy",
    "magnetization",
    "boundary-magnetization",
    "spin-profile",
    "bond-profile",
    "spin-correlator",
    "bond-correlator",
)
PRIORS: Final[tuple[str, ...]] = ("quantum", "finite-transfer", "transfer-ground")


@dataclass(frozen=True)
class MeasurementPoint:
    """One fully resolved protocol point."""

    name: str
    gamma: float | None = None

    @property
    def identifier(self) -> str:
        if self.name != "gaussian":
            return self.name
        if self.gamma is None:
            raise ValueError("gaussian measurement is missing gamma")
        return f"gaussian({self.gamma:.17g})"


def resolve_measurements(
    names: tuple[str, ...],
    p: float,
    gaussian_fractions: tuple[float, ...],
) -> tuple[MeasurementPoint, ...]:
    """Resolve named protocols and fractions of the allowed Gaussian range."""
    points: list[MeasurementPoint] = []
    for name in names:
        if name not in MEASUREMENTS or name == "gaussian":
            raise ValueError(f"invalid named measurement {name!r}")
        points.append(MeasurementPoint(name))
    maximum = float(_core.protocol_parameters("homodyne", p)["gamma"])
    points.extend(MeasurementPoint("gaussian", fraction * maximum) for fraction in gaussian_fractions)
    unique: dict[str, MeasurementPoint] = {}
    for point in points:
        parameters = _core.protocol_parameters(point.name, p, point.gamma)
        resolved = MeasurementPoint(point.name, parameters["gamma"] if point.name == "gaussian" else None)
        unique[resolved.identifier] = resolved
    return tuple(unique.values())


def verify_rust_registries() -> None:
    """Fail if the Python and Rust closed registries drift."""
    comparisons = (
        (MEASUREMENTS, tuple(_core.measurement_registry()), "measurement"),
        (NOISES, tuple(_core.noise_registry()), "noise"),
        (UPDATES, tuple(_core.update_registry()), "update"),
        (OBSERVABLES, tuple(_core.observable_registry()), "observable"),
    )
    for python_values, rust_values, label in comparisons:
        if python_values != rust_values:
            raise RuntimeError(
                f"{label} registry mismatch: Python={python_values!r}, Rust={rust_values!r}"
            )
