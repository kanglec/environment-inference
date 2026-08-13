"""Synthetic Rust-analysis CSV fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dcft_analysis.campaign import CSV_COLUMNS
from dcft_analysis.scaling import CORRELATOR_COLUMNS


def write_observables(
    root: Path,
    noise: str = "z",
    p_tag: str = "p005",
    *,
    lx: int = 4,
    lt: int = 6,
    samples: int = 12,
) -> Path:
    """Write a small valid facsimile of the Rust analysis writer's CSV."""
    rows = []
    for r in range(lx // 2 + 1):
        rows.append(
            {
                "samples": samples,
                "lx": lx,
                "lt": lt,
                "energy_density": -1.1,
                "magnetization_density": 0.02,
                "boundary_magnetization": 0.03,
                "local_spin_fidelity": 0.2,
                "local_bond_fidelity": 0.7,
                "r": r,
                "spin_linear_corr": 1.0 / (r + 1),
                "spin_fidelity_corr": 1.0 / (r + 1),
                "spin_ea_corr": 0.2 / (r + 1),
                "bond_linear_corr": 0.8 / (r + 1),
                "bond_fidelity_corr": 0.8 / (r + 1),
                "bond_ea_corr": 0.3 / (r + 1),
            }
        )
    path = root / noise / p_tag / "analysis" / "observables.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_writer():
    return write_observables


def write_campaign_tables(
    root: Path,
    L: int,
    *,
    noises: tuple[str, ...] = ("z",),
    p_values: tuple[float, ...] = (0.0,),
    decay_exponent: float = 0.5,
    analysis_name: str = "campaign-analysis",
) -> tuple[Path, Path]:
    """Write synthetic outputs of dcft-analyze-campaign for one square size."""
    scalar_rows = []
    correlator_rows = []
    amplitudes = dict(zip(CORRELATOR_COLUMNS, (1.0, 0.8, 0.9, 0.7, 0.3, 0.2), strict=True))
    for noise in noises:
        for p in p_values:
            p_tag = f"p{round(100 * p):03d}"
            scalar_rows.append(
                {
                    "noise": noise,
                    "p": p,
                    "p_tag": p_tag,
                    "samples": 20,
                    "lx": L,
                    "lt": L,
                    "E_D": -1.0 - p,
                    "M": 0.01,
                    "M_partial": 0.02,
                    "F_sigma_loc": 0.2 + p,
                    "F_epsilon_loc": 0.6 + p / 2,
                    "source_csv": f"{noise}/{p_tag}/analysis/observables.csv",
                }
            )
            for r in range(L // 2 + 1):
                distance_scale = max(1, 2 * r)
                row = {"noise": noise, "p": p, "p_tag": p_tag, "r": r}
                row.update(
                    {
                        column: amplitude * distance_scale ** (-decay_exponent)
                        for column, amplitude in amplitudes.items()
                    }
                )
                correlator_rows.append(row)
    tables = root / f"L{L}" / analysis_name / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    scalar_path = tables / "scalar_observables.csv"
    correlator_path = tables / "correlators.csv"
    pd.DataFrame(scalar_rows).to_csv(scalar_path, index=False)
    pd.DataFrame(correlator_rows).to_csv(correlator_path, index=False)
    return scalar_path, correlator_path


@pytest.fixture
def scaling_writer():
    return write_campaign_tables
