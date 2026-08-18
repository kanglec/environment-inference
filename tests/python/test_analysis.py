from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dcft.analysis import (
    _diagnostic_summary_rows,
    _is_identical_constant_saturation,
    _protocol_difference_rows,
)
from dcft.artifacts import canonical_json
from dcft.config import load_config


def test_infinite_rhat_is_explicitly_labeled_not_serialized(
    smoke_config_path: Path,
) -> None:
    config = load_config(smoke_config_path)
    rows = []
    for global_id in range(2):
        for multiplier in (1, 2, 4):
            for replica in range(2):
                role = "production" if multiplier == 1 and replica == 0 else "finite-inner"
                value = float(replica)
                rows.append(
                    {
                        "global_id": global_id,
                        "lx": 4,
                        "lt": 8,
                        "noise": "z",
                        "p": 0.1,
                        "measurement": "heterodyne",
                        "protocol_id": "heterodyne",
                        "update": "metropolis",
                        "inner_budget_multiplier": multiplier,
                        "replica": replica,
                        "chain_role": role,
                        "initialization": "planted",
                        "energy": value,
                        "magnetization": value,
                        "boundary_magnetization": value,
                        "planted_spin_overlap": value,
                        "planted_bond_overlap": value,
                        "spin_profile": [value] * 4,
                        "bond_profile": [value] * 4,
                        "energy_trace": [],
                        "magnetization_trace": [],
                        "boundary_magnetization_trace": [],
                        "planted_spin_overlap_trace": [],
                        "planted_bond_overlap_trace": [],
                    }
                )
        for replica, initialization in enumerate(("all-plus", "all-minus")):
            value = float(replica)
            rows.append(
                {
                    **rows[-1],
                    "inner_budget_multiplier": 1,
                    "replica": replica,
                    "chain_role": "convergence",
                    "initialization": initialization,
                    "energy_trace": [value] * 4,
                    "magnetization_trace": [value] * 4,
                    "boundary_magnetization_trace": [value] * 4,
                    "planted_spin_overlap_trace": [value] * 4,
                    "planted_bond_overlap_trace": [value] * 4,
                }
            )
    summaries = _diagnostic_summary_rows(config, rows)
    energy = next(row for row in summaries if row["metric"] == "energy")
    assert energy["infinite_split_rhat_count"] == 2
    assert energy["identical_saturation_count"] == 0
    assert energy["unresolved_infinite_split_rhat_count"] == 2
    assert energy["maximum_split_rhat"] is None
    assert energy["rhat_converged"] is False
    assert energy["rhat_or_saturation_converged"] is False
    assert not any(
        isinstance(value, float) and not math.isfinite(value)
        for row in summaries
        for value in row.values()
    )
    canonical_json(summaries)


def test_identical_constant_saturation_requires_same_value_everywhere() -> None:
    assert _is_identical_constant_saturation(np.ones((4, 64)))
    assert not _is_identical_constant_saturation(
        np.asarray([[1.0] * 64, [1.0] * 64, [-1.0] * 64, [-1.0] * 64])
    )
    moving = np.ones((4, 64))
    moving[0, -1] = -1.0
    assert not _is_identical_constant_saturation(moving)


def test_protocol_difference_preserves_shared_outer_covariance(
    smoke_config_path: Path,
) -> None:
    config = load_config(smoke_config_path)
    grouped = {}
    for measurement, offset, gamma in (
        ("heterodyne", 0.0, 0.2),
        ("homodyne", 0.1, 0.4),
    ):
        records = []
        for global_id, shared in enumerate((0.2, 0.4, 0.6, 0.8)):
            profile = [[shared + offset] * 4 for _ in range(3)]
            records.append(
                {
                    "global_id": global_id,
                    "separations": [0, 1, 2],
                    "spin_correlator_profile": profile,
                    "bond_correlator_profile": profile,
                }
            )
        grouped[(4, 8, "z", 0.1, measurement, measurement, "metropolis", gamma)] = records
    rows = _protocol_difference_rows(config, grouped, [], {})
    assert rows
    assert all(row["difference"] == pytest.approx(0.1) for row in rows)
    assert all(row["difference_standard_error"] == pytest.approx(0.0) for row in rows)


def test_sampled_exact_protocol_difference_uses_common_random_numbers(
    smoke_config_path: Path,
) -> None:
    config = load_config(smoke_config_path)
    exact_records = []
    for measurement, offset in (("heterodyne", 0.0), ("homodyne", 0.2)):
        for separation in (0, 1, 2):
            for global_id, shared in enumerate((0.1, 0.3, 0.5, 0.7)):
                exact_records.append(
                    {
                        "global_id": global_id,
                        "lx": 4,
                        "noise": "z",
                        "p": 0.1,
                        "measurement": measurement,
                        "prior": "finite-transfer",
                        "observable": "spin-pair",
                        "separation": separation,
                        "absolute_contribution": shared + offset,
                        "record_mode": "sampled-gaussian",
                    }
                )
    rows = _protocol_difference_rows(config, {}, exact_records, {})
    assert rows
    assert {row["source"] for row in rows} == {"ed"}
    assert all(row["difference"] == pytest.approx(0.2) for row in rows)
    assert all(row["difference_standard_error"] == pytest.approx(0.0) for row in rows)
