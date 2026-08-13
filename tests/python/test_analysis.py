from __future__ import annotations

import math
from pathlib import Path

from dcft.analysis import _diagnostic_summary_rows
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
                        "energy": value,
                        "magnetization": value,
                        "boundary_magnetization": value,
                        "planted_spin_overlap": value,
                        "planted_bond_overlap": value,
                        "spin_profile": [value] * 4,
                        "bond_profile": [value] * 4,
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
    assert energy["maximum_split_rhat"] is None
    assert energy["rhat_converged"] is False
    assert not any(
        isinstance(value, float) and not math.isfinite(value)
        for row in summaries
        for value in row.values()
    )
    canonical_json(summaries)
