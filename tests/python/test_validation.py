from __future__ import annotations

from pathlib import Path

from dcft.analysis import _finite_inner_absolute_tolerance
from dcft.config import load_config
from dcft.validation import _diagnostic_checks, _protocol_difference_checks


def test_frozen_chain_diagnostics_block_promotion(smoke_config_path: Path) -> None:
    config = load_config(smoke_config_path)
    scalar = {
        "lx": 4,
        "noise": "z",
        "p": 0.1,
        "measurement": "heterodyne",
        "protocol_id": "heterodyne",
        "update": "metropolis",
    }
    diagnostics = []
    for metric in ("energy", "boundary_magnetization", "planted_spin_overlap"):
        diagnostics.append(
            {
                **scalar,
                "metric": metric,
                "maximum_split_rhat": None,
                "rhat_converged": False,
                "minimum_trace_transitions": 0,
                "movement_detected": False,
                "minimum_trace_length": 32,
                "overdispersed_initializations_complete": True,
                "budget_bias_envelope": 0.0,
                "budget_complete": True,
                "budget_consistent": True,
                "budget_outer_records": 4,
            }
        )
    checks = _diagnostic_checks(config, [scalar], diagnostics)
    failures = {check.name for check in checks if not check.passed}
    assert "replicated-chain-rhat" in failures
    assert "diagnostic-chain-movement" in failures


def test_zz_symmetry_zero_spin_profile_records_bias_without_false_failure(
    smoke_config_path: Path,
) -> None:
    config = load_config(smoke_config_path)
    scalar = {
        "lx": 4,
        "noise": "zz",
        "p": 0.1,
        "measurement": "heterodyne",
        "protocol_id": "heterodyne",
        "update": "metropolis",
    }
    diagnostics = [
        {
            **scalar,
            "metric": metric,
            "budget_bias_envelope": 0.04,
            "budget_complete": True,
            "budget_consistent": metric != "spin_absolute_profile",
            "diagnostic_outer_records": config.mc.diagnostic_outer_records,
        }
        for metric in ("spin_absolute_profile", "bond_absolute_profile")
    ]

    checks = _diagnostic_checks(config, [scalar], diagnostics)
    symmetry_checks = [
        check for check in checks if check.name == "finite-inner-symmetry-bias-recorded"
    ]

    assert len(symmetry_checks) == 1
    assert symmetry_checks[0].passed
    assert not any(
        check.name == "finite-inner-budget-consistency"
        and check.scope.endswith("spin_absolute_profile")
        for check in checks
    )


def test_finite_inner_tolerance_respects_extensive_and_bounded_scales() -> None:
    assert _finite_inner_absolute_tolerance("energy", 10, 40) == 2.0
    assert _finite_inner_absolute_tolerance("planted_spin_overlap", 10, 40) == 0.01


def test_saturated_overlap_accepts_identical_overdispersed_saturation(
    smoke_config_path: Path,
) -> None:
    config = load_config(smoke_config_path)
    scalar = {
        "lx": 4,
        "noise": "zz",
        "p": 0.3,
        "measurement": "local-x",
        "protocol_id": "local-x",
        "update": "metropolis-global",
    }
    common = {
        **scalar,
        "maximum_split_rhat": 1.001,
        "rhat_converged": True,
        "minimum_trace_length": 32,
        "overdispersed_initializations_complete": True,
        "budget_bias_envelope": 0.001,
        "budget_complete": True,
        "budget_consistent": True,
        "budget_outer_records": 4,
        "diagnostic_outer_records": config.mc.diagnostic_outer_records,
    }
    diagnostics = [
        {
            **common,
            "metric": "energy",
            "minimum_trace_transitions": 10,
            "movement_detected": True,
        },
        {
            **common,
            "metric": "planted_bond_overlap",
            "rhat_converged": False,
            "rhat_or_saturation_converged": True,
            "identical_saturation_count": 1,
            "minimum_trace_transitions": 0,
            "movement_detected": False,
        },
        {**common, "metric": "spin_absolute_profile"},
        {**common, "metric": "bond_absolute_profile"},
    ]

    checks = _diagnostic_checks(config, [scalar], diagnostics)
    saturation = [check for check in checks if check.name == "diagnostic-movement-or-saturation"]

    assert len(saturation) == 1
    assert saturation[0].passed


def test_paired_ordering_uses_the_simultaneous_difference_band() -> None:
    curve_rows = [
        {
            "lx": 4,
            "noise": "z",
            "p": 0.1,
            "update": "metropolis",
            "measurement": measurement,
            "family": "spin",
            "separation": 2,
        }
        for measurement in ("heterodyne", "homodyne")
    ]
    difference = {
        "source": "mc",
        "lx": 4,
        "noise": "z",
        "p": 0.1,
        "update": "metropolis",
        "prior": None,
        "family": "spin",
        "separation": 2,
        "difference": -0.2,
        "difference_simultaneous_lower": -0.25,
        "difference_simultaneous_upper": -0.15,
    }
    checks = _protocol_difference_checks(curve_rows, [], [difference])
    assert any(check.name == "homodyne-ordering-mc" and not check.passed for check in checks)
