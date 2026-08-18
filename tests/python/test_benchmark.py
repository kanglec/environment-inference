from __future__ import annotations

from typing import Any

from dcft.benchmark import _recommend_update


def _method(
    update: str,
    *,
    efficiency: float,
    moves: bool,
    rhat: float | None,
) -> dict[str, Any]:
    correlation = {"window_lag": 0}
    return {
        "update": update,
        "autocorrelation": {
            "effective_overlap_samples_per_second": efficiency,
            "probes": 64,
            "energy": correlation,
            "planted_overlap": correlation,
            "energy_trace_moves": moves,
            "planted_overlap_trace_moves": moves,
        },
        "thermalization": {
            "energy_split_rhat": rhat,
            "boundary_magnetization_split_rhat": rhat,
            "planted_overlap_split_rhat": rhat,
        },
        "acceptance": {"tnmc_conditionals_regularized": 0},
    }


def test_recommendation_rejects_fast_frozen_kernel() -> None:
    frozen = _method("frozen", efficiency=1_000.0, moves=False, rhat=None)
    healthy = _method("healthy", efficiency=10.0, moves=True, rhat=1.001)
    recommendation = _recommend_update([frozen, healthy], noise="z")
    assert recommendation["status"] == "qualified"
    assert recommendation["update"] == "healthy"
    assert frozen["qualification"]["eligible_for_recommendation"] is False
    assert healthy["qualification"]["eligible_for_recommendation"] is True


def test_recommendation_reports_when_no_method_qualifies() -> None:
    frozen = _method("frozen", efficiency=1_000.0, moves=False, rhat=None)
    recommendation = _recommend_update([frozen], noise="zz")
    assert recommendation["status"] == "no-qualified-method"
    assert recommendation["update"] is None
    assert recommendation["rejections"]["frozen"]
