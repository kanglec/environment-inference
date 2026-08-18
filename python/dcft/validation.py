"""Scientific and structural campaign validation matrix."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import pyarrow as pa

from . import _core
from .artifacts import (
    Artifact,
    discover_artifacts,
    read_table,
    source_digest,
    verify_artifact,
    write_artifact,
)
from .config import CampaignConfig
from .exact import build_priors, heterodyne_purity_from_kernel
from .planning import read_plan, read_state, update_task_state


class ValidationError(RuntimeError):
    """A validation campaign contains failed required checks."""


@dataclass(frozen=True)
class Check:
    name: str
    scope: str
    value: float | None
    tolerance: float | None
    passed: bool
    detail: str

    def row(self) -> dict[str, object]:
        return {
            "check": self.name,
            "scope": self.scope,
            "value": self.value,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "detail": self.detail,
        }


def _current_artifacts(config: CampaignConfig, kind: str | None = None) -> list[Artifact]:
    state = read_state(config)
    referenced = {
        str(artifact_id)
        for task_state in state["tasks"].values()
        for artifact_id in task_state["artifacts"]
    }
    return [
        artifact
        for artifact in discover_artifacts(config.campaign.output_root)
        if artifact.artifact_id in referenced
        and (kind is None or artifact.manifest["kind"] == kind)
    ]


def _tables(config: CampaignConfig, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in _current_artifacts(config, kind):
        rows.extend(cast(list[dict[str, Any]], read_table(artifact).to_pylist()))
    return rows


def _structural_checks(artifacts: Iterable[Artifact]) -> list[Check]:
    checks: list[Check] = []
    for artifact in artifacts:
        try:
            verify_artifact(artifact.path)
        except Exception as error:
            checks.append(
                Check("artifact-integrity", artifact.artifact_id, None, None, False, str(error))
            )
        else:
            checks.append(
                Check(
                    "artifact-integrity",
                    artifact.artifact_id,
                    0.0,
                    0.0,
                    True,
                    f"{artifact.manifest['kind']} checksums and row count verified",
                )
            )
    return checks


def _ed_checks(config: CampaignConfig, rows: list[dict[str, Any]]) -> list[Check]:
    checks: list[Check] = []
    for index, row in enumerate(rows):
        scope = (
            f"L={row['lx']}/{row['noise']}/p={row['p']}/{row['prior']}/"
            f"{row['protocol_id']}/{row['observable']}/r={row['separation']}"
        )
        diagnostics = {
            "ed-trace": abs(float(row["trace_error"])),
            "ed-hermiticity": abs(float(row["hermiticity_error"])),
            "ed-positivity": max(0.0, -float(row["minimum_eigenvalue"])),
        }
        for name, value in diagnostics.items():
            checks.append(
                Check(
                    name,
                    scope,
                    value,
                    config.ed.positivity_tolerance,
                    value <= config.ed.positivity_tolerance,
                    "dense density-matrix diagnostic",
                )
            )
        if row["ground_residual"] is not None:
            residual = float(row["ground_residual"])
            checks.append(
                Check(
                    "ed-ground-residual",
                    scope,
                    residual,
                    1e-10,
                    residual <= 1e-10,
                    "||H psi - E psi||_2",
                )
            )
        violation = float(row["measurement_witness"]) - float(row["physical_fidelity"])
        tolerance = max(
            5.0 * float(row["witness_standard_error"]),
            20.0 * config.ed.positivity_tolerance,
        )
        checks.append(
            Check(
                "fidelity-bound",
                scope,
                violation,
                tolerance,
                violation <= tolerance,
                "measurement witness must not exceed physical fidelity",
            )
        )
        if index > 100_000:
            raise RuntimeError("unexpectedly large ED result table")

    return checks


def _purity_checks(config: CampaignConfig, rows: list[dict[str, Any]]) -> list[Check]:
    kx, kt = _core.lattice_couplings(config.lattice.regularization, config.lattice.delta_tau)
    cache: dict[int, Any] = {}
    unique: dict[tuple[int, int, str, float, str], float] = {}
    for row in rows:
        if row["measurement"] != "heterodyne":
            continue
        key = (
            int(row["lx"]),
            config.lattice.lt(int(row["lx"])),
            str(row["noise"]),
            float(row["p"]),
            str(row["prior"]),
        )
        unique[key] = float(row["purity"])
    checks: list[Check] = []
    for (lx, lt, noise, p, prior_name), physical in sorted(
        unique.items(), key=lambda item: repr(item[0])
    ):
        if lx not in cache:
            cache[lx] = build_priors(lx, lt, kx, kt)[1]
        prior = cache[lx].named(prior_name)
        kernel = heterodyne_purity_from_kernel(prior, p, noise)
        difference = abs(kernel - physical)
        checks.append(
            Check(
                "heterodyne-purity-calibration",
                f"L={lx}/{noise}/p={p}/{prior_name}",
                difference,
                2e-10,
                difference <= 2e-10,
                "two-replica Gaussian kernel versus direct density-matrix purity",
            )
        )
    return checks


def _mc_checks(scalar_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]]) -> list[Check]:
    checks: list[Check] = []
    for row in scalar_rows:
        scope = f"L={row['lx']}/{row['noise']}/p={row['p']}/{row['protocol_id']}/{row['update']}"
        energy_tolerance = max(5.0 * float(row["energy_shift_residual_standard_error"]), 1e-10)
        energy_residual = abs(float(row["energy_shift_residual"]))
        checks.append(
            Check(
                "annealed-energy-shift",
                scope,
                energy_residual,
                energy_tolerance,
                energy_residual <= energy_tolerance,
                "fixed-record energy versus clean energy minus protocol shift",
            )
        )
        q_difference = abs(float(row["q_ea_planted"]) - float(row["q_ea_direct_diagnostic"]))
        q_tolerance = (
            5.0
            * math.hypot(
                float(row["q_ea_planted_standard_error"]),
                float(row["q_ea_direct_standard_error"]),
            )
            + 0.05
        )
        checks.append(
            Check(
                "nishimori-planted-estimator",
                scope,
                q_difference,
                q_tolerance,
                q_difference <= q_tolerance,
                "planted q_EA versus finite-inner-chain square diagnostic; 0.05 bias envelope",
            )
        )
        checks.append(
            Check(
                "outer-block-correlation-scale",
                scope,
                float(row["outer_block_length_used"]),
                float(row["outer_minimum_correlation_block"]),
                bool(row["outer_block_correlation_adequate"]),
                "configured block length must cover twice the slowest clean-chain tau_int",
            )
        )
        ratio = row.get("outer_block_standard_error_ratio")
        checks.append(
            Check(
                "outer-block-stability",
                scope,
                None if ratio is None else float(ratio),
                2.0,
                bool(row.get("outer_block_stable", False)),
                "standard errors must be stable from half to twice the configured block length",
            )
        )
        if row["update"] in {"tnmc", "tnmc-global"}:
            regularized = int(row.get("tnmc_conditionals_regularized", 0))
            checks.append(
                Check(
                    "tnmc-conditional-positivity",
                    scope,
                    float(regularized),
                    0.0,
                    regularized == 0,
                    "production TNMC conditionals must not require defensive regularization",
                )
            )
    for row in curve_rows:
        residual = abs(float(row["annealed_residual"]))
        tolerance = max(5.0 * float(row["annealed_residual_standard_error"]), 1e-10)
        checks.append(
            Check(
                "annealed-linear-correlator",
                f"L={row['lx']}/{row['noise']}/{row['protocol_id']}/{row['update']}/{row['family']}/r={row['separation']}",
                residual,
                tolerance,
                residual <= tolerance,
                "posterior linear correlator versus clean prior",
            )
        )
        bias = row.get("witness_inner_bias_envelope")
        checks.append(
            Check(
                "finite-inner-witness-envelope",
                f"L={row['lx']}/{row['noise']}/{row['protocol_id']}/{row['update']}/{row['family']}/r={row['separation']}",
                None if bias is None else float(bias),
                None,
                bias is not None and math.isfinite(float(bias)),
                "every nonlinear witness requires a bootstrapped 1x/2x/4x bias envelope",
            )
        )
        if bias is not None:
            bias_tolerance = max(5.0 * float(row["witness_standard_error"]), 0.05)
            checks.append(
                Check(
                    "finite-inner-witness-bias",
                    f"L={row['lx']}/{row['noise']}/{row['protocol_id']}/{row['update']}/{row['family']}/r={row['separation']}",
                    float(bias),
                    bias_tolerance,
                    float(bias) <= bias_tolerance,
                    "the conservative finite-inner envelope must be small relative to uncertainty",
                )
            )

    return checks


def _diagnostic_checks(
    config: CampaignConfig,
    scalar_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> list[Check]:
    """Gate promotion on explicit overdispersed and finite-inner evidence."""
    index = {
        (
            int(row["lx"]),
            str(row["noise"]),
            float(row["p"]),
            str(row["measurement"]),
            str(row["protocol_id"]),
            str(row["update"]),
            str(row["metric"]),
        ): row
        for row in diagnostic_rows
    }
    points = {
        (
            int(row["lx"]),
            str(row["noise"]),
            float(row["p"]),
            str(row["measurement"]),
            str(row["protocol_id"]),
            str(row["update"]),
        )
        for row in scalar_rows
    }
    checks: list[Check] = []
    for point in sorted(points, key=repr):
        lx, noise, p, _measurement, protocol_id, update = point
        scope = f"L={lx}/{noise}/p={p}/{protocol_id}/{update}"
        required_metrics = ["energy"]
        if noise == "z":
            required_metrics.extend(("boundary_magnetization", "planted_spin_overlap"))
        else:
            required_metrics.append("planted_bond_overlap")
        for metric in required_metrics:
            row = index.get((*point, metric))
            checks.append(
                Check(
                    "analysis-diagnostics-present",
                    f"{scope}/{metric}",
                    0.0 if row is not None else 1.0,
                    0.0,
                    row is not None,
                    "every production point requires its relevant diagnostic summary",
                )
            )
            if row is None:
                continue
            movement_detected = bool(row.get("movement_detected", False))
            rhat_converged = bool(row.get("rhat_converged", False))
            rhat_or_saturation_converged = rhat_converged or bool(
                row.get("rhat_or_saturation_converged", False)
            )
            if metric == "energy":
                diagnostic_converged = rhat_converged
                movement_check = Check(
                    "diagnostic-chain-movement",
                    f"{scope}/{metric}",
                    (
                        None
                        if row.get("minimum_trace_transitions") is None
                        else float(row["minimum_trace_transitions"])
                    ),
                    1.0,
                    movement_detected,
                    "every retained energy trace must change at least once",
                )
            else:
                diagnostic_converged = rhat_or_saturation_converged
                movement_check = Check(
                    "diagnostic-movement-or-saturation",
                    f"{scope}/{metric}",
                    (
                        None
                        if row.get("minimum_trace_transitions") is None
                        else float(row["minimum_trace_transitions"])
                    ),
                    0.0,
                    movement_detected or diagnostic_converged,
                    "a discrete overlap may saturate when overdispersed chains "
                    "have a finite passing split R-hat or every replica has "
                    "the same exact constant; energy movement is checked "
                    "separately",
                )
            checks.extend(
                (
                    Check(
                        "replicated-chain-rhat",
                        f"{scope}/{metric}",
                        (
                            None
                            if row.get("maximum_split_rhat") is None
                            else float(row["maximum_split_rhat"])
                        ),
                        1.01,
                        diagnostic_converged,
                        "overdispersed all-plus/all-minus/random chains require "
                        "a passing split R-hat, except for identical exact "
                        "saturation of a non-energy diagnostic",
                    ),
                    movement_check,
                    Check(
                        "diagnostic-trace-length",
                        f"{scope}/{metric}",
                        (
                            None
                            if row.get("minimum_trace_length") is None
                            else float(row["minimum_trace_length"])
                        ),
                        16.0,
                        int(row.get("minimum_trace_length") or 0) >= 16,
                        "each convergence trace must retain at least sixteen measurements",
                    ),
                    Check(
                        "diagnostic-record-completeness",
                        f"{scope}/{metric}",
                        float(row.get("diagnostic_outer_records", 0)),
                        float(config.mc.diagnostic_outer_records),
                        int(row.get("diagnostic_outer_records", 0))
                        == config.mc.diagnostic_outer_records,
                        "every configured diagnostic outer id must be represented",
                    ),
                    Check(
                        "overdispersed-initializations",
                        f"{scope}/{metric}",
                        0.0
                        if bool(row.get("overdispersed_initializations_complete", False))
                        else 1.0,
                        0.0,
                        bool(row.get("overdispersed_initializations_complete", False)),
                        "replicas must start all-plus, all-minus, then from deterministic random states",
                    ),
                    Check(
                        "finite-inner-budget-consistency",
                        f"{scope}/{metric}",
                        (
                            None
                            if row.get("budget_bias_envelope") is None
                            else float(row["budget_bias_envelope"])
                        ),
                        None,
                        bool(row.get("budget_complete", False))
                        and bool(row.get("budget_consistent", False)),
                        "paired 1x and 2x estimates must agree statistically with the 4x budget",
                    ),
                    Check(
                        "diagnostic-outer-block-evidence",
                        f"{scope}/{metric}",
                        float(row.get("budget_outer_records", 0)),
                        float(2 * config.statistics.block_length),
                        int(row.get("budget_outer_records", 0))
                        >= 2 * config.statistics.block_length,
                        "finite-inner bootstrap must contain at least two configured outer blocks",
                    ),
                )
            )
        for metric in ("spin_absolute_profile", "bond_absolute_profile"):
            row = index.get((*point, metric))
            checks.append(
                Check(
                    "analysis-diagnostics-present",
                    f"{scope}/{metric}",
                    0.0 if row is not None else 1.0,
                    0.0,
                    row is not None,
                    "nonlinear production estimators require finite-inner diagnostics",
                )
            )
            if row is None:
                continue
            bias = row.get("budget_bias_envelope")
            symmetry_zero_profile = noise == "zz" and metric == "spin_absolute_profile"
            if symmetry_zero_profile:
                budget_check = Check(
                    "finite-inner-symmetry-bias-recorded",
                    f"{scope}/{metric}",
                    None if bias is None else float(bias),
                    None,
                    bool(row.get("budget_complete", False))
                    and bias is not None
                    and math.isfinite(float(bias)),
                    "the ZZ spin one-point function is symmetry-forbidden, so the "
                    "positive finite-sample |m_z| bias must be recorded rather than "
                    "required to agree across inner budgets",
                )
            else:
                budget_check = Check(
                    "finite-inner-budget-consistency",
                    f"{scope}/{metric}",
                    None if bias is None else float(bias),
                    None,
                    bool(row.get("budget_complete", False))
                    and bool(row.get("budget_consistent", False)),
                    "nonlinear 1x and 2x estimates must agree statistically with 4x",
                )
            checks.extend(
                (
                    budget_check,
                    Check(
                        "diagnostic-record-completeness",
                        f"{scope}/{metric}",
                        float(row.get("diagnostic_outer_records", 0)),
                        float(config.mc.diagnostic_outer_records),
                        int(row.get("diagnostic_outer_records", 0))
                        == config.mc.diagnostic_outer_records,
                        "every configured diagnostic outer id must be represented",
                    ),
                )
            )
    return checks


def _protocol_difference_checks(
    curve_rows: list[dict[str, Any]],
    ed_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[Check]:
    checks: list[Check] = []
    mc_measurements: dict[tuple[Any, ...], set[str]] = {}
    for row in curve_rows:
        key = (
            row["lx"],
            row["noise"],
            row["p"],
            row["update"],
            row["family"],
            row["separation"],
        )
        mc_measurements.setdefault(key, set()).add(str(row["measurement"]))
    expected_mc = {
        key
        for key, measurements in mc_measurements.items()
        if {"heterodyne", "homodyne"} <= measurements
    }
    available_mc = {
        (
            row["lx"],
            row["noise"],
            row["p"],
            row["update"],
            row["family"],
            row["separation"],
        )
        for row in rows
        if row["source"] == "mc"
    }
    for key in sorted(expected_mc, key=repr):
        checks.append(
            Check(
                "paired-protocol-differences-present",
                f"mc/{key!r}",
                0.0 if key in available_mc else 1.0,
                0.0,
                key in available_mc,
                "every paired MC curve point requires a common-resample difference",
            )
        )

    ed_measurements: dict[tuple[Any, ...], set[str]] = {}
    for row in ed_rows:
        if row["observable"] not in {"spin-pair", "bond-pair"}:
            continue
        key = (
            row["lx"],
            row["noise"],
            row["p"],
            row["prior"],
            str(row["observable"]).removesuffix("-pair"),
            row["separation"],
        )
        ed_measurements.setdefault(key, set()).add(str(row["measurement"]))
    expected_ed = {
        key
        for key, measurements in ed_measurements.items()
        if {"heterodyne", "homodyne"} <= measurements
    }
    available_ed = {
        (
            row["lx"],
            row["noise"],
            row["p"],
            row["prior"],
            row["family"],
            row["separation"],
        )
        for row in rows
        if row["source"] == "ed"
    }
    for key in sorted(expected_ed, key=repr):
        checks.append(
            Check(
                "paired-protocol-differences-present",
                f"ed/{key!r}",
                0.0 if key in available_ed else 1.0,
                0.0,
                key in available_ed,
                "every sampled ED curve point requires a common-random-number difference",
            )
        )
    numerical_tolerance = 64.0 * math.ulp(1.0)
    for row in rows:
        upper = float(row["difference_simultaneous_upper"])
        source = str(row["source"])
        scope = (
            f"{source}/L={row['lx']}/{row['noise']}/p={row['p']}/"
            f"{row.get('prior') or row.get('update')}/{row['family']}/r={row['separation']}"
        )
        checks.append(
            Check(
                f"homodyne-ordering-{source}",
                scope,
                float(row["difference"]),
                max(0.0, -float(row["difference_simultaneous_lower"])),
                upper >= -numerical_tolerance,
                "a resolved negative homodyne-minus-heterodyne simultaneous band is forbidden",
            )
        )
    return checks


def _mc_exact_checks(
    curve_rows: list[dict[str, Any]], ed_rows: list[dict[str, Any]]
) -> list[Check]:
    exact = {
        (
            row["lx"],
            row["noise"],
            row["p"],
            row["protocol_id"],
            row["observable"].replace("-pair", ""),
            row["separation"],
        ): row
        for row in ed_rows
        if row["prior"] == "finite-transfer" and row["observable"] in {"spin-pair", "bond-pair"}
    }
    checks: list[Check] = []
    for row in curve_rows:
        key = (
            row["lx"],
            row["noise"],
            row["p"],
            row["protocol_id"],
            row["family"],
            row["separation"],
        )
        reference = exact.get(key)
        if reference is None:
            continue
        difference = abs(float(row["witness"]) - float(reference["measurement_witness"]))
        tolerance = (
            5.0
            * math.hypot(
                float(row["witness_standard_error"]),
                float(reference["witness_standard_error"]),
            )
            + 0.02
            + float(row.get("witness_inner_bias_envelope") or 0.0)
        )
        checks.append(
            Check(
                "matched-finite-transfer-mc-ed",
                repr(key),
                difference,
                tolerance,
                difference <= tolerance,
                "MC witness versus exact posterior sum for the identical finite-torus prior",
            )
        )
    return checks


def _simultaneous_comparison_checks(rows: list[dict[str, Any]]) -> list[Check]:
    checks: list[Check] = []
    numerical_tolerance = 64.0 * math.ulp(1.0)
    for row in rows:
        scope = (
            f"L={row['lx']}/{row['noise']}/p={row['p']}/{row['protocol_id']}/"
            f"{row['update']}/{row['family']}/r={row['separation']}"
        )
        lower = float(row["difference_simultaneous_lower"])
        upper = float(row["difference_simultaneous_upper"])
        checks.append(
            Check(
                "matched-finite-transfer-simultaneous",
                scope,
                float(row["difference"]),
                max(abs(lower), abs(upper)),
                lower - numerical_tolerance <= 0.0 <= upper + numerical_tolerance,
                "zero must lie in the simultaneous MC-minus-exact curve interval",
            )
        )
        central_fidelity_gap = (
            float(row["difference"])
            + float(row["exact_finite_transfer_witness"])
            - float(row["exact_physical_fidelity"])
        )
        fidelity_tolerance = max(float(row["difference"]) - lower, numerical_tolerance)
        checks.append(
            Check(
                "mc-fidelity-bound-simultaneous",
                scope,
                central_fidelity_gap,
                fidelity_tolerance,
                central_fidelity_gap <= fidelity_tolerance,
                "MC central-value bound violation must be covered by the simultaneous curve band",
            )
        )
    return checks


def validate_campaign(
    config: CampaignConfig, *, raise_on_failure: bool = True
) -> dict[str, object]:
    plan = read_plan(config)
    state = read_state(config)
    validation_task = next(task for task in plan.tasks if task.kind == "validation")
    incomplete = [
        dependency
        for dependency in validation_task.dependencies
        if state["tasks"][dependency]["status"] != "complete"
    ]
    if incomplete:
        raise ValidationError(f"validation dependencies are incomplete: {incomplete[:5]}")
    artifacts = _current_artifacts(config)
    current_digest = source_digest(config.campaign.project_root)
    stale = [
        artifact.artifact_id
        for artifact in artifacts
        if artifact.manifest["kind"] != "validation-report"
        and artifact.manifest["source_digest"] != current_digest
    ]
    if stale:
        raise ValidationError(
            "validation dependencies were generated by another source digest; "
            "rerun compute and analysis"
        )
    # The report is not yet present on a first run; exclude old reports on a resumed audit.
    parents = [
        artifact for artifact in artifacts if artifact.manifest["kind"] != "validation-report"
    ]
    checks = _structural_checks(parents)
    ed_rows = _tables(config, "ed-results")
    if ed_rows:
        checks.extend(_ed_checks(config, ed_rows))
        checks.extend(_purity_checks(config, ed_rows))
    scalar_rows = _tables(config, "analysis-scalars")
    curve_rows = _tables(config, "analysis-curves")
    checks.extend(_mc_checks(scalar_rows, curve_rows))
    diagnostic_rows = _tables(config, "analysis-diagnostics")
    checks.extend(_diagnostic_checks(config, scalar_rows, diagnostic_rows))
    checks.extend(
        _protocol_difference_checks(
            curve_rows,
            ed_rows,
            _tables(config, "analysis-protocol-differences"),
        )
    )
    if ed_rows:
        checks.extend(_mc_exact_checks(curve_rows, ed_rows))
        checks.extend(_simultaneous_comparison_checks(_tables(config, "comparison-curves")))
    local_x_rows = _tables(config, "ed-records")
    for row in local_x_rows:
        if (
            row["measurement"] == "local-x"
            and int(row["lx"]) <= config.ed.local_x_enumeration_limit
        ):
            passed = row["record_mode"] == "enumerated-binary"
            checks.append(
                Check(
                    "local-x-enumeration-mode",
                    f"L={row['lx']}/{row['noise']}/{row['prior']}",
                    0.0 if passed else 1.0,
                    0.0,
                    passed,
                    "N_noise <= 14 must use exact outcome enumeration",
                )
            )
            break
    max_mc = max((int(row["lx"]) for row in scalar_rows), default=0)
    if config.campaign.preset == "scaling":
        checks.append(
            Check(
                "mc-beyond-ed-range",
                config.campaign.name,
                float(max_mc),
                float(config.ed.max_sites),
                max_mc > config.ed.max_sites,
                "MC-only smoke includes a lattice beyond the configured ED range",
            )
        )
    failures = [check for check in checks if not check.passed]
    schema = pa.schema(
        [
            pa.field("check", pa.string(), nullable=False),
            pa.field("scope", pa.string(), nullable=False),
            pa.field("value", pa.float64()),
            pa.field("tolerance", pa.float64()),
            pa.field("passed", pa.bool_(), nullable=False),
            pa.field("detail", pa.string(), nullable=False),
        ]
    )
    report = write_artifact(
        config.campaign.output_root,
        "validation-report",
        pa.Table.from_pylist([check.row() for check in checks], schema=schema),
        metadata={
            "validation_task": validation_task.task_id,
            "required_checks": len(checks),
            "failures": len(failures),
            "promotion_allowed": not failures,
        },
        project_root=config.campaign.project_root,
        parents=tuple(artifact.artifact_id for artifact in parents),
        partition_by=("passed",),
    )
    status = "complete" if not failures else "failed"
    update_task_state(
        config,
        validation_task.task_id,
        status,
        artifacts=(report.artifact_id,),
        error=None if not failures else f"{len(failures)} required checks failed",
    )
    result = {
        "status": status,
        "checks": len(checks),
        "failures": len(failures),
        "report_artifact": report.artifact_id,
        "failed_checks": [f"{check.name}: {check.scope}" for check in failures[:20]],
    }
    if failures and raise_on_failure:
        raise ValidationError(f"{len(failures)} validation checks failed; report={report.path}")
    return result
