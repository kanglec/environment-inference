"""Machine-local update-method speed and mixing benchmarks."""

from __future__ import annotations

import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, _core
from .artifacts import source_digest
from .config import CampaignConfig
from .statistics import AutocorrelationEstimate, geyer_autocorrelation, split_rhat


class BenchmarkError(RuntimeError):
    """An update benchmark request is invalid."""


def _autocorrelation_payload(estimate: AutocorrelationEstimate) -> dict[str, object]:
    return {
        "tau_saved": estimate.tau_saved,
        "tau_sweeps": estimate.tau_sweeps,
        "effective_samples": estimate.effective_samples,
        "window_lag": estimate.window_lag,
        "rule": estimate.rule,
    }


def _rhat(values: Any) -> float | None:
    result = split_rhat(np.asarray(values, dtype=np.float64))
    return result if math.isfinite(result) else None


def _acceptance(accepted: int, proposed: int) -> float | None:
    return accepted / proposed if proposed else None


def benchmark_updates(
    config: CampaignConfig,
    *,
    lx: int | None = None,
    noise: str | None = None,
    p: float | None = None,
    measurement: str | None = None,
    gamma: float | None = None,
    updates: tuple[str, ...] | None = None,
    warmup_sweeps: int = 64,
    speed_sweeps: int = 128,
    probes: int = 256,
    probe_interval: int = 1,
    thermalization_sweeps: int = 64,
    thermalization_measurements: int = 64,
    chains: int = 4,
    workers: int | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Compare update cost, autocorrelation, and multi-chain thermalization."""
    selected_lx = config.lattice.sizes[0] if lx is None else lx
    if selected_lx not in config.lattice.sizes:
        raise BenchmarkError("benchmark lx must be listed in lattice.sizes")
    selected_noise = config.protocols.noises[0] if noise is None else noise
    selected_p = config.protocols.p_values[0] if p is None else p
    selected_measurement = config.protocols.measurements[0] if measurement is None else measurement
    selected_updates = config.mc.updates if updates is None else updates
    selected_workers = config.execution.local_workers if workers is None else workers
    if selected_noise not in _core.noise_registry():
        raise BenchmarkError(f"unknown noise {selected_noise!r}")
    if selected_measurement not in _core.measurement_registry():
        raise BenchmarkError(f"unknown measurement {selected_measurement!r}")
    if selected_measurement == "gaussian" and gamma is None:
        raise BenchmarkError("gaussian benchmark requires --gamma")
    if any(update not in _core.update_registry() for update in selected_updates):
        raise BenchmarkError("benchmark updates contain an unknown method")
    if len(set(selected_updates)) != len(selected_updates):
        raise BenchmarkError("benchmark updates cannot contain duplicates")
    if selected_workers < 1:
        raise BenchmarkError("benchmark workers must be positive")

    lt = config.lattice.lt(selected_lx)
    kx, kt = _core.lattice_couplings(
        config.lattice.regularization,
        config.lattice.delta_tau,
    )
    planted = _core.clean_configurations(
        selected_lx,
        lt,
        kx,
        kt,
        config.campaign.seed,
        config.mc.clean_thermalization_sweeps,
        config.mc.clean_saving_interval,
        1,
    )[0]
    boundary = _core.boundary_from_packed(selected_lx, lt, planted)
    record = _core.generate_record(
        boundary,
        selected_noise,
        selected_measurement,
        selected_p,
        config.campaign.seed,
        0,
        gamma,
    )

    methods: list[dict[str, Any]] = []
    for update in selected_updates:
        raw = _core.benchmark_update_method(
            selected_lx,
            lt,
            kx,
            kt,
            selected_noise,
            list(record["record_couplings"]),
            planted,
            update,
            config.campaign.seed,
            0,
            warmup_sweeps,
            speed_sweeps,
            probes,
            probe_interval,
            thermalization_sweeps,
            thermalization_measurements,
            chains,
            config.mc.tnmc_bond_dimension,
            selected_workers,
        )
        elapsed = float(raw["speed_elapsed_seconds"])
        sweep_rate = speed_sweeps / elapsed
        energy_autocorrelation = geyer_autocorrelation(
            raw["energy_trace"], saving_interval_sweeps=probe_interval
        )
        overlap_autocorrelation = geyer_autocorrelation(
            raw["planted_overlap_trace"], saving_interval_sweeps=probe_interval
        )
        serial_elapsed = float(raw["thermalization_serial_elapsed_seconds"])
        parallel_elapsed = float(raw["thermalization_elapsed_seconds"])
        methods.append(
            {
                "update": update,
                "sweep_speed": {
                    "sweeps": speed_sweeps,
                    "elapsed_seconds": elapsed,
                    "sweeps_per_second": sweep_rate,
                },
                "autocorrelation": {
                    "energy": _autocorrelation_payload(energy_autocorrelation),
                    "planted_overlap": _autocorrelation_payload(overlap_autocorrelation),
                    "effective_overlap_samples_per_second": sweep_rate
                    / (2.0 * overlap_autocorrelation.tau_sweeps),
                    "probes": probes,
                    "probe_interval_sweeps": probe_interval,
                },
                "thermalization": {
                    "sweeps": thermalization_sweeps,
                    "chains": chains,
                    "measurements_per_chain": thermalization_measurements,
                    "parallel_elapsed_seconds": parallel_elapsed,
                    "energy_split_rhat": _rhat(raw["thermalization_energy"]),
                    "boundary_magnetization_split_rhat": _rhat(
                        raw["thermalization_boundary_magnetization"]
                    ),
                    "planted_overlap_split_rhat": _rhat(raw["thermalization_planted_overlap"]),
                    "initialization": "overdispersed: all-plus, all-minus, then random",
                },
                "parallelism": {
                    "independent_chains": chains,
                    "rayon_workers": selected_workers,
                    "serial_elapsed_seconds": serial_elapsed,
                    "parallel_elapsed_seconds": parallel_elapsed,
                    "speedup": serial_elapsed / parallel_elapsed,
                },
                "acceptance": {
                    "local": _acceptance(int(raw["local_accepted"]), int(raw["local_proposed"])),
                    "corrected_wolff": _acceptance(
                        int(raw["cluster_accepted"]), int(raw["cluster_proposed"])
                    ),
                    "global": _acceptance(
                        int(raw["global_accepted"]), int(raw["global_attempted"])
                    ),
                    "tnmc": _acceptance(int(raw["tnmc_accepted"]), int(raw["tnmc_proposed"])),
                    "tnmc_conditionals_regularized": int(raw["tnmc_conditionals_regularized"]),
                },
            }
        )
    recommendation = max(
        methods,
        key=lambda method: float(method["autocorrelation"]["effective_overlap_samples_per_second"]),
    )["update"]
    report: dict[str, Any] = {
        "schema": "DCFT_UPDATE_BENCHMARK_V1",
        "software": {
            "package_version": __version__,
            "rust_core_version": _core.version(),
            "source_digest": source_digest(config.campaign.project_root),
            "rng_contract": _core.rng_contract(),
        },
        "machine": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "rayon_workers": selected_workers,
        },
        "parameters": {
            "lx": selected_lx,
            "lt": lt,
            "kx": kx,
            "kt": kt,
            "noise": selected_noise,
            "p": selected_p,
            "measurement": selected_measurement,
            "gamma": gamma,
            "seed": config.campaign.seed,
            "warmup_sweeps": warmup_sweeps,
            "tnmc_bond_dimension": config.mc.tnmc_bond_dimension,
        },
        "methods": methods,
        "recommendation": {
            "update": recommendation,
            "criterion": "largest planted-overlap effective samples per wall second",
            "warning": "also require acceptable split R-hat and numerical diagnostics",
        },
    }
    if output is not None:
        destination = output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["output"] = str(destination)
    return report
