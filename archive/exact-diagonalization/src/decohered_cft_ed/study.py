"""Command-line driver for the small-system TFIM comparison study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .numerics import (
    density_square_root,
    fidelity_average_from_sqrt,
    linear_expectation,
    measurement_witness,
    observable_families,
    tfim_ground_state,
    transfer_boundary_prior,
    z_noise_density,
)


def _csv_values(text: str, converter):
    return [converter(value.strip()) for value in text.split(",") if value.strip()]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_study(
    *,
    sizes: list[int],
    probabilities: list[float],
    delta_tau: float,
    l_tau_multiplier: int,
    witness_samples: int,
    seed: int,
    output: Path,
    measurements: tuple[str, ...] = ("heterodyne",),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []

    for n_sites in sizes:
        energy, state = tfim_ground_state(n_sites)
        quantum_prior = state * state
        l_tau = l_tau_multiplier * n_sites
        classical_prior, transfer_ground_prior = transfer_boundary_prior(
            n_sites, l_tau=l_tau, delta_tau=delta_tau
        )
        prior_rows.append(
            {
                "n_sites": n_sites,
                "l_tau": l_tau,
                "delta_tau": delta_tau,
                "ground_energy": energy,
                "tv_finite_classical_vs_quantum": 0.5
                * float(np.sum(np.abs(classical_prior - quantum_prior))),
                "tv_transfer_ground_vs_quantum": 0.5
                * float(np.sum(np.abs(transfer_ground_prior - quantum_prior))),
                "tv_finite_vs_transfer_ground": 0.5
                * float(np.sum(np.abs(classical_prior - transfer_ground_prior))),
            }
        )
        families = observable_families(n_sites)
        priors = {"quantum": quantum_prior, "finite_classical": classical_prior}

        for p_index, p in enumerate(probabilities):
            for prior_index, (prior_name, prior) in enumerate(priors.items()):
                amplitude = np.sqrt(prior)
                rho = z_noise_density(amplitude, n_sites, p)
                sqrt_rho = density_square_root(rho)
                exact_by_family = {}
                for family in families:
                    exact_by_family[(family.name, family.separation)] = float(
                        np.mean(
                            [
                                fidelity_average_from_sqrt(rho, sqrt_rho, values)
                                for values in family.eigenvalues
                            ]
                        )
                    )
                for measurement_index, measurement in enumerate(measurements):
                    witness = measurement_witness(
                        prior,
                        families,
                        n_sites=n_sites,
                        p=p,
                        measurement=measurement,
                        samples=witness_samples,
                        seed=(
                            seed
                            + 100_000 * n_sites
                            + 1_000 * p_index
                            + 10 * measurement_index
                            + prior_index
                        ),
                    )
                    for family in families:
                        fidelity = exact_by_family[(family.name, family.separation)]
                        linear = linear_expectation(prior, family)
                        estimate = witness[(family.name, family.separation)]
                        result_rows.append(
                            {
                                "n_sites": n_sites,
                                "l_tau": l_tau,
                                "delta_tau": delta_tau,
                                "p": p,
                                "prior": prior_name,
                                "measurement": measurement,
                                "observable": family.name,
                                "r": "" if family.separation is None else family.separation,
                                "linear_exact": linear,
                                "linear_witness_sample": estimate.linear_sample_mean,
                                "linear_witness_se": estimate.linear_standard_error,
                                "fidelity_exact": fidelity,
                                "gaussian_witness": estimate.value,
                                "gaussian_witness_se": estimate.standard_error,
                                "fidelity_gap": fidelity - estimate.value,
                                "witness_fraction": estimate.value / fidelity
                                if fidelity > 0.0
                                else "",
                            }
                        )

    _write_csv(
        output / "reference.csv",
        result_rows,
        [
            "n_sites",
            "l_tau",
            "delta_tau",
            "p",
            "prior",
            "measurement",
            "observable",
            "r",
            "linear_exact",
            "linear_witness_sample",
            "linear_witness_se",
            "fidelity_exact",
            "gaussian_witness",
            "gaussian_witness_se",
            "fidelity_gap",
            "witness_fraction",
        ],
    )
    _write_csv(
        output / "prior_diagnostics.csv",
        prior_rows,
        [
            "n_sites",
            "l_tau",
            "delta_tau",
            "ground_energy",
            "tv_finite_classical_vs_quantum",
            "tv_transfer_ground_vs_quantum",
            "tv_finite_vs_transfer_ground",
        ],
    )
    metadata = {
        "model": "periodic critical TFIM, J=h=1",
        "noise": "iid Z",
        "sizes": sizes,
        "p_values": probabilities,
        "delta_tau": delta_tau,
        "l_tau_multiplier": l_tau_multiplier,
        "witness_samples": witness_samples,
        "measurements": measurements,
        "seed": seed,
        "interpretation": (
            "Gaussian witnesses use exact posterior sums; only their outer record integral is "
            "sampled. Local-X records are fully enumerated. finite_classical uses the exact "
            "finite-torus transfer-matrix "
            "boundary prior matching the Monte Carlo lattice."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="4,6,8")
    parser.add_argument("--p-values", default="0,0.1,0.2,0.3,0.4,0.49")
    parser.add_argument("--delta-tau", type=float, default=0.2)
    parser.add_argument("--l-tau-multiplier", type=int, default=16)
    parser.add_argument("--witness-samples", type=int, default=100_000)
    parser.add_argument(
        "--measurements",
        default="heterodyne,homodyne,local_x",
        help="comma-separated: heterodyne, homodyne, local_x",
    )
    parser.add_argument("--seed", type=int, default=2254)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = _csv_values(args.sizes, int)
    probabilities = _csv_values(args.p_values, float)
    measurements = _csv_values(args.measurements, str)
    if any(size < 2 for size in sizes):
        parser.error("all sizes must be at least 2")
    if any(not 0.0 <= p < 0.5 for p in probabilities):
        parser.error("all p values must satisfy 0 <= p < 1/2")
    if not measurements or any(
        measurement not in {"heterodyne", "homodyne", "local_x"} for measurement in measurements
    ):
        parser.error("measurements must be heterodyne, homodyne, and/or local_x")
    run_study(
        sizes=sizes,
        probabilities=probabilities,
        delta_tau=args.delta_tau,
        l_tau_multiplier=args.l_tau_multiplier,
        witness_samples=args.witness_samples,
        seed=args.seed,
        output=args.output,
        measurements=measurements,
    )


if __name__ == "__main__":
    main()
