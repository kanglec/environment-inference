"""Compare small-system Metropolis data with exact TFIM references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POINT_RE = re.compile(r"L(?P<size>\d+)(?:_[^/]+)?/p(?P<tag>\d{3})/disorder_records\.csv$")


def _mean_se(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, se


def load_monte_carlo(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("L*/p*/disorder_records.csv")):
        match = POINT_RE.search(path.as_posix())
        if match is None:
            raise ValueError(f"unexpected Monte Carlo path: {path}")
        size = int(match.group("size"))
        p = int(match.group("tag")) / 100.0
        frame = pd.read_csv(path)
        independent = frame.drop_duplicates("disorder_id")
        for observable, linear_column, witness_column in (
            ("local_spin", "local_spin_linear", "local_spin_fidelity"),
            ("local_bond", "local_bond_linear", "local_bond_fidelity"),
        ):
            linear, linear_se = _mean_se(independent[linear_column])
            witness, witness_se = _mean_se(independent[witness_column])
            rows.append(
                {
                    "n_sites": size,
                    "p": p,
                    "observable": observable,
                    "r": np.nan,
                    "mc_linear": linear,
                    "mc_linear_se": linear_se,
                    "mc_witness": witness,
                    "mc_witness_se": witness_se,
                    "mc_disorder_samples": independent.shape[0],
                }
            )
        for r, group in frame.groupby("r", sort=True):
            for observable, linear_column, witness_column in (
                ("spin_pair", "spin_linear_corr", "spin_fidelity_corr"),
                ("bond_pair", "bond_linear_corr", "bond_fidelity_corr"),
            ):
                linear, linear_se = _mean_se(group[linear_column])
                witness, witness_se = _mean_se(group[witness_column])
                rows.append(
                    {
                        "n_sites": size,
                        "p": p,
                        "observable": observable,
                        "r": int(r),
                        "mc_linear": linear,
                        "mc_linear_se": linear_se,
                        "mc_witness": witness,
                        "mc_witness_se": witness_se,
                        "mc_disorder_samples": group.shape[0],
                    }
                )
    if not rows:
        raise ValueError(f"no disorder record CSV files found below {root}")
    return pd.DataFrame(rows)


def _reference_for_prior(reference: pd.DataFrame, prior: str, prefix: str) -> pd.DataFrame:
    subset = reference[reference["prior"] == prior].copy()
    subset["r"] = pd.to_numeric(subset["r"], errors="coerce")
    columns = {
        "linear_exact": f"{prefix}_linear_exact",
        "fidelity_exact": f"{prefix}_fidelity_exact",
        "gaussian_witness": f"{prefix}_witness",
        "gaussian_witness_se": f"{prefix}_witness_se",
    }
    return subset[["n_sites", "p", "observable", "r", *columns]].rename(columns=columns)


def build_comparison(reference_path: Path, mc_root: Path) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    mc = load_monte_carlo(mc_root)
    finite = _reference_for_prior(reference, "finite_classical", "finite")
    quantum = _reference_for_prior(reference, "quantum", "quantum")
    keys = ["n_sites", "p", "observable", "r"]
    result = mc.merge(finite, on=keys, how="left", validate="one_to_one")
    result = result.merge(quantum, on=keys, how="left", validate="one_to_one")
    if result.filter(like="_exact").isna().any().any():
        raise ValueError("Monte Carlo rows did not match exact reference rows")
    result["mc_finite_witness_difference"] = result.mc_witness - result.finite_witness
    result["mc_finite_witness_combined_se"] = np.hypot(
        result.mc_witness_se, result.finite_witness_se
    )
    result["mc_finite_witness_z"] = (
        result.mc_finite_witness_difference / result.mc_finite_witness_combined_se
    )
    result["mc_linear_difference"] = result.mc_linear - result.finite_linear_exact
    result["mc_linear_z"] = result.mc_linear_difference / result.mc_linear_se
    result["trotter_witness_difference"] = result.finite_witness - result.quantum_witness
    result["method_gap"] = result.quantum_fidelity_exact - result.quantum_witness
    result["method_fraction"] = result.quantum_witness / result.quantum_fidelity_exact
    result["mc_total_gap"] = result.quantum_fidelity_exact - result.mc_witness
    result["mc_fraction_of_quantum_fidelity"] = (
        result.mc_witness / result.quantum_fidelity_exact
    )
    result["known_inner_absolute_bias"] = (
        (result.p == 0.0) & (result.observable == "local_spin")
    )
    return result.sort_values(keys, na_position="first").reset_index(drop=True)


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def plot_local_fidelity(data: pd.DataFrame, output: Path) -> None:
    local = data[data.observable.isin(["local_spin", "local_bond"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    labels = {"local_spin": r"$A=Z_x$", "local_bond": r"$A=Z_xZ_{x+1}$"}
    colors = {4: "tab:blue", 6: "tab:orange", 8: "tab:green"}
    for axis, observable in zip(axes, labels, strict=True):
        subset = local[local.observable == observable]
        for size, group in subset.groupby("n_sites"):
            group = group.sort_values("p")
            color = colors[int(size)]
            axis.plot(
                group.p,
                group.quantum_fidelity_exact,
                color=color,
                linewidth=1.8,
                label=f"exact F, L={size}",
            )
            axis.plot(group.p, group.quantum_witness, color=color, linestyle="--", linewidth=1.4)
            axis.errorbar(
                group.p,
                group.mc_witness,
                yerr=group.mc_witness_se,
                color=color,
                marker="o",
                linestyle="none",
                markersize=3.5,
                capsize=2,
            )
        axis.set_title(labels[observable])
        axis.set_xlabel("Z-noise probability p")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("fidelity / Gaussian witness")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Exact TFIM fidelity (solid), exact Gaussian witness (dashed), MC (points)")
    _save(fig, output, "local_fidelity_comparison")


def plot_validation(data: pd.DataFrame, output: Path) -> None:
    valid = data[~data.known_inner_absolute_bias].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for size, group in valid.groupby("n_sites"):
        axes[0].scatter(group.finite_witness, group.mc_witness, s=13, alpha=0.65, label=f"L={size}")
        axes[1].scatter(
            group.finite_linear_exact,
            group.mc_linear,
            s=13,
            alpha=0.65,
            label=f"L={size}",
        )
    for axis in axes:
        low, high = axis.get_xlim()
        low = min(low, axis.get_ylim()[0])
        high = max(high, axis.get_ylim()[1])
        axis.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.grid(alpha=0.25)
    axes[0].set_xlabel("exact finite-classical Gaussian witness")
    axes[0].set_ylabel("Metropolis estimate")
    axes[1].set_xlabel("exact finite-classical linear expectation")
    axes[1].set_ylabel("Metropolis estimate")
    axes[0].legend(fontsize=8)
    fig.suptitle("Numerical validation against the matched finite path-integral model")
    _save(fig, output, "monte_carlo_validation")


def plot_pair_profiles(data: pd.DataFrame, output: Path) -> None:
    selected = data[
        (data.n_sites == 8)
        & (data.observable.isin(["spin_pair", "bond_pair"]))
        & (data.p.isin([0.1, 0.3, 0.49]))
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for row, observable in enumerate(("spin_pair", "bond_pair")):
        for col, p in enumerate((0.1, 0.3, 0.49)):
            axis = axes[row, col]
            group = selected[
                (selected.observable == observable) & (selected.p == p)
            ].sort_values("r")
            axis.plot(group.r, group.quantum_fidelity_exact, "-", label="exact F")
            axis.plot(group.r, group.quantum_witness, "--", label="exact witness")
            axis.errorbar(
                group.r,
                group.mc_witness,
                yerr=group.mc_witness_se,
                fmt="o",
                markersize=3.5,
                capsize=2,
                label="Metropolis",
            )
            axis.set_title(f"{observable.replace('_', ' ')}, p={p:g}")
            axis.grid(alpha=0.25)
            if row == 1:
                axis.set_xlabel("separation r")
            if col == 0:
                axis.set_ylabel("fidelity / witness")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("L=8 two-point observables")
    _save(fig, output, "pair_fidelity_profiles_L8")


def write_summary(data: pd.DataFrame, prior_path: Path, output: Path) -> dict[str, object]:
    valid = data[~data.known_inner_absolute_bias]
    nontrivial = valid[
        ~(
            valid.observable.isin(["spin_pair", "bond_pair"])
            & valid.r.eq(0)
        )
    ]
    positive_noise = nontrivial[nontrivial.p > 0.0]
    witness_statistical = valid[
        np.isfinite(valid.mc_finite_witness_z)
        & (valid.mc_finite_witness_combined_se > 1e-14)
    ]
    linear_statistical = valid[
        np.isfinite(valid.mc_linear_z) & (valid.mc_linear_se > 1e-14)
    ]
    prior = pd.read_csv(prior_path)
    summary = {
        "rows": int(data.shape[0]),
        "disorder_samples_per_point": int(data.mc_disorder_samples.min()),
        "mc_validation": {
            "excluded": "local_spin at p=0 (known absolute-value estimator bias)",
            "witness_rmse": float(np.sqrt(np.mean(valid.mc_finite_witness_difference**2))),
            "witness_max_abs_difference": float(np.max(np.abs(valid.mc_finite_witness_difference))),
            "witness_max_abs_z": float(
                np.max(np.abs(witness_statistical.mc_finite_witness_z))
            ),
            "witness_fraction_within_2sigma": float(
                np.mean(np.abs(witness_statistical.mc_finite_witness_z) <= 2.0)
            ),
            "linear_rmse": float(np.sqrt(np.mean(valid.mc_linear_difference**2))),
            "linear_max_abs_difference": float(np.max(np.abs(valid.mc_linear_difference))),
            "linear_max_abs_z": float(np.max(np.abs(linear_statistical.mc_linear_z))),
            "linear_fraction_within_2sigma": float(
                np.mean(np.abs(linear_statistical.mc_linear_z) <= 2.0)
            ),
        },
        "method_quality_positive_p_nontrivial": {
            "minimum_witness_fraction": float(np.min(positive_noise.method_fraction)),
            "median_witness_fraction": float(np.median(positive_noise.method_fraction)),
            "maximum_witness_fraction": float(np.max(positive_noise.method_fraction)),
            "maximum_absolute_fidelity_gap": float(np.max(positive_noise.method_gap)),
        },
        "path_integral_mismatch": {
            "maximum_abs_witness_difference": float(
                np.max(np.abs(data.trotter_witness_difference))
            ),
            "prior_diagnostics": prior.to_dict(orient="records"),
        },
        "p0_local_spin_mc_bias": data[
            (data.p == 0.0) & (data.observable == "local_spin")
        ][["n_sites", "mc_witness", "mc_witness_se", "quantum_witness"]].to_dict(
            orient="records"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prior-diagnostics", type=Path, required=True)
    parser.add_argument("--mc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = build_comparison(args.reference, args.mc_root)
    data.to_csv(args.output / "comparison.csv", index=False, float_format="%.17g")
    plot_local_fidelity(data, args.output)
    plot_validation(data, args.output)
    plot_pair_profiles(data, args.output)
    summary = write_summary(data, args.prior_diagnostics, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
