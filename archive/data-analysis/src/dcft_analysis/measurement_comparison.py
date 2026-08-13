"""Compare heterodyne, homodyne, and local-X TFIM witnesses."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POINT_RE = re.compile(
    r"(?:(?P<measurement>heterodyne|homodyne|local[-_]x)/)?"
    r"L(?P<size>\d+)/p(?P<tag>\d{3})/disorder_records\.csv$"
)


def _mean_se(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, se


def load_monte_carlo(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("**/L*/p*/disorder_records.csv")):
        relative = path.relative_to(root).as_posix()
        match = POINT_RE.search(relative)
        if match is None:
            raise ValueError(f"unexpected Monte Carlo path: {path}")
        measurement = (match.group("measurement") or "heterodyne").replace("-", "_")
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
                    "measurement": measurement,
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
                        "measurement": measurement,
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
    subset = reference[reference.prior == prior].copy()
    subset["r"] = pd.to_numeric(subset.r, errors="coerce")
    if "measurement" not in subset:
        subset["measurement"] = "heterodyne"
    columns = {
        "linear_exact": f"{prefix}_linear_exact",
        "fidelity_exact": f"{prefix}_fidelity_exact",
        "gaussian_witness": f"{prefix}_witness",
        "gaussian_witness_se": f"{prefix}_witness_se",
    }
    keys = ["measurement", "n_sites", "p", "observable", "r"]
    return subset[[*keys, *columns]].rename(columns=columns)


def build_comparison(reference_path: Path, mc_root: Path) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    mc = load_monte_carlo(mc_root)
    keys = ["measurement", "n_sites", "p", "observable", "r"]
    finite = _reference_for_prior(reference, "finite_classical", "finite")
    quantum = _reference_for_prior(reference, "quantum", "quantum")
    result = mc.merge(finite, on=keys, how="left", validate="one_to_one")
    result = result.merge(quantum, on=keys, how="left", validate="one_to_one")
    if result.filter(like="_exact").isna().any().any():
        raise ValueError("Monte Carlo rows did not match exact reference rows")
    result["mc_finite_difference"] = result.mc_witness - result.finite_witness
    result["mc_finite_combined_se"] = np.hypot(result.mc_witness_se, result.finite_witness_se)
    result["mc_finite_z"] = result.mc_finite_difference / result.mc_finite_combined_se
    result["mc_linear_difference"] = result.mc_linear - result.finite_linear_exact
    result["method_gap"] = result.quantum_fidelity_exact - result.quantum_witness
    result["method_fraction"] = result.quantum_witness / result.quantum_fidelity_exact
    result["trotter_difference"] = result.finite_witness - result.quantum_witness
    result["known_inner_absolute_bias"] = (result.p == 0.0) & (result.observable == "local_spin")
    return result.sort_values(keys, na_position="first").reset_index(drop=True)


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def plot_local(data: pd.DataFrame, output: Path) -> None:
    measurements = ["heterodyne", "homodyne", "local_x"]
    observables = ["local_spin", "local_bond"]
    titles = {"heterodyne": "Heterodyne", "homodyne": "Homodyne", "local_x": "Local X"}
    labels = {"local_spin": r"$Z_x$", "local_bond": r"$Z_xZ_{x+1}$"}
    colors = {4: "tab:blue", 6: "tab:orange", 8: "tab:green"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey="row")
    for row, observable in enumerate(observables):
        for col, measurement in enumerate(measurements):
            axis = axes[row, col]
            subset = data[(data.observable == observable) & (data.measurement == measurement)]
            for size, group in subset.groupby("n_sites"):
                group = group.sort_values("p")
                color = colors.get(int(size))
                axis.plot(group.p, group.quantum_fidelity_exact, color=color, linewidth=1.5)
                axis.plot(group.p, group.quantum_witness, color=color, linestyle="--")
                axis.errorbar(
                    group.p,
                    group.mc_witness,
                    yerr=group.mc_witness_se,
                    color=color,
                    marker="o",
                    linestyle="none",
                    markersize=3,
                    capsize=2,
                    label=f"L={size}",
                )
            if row == 0:
                axis.set_title(titles[measurement])
            if row == 1:
                axis.set_xlabel("Z-noise probability p")
            if col == 0:
                axis.set_ylabel(f"{labels[observable]} fidelity / witness")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Exact fidelity (solid), exact measurement witness (dashed), MC (points)")
    _save(fig, output, "local_measurement_comparison")


def plot_validation(data: pd.DataFrame, output: Path) -> None:
    measurements = ["heterodyne", "homodyne", "local_x"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    valid = data[~data.known_inner_absolute_bias]
    for axis, measurement in zip(axes, measurements, strict=True):
        subset = valid[valid.measurement == measurement]
        for size, group in subset.groupby("n_sites"):
            axis.scatter(
                group.finite_witness,
                group.mc_witness,
                s=12,
                alpha=0.65,
                label=f"L={size}",
            )
        low = min(axis.get_xlim()[0], axis.get_ylim()[0])
        high = max(axis.get_xlim()[1], axis.get_ylim()[1])
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_title(measurement.replace("_", " ").title())
        axis.set_xlabel("exact finite-classical witness")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Monte Carlo witness")
    axes[0].legend(fontsize=8)
    fig.suptitle("Monte Carlo validation against exact finite path-integral posteriors")
    _save(fig, output, "measurement_monte_carlo_validation")


def plot_direct_comparison(data: pd.DataFrame, output: Path) -> None:
    local = data[
        (data.n_sites == data.n_sites.max()) & data.observable.isin(["local_spin", "local_bond"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    styles = {"heterodyne": "--", "homodyne": "-", "local_x": ":"}
    for axis, observable in zip(axes, ["local_spin", "local_bond"], strict=True):
        subset = local[local.observable == observable]
        fidelity = subset.drop_duplicates("p").sort_values("p")
        axis.plot(
            fidelity.p,
            fidelity.quantum_fidelity_exact,
            color="black",
            label="exact fidelity",
        )
        for measurement, group in subset.groupby("measurement"):
            group = group.sort_values("p")
            axis.plot(
                group.p,
                group.quantum_witness,
                styles[measurement],
                label=measurement.replace("_", " "),
            )
        axis.set_title(observable.replace("_", " "))
        axis.set_xlabel("Z-noise probability p")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("exact quantum value")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Direct measurement comparison at L={int(data.n_sites.max())}")
    _save(fig, output, "direct_exact_measurement_comparison")


def write_summary(data: pd.DataFrame, output: Path) -> dict[str, object]:
    valid = data[~data.known_inner_absolute_bias]
    nontrivial = valid[
        (valid.p > 0.0) & ~(valid.observable.isin(["spin_pair", "bond_pair"]) & valid.r.eq(0))
    ]
    validation = {}
    quality = {}
    for measurement, group in valid.groupby("measurement"):
        statistical = group[np.isfinite(group.mc_finite_z) & (group.mc_finite_combined_se > 1e-14)]
        validation[measurement] = {
            "rows": int(group.shape[0]),
            "disorder_samples_min": int(group.mc_disorder_samples.min()),
            "rmse": float(np.sqrt(np.mean(group.mc_finite_difference**2))),
            "max_abs_difference": float(np.max(np.abs(group.mc_finite_difference))),
            "fraction_within_2sigma": float(np.mean(np.abs(statistical.mc_finite_z) <= 2.0)),
        }
        method = nontrivial[nontrivial.measurement == measurement]
        quality[measurement] = {
            "minimum_fidelity_fraction": float(method.method_fraction.min()),
            "median_fidelity_fraction": float(method.method_fraction.median()),
            "maximum_gap": float(method.method_gap.max()),
        }

    pivot = nontrivial.pivot_table(
        index=["n_sites", "p", "observable", "r"],
        columns="measurement",
        values="quantum_witness",
    ).dropna()
    hom_minus_het = pivot.homodyne - pivot.heterodyne
    local_minus_hom = pivot.local_x - pivot.homodyne
    summary = {
        "rows": int(data.shape[0]),
        "mc_validation_by_measurement": validation,
        "method_quality_by_measurement": quality,
        "exact_measurement_ordering": {
            "homodyne_minus_heterodyne_minimum": float(hom_minus_het.min()),
            "homodyne_minus_heterodyne_median": float(hom_minus_het.median()),
            "homodyne_below_heterodyne_count": int((hom_minus_het < -0.005).sum()),
            "local_x_minus_homodyne_median": float(local_minus_hom.median()),
            "local_x_beats_homodyne_fraction": float((local_minus_hom > 0.0).mean()),
        },
        "maximum_abs_trotter_witness_difference": {
            measurement: float(np.max(np.abs(group.trotter_difference)))
            for measurement, group in data.groupby("measurement")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = build_comparison(args.reference, args.mc_root)
    data.to_csv(args.output / "comparison.csv", index=False, float_format="%.17g")
    plot_local(data, args.output)
    plot_validation(data, args.output)
    plot_direct_comparison(data, args.output)
    print(json.dumps(write_summary(data, args.output), indent=2))


if __name__ == "__main__":
    main()
