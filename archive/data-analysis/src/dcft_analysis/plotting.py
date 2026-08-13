"""Deterministic publication-oriented campaign figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DPI = 220
NOISE_LABELS = {"z": r"$Z$ noise", "zz": r"$ZZ$ noise"}

CORRELATOR_LABELS = {
    "C_sigma_lin": r"$C_\sigma^{\mathrm{lin}}(p,r)$",
    "C_epsilon_lin": r"$C_\epsilon^{\mathrm{lin}}(p,r)$",
    "F_sigma": r"$F_\sigma(p,r)$ (approx.)",
    "F_epsilon": r"$F_\epsilon(p,r)$ (approx.)",
    "C_sigma_EA": r"$C_\sigma^{\mathrm{EA}}(p,r)$",
    "C_epsilon_EA": r"$C_\epsilon^{\mathrm{EA}}(p,r)$",
}


def configure_style() -> None:
    """Set a stable, readable Matplotlib style without requiring LaTeX."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_pair(fig: plt.Figure, stem: Path, output_root: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    for extension in ("png", "pdf"):
        path = stem.with_suffix(f".{extension}")
        metadata = {"Creator": "dcft-campaign-analysis"}
        if extension == "png":
            metadata["Software"] = "dcft-campaign-analysis"
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
        artifacts.append(str(path.relative_to(output_root)))
    plt.close(fig)
    return artifacts


def _noise_groups(frame: pd.DataFrame):
    for noise in sorted(frame["noise"].unique()):
        yield noise, frame.loc[frame["noise"] == noise].sort_values("p")


def _plot_scalar_observables(scalars: pd.DataFrame, figures: Path, out: Path) -> list[str]:
    specs = [
        ("E_D", r"$E_D(p)$", "Disordered energy density"),
        ("M", r"$M(p)$", "Bulk magnetization density"),
        ("M_partial", r"$M_\partial(p)$", "Boundary magnetization density"),
    ]
    noise_groups = list(_noise_groups(scalars))
    fig, axes = plt.subplots(
        len(noise_groups),
        3,
        figsize=(11.2, 3.05 * len(noise_groups)),
        constrained_layout=True,
        squeeze=False,
        sharex="col",
    )
    for row, (noise, group) in enumerate(noise_groups):
        for axis, (column, label, title) in zip(axes[row], specs, strict=True):
            axis.plot(group["p"], group[column], marker="o")
            axis.set(
                title=f"{NOISE_LABELS[noise]}: {title}",
                xlabel=r"decoherence probability $p$",
                ylabel=label,
            )
    fig.suptitle("Annealed scalar observables (finite-size campaign)", fontsize=12)
    return _save_pair(fig, figures / "scalar_observables_vs_p", out)


def _plot_local_fidelity(scalars: pd.DataFrame, figures: Path, out: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.35), constrained_layout=True, sharex=True)
    specs = [
        ("F_sigma_loc", r"$F_\sigma^{\mathrm{loc}}(p)$", "Local spin fidelity"),
        ("F_epsilon_loc", r"$F_\epsilon^{\mathrm{loc}}(p)$", "Local bond fidelity"),
    ]
    for axis, (column, label, title) in zip(axes, specs, strict=True):
        for noise, group in _noise_groups(scalars):
            axis.plot(group["p"], group[column], marker="o", label=NOISE_LABELS[noise])
        axis.set(title=title, xlabel=r"decoherence probability $p$", ylabel=label)
        axis.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("Fidelity observables from the all-to-all replica approximation", fontsize=12)
    return _save_pair(fig, figures / "local_fidelity_vs_p", out)


def _p_colors(p_values: list[float]) -> dict[float, tuple[float, float, float, float]]:
    if len(p_values) == 1:
        return {p_values[0]: plt.colormaps["viridis"](0.55)}
    positions = np.linspace(0.05, 0.95, len(p_values))
    return {
        p: plt.colormaps["viridis"](position)
        for p, position in zip(p_values, positions, strict=True)
    }


def _plot_correlators_by_r(correlators: pd.DataFrame, figures: Path, out: Path) -> list[str]:
    artifacts: list[str] = []
    families = {
        "spin": ["C_sigma_lin", "F_sigma", "C_sigma_EA"],
        "bond": ["C_epsilon_lin", "F_epsilon", "C_epsilon_EA"],
    }
    for noise, noise_frame in _noise_groups(correlators):
        p_values = sorted(noise_frame["p"].unique().tolist())
        colors = _p_colors(p_values)
        for sector, columns in families.items():
            fig, axes = plt.subplots(
                1, 3, figsize=(11.2, 3.45), constrained_layout=True, sharex=True
            )
            for axis, column in zip(axes, columns, strict=True):
                for p in p_values:
                    group = noise_frame.loc[noise_frame["p"] == p].sort_values("r")
                    axis.plot(
                        group["r"],
                        group[column],
                        marker="o",
                        color=colors[p],
                        label=f"p = {p:.2f}",
                    )
                axis.set(
                    title=CORRELATOR_LABELS[column],
                    xlabel=r"boundary separation $r$",
                    ylabel=CORRELATOR_LABELS[column],
                )
                axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            axes[-1].legend(ncol=2 if len(p_values) > 8 else 1, fontsize=7)
            qualifier = "linear check, approximate fidelity, and Edwards–Anderson overlap"
            fig.suptitle(
                f"{NOISE_LABELS[noise]}: {sector} correlators — {qualifier}", fontsize=11.5
            )
            artifacts.extend(_save_pair(fig, figures / f"{sector}_correlators_vs_r_{noise}", out))
    return artifacts


def _selected_separations(frame: pd.DataFrame) -> list[int]:
    r_max = int(frame["r"].max())
    short = 1 if r_max >= 1 else 0
    return sorted({short, r_max})


def _plot_fixed_r(correlators: pd.DataFrame, figures: Path, out: Path) -> list[str]:
    artifacts: list[str] = []
    columns = list(CORRELATOR_LABELS)
    for noise, noise_frame in _noise_groups(correlators):
        separations = _selected_separations(noise_frame)
        fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.35), constrained_layout=True, sharex=True)
        for axis, column in zip(axes.flat, columns, strict=True):
            for r in separations:
                group = noise_frame.loc[noise_frame["r"] == r].sort_values("p")
                axis.plot(group["p"], group[column], marker="o", label=f"r = {r}")
            axis.set(
                title=CORRELATOR_LABELS[column],
                xlabel=r"decoherence probability $p$",
                ylabel=CORRELATOR_LABELS[column],
            )
        axes.flat[0].legend()
        fig.suptitle(
            f"{NOISE_LABELS[noise]}: fixed-separation p dependence "
            f"(including largest available r = {max(separations)})",
            fontsize=11.5,
        )
        artifacts.extend(_save_pair(fig, figures / f"correlators_vs_p_fixed_r_{noise}", out))
    return artifacts


def _plot_linear_consistency(correlators: pd.DataFrame, figures: Path, out: Path) -> list[str]:
    artifacts: list[str] = []
    specs = [
        ("C_sigma_lin", r"$C_\sigma^{\mathrm{lin}}(p,r)$"),
        ("C_epsilon_lin", r"$C_\epsilon^{\mathrm{lin}}(p,r)$"),
    ]
    for noise, noise_frame in _noise_groups(correlators):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.55), constrained_layout=True, sharex=True)
        r_values = sorted(noise_frame["r"].unique().tolist())
        colors = _p_colors([float(r) for r in r_values])
        for axis, (column, label) in zip(axes, specs, strict=True):
            for r in r_values:
                group = noise_frame.loc[noise_frame["r"] == r].sort_values("p")
                axis.plot(
                    group["p"],
                    group[column],
                    marker="o",
                    color=colors[float(r)],
                    label=f"r = {r}",
                )
            axis.set(
                title=label,
                xlabel=r"decoherence probability $p$",
                ylabel=label,
            )
        axes[-1].legend(ncol=2 if len(r_values) > 5 else 1, fontsize=7)
        fig.suptitle(
            f"{NOISE_LABELS[noise]}: annealed linear sanity check (expected p-independent)",
            fontsize=11.5,
        )
        artifacts.extend(_save_pair(fig, figures / f"linear_p_independence_check_{noise}", out))
    return artifacts


def plot_campaign(scalars: pd.DataFrame, correlators: pd.DataFrame, out: Path) -> list[str]:
    """Generate every campaign figure in both PNG and PDF formats."""
    configure_style()
    figures = out / "figures"
    artifacts: list[str] = []
    artifacts.extend(_plot_scalar_observables(scalars, figures, out))
    artifacts.extend(_plot_local_fidelity(scalars, figures, out))
    artifacts.extend(_plot_correlators_by_r(correlators, figures, out))
    artifacts.extend(_plot_fixed_r(correlators, figures, out))
    artifacts.extend(_plot_linear_consistency(correlators, figures, out))
    return artifacts
