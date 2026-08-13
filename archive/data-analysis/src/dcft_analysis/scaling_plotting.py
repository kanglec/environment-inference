"""Publication-oriented figures for square-system finite-size scaling."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import CORRELATOR_LABELS, NOISE_LABELS, _save_pair, configure_style
from .scaling import CORRELATOR_COLUMNS, ScalingAnalysis

SECTORS = {
    "spin": ["C_sigma_lin", "F_sigma", "C_sigma_EA"],
    "bond": ["C_epsilon_lin", "F_epsilon", "C_epsilon_EA"],
}


def _selected_p(frame: pd.DataFrame, maximum: int = 6) -> list[float]:
    values = sorted(frame["p"].unique().astype(float).tolist())
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum).round().astype(int)
    return [values[index] for index in sorted(set(indices.tolist()))]


def _plot_scalar_by_size(analysis: ScalingAnalysis, figures: Path, out: Path) -> list[str]:
    artifacts: list[str] = []
    specs = [
        ("E_D", r"$E_D(p)$", "Disordered energy density"),
        ("M", r"$M(p)$", "Bulk magnetization density"),
        ("M_partial", r"$M_\partial(p)$", "Boundary magnetization density"),
    ]
    for noise in analysis.noises:
        noise_frame = analysis.scalars.loc[analysis.scalars["noise"] == noise]
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True, sharex=True)
        for axis, (column, label, title) in zip(axes, specs, strict=True):
            for L, group in noise_frame.groupby("L", sort=True):
                axis.plot(group["p"], group[column], marker="o", label=f"L = {L}")
            axis.set(
                title=title,
                xlabel=r"decoherence probability $p$",
                ylabel=label,
            )
        axes[-1].legend()
        fig.suptitle(f"{NOISE_LABELS[noise]}: scalar observables by system size", fontsize=12)
        artifacts.extend(_save_pair(fig, figures / f"scalar_observables_by_size_{noise}", out))
    return artifacts


def _plot_local_fidelity(analysis: ScalingAnalysis, figures: Path, out: Path) -> list[str]:
    artifacts: list[str] = []
    specs = [
        ("F_sigma_loc", r"$F_\sigma^{\mathrm{loc}}(p)$", "Local spin fidelity"),
        ("F_epsilon_loc", r"$F_\epsilon^{\mathrm{loc}}(p)$", "Local bond fidelity"),
    ]
    for noise in analysis.noises:
        noise_frame = analysis.scalars.loc[analysis.scalars["noise"] == noise]
        fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), constrained_layout=True, sharex=True)
        for axis, (column, label, title) in zip(axes, specs, strict=True):
            for L, group in noise_frame.groupby("L", sort=True):
                axis.plot(group["p"], group[column], marker="o", label=f"L = {L}")
            axis.set(
                title=title,
                xlabel=r"decoherence probability $p$",
                ylabel=label,
            )
            axis.set_ylim(bottom=0)
        axes[-1].legend()
        fig.suptitle(
            f"{NOISE_LABELS[noise]}: approximate local fidelity by system size", fontsize=12
        )
        artifacts.extend(_save_pair(fig, figures / f"local_fidelity_by_size_{noise}", out))
    return artifacts


def _plot_long_distance_vs_p(analysis: ScalingAnalysis, figures: Path, out: Path) -> list[str]:
    artifacts: list[str] = []
    for noise in analysis.noises:
        noise_frame = analysis.long_distance.loc[analysis.long_distance["noise"] == noise]
        fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), constrained_layout=True, sharex=True)
        for axis, column in zip(axes.flat, CORRELATOR_COLUMNS, strict=True):
            for L, group in noise_frame.groupby("L", sort=True):
                axis.plot(group["p"], group[column], marker="o", label=f"L = {L}")
            axis.set(
                title=CORRELATOR_LABELS[column],
                xlabel=r"decoherence probability $p$",
                ylabel=rf"{CORRELATOR_LABELS[column]} at $r_{{\max}}=\lfloor L/2\rfloor$",
            )
        axes.flat[-1].legend()
        fig.suptitle(f"{NOISE_LABELS[noise]}: largest-separation correlators by size", fontsize=12)
        artifacts.extend(_save_pair(fig, figures / f"long_distance_vs_p_{noise}", out))
    return artifacts


def _plot_long_distance_size_scaling(
    analysis: ScalingAnalysis, figures: Path, out: Path
) -> list[str]:
    artifacts: list[str] = []
    for noise in analysis.noises:
        noise_frame = analysis.long_distance.loc[analysis.long_distance["noise"] == noise]
        p_values = _selected_p(noise_frame)
        for sector, columns in SECTORS.items():
            fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)
            for axis, column in zip(axes, columns, strict=True):
                for p in p_values:
                    group = noise_frame.loc[
                        (noise_frame["p"] == p) & (noise_frame[column] > 0)
                    ].sort_values("L")
                    if group.empty:
                        continue
                    axis.loglog(
                        group["L"],
                        group[column],
                        marker="o",
                        label=f"p = {p:.2f}",
                    )
                axis.set(
                    title=CORRELATOR_LABELS[column],
                    xlabel=r"system size $L$",
                    ylabel=rf"{CORRELATOR_LABELS[column]} at $r_{{\max}}$",
                )
            axes[-1].legend(ncol=2 if len(p_values) > 4 else 1, fontsize=7)
            fig.suptitle(
                f"{NOISE_LABELS[noise]}: {sector} long-distance size scaling "
                "(positive values only)",
                fontsize=11.5,
            )
            artifacts.extend(
                _save_pair(fig, figures / f"long_distance_size_scaling_{sector}_{noise}", out)
            )
    return artifacts


def _plot_fitted_dimensions(analysis: ScalingAnalysis, figures: Path, out: Path) -> list[str]:
    artifacts: list[str] = []
    valid = analysis.fits.loc[analysis.fits["fit_status"] == "ok"]
    for noise in sorted(valid["noise"].unique()):
        noise_frame = valid.loc[valid["noise"] == noise]
        fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.35), constrained_layout=True, sharex=True)
        for axis, column in zip(axes.flat, CORRELATOR_COLUMNS, strict=True):
            group = noise_frame.loc[noise_frame["observable"] == column].sort_values("p")
            axis.plot(group["p"], group["scaling_dimension"], marker="o")
            axis.set(
                title=CORRELATOR_LABELS[column],
                xlabel=r"decoherence probability $p$",
                ylabel=r"exploratory $\Delta$",
            )
        fig.suptitle(
            rf"{NOISE_LABELS[noise]}: fits of $C(r_{{max}};L) \propto L^{{-2\Delta}}$ "
            "(no uncertainty estimates)",
            fontsize=11.5,
        )
        artifacts.extend(_save_pair(fig, figures / f"fitted_scaling_dimensions_{noise}", out))
    return artifacts


def _plot_chord_distance(analysis: ScalingAnalysis, figures: Path, out: Path) -> list[str]:
    artifacts: list[str] = []
    for noise in analysis.noises:
        noise_frame = analysis.correlators.loc[analysis.correlators["noise"] == noise]
        for p in _selected_p(noise_frame):
            p_frame = noise_frame.loc[noise_frame["p"] == p]
            p_tag = str(p_frame["p_tag"].iloc[0])
            for sector, columns in SECTORS.items():
                fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)
                for axis, column in zip(axes, columns, strict=True):
                    plotted_values: list[float] = []
                    for L, group in p_frame.groupby("L", sort=True):
                        group = group.loc[group["chord_distance"] > 0].sort_values("chord_distance")
                        if group.empty:
                            continue
                        axis.plot(
                            group["chord_distance"],
                            group[column],
                            marker="o",
                            label=f"L = {L}",
                        )
                        plotted_values.extend(group[column].astype(float).tolist())
                    axis.set_xscale("log")
                    if plotted_values and min(plotted_values) > 0:
                        axis.set_yscale("log")
                    axis.set(
                        title=CORRELATOR_LABELS[column],
                        xlabel=r"chord distance $d_L(r)$",
                        ylabel=CORRELATOR_LABELS[column],
                    )
                axes[-1].legend()
                fig.suptitle(
                    f"{NOISE_LABELS[noise]}, p = {p:.2f}: {sector} correlators versus "
                    "periodic chord distance",
                    fontsize=11.5,
                )
                artifacts.extend(
                    _save_pair(
                        fig,
                        figures / f"chord_distance_{sector}_{noise}_{p_tag}",
                        out,
                    )
                )
    return artifacts


def plot_scaling_analysis(analysis: ScalingAnalysis, out: Path) -> list[str]:
    """Generate the complete finite-size-scaling figure set as PNG and PDF."""
    configure_style()
    figures = out / "figures"
    artifacts: list[str] = []
    artifacts.extend(_plot_scalar_by_size(analysis, figures, out))
    artifacts.extend(_plot_local_fidelity(analysis, figures, out))
    artifacts.extend(_plot_long_distance_vs_p(analysis, figures, out))
    artifacts.extend(_plot_long_distance_size_scaling(analysis, figures, out))
    artifacts.extend(_plot_fitted_dimensions(analysis, figures, out))
    artifacts.extend(_plot_chord_distance(analysis, figures, out))
    return artifacts
