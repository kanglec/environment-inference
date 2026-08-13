"""Write tidy tables and human/machine-readable campaign summaries."""

from __future__ import annotations

import json
from pathlib import Path

from .campaign import Campaign

FIGURE_DESCRIPTIONS = {
    "scalar_observables_vs_p": (
        "Disordered energy, bulk magnetization, and boundary magnetization versus p. "
        "The disordered energy includes the explicit disorder contribution."
    ),
    "local_fidelity_vs_p": (
        "Local spin and bond fidelity averages. These nonlinear observables use the "
        "all-to-all replica approximation described in notes/main.tex."
    ),
    "spin_correlators_vs_r": (
        "Spin linear, approximate fidelity, and Edwards–Anderson correlators versus "
        "boundary separation for every available p."
    ),
    "bond_correlators_vs_r": (
        "Bond linear, approximate fidelity, and Edwards–Anderson correlators versus "
        "boundary separation for every available p."
    ),
    "correlators_vs_p_fixed_r": (
        "All six correlator families versus p at a short and the largest available "
        "boundary separation."
    ),
    "linear_p_independence_check": (
        "Annealed linear correlators versus p. They are expected to be p-independent "
        "up to thermalization and sampling effects."
    ),
}


def write_tables(campaign: Campaign, out: Path) -> list[str]:
    """Write one scalar row per point and one correlator row per point/separation."""
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    scalar_path = tables / "scalar_observables.csv"
    correlator_path = tables / "correlators.csv"
    campaign.scalars.to_csv(scalar_path, index=False, float_format="%.17g")
    campaign.correlators.to_csv(correlator_path, index=False, float_format="%.17g")
    return [str(scalar_path.relative_to(out)), str(correlator_path.relative_to(out))]


def _figure_description(path: str) -> str:
    stem = Path(path).stem
    for prefix, description in FIGURE_DESCRIPTIONS.items():
        if stem.startswith(prefix):
            return description
    return "Campaign analysis figure."


def write_manifest(
    campaign: Campaign,
    out: Path,
    *,
    table_artifacts: list[str],
    figure_artifacts: list[str],
    allow_incomplete: bool,
) -> list[str]:
    """Write JSON and Markdown summaries of validation and generated artifacts."""
    point_records = [
        {
            "noise": point.noise,
            "p": point.p,
            "p_tag": point.p_tag,
            "source_csv": str(point.path.relative_to(campaign.root)),
        }
        for point in campaign.points
    ]
    manifest = {
        "campaign_root": str(campaign.root),
        "validation_mode": "allow-incomplete" if allow_incomplete else "strict",
        "metadata": campaign.metadata,
        "discovered_point_count": len(campaign.points),
        "discovered_points": point_records,
        "missing_points": [
            {"noise": noise, "p_tag": p_tag} for noise, p_tag in campaign.missing_points
        ],
        "warnings": campaign.warnings,
        "uncertainty_estimates": False,
        "artifacts": {
            "tables": table_artifacts,
            "figures": figure_artifacts,
            "manifest": ["summary.json", "summary.md"],
        },
        "interpretation": {
            "linear_correlators": (
                "Annealed linear checks expected to be p-independent up to simulation error."
            ),
            "fidelity_observables": (
                "Approximate nonlinear fidelity observables from the all-to-all replica "
                "approximation."
            ),
            "ea_correlators": "Annealed Edwards–Anderson-type two-marked-replica correlators.",
            "finite_size_caution": (
                "These finite-size results do not by themselves establish a transition or "
                "critical behavior."
            ),
        },
    }
    json_path = out / "summary.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    lx = campaign.metadata["lx"]
    lt = campaign.metadata["lt"]
    lines = [
        "# Decohered-CFT campaign analysis",
        "",
        f"- Campaign root: `{campaign.root}`",
        f"- Validation mode: `{'allow-incomplete' if allow_incomplete else 'strict'}`",
        f"- Lattice: $L_x={lx}$, $L_\\tau={lt}$",
        f"- Disorder samples per point: {campaign.metadata['samples_per_point']}",
        f"- Valid discovered points: {len(campaign.points)}",
        f"- Missing expected points: {len(campaign.missing_points)}",
        "- Uncertainty estimates/error bars: not computed",
        "",
        "## Discovered points",
        "",
        "| noise | p | tag | source |",
        "|---|---:|---|---|",
    ]
    lines.extend(
        f"| {point['noise']} | {point['p']:.2f} | {point['p_tag']} | `{point['source_csv']}` |"
        for point in point_records
    )
    lines.extend(["", "## Missing points", ""])
    if campaign.missing_points:
        lines.extend(f"- `{noise}/{p_tag}`" for noise, p_tag in campaign.missing_points)
    else:
        lines.append("None detected for the expected campaign grid.")
    if campaign.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in campaign.warnings)
    lines.extend(
        [
            "",
            "## Tables",
            "",
            *[f"- `{artifact}`" for artifact in table_artifacts],
            "",
            "## Figures",
            "",
        ]
    )
    seen_stems: set[str] = set()
    for artifact in figure_artifacts:
        stem = str(Path(artifact).with_suffix(""))
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        lines.append(f"- `{stem}.{{png,pdf}}`: {_figure_description(artifact)}")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The fidelity plots are approximate nonlinear observables inherited from the "
            "all-to-all replica approximation. The EA plots are annealed Edwards–Anderson-type "
            "correlators. Linear correlators are sanity checks rather than mixed-state transition "
            "diagnostics. No bootstrap, jackknife, "
            "confidence intervals, uncertainty estimates, or error bars are computed. A "
            "finite-size "
            "pilot—especially $L=8$—cannot alone establish a transition or critical behavior.",
            "",
        ]
    )
    markdown_path = out / "summary.md"
    markdown_path.write_text("\n".join(lines))
    return [str(json_path.relative_to(out)), str(markdown_path.relative_to(out))]
