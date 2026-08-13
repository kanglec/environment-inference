"""Tables and manifests for multi-size scaling analysis."""

from __future__ import annotations

import json
from pathlib import Path

from .scaling import ScalingAnalysis

FIGURE_DESCRIPTIONS = {
    "scalar_observables_by_size": "Scalar observables versus p with one curve per system size.",
    "local_fidelity_by_size": (
        "Approximate local spin and bond fidelity versus p with one curve per size."
    ),
    "long_distance_vs_p": (
        "All six correlator families at the largest periodic separation versus p."
    ),
    "long_distance_size_scaling": (
        "Log-log size dependence of positive largest-separation correlators for selected p values."
    ),
    "fitted_scaling_dimensions": (
        "Exploratory scaling dimensions from unweighted log-log fits, without uncertainties."
    ),
    "chord_distance": (
        "Multi-size correlators versus periodic chord distance for a selected p value."
    ),
}


def write_scaling_tables(analysis: ScalingAnalysis, out: Path) -> list[str]:
    """Write combined, derived, and fitted machine-readable scaling tables."""
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "scalar_observables_by_size.csv": analysis.scalars,
        "correlators_by_size.csv": analysis.correlators,
        "long_distance_observables.csv": analysis.long_distance,
        "power_law_fits.csv": analysis.fits,
    }
    artifacts: list[str] = []
    for name, frame in outputs.items():
        path = tables / name
        frame.to_csv(path, index=False, float_format="%.17g")
        artifacts.append(str(path.relative_to(out)))
    return artifacts


def _description(path: str) -> str:
    stem = Path(path).stem
    for prefix, description in FIGURE_DESCRIPTIONS.items():
        if stem.startswith(prefix):
            return description
    return "Finite-size-scaling figure."


def write_scaling_manifest(
    analysis: ScalingAnalysis,
    out: Path,
    *,
    table_artifacts: list[str],
    figure_artifacts: list[str],
    allow_incomplete: bool,
) -> list[str]:
    """Write JSON and Markdown scaling summaries."""
    sources = [
        {
            "L": source.L,
            "scalar_table": str(source.scalar_path.relative_to(analysis.root)),
            "correlator_table": str(source.correlator_path.relative_to(analysis.root)),
        }
        for source in analysis.sources
    ]
    successful_fits = int((analysis.fits["fit_status"] == "ok").sum())
    manifest = {
        "scaling_root": str(analysis.root),
        "validation_mode": "allow-incomplete" if allow_incomplete else "strict",
        "square_systems_only": True,
        "sizes": analysis.sizes,
        "noises": analysis.noises,
        "p_values": analysis.p_values,
        "campaign_table_pairs": sources,
        "point_count": int(len(analysis.scalars)),
        "missing_points": [
            {"L": L, "noise": noise, "p_tag": p_tag} for L, noise, p_tag in analysis.missing_points
        ],
        "warnings": analysis.warnings,
        "uncertainty_estimates": False,
        "long_distance_definition": "largest available separation r=floor(L/2)",
        "chord_distance_definition": "d_L(r) = (L/pi) sin(pi r/L)",
        "power_law_fit": {
            "model": "C(r_max; L) = amplitude * L^(-decay_exponent)",
            "scaling_dimension": "Delta = decay_exponent / 2",
            "minimum_positive_sizes": 2,
            "weighting": "unweighted ordinary least squares in log space",
            "successful_fit_count": successful_fits,
        },
        "artifacts": {
            "tables": table_artifacts,
            "figures": figure_artifacts,
            "manifest": ["summary.json", "summary.md"],
        },
        "interpretation_limit": (
            "Fits and apparent crossings are exploratory point-estimate diagnostics. Without "
            "uncertainty estimates, covariance treatment, and corrections to scaling, they do "
            "not establish a transition, critical point, or universal exponent."
        ),
    }
    json_path = out / "summary.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Decohered-CFT finite-size-scaling analysis",
        "",
        f"- Scaling root: `{analysis.root}`",
        f"- Validation mode: `{'allow-incomplete' if allow_incomplete else 'strict'}`",
        f"- Square sizes: {', '.join(f'L={L}' for L in analysis.sizes)}",
        f"- Noise types: {', '.join(analysis.noises)}",
        f"- p values: {', '.join(f'{p:.2f}' for p in analysis.p_values)}",
        f"- Combined size/noise/p points: {len(analysis.scalars)}",
        f"- Successful positive-value power-law fits: {successful_fits}",
        "- Uncertainty estimates/error bars: not computed",
        "",
        "## Input campaign tables",
        "",
        "| L | scalar table | correlator table |",
        "|---:|---|---|",
    ]
    lines.extend(
        f"| {source['L']} | `{source['scalar_table']}` | `{source['correlator_table']}` |"
        for source in sources
    )
    lines.extend(["", "## Missing points", ""])
    if analysis.missing_points:
        lines.extend(f"- `L={L}/{noise}/{p_tag}`" for L, noise, p_tag in analysis.missing_points)
    else:
        lines.append("None detected in the inferred size/noise/p grid.")
    if analysis.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in analysis.warnings)
    lines.extend(
        [
            "",
            "## Derived definitions",
            "",
            r"- Periodic chord distance: $d_L(r)=\frac{L}{\pi}\sin(\pi r/L)$.",
            r"- Long distance: the largest available separation $r=\lfloor L/2\rfloor$.",
            r"- Fit model: $C(r_{\max};L)=A L^{-2\Delta}$, using positive values only.",
            "- Fits use unweighted ordinary least squares in log space and at least two sizes.",
            "",
            "## Tables",
            "",
            *[f"- `{artifact}`" for artifact in table_artifacts],
            "",
            "## Figures",
            "",
        ]
    )
    seen: set[str] = set()
    for artifact in figure_artifacts:
        stem = str(Path(artifact).with_suffix(""))
        if stem in seen:
            continue
        seen.add(stem)
        lines.append(f"- `{stem}.{{png,pdf}}`: {_description(artifact)}")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "These are exploratory finite-size comparisons of point estimates. The fidelity and "
            "EA observables retain the all-to-all replica approximation. Linear correlators remain "
            "annealed sanity checks. Without uncertainty "
            "estimates, covariance treatment, systematic fit-window studies, or corrections to "
            "scaling, an apparent crossing or fitted exponent does not establish a transition, "
            "critical point, or universal exponent.",
            "",
        ]
    )
    markdown_path = out / "summary.md"
    markdown_path.write_text("\n".join(lines))
    return [str(json_path.relative_to(out)), str(markdown_path.relative_to(out))]
