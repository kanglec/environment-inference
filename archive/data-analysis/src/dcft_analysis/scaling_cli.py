"""Command-line entry point for multi-size scaling analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .campaign import CampaignValidationError
from .scaling import load_scaling_analysis
from .scaling_plotting import plot_scaling_analysis
from .scaling_report import write_scaling_manifest, write_scaling_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dcft-analyze-scaling",
        description=(
            "Combine campaign-analysis CSV tables from multiple square system sizes and "
            "produce exploratory finite-size-scaling tables, fits, figures, and a manifest."
        ),
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="root beneath which per-size campaign-analysis tables can be found",
    )
    parser.add_argument("--out", required=True, type=Path, help="scaling-analysis output directory")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow missing size/noise/p combinations; malformed or duplicate data remain fatal",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    analysis = load_scaling_analysis(args.root, allow_incomplete=args.allow_incomplete)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    table_artifacts = write_scaling_tables(analysis, out)
    figure_artifacts = plot_scaling_analysis(analysis, out)
    write_scaling_manifest(
        analysis,
        out,
        table_artifacts=table_artifacts,
        figure_artifacts=figure_artifacts,
        allow_incomplete=args.allow_incomplete,
    )
    print(
        f"Analyzed {len(analysis.scalars)} point(s) across square sizes "
        f"{', '.join(str(L) for L in analysis.sizes)}."
    )
    print(f"Wrote {len(table_artifacts)} tables and {len(figure_artifacts)} figure files to {out}")
    for warning in analysis.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except CampaignValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
