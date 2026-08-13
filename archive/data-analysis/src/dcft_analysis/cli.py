"""Command-line entry point for campaign analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .campaign import CampaignValidationError, load_campaign, parse_expected_p
from .plotting import plot_campaign
from .report import write_manifest, write_tables


def _p_argument(value: str) -> int:
    try:
        return parse_expected_p(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dcft-analyze-campaign",
        description=(
            "Validate and combine decohered-CFT Monte Carlo campaign CSVs, then write "
            "publication-oriented figures and a campaign manifest."
        ),
    )
    parser.add_argument("--root", required=True, type=Path, help="lattice root, e.g. /data/L8")
    parser.add_argument("--out", required=True, type=Path, help="analysis output directory")
    parser.add_argument(
        "--expected-noise",
        action="append",
        choices=("z", "zz"),
        dest="expected_noises",
        help="expected noise type; repeat as needed (default: infer discovered types)",
    )
    parser.add_argument(
        "--expected-p",
        action="append",
        type=_p_argument,
        dest="expected_p",
        metavar="P",
        help="expected p in hundredth increments; repeat as needed (default: infer union)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="continue past missing grid points; all malformed/inconsistent data remain fatal",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    campaign = load_campaign(
        args.root,
        expected_noises=args.expected_noises,
        expected_p=args.expected_p,
        allow_incomplete=args.allow_incomplete,
    )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    table_artifacts = write_tables(campaign, out)
    figure_artifacts = plot_campaign(campaign.scalars, campaign.correlators, out)
    write_manifest(
        campaign,
        out,
        table_artifacts=table_artifacts,
        figure_artifacts=figure_artifacts,
        allow_incomplete=args.allow_incomplete,
    )
    print(
        f"Analyzed {len(campaign.points)} point(s) on "
        f"Lx={campaign.metadata['lx']}, Lt={campaign.metadata['lt']}."
    )
    print(f"Wrote {len(table_artifacts)} tables and {len(figure_artifacts)} figure files to {out}")
    if campaign.warnings:
        for warning in campaign.warnings:
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
