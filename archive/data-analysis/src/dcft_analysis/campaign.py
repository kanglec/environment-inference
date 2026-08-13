"""Discovery, validation, and table construction for simulation campaigns."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_NOISES = ("z", "zz")
P_TAG_RE = re.compile(r"^p(?P<hundredths>\d{3})$")
LATTICE_TAG_RE = re.compile(r"^L(?P<lx>\d+)(?:x(?P<lt>\d+))?$")

CSV_COLUMNS = [
    "samples",
    "lx",
    "lt",
    "energy_density",
    "magnetization_density",
    "boundary_magnetization",
    "local_spin_fidelity",
    "local_bond_fidelity",
    "r",
    "spin_linear_corr",
    "spin_fidelity_corr",
    "spin_ea_corr",
    "bond_linear_corr",
    "bond_fidelity_corr",
    "bond_ea_corr",
]

SCALAR_SOURCE_COLUMNS = [
    "samples",
    "lx",
    "lt",
    "energy_density",
    "magnetization_density",
    "boundary_magnetization",
    "local_spin_fidelity",
    "local_bond_fidelity",
]

SCALAR_RENAMES = {
    "energy_density": "E_D",
    "magnetization_density": "M",
    "boundary_magnetization": "M_partial",
    "local_spin_fidelity": "F_sigma_loc",
    "local_bond_fidelity": "F_epsilon_loc",
}

CORRELATOR_RENAMES = {
    "spin_linear_corr": "C_sigma_lin",
    "bond_linear_corr": "C_epsilon_lin",
    "spin_fidelity_corr": "F_sigma",
    "bond_fidelity_corr": "F_epsilon",
    "spin_ea_corr": "C_sigma_EA",
    "bond_ea_corr": "C_epsilon_EA",
}


class CampaignValidationError(ValueError):
    """Raised when a campaign cannot be interpreted without ambiguity."""

    def __init__(self, problems: Iterable[str]):
        self.problems = list(problems)
        message = "Campaign validation failed:\n" + "\n".join(
            f"  - {problem}" for problem in self.problems
        )
        super().__init__(message)


@dataclass(frozen=True, order=True)
class Point:
    """One simulation point discovered from a campaign path."""

    noise: str
    p_hundredths: int
    path: Path

    @property
    def p(self) -> float:
        return self.p_hundredths / 100.0

    @property
    def p_tag(self) -> str:
        return f"p{self.p_hundredths:03d}"


@dataclass
class Campaign:
    """Validated campaign tables and discovery information."""

    root: Path
    points: list[Point]
    scalars: pd.DataFrame
    correlators: pd.DataFrame
    missing_points: list[tuple[str, str]]
    warnings: list[str]

    @property
    def metadata(self) -> dict[str, int]:
        first = self.scalars.iloc[0]
        return {
            "lx": int(first["lx"]),
            "lt": int(first["lt"]),
            "samples_per_point": int(first["samples"]),
        }


def parse_p_tag(tag: str) -> tuple[int, float]:
    """Parse the pipeline's pNNN tag and enforce 0 <= p < 0.5."""
    match = P_TAG_RE.fullmatch(tag)
    if match is None:
        raise ValueError(f"invalid p tag {tag!r}: expected pNNN (for example p005)")
    hundredths = int(match.group("hundredths"))
    if not 0 <= hundredths < 50:
        raise ValueError(f"invalid p tag {tag!r}: expected p000 through p049")
    return hundredths, hundredths / 100.0


def parse_expected_p(value: str) -> int:
    """Convert a CLI p value to the pipeline's integer-hundredths representation."""
    try:
        p = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid p value {value!r}") from exc
    scaled = p * 100
    if p < 0 or p >= Decimal("0.5") or scaled != scaled.to_integral_value():
        raise ValueError(f"invalid p value {value!r}: expected 0 <= p < 0.5 in increments of 0.01")
    return int(scaled)


def _point_from_csv(root: Path, path: Path) -> Point:
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) < 4 or parts[-1] != "observables.csv" or parts[-2] != "analysis":
        raise ValueError(f"{relative}: expected a {{noise}}/pNNN/analysis/observables.csv suffix")
    noise, p_tag = parts[-4], parts[-3]
    if noise not in SUPPORTED_NOISES:
        raise ValueError(f"{relative}: unsupported noise directory {noise!r}; expected z or zz")
    hundredths, _ = parse_p_tag(p_tag)
    return Point(noise=noise, p_hundredths=hundredths, path=path)


def _point_directories_without_csv(root: Path) -> list[tuple[str, int, Path]]:
    missing: list[tuple[str, int, Path]] = []
    for noise in SUPPORTED_NOISES:
        noise_dir = root / noise
        if not noise_dir.is_dir():
            continue
        for point_dir in sorted(path for path in noise_dir.iterdir() if path.is_dir()):
            if not point_dir.name.startswith("p"):
                continue
            try:
                p_hundredths, _ = parse_p_tag(point_dir.name)
            except ValueError:
                continue
            expected_csv = point_dir / "analysis" / "observables.csv"
            if not expected_csv.is_file():
                missing.append((noise, p_hundredths, point_dir))
    return missing


def discover_points(root: Path) -> tuple[list[Point], list[str]]:
    """Recursively discover valid point CSVs and report malformed candidates."""
    root = root.resolve()
    if not root.is_dir():
        raise CampaignValidationError(
            [f"lattice root does not exist or is not a directory: {root}"]
        )

    candidates = sorted(root.rglob("observables.csv"))
    points: list[Point] = []
    problems: list[str] = []
    for candidate in candidates:
        try:
            points.append(_point_from_csv(root, candidate))
        except ValueError as exc:
            problems.append(str(exc))

    by_key: dict[tuple[str, int], list[Path]] = {}
    for point in points:
        by_key.setdefault((point.noise, point.p_hundredths), []).append(point.path)
    for (noise, p_hundredths), paths in sorted(by_key.items()):
        if len(paths) > 1:
            rendered = ", ".join(str(path.relative_to(root)) for path in paths)
            problems.append(f"duplicate point ({noise}, p{p_hundredths:03d}) found at: {rendered}")

    # A point directory already created by the pipeline but lacking its analysis CSV
    # is a distinct and useful failure mode.
    for noise in SUPPORTED_NOISES:
        noise_dir = root / noise
        if not noise_dir.is_dir():
            continue
        for point_dir in sorted(path for path in noise_dir.iterdir() if path.is_dir()):
            if not point_dir.name.startswith("p"):
                continue
            try:
                parse_p_tag(point_dir.name)
            except ValueError as exc:
                problems.append(f"{point_dir.relative_to(root)}: {exc}")
                continue
    for _, _, point_dir in _point_directories_without_csv(root):
        problems.append(
            f"missing analysis CSV for existing point directory: {point_dir.relative_to(root)}"
        )

    unique_points = [
        Point(noise=noise, p_hundredths=p_hundredths, path=paths[0])
        for (noise, p_hundredths), paths in sorted(by_key.items())
    ]
    if not unique_points:
        problems.append(f"no valid observables.csv points found beneath {root}")
    return unique_points, problems


def _constant_value(frame: pd.DataFrame, column: str, relative: Path) -> float:
    values = frame[column]
    if values.isna().any():
        raise CampaignValidationError(
            [f"{relative}: scalar column {column!r} mixes missing and non-missing values"]
        )
    first = float(values.iloc[0])
    if not np.all(values.to_numpy(dtype=float) == first):
        raise CampaignValidationError(
            [f"{relative}: scalar column {column!r} is not constant across r"]
        )
    return first


def read_point_csv(point: Point, root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    """Read and fully validate one Rust-generated analysis CSV."""
    relative = point.path.relative_to(root)
    try:
        frame = pd.read_csv(point.path, keep_default_na=False)
    except Exception as exc:
        raise CampaignValidationError([f"{relative}: could not read CSV: {exc}"]) from exc

    if list(frame.columns) != CSV_COLUMNS:
        missing = [column for column in CSV_COLUMNS if column not in frame.columns]
        extra = [column for column in frame.columns if column not in CSV_COLUMNS]
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if not missing and not extra:
            details.append("columns are out of Rust-writer order")
        raise CampaignValidationError([f"{relative}: invalid CSV schema ({'; '.join(details)})"])
    if frame.empty:
        raise CampaignValidationError([f"{relative}: CSV has no data rows"])

    for column in CSV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    bad_required = [column for column in CSV_COLUMNS if frame[column].isna().any()]
    if bad_required:
        raise CampaignValidationError(
            [f"{relative}: non-numeric or missing values in columns {bad_required}"]
        )
    finite_columns = [
        column for column in CSV_COLUMNS if column not in ("samples", "lx", "lt", "r")
    ]
    bad_finite = [
        column
        for column in finite_columns
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all()
    ]
    if bad_finite:
        raise CampaignValidationError([f"{relative}: non-finite values in columns {bad_finite}"])

    for column in ("samples", "lx", "lt", "r"):
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise CampaignValidationError([f"{relative}: {column!r} must contain integers"])
        frame[column] = frame[column].astype(int)

    scalars = {column: _constant_value(frame, column, relative) for column in SCALAR_SOURCE_COLUMNS}
    samples, lx, lt = (int(scalars[key]) for key in ("samples", "lx", "lt"))
    if samples < 1 or lx < 1 or lt < 1:
        raise CampaignValidationError(
            [f"{relative}: samples, lx, and lt must all be positive (got {samples}, {lx}, {lt})"]
        )

    duplicated_r = sorted(frame.loc[frame["r"].duplicated(keep=False), "r"].unique())
    if duplicated_r:
        raise CampaignValidationError([f"{relative}: duplicate r values {duplicated_r}"])
    expected_r = list(range(lx // 2 + 1))
    actual_r = sorted(frame["r"].tolist())
    if actual_r != expected_r:
        missing = sorted(set(expected_r) - set(actual_r))
        extra = sorted(set(actual_r) - set(expected_r))
        raise CampaignValidationError(
            [
                f"{relative}: invalid r range; expected 0..{lx // 2}, "
                f"missing={missing}, extra={extra}"
            ]
        )

    bounded_columns = [
        "magnetization_density",
        "boundary_magnetization",
        "local_spin_fidelity",
        "local_bond_fidelity",
        *CORRELATOR_RENAMES,
    ]
    for column in bounded_columns:
        values = frame[column].to_numpy(dtype=float)
        if np.any(values < -1.000000000001) or np.any(values > 1.000000000001):
            raise CampaignValidationError(
                [f"{relative}: physical observable {column!r} lies outside [-1, 1]"]
            )
    for column in (
        "local_spin_fidelity",
        "local_bond_fidelity",
        "spin_fidelity_corr",
        "bond_fidelity_corr",
    ):
        value = frame[column].to_numpy(dtype=float)
        if np.any(value < -1e-12):
            raise CampaignValidationError(
                [f"{relative}: fidelity observable {column!r} is negative"]
            )

    scalar_row: dict[str, object] = {
        "noise": point.noise,
        "p": point.p,
        "p_tag": point.p_tag,
        **scalars,
        "source_csv": str(relative),
    }
    for column in ("samples", "lx", "lt"):
        scalar_row[column] = int(scalar_row[column])
    scalar_row = {SCALAR_RENAMES.get(key, key): value for key, value in scalar_row.items()}

    correlators = frame[["r", *CORRELATOR_RENAMES]].rename(columns=CORRELATOR_RENAMES).copy()
    correlators.insert(0, "p_tag", point.p_tag)
    correlators.insert(0, "p", point.p)
    correlators.insert(0, "noise", point.noise)
    return scalar_row, correlators


def _expected_grid(
    points: list[Point],
    expected_noises: Iterable[str] | None,
    expected_p: Iterable[int] | None,
) -> tuple[list[str], list[int], list[str]]:
    discovered_noises = sorted({point.noise for point in points})
    discovered_p = sorted({point.p_hundredths for point in points})
    noises = list(dict.fromkeys(expected_noises or discovered_noises))
    p_values = list(dict.fromkeys(expected_p or discovered_p))
    problems: list[str] = []
    invalid_noises = [noise for noise in noises if noise not in SUPPORTED_NOISES]
    if invalid_noises:
        problems.append(f"unsupported expected noise values: {invalid_noises}")

    point_keys = {(point.noise, point.p_hundredths) for point in points}
    expected_keys = {(noise, p) for noise in noises for p in p_values}
    unexpected = sorted(point_keys - expected_keys)
    if unexpected:
        rendered = ", ".join(f"({noise}, p{p:03d})" for noise, p in unexpected)
        problems.append(f"discovered points outside the explicitly expected grid: {rendered}")
    return noises, p_values, problems


def load_campaign(
    root: Path,
    *,
    expected_noises: Iterable[str] | None = None,
    expected_p: Iterable[int] | None = None,
    allow_incomplete: bool = False,
) -> Campaign:
    """Discover and validate a complete campaign, returning tidy tables."""
    root = root.resolve()
    points, discovery_problems = discover_points(root)
    noises, p_values, grid_problems = _expected_grid(points, expected_noises, expected_p)

    point_keys = {(point.noise, point.p_hundredths) for point in points}
    missing = {
        (noise, f"p{p:03d}") for noise in noises for p in p_values if (noise, p) not in point_keys
    }
    missing.update(
        (noise, f"p{p_hundredths:03d}")
        for noise, p_hundredths, _ in _point_directories_without_csv(root)
    )
    missing = sorted(missing)
    warnings: list[str] = []
    missing_directory_problems = [
        problem
        for problem in discovery_problems
        if problem.startswith("missing analysis CSV for existing point directory:")
    ]
    problems = [
        problem for problem in discovery_problems if problem not in missing_directory_problems
    ]
    problems.extend(grid_problems)
    if missing_directory_problems:
        if allow_incomplete:
            warnings.extend(missing_directory_problems)
        else:
            problems.extend(missing_directory_problems)
    if missing:
        rendered = ", ".join(f"({noise}, {p_tag})" for noise, p_tag in missing)
        message = f"missing points from expected campaign grid: {rendered}"
        if allow_incomplete:
            warnings.append(message)
        else:
            problems.append(message)
    if problems:
        raise CampaignValidationError(problems)

    scalar_rows: list[dict[str, object]] = []
    correlator_frames: list[pd.DataFrame] = []
    read_problems: list[str] = []
    for point in points:
        try:
            scalar, correlators = read_point_csv(point, root)
            scalar_rows.append(scalar)
            correlator_frames.append(correlators)
        except CampaignValidationError as exc:
            read_problems.extend(exc.problems)
    if read_problems:
        raise CampaignValidationError(read_problems)

    scalars = pd.DataFrame(scalar_rows).sort_values(["noise", "p"]).reset_index(drop=True)
    correlators = (
        pd.concat(correlator_frames, ignore_index=True)
        .sort_values(["noise", "p", "r"])
        .reset_index(drop=True)
    )

    metadata_columns = ["samples", "lx", "lt"]
    inconsistent = [column for column in metadata_columns if scalars[column].nunique() != 1]
    if inconsistent:
        details = ", ".join(
            f"{column}={sorted(scalars[column].unique().tolist())}" for column in inconsistent
        )
        raise CampaignValidationError([f"inconsistent metadata across campaign points: {details}"])

    match = LATTICE_TAG_RE.fullmatch(root.name)
    if match:
        path_lx = int(match.group("lx"))
        path_lt = int(match.group("lt") or path_lx)
        actual_lx = int(scalars.iloc[0]["lx"])
        actual_lt = int(scalars.iloc[0]["lt"])
        if (path_lx, path_lt) != (actual_lx, actual_lt):
            raise CampaignValidationError(
                [
                    f"lattice directory {root.name!r} implies lx={path_lx}, lt={path_lt}, "
                    f"but CSV metadata says lx={actual_lx}, lt={actual_lt}"
                ]
            )

    return Campaign(
        root=root,
        points=points,
        scalars=scalars,
        correlators=correlators,
        missing_points=missing,
        warnings=warnings,
    )
