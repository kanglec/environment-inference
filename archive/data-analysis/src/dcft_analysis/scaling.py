"""Multi-size scaling analysis from validated campaign-analysis tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import CampaignValidationError, parse_p_tag

SCALAR_REQUIRED = [
    "noise",
    "p",
    "p_tag",
    "samples",
    "lx",
    "lt",
    "E_D",
    "M",
    "M_partial",
    "F_sigma_loc",
    "F_epsilon_loc",
]

CORRELATOR_COLUMNS = [
    "C_sigma_lin",
    "C_epsilon_lin",
    "F_sigma",
    "F_epsilon",
    "C_sigma_EA",
    "C_epsilon_EA",
]

CORRELATOR_REQUIRED = ["noise", "p", "p_tag", "r", *CORRELATOR_COLUMNS]
KEY_COLUMNS = ["noise", "p", "p_tag"]


@dataclass(frozen=True)
class CampaignTablePair:
    """One campaign-analysis scalar/correlator table pair."""

    scalar_path: Path
    correlator_path: Path
    L: int


@dataclass
class ScalingAnalysis:
    """Validated and derived multi-size scaling tables."""

    root: Path
    sources: list[CampaignTablePair]
    scalars: pd.DataFrame
    correlators: pd.DataFrame
    long_distance: pd.DataFrame
    fits: pd.DataFrame
    missing_points: list[tuple[int, str, str]]
    warnings: list[str]

    @property
    def sizes(self) -> list[int]:
        return sorted(self.scalars["L"].unique().astype(int).tolist())

    @property
    def noises(self) -> list[str]:
        return sorted(self.scalars["noise"].unique().tolist())

    @property
    def p_values(self) -> list[float]:
        return sorted(self.scalars["p"].unique().astype(float).tolist())


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except Exception as exc:
        raise CampaignValidationError([f"{path}: could not read CSV: {exc}"]) from exc
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise CampaignValidationError([f"{path}: missing required columns {missing}"])
    if frame.empty:
        raise CampaignValidationError([f"{path}: table has no data rows"])
    return frame


def _numeric(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise CampaignValidationError(
                [f"{path}: column {column!r} contains missing, non-numeric, or non-finite values"]
            )
        frame[column] = converted


def _validate_tags(frame: pd.DataFrame, path: Path) -> None:
    problems: list[str] = []
    for p, p_tag in frame[["p", "p_tag"]].drop_duplicates().itertuples(index=False):
        try:
            hundredths, tag_p = parse_p_tag(str(p_tag))
        except ValueError as exc:
            problems.append(f"{path}: {exc}")
            continue
        if not np.isclose(float(p), tag_p, rtol=0, atol=1e-12):
            problems.append(
                f"{path}: p={p} is inconsistent with p_tag={p_tag} "
                f"(which denotes {hundredths / 100:.2f})"
            )
    if problems:
        raise CampaignValidationError(problems)


def _validate_scalar_table(path: Path) -> tuple[pd.DataFrame, int]:
    frame = _read_csv(path, SCALAR_REQUIRED)
    numeric = [
        "p",
        "samples",
        "lx",
        "lt",
        "E_D",
        "M",
        "M_partial",
        "F_sigma_loc",
        "F_epsilon_loc",
    ]
    _numeric(frame, numeric, path)

    for column in ("samples", "lx", "lt"):
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all() or np.any(values < 1):
            raise CampaignValidationError([f"{path}: {column!r} must contain positive integers"])
        frame[column] = frame[column].astype(int)
    if frame["lx"].nunique() != 1 or frame["lt"].nunique() != 1:
        raise CampaignValidationError([f"{path}: lx and lt must be constant within the table"])
    L = int(frame["lx"].iloc[0])
    lt = int(frame["lt"].iloc[0])
    if lt != L:
        raise CampaignValidationError(
            [f"{path}: scaling analysis requires square systems, but lx={L} and lt={lt}"]
        )
    if frame[KEY_COLUMNS].duplicated().any():
        duplicate = frame.loc[frame[KEY_COLUMNS].duplicated(keep=False), KEY_COLUMNS]
        raise CampaignValidationError(
            [f"{path}: duplicate scalar keys {duplicate.drop_duplicates().to_dict('records')}"]
        )
    unsupported = sorted(set(frame["noise"]) - {"z", "zz"})
    if unsupported:
        raise CampaignValidationError([f"{path}: unsupported noise values {unsupported}"])
    _validate_tags(frame, path)
    frame.insert(0, "L", L)
    return frame, L


def _validate_correlator_table(path: Path, scalar: pd.DataFrame, L: int) -> pd.DataFrame:
    frame = _read_csv(path, CORRELATOR_REQUIRED)
    _numeric(frame, ["p", "r", *CORRELATOR_COLUMNS], path)
    r_values = frame["r"].to_numpy(dtype=float)
    if not np.equal(r_values, np.floor(r_values)).all():
        raise CampaignValidationError([f"{path}: r must contain integers"])
    frame["r"] = frame["r"].astype(int)
    _validate_tags(frame, path)

    scalar_keys = set(map(tuple, scalar[KEY_COLUMNS].itertuples(index=False, name=None)))
    correlator_keys = set(map(tuple, frame[KEY_COLUMNS].itertuples(index=False, name=None)))
    if scalar_keys != correlator_keys:
        only_scalar = sorted(scalar_keys - correlator_keys)
        only_correlator = sorted(correlator_keys - scalar_keys)
        raise CampaignValidationError(
            [
                f"{path}: scalar/correlator point keys differ; "
                f"missing_correlators={only_scalar}, missing_scalars={only_correlator}"
            ]
        )

    expected_r = list(range(L // 2 + 1))
    problems: list[str] = []
    for key, group in frame.groupby(KEY_COLUMNS, sort=True):
        actual = sorted(group["r"].tolist())
        if actual != expected_r:
            problems.append(f"{path}: point {key} has r={actual}, expected {expected_r}")
    if frame[[*KEY_COLUMNS, "r"]].duplicated().any():
        problems.append(f"{path}: duplicate (noise, p, p_tag, r) rows")
    if problems:
        raise CampaignValidationError(problems)

    frame.insert(0, "L", L)
    frame.insert(5, "r_over_L", frame["r"] / L)
    frame.insert(6, "chord_distance", L / np.pi * np.sin(np.pi * frame["r"] / L))
    return frame


def discover_campaign_tables(root: Path) -> list[tuple[Path, Path]]:
    """Find campaign-analysis table pairs beneath a multi-size root."""
    root = root.resolve()
    if not root.is_dir():
        raise CampaignValidationError(
            [f"scaling root does not exist or is not a directory: {root}"]
        )
    scalar_paths = sorted(root.rglob("tables/scalar_observables.csv"))
    if not scalar_paths:
        raise CampaignValidationError(
            [f"no tables/scalar_observables.csv files found beneath {root}"]
        )
    pairs: list[tuple[Path, Path]] = []
    problems: list[str] = []
    for scalar_path in scalar_paths:
        correlator_path = scalar_path.with_name("correlators.csv")
        if not correlator_path.is_file():
            problems.append(f"{scalar_path}: sibling correlators.csv is missing")
        else:
            pairs.append((scalar_path, correlator_path))
    if problems:
        raise CampaignValidationError(problems)
    return pairs


def _derive_long_distance(correlators: pd.DataFrame) -> pd.DataFrame:
    index = correlators.groupby(["L", *KEY_COLUMNS], sort=True)["r"].idxmax()
    long_distance = correlators.loc[index].copy()
    long_distance = long_distance.rename(columns={"r": "r_max"})
    return long_distance.sort_values(["noise", "p", "L"]).reset_index(drop=True)


def fit_long_distance_power_laws(long_distance: pd.DataFrame) -> pd.DataFrame:
    """Fit C(r_max; L) = amplitude * L^(-decay_exponent) for positive values."""
    rows: list[dict[str, object]] = []
    for (noise, p, p_tag), group in long_distance.groupby(KEY_COLUMNS, sort=True):
        for observable in CORRELATOR_COLUMNS:
            valid = group.loc[group[observable] > 0, ["L", observable]].drop_duplicates("L")
            row: dict[str, object] = {
                "noise": noise,
                "p": float(p),
                "p_tag": p_tag,
                "observable": observable,
                "n_sizes_total": int(group["L"].nunique()),
                "n_sizes_fit": int(valid["L"].nunique()),
                "L_min": int(valid["L"].min()) if not valid.empty else np.nan,
                "L_max": int(valid["L"].max()) if not valid.empty else np.nan,
                "amplitude": np.nan,
                "decay_exponent": np.nan,
                "scaling_dimension": np.nan,
                "r_squared": np.nan,
                "fit_status": "insufficient_positive_sizes",
            }
            if len(valid) >= 2:
                x = np.log(valid["L"].to_numpy(dtype=float))
                y = np.log(valid[observable].to_numpy(dtype=float))
                slope, intercept = np.polyfit(x, y, 1)
                prediction = slope * x + intercept
                residual = float(np.sum((y - prediction) ** 2))
                total = float(np.sum((y - y.mean()) ** 2))
                row.update(
                    {
                        "amplitude": float(np.exp(intercept)),
                        "decay_exponent": float(-slope),
                        "scaling_dimension": float(-slope / 2),
                        "r_squared": float(1 - residual / total) if total > 0 else np.nan,
                        "fit_status": "ok",
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["noise", "p", "observable"]).reset_index(drop=True)


def load_scaling_analysis(root: Path, *, allow_incomplete: bool = False) -> ScalingAnalysis:
    """Load all campaign tables under root and derive multi-size scaling products."""
    root = root.resolve()
    pairs = discover_campaign_tables(root)
    sources: list[CampaignTablePair] = []
    scalar_frames: list[pd.DataFrame] = []
    correlator_frames: list[pd.DataFrame] = []
    problems: list[str] = []
    for scalar_path, correlator_path in pairs:
        try:
            scalar, L = _validate_scalar_table(scalar_path)
            correlator = _validate_correlator_table(correlator_path, scalar, L)
            relative_source = str(scalar_path.parent.parent.relative_to(root))
            scalar["source_campaign"] = relative_source
            correlator["source_campaign"] = relative_source
            scalar_frames.append(scalar)
            correlator_frames.append(correlator)
            sources.append(CampaignTablePair(scalar_path, correlator_path, L))
        except CampaignValidationError as exc:
            problems.extend(exc.problems)
    if problems:
        raise CampaignValidationError(problems)

    scalars = pd.concat(scalar_frames, ignore_index=True)
    correlators = pd.concat(correlator_frames, ignore_index=True)
    scalar_key = ["L", *KEY_COLUMNS]
    if scalars[scalar_key].duplicated().any():
        duplicates = scalars.loc[scalars[scalar_key].duplicated(keep=False), scalar_key]
        records = duplicates.to_dict("records")
        raise CampaignValidationError(
            [f"duplicate size/noise/p points across campaign tables: {records}"]
        )
    if correlators[[*scalar_key, "r"]].duplicated().any():
        raise CampaignValidationError(["duplicate size/noise/p/r rows across campaign tables"])

    sizes = sorted(scalars["L"].unique().astype(int).tolist())
    if len(sizes) < 2:
        raise CampaignValidationError(
            [f"scaling analysis requires at least two distinct sizes; found {sizes}"]
        )
    noises = sorted(scalars["noise"].unique().tolist())
    p_tags = sorted(scalars["p_tag"].unique().tolist())
    actual = set(map(tuple, scalars[["L", "noise", "p_tag"]].itertuples(index=False, name=None)))
    missing = sorted(
        (L, noise, p_tag)
        for L in sizes
        for noise in noises
        for p_tag in p_tags
        if (L, noise, p_tag) not in actual
    )
    warnings: list[str] = []
    if missing:
        rendered = ", ".join(f"(L={L}, {noise}, {p_tag})" for L, noise, p_tag in missing)
        message = f"missing points from the inferred size/noise/p grid: {rendered}"
        if allow_incomplete:
            warnings.append(message)
        else:
            raise CampaignValidationError([message])

    scalars = scalars.sort_values(["noise", "p", "L"]).reset_index(drop=True)
    correlators = correlators.sort_values(["noise", "p", "L", "r"]).reset_index(drop=True)
    long_distance = _derive_long_distance(correlators)
    fits = fit_long_distance_power_laws(long_distance)
    return ScalingAnalysis(
        root=root,
        sources=sorted(sources, key=lambda source: source.L),
        scalars=scalars,
        correlators=correlators,
        long_distance=long_distance,
        fits=fits,
        missing_points=missing,
        warnings=warnings,
    )
