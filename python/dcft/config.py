"""Strict TOML configuration for local and cluster campaigns."""

from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from .registries import MEASUREMENTS, NOISES, PRIORS, UPDATES

T = TypeVar("T")


class ConfigError(ValueError):
    """A campaign configuration violates the numerical contract."""


@dataclass(frozen=True)
class CampaignSection:
    name: str
    preset: str
    seed: int
    output_root: Path
    project_root: Path
    scratch_root: Path


@dataclass(frozen=True)
class LatticeSection:
    sizes: tuple[int, ...]
    lt_factor: int
    regularization: str
    delta_tau: float | None

    def lt(self, lx: int) -> int:
        return self.lt_factor * lx


@dataclass(frozen=True)
class ProtocolSection:
    noises: tuple[str, ...]
    p_values: tuple[float, ...]
    measurements: tuple[str, ...]
    gaussian_fractions: tuple[float, ...]


@dataclass(frozen=True)
class McSection:
    enabled: bool
    outer_records: int
    clean_thermalization_sweeps: int
    clean_saving_interval: int
    posterior_decorrelation_gap: int
    inner_measurements: int
    inner_saving_interval: int
    updates: tuple[str, ...]
    separations: tuple[int, ...] | None
    inner_budget_multipliers: tuple[int, ...]
    diagnostic_outer_records: int
    replicated_chains: int
    tnmc_bond_dimension: int


@dataclass(frozen=True)
class EdSection:
    enabled: bool
    max_sites: int
    gaussian_outer_records: int
    local_x_enumeration_limit: int
    sampled_binary_records: int
    priors: tuple[str, ...]
    positivity_tolerance: float


@dataclass(frozen=True)
class StatisticsSection:
    bootstrap_resamples: int
    block_length: int
    confidence: float


@dataclass(frozen=True)
class ExecutionSection:
    mc_chunks: int
    local_workers: int


@dataclass(frozen=True)
class ClusterSection:
    scheduler: str
    partition: str
    account: str
    time_limit: str
    cpus_per_task: int
    memory: str


@dataclass(frozen=True)
class CampaignConfig:
    schema_version: int
    source: Path
    campaign: CampaignSection
    lattice: LatticeSection
    protocols: ProtocolSection
    mc: McSection
    ed: EdSection
    statistics: StatisticsSection
    execution: ExecutionSection
    cluster: ClusterSection

    def separations_for(self, lx: int) -> tuple[int, ...]:
        if self.mc.separations is None:
            return tuple(range(lx // 2 + 1))
        values = tuple(sorted(set(value % lx for value in self.mc.separations)))
        if not values:
            raise ConfigError("separation grid is empty")
        return values


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] table")
    return cast(dict[str, Any], value)


def _required(table: dict[str, Any], name: str, expected: type[T]) -> T:
    value = table.get(name)
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise ConfigError(f"{name} must be {expected.__name__}")
    return value


def _number(table: dict[str, Any], name: str) -> float:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be numeric")
    return float(value)


def _tuple_of(table: dict[str, Any], name: str, expected: type[T]) -> tuple[T, ...]:
    value = table.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, expected) or (expected is int and isinstance(item, bool))
        for item in value
    ):
        raise ConfigError(f"{name} must be an array of {expected.__name__}")
    return tuple(cast(list[T], value))


def _numeric_tuple(table: dict[str, Any], name: str) -> tuple[float, ...]:
    value = table.get(name)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in value
    ):
        raise ConfigError(f"{name} must be an array of numbers")
    return tuple(float(item) for item in value)


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_config(path: str | Path) -> CampaignConfig:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)

    schema_version = _required(raw, "schema_version", int)
    campaign_raw = _table(raw, "campaign")
    lattice_raw = _table(raw, "lattice")
    protocol_raw = _table(raw, "protocols")
    mc_raw = _table(raw, "mc")
    ed_raw = _table(raw, "ed")
    stats_raw = _table(raw, "statistics")
    execution_raw = _table(raw, "execution")
    cluster_raw = _table(raw, "cluster")

    campaign = CampaignSection(
        name=_required(campaign_raw, "name", str),
        preset=_required(campaign_raw, "preset", str),
        seed=_required(campaign_raw, "seed", int),
        output_root=_resolve_path(source, _required(campaign_raw, "output_root", str)),
        project_root=_resolve_path(source, _required(campaign_raw, "project_root", str)),
        scratch_root=_resolve_path(source, _required(campaign_raw, "scratch_root", str)),
    )
    delta_tau_raw = lattice_raw.get("delta_tau")
    lattice = LatticeSection(
        sizes=_tuple_of(lattice_raw, "sizes", int),
        lt_factor=_required(lattice_raw, "lt_factor", int),
        regularization=_required(lattice_raw, "regularization", str),
        delta_tau=None if delta_tau_raw is None else float(delta_tau_raw),
    )
    protocols = ProtocolSection(
        noises=_tuple_of(protocol_raw, "noises", str),
        p_values=_numeric_tuple(protocol_raw, "p_values"),
        measurements=_tuple_of(protocol_raw, "measurements", str),
        gaussian_fractions=_numeric_tuple(protocol_raw, "gaussian_fractions"),
    )
    raw_separations = mc_raw.get("separations")
    if raw_separations == "all":
        separations = None
    elif isinstance(raw_separations, list) and all(
        isinstance(value, int) and not isinstance(value, bool) for value in raw_separations
    ):
        separations = tuple(cast(list[int], raw_separations))
    else:
        raise ConfigError("mc.separations must be 'all' or an integer array")
    mc = McSection(
        enabled=_required(mc_raw, "enabled", bool),
        outer_records=_required(mc_raw, "outer_records", int),
        clean_thermalization_sweeps=_required(mc_raw, "clean_thermalization_sweeps", int),
        clean_saving_interval=_required(mc_raw, "clean_saving_interval", int),
        posterior_decorrelation_gap=_required(mc_raw, "posterior_decorrelation_gap", int),
        inner_measurements=_required(mc_raw, "inner_measurements", int),
        inner_saving_interval=_required(mc_raw, "inner_saving_interval", int),
        updates=_tuple_of(mc_raw, "updates", str),
        separations=separations,
        inner_budget_multipliers=_tuple_of(mc_raw, "inner_budget_multipliers", int),
        diagnostic_outer_records=_required(mc_raw, "diagnostic_outer_records", int),
        replicated_chains=_required(mc_raw, "replicated_chains", int),
        tnmc_bond_dimension=_required(mc_raw, "tnmc_bond_dimension", int),
    )
    ed = EdSection(
        enabled=_required(ed_raw, "enabled", bool),
        max_sites=_required(ed_raw, "max_sites", int),
        gaussian_outer_records=_required(ed_raw, "gaussian_outer_records", int),
        local_x_enumeration_limit=_required(ed_raw, "local_x_enumeration_limit", int),
        sampled_binary_records=_required(ed_raw, "sampled_binary_records", int),
        priors=_tuple_of(ed_raw, "priors", str),
        positivity_tolerance=_number(ed_raw, "positivity_tolerance"),
    )
    statistics = StatisticsSection(
        bootstrap_resamples=_required(stats_raw, "bootstrap_resamples", int),
        block_length=_required(stats_raw, "block_length", int),
        confidence=_number(stats_raw, "confidence"),
    )
    execution = ExecutionSection(
        mc_chunks=_required(execution_raw, "mc_chunks", int),
        local_workers=_required(execution_raw, "local_workers", int),
    )
    cluster = ClusterSection(
        scheduler=_required(cluster_raw, "scheduler", str),
        partition=_required(cluster_raw, "partition", str),
        account=_required(cluster_raw, "account", str),
        time_limit=_required(cluster_raw, "time_limit", str),
        cpus_per_task=_required(cluster_raw, "cpus_per_task", int),
        memory=_required(cluster_raw, "memory", str),
    )
    config = CampaignConfig(
        schema_version=schema_version,
        source=source,
        campaign=campaign,
        lattice=lattice,
        protocols=protocols,
        mc=mc,
        ed=ed,
        statistics=statistics,
        execution=execution,
        cluster=cluster,
    )
    validate_config(config)
    return config


def validate_config(config: CampaignConfig) -> None:
    problems: list[str] = []

    def duplicated(values: tuple[Any, ...]) -> bool:
        return len(values) != len(set(values))

    if config.schema_version != 2:
        problems.append("schema_version must equal 2")
    if config.campaign.preset not in {"comparison", "scaling", "test"}:
        problems.append("campaign.preset must be comparison, scaling, or test")
    if config.campaign.output_root == config.campaign.scratch_root:
        problems.append("output_root and scratch_root must be separate")
    if config.campaign.project_root == config.campaign.scratch_root:
        problems.append("project_root and scratch_root must be separate")
    if not config.lattice.sizes or any(size < 2 for size in config.lattice.sizes):
        problems.append("lattice.sizes must contain sizes >= 2")
    if duplicated(config.lattice.sizes):
        problems.append("lattice.sizes cannot contain duplicates")
    if config.lattice.lt_factor < 1:
        problems.append("lattice.lt_factor must be positive")
    if config.lattice.regularization not in {"isotropic", "trotter"}:
        problems.append("regularization must be isotropic or trotter")
    if config.lattice.regularization == "trotter" and (
        config.lattice.delta_tau is None
        or not math.isfinite(config.lattice.delta_tau)
        or config.lattice.delta_tau <= 0
    ):
        problems.append("trotter regularization requires positive delta_tau")
    if not config.protocols.noises or any(noise not in NOISES for noise in config.protocols.noises):
        problems.append(f"protocol noises must be drawn from {NOISES}")
    if duplicated(config.protocols.noises):
        problems.append("protocol noises cannot contain duplicates")
    if not config.protocols.p_values:
        problems.append("protocol p_values cannot be empty")
    if any(not 0.0 <= p < 0.5 for p in config.protocols.p_values):
        problems.append("all p values must satisfy 0 <= p < 1/2")
    if duplicated(config.protocols.p_values):
        problems.append("protocol p_values cannot contain duplicates")
    named_measurements = tuple(value for value in MEASUREMENTS if value != "gaussian")
    if not config.protocols.measurements or any(
        measurement not in named_measurements for measurement in config.protocols.measurements
    ):
        problems.append(
            f"named measurements must be drawn from {named_measurements}; "
            "use gaussian_fractions for gaussian(gamma)"
        )
    if duplicated(config.protocols.measurements):
        problems.append("named measurements cannot contain duplicates")
    if any(not 0.0 <= fraction <= 1.0 for fraction in config.protocols.gaussian_fractions):
        problems.append("gaussian fractions must lie in [0, 1]")
    if duplicated(config.protocols.gaussian_fractions):
        problems.append("gaussian fractions cannot contain duplicates")
    if not 0 <= config.campaign.seed < 1 << 64:
        problems.append("campaign.seed must fit an unsigned 64-bit integer")
    if not config.mc.enabled:
        problems.append("campaign analysis currently requires mc.enabled = true")
    if not config.mc.updates:
        problems.append("mc.updates cannot be empty")
    if any(update not in UPDATES for update in config.mc.updates):
        problems.append(f"updates must be drawn from {UPDATES}")
    if duplicated(config.mc.updates):
        problems.append("mc.updates cannot contain duplicates")
    positive_mc = (
        config.mc.outer_records,
        config.mc.clean_saving_interval,
        config.mc.posterior_decorrelation_gap,
        config.mc.inner_measurements,
        config.mc.inner_saving_interval,
        config.mc.replicated_chains,
        config.mc.tnmc_bond_dimension,
    )
    if any(value < 1 for value in positive_mc):
        problems.append(
            "MC counts, intervals, planted decorrelation gap, and TNMC bond dimension "
            "must be positive"
        )
    if config.mc.diagnostic_outer_records > config.mc.outer_records:
        problems.append("diagnostic_outer_records cannot exceed outer_records")
    if config.mc.diagnostic_outer_records < 2 or config.mc.replicated_chains < 2:
        problems.append("diagnostics require at least two outer records and two chains")
    if config.mc.separations is not None and any(
        separation < 0 for separation in config.mc.separations
    ):
        problems.append("configured separations must be nonnegative")
    if config.mc.inner_budget_multipliers != (1, 2, 4):
        problems.append("inner_budget_multipliers must be exactly [1, 2, 4]")
    if not 1 <= config.ed.local_x_enumeration_limit <= 14:
        problems.append("local_x_enumeration_limit must be between 1 and 14")
    if any(prior not in PRIORS for prior in config.ed.priors):
        problems.append(f"ED priors must be drawn from {PRIORS}")
    if duplicated(config.ed.priors):
        problems.append("ED priors cannot contain duplicates")
    if config.ed.enabled and (
        config.ed.max_sites < 2
        or config.ed.gaussian_outer_records < 2
        or config.ed.sampled_binary_records < 2
        or not config.ed.priors
    ):
        problems.append("enabled ED requires sites, record budgets, and at least one prior")
    if config.statistics.bootstrap_resamples < 20 or config.statistics.block_length < 1:
        problems.append("bootstrap_resamples must be >= 20 and block_length positive")
    if not 0.0 < config.statistics.confidence < 1.0:
        problems.append("statistics.confidence must lie in (0, 1)")
    if config.execution.mc_chunks < 1 or config.execution.local_workers < 1:
        problems.append("execution mc_chunks and local_workers must be positive")
    if config.execution.mc_chunks > config.mc.outer_records:
        problems.append("execution.mc_chunks cannot exceed mc.outer_records")
    if config.cluster.scheduler != "slurm":
        problems.append("only the slurm cluster scheduler is supported")
    if config.cluster.cpus_per_task < 1:
        problems.append("cluster cpus_per_task must be positive")
    if problems:
        raise ConfigError("; ".join(problems))


def as_serializable(config: CampaignConfig) -> dict[str, Any]:
    """Return a canonical JSON-compatible representation."""

    def convert(value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return {
                field.name: convert(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if field.name != "source"
            }
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        return value

    return cast(dict[str, Any], convert(config))


def scientific_config(config: CampaignConfig) -> dict[str, Any]:
    """Return the path- and scheduler-independent scientific request."""
    payload = as_serializable(config)
    campaign = cast(dict[str, Any], payload["campaign"])
    payload["campaign"] = {
        key: value
        for key, value in campaign.items()
        if key not in {"output_root", "project_root", "scratch_root"}
    }
    payload.pop("execution")
    payload.pop("cluster")
    return payload
