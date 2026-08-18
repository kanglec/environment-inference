"""Versioned Arrow schemas for immutable scientific datasets."""

from __future__ import annotations

from typing import Final

import pyarrow as pa

SCHEMA_METADATA: Final[dict[bytes, bytes]] = {
    b"dcft.schema": b"DCFT_PARQUET_V1",
    b"dcft.configuration.bits": b"bit 0 means +1; bit 1 means -1; x-fast then t",
}


def _schema(fields: list[tuple[str, pa.DataType, bool]]) -> pa.Schema:
    return pa.schema(
        [pa.field(name, kind, nullable=nullable) for name, kind, nullable in fields],
        metadata=SCHEMA_METADATA,
    )


CLEAN_SCHEMA: Final[pa.Schema] = _schema(
    [
        ("global_id", pa.uint64(), False),
        ("lx", pa.uint32(), False),
        ("lt", pa.uint32(), False),
        ("configuration", pa.binary(), False),
        ("configuration_hash", pa.string(), False),
        ("regularization", pa.string(), False),
        ("kx", pa.float64(), False),
        ("kt", pa.float64(), False),
        ("delta_tau", pa.float64(), True),
        ("seed", pa.uint64(), False),
        ("clean_thermalization_sweeps", pa.uint64(), False),
        ("clean_saving_interval", pa.uint64(), False),
    ]
)


MC_RECORD_SCHEMA: Final[pa.Schema] = _schema(
    [
        ("global_id", pa.uint64(), False),
        ("lx", pa.uint32(), False),
        ("lt", pa.uint32(), False),
        ("noise", pa.string(), False),
        ("measurement", pa.string(), False),
        ("protocol_id", pa.string(), False),
        ("update", pa.string(), False),
        ("tnmc_bond_dimension", pa.uint32(), False),
        ("p", pa.float64(), False),
        ("lambda", pa.float64(), False),
        ("gamma", pa.float64(), True),
        ("kappa", pa.float64(), True),
        ("protocol_coupling", pa.float64(), True),
        ("planted_configuration_hash", pa.string(), False),
        ("initial_configuration_hash", pa.string(), False),
        ("chain_role", pa.string(), False),
        ("initialization", pa.string(), False),
        ("planted_variables", pa.list_(pa.int8()), False),
        ("raw_record", pa.list_(pa.float64()), False),
        ("record_couplings", pa.list_(pa.float64()), False),
        ("standard_variates", pa.list_(pa.float64()), False),
        ("inner_budget_multiplier", pa.uint8(), False),
        ("replica", pa.uint16(), False),
        ("posterior_decorrelation_gap", pa.uint64(), False),
        ("inner_measurements", pa.uint64(), False),
        ("inner_saving_interval", pa.uint64(), False),
        ("energy", pa.float64(), False),
        ("magnetization", pa.float64(), False),
        ("boundary_magnetization", pa.float64(), False),
        ("spin_profile", pa.list_(pa.float64()), False),
        ("bond_profile", pa.list_(pa.float64()), False),
        ("separations", pa.list_(pa.uint32()), False),
        ("spin_correlator_profile", pa.list_(pa.list_(pa.float64())), False),
        ("bond_correlator_profile", pa.list_(pa.list_(pa.float64())), False),
        ("planted_spin_overlap", pa.float64(), False),
        ("planted_bond_overlap", pa.float64(), False),
        ("planted_spin_correlator", pa.list_(pa.float64()), False),
        ("planted_bond_correlator", pa.list_(pa.float64()), False),
        ("energy_trace", pa.list_(pa.float64()), False),
        ("magnetization_trace", pa.list_(pa.float64()), False),
        ("boundary_magnetization_trace", pa.list_(pa.float64()), False),
        ("planted_spin_overlap_trace", pa.list_(pa.float64()), False),
        ("planted_bond_overlap_trace", pa.list_(pa.float64()), False),
        ("sweeps", pa.uint64(), False),
        ("local_proposed", pa.uint64(), False),
        ("local_accepted", pa.uint64(), False),
        ("cluster_proposed", pa.uint64(), False),
        ("cluster_accepted", pa.uint64(), False),
        ("cluster_sites_proposed", pa.uint64(), False),
        ("global_proposed", pa.uint64(), False),
        ("global_attempted", pa.uint64(), False),
        ("global_accepted", pa.uint64(), False),
        ("tnmc_proposed", pa.uint64(), False),
        ("tnmc_accepted", pa.uint64(), False),
        ("tnmc_sites_proposed", pa.uint64(), False),
        ("tnmc_conditionals_regularized", pa.uint64(), False),
    ]
)


ED_RESULT_SCHEMA: Final[pa.Schema] = _schema(
    [
        ("lx", pa.uint32(), False),
        ("noise", pa.string(), False),
        ("measurement", pa.string(), False),
        ("protocol_id", pa.string(), False),
        ("prior", pa.string(), False),
        ("observable", pa.string(), False),
        ("separation", pa.uint32(), True),
        ("p", pa.float64(), False),
        ("lambda", pa.float64(), False),
        ("gamma", pa.float64(), True),
        ("kappa", pa.float64(), True),
        ("linear_exact", pa.float64(), False),
        ("physical_fidelity", pa.float64(), False),
        ("measurement_witness", pa.float64(), False),
        ("witness_standard_error", pa.float64(), False),
        ("fidelity_gap", pa.float64(), False),
        ("entropy", pa.float64(), False),
        ("purity", pa.float64(), False),
        ("ground_energy", pa.float64(), True),
        ("ground_residual", pa.float64(), True),
        ("trace_error", pa.float64(), False),
        ("hermiticity_error", pa.float64(), False),
        ("minimum_eigenvalue", pa.float64(), False),
        ("record_mode", pa.string(), False),
        ("outer_records", pa.uint64(), False),
    ]
)


ED_RECORD_SCHEMA: Final[pa.Schema] = _schema(
    [
        ("global_id", pa.uint64(), False),
        ("lx", pa.uint32(), False),
        ("noise", pa.string(), False),
        ("measurement", pa.string(), False),
        ("protocol_id", pa.string(), False),
        ("prior", pa.string(), False),
        ("p", pa.float64(), False),
        ("gamma", pa.float64(), True),
        ("planted_state", pa.uint64(), True),
        ("raw_record", pa.list_(pa.float64()), False),
        ("standard_variates", pa.list_(pa.float64()), False),
        ("observable", pa.string(), False),
        ("separation", pa.uint32(), True),
        ("posterior_mean", pa.float64(), False),
        ("absolute_contribution", pa.float64(), False),
        ("planted_contribution", pa.float64(), False),
        ("record_weight", pa.float64(), False),
        ("record_mode", pa.string(), False),
    ]
)


PRIOR_DIAGNOSTIC_SCHEMA: Final[pa.Schema] = _schema(
    [
        ("lx", pa.uint32(), False),
        ("lt", pa.uint32(), False),
        ("regularization", pa.string(), False),
        ("delta_tau", pa.float64(), True),
        ("left_prior", pa.string(), False),
        ("right_prior", pa.string(), False),
        ("total_variation", pa.float64(), False),
    ]
)


SCHEMAS: Final[dict[str, pa.Schema]] = {
    "clean": CLEAN_SCHEMA,
    "mc-records": MC_RECORD_SCHEMA,
    "ed-results": ED_RESULT_SCHEMA,
    "ed-records": ED_RECORD_SCHEMA,
    "prior-diagnostics": PRIOR_DIAGNOSTIC_SCHEMA,
}


def table_from_rows(kind: str, rows: list[dict[str, object]]) -> pa.Table:
    schema = SCHEMAS.get(kind)
    if schema is None:
        return pa.Table.from_pylist(rows)
    return pa.Table.from_pylist(rows, schema=schema)
