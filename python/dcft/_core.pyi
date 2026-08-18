from typing import Any

def version() -> str: ...
def rng_contract() -> tuple[str, str]: ...
def measurement_registry() -> list[str]: ...
def noise_registry() -> list[str]: ...
def update_registry() -> list[str]: ...
def observable_registry() -> list[str]: ...
def protocol_parameters(
    measurement: str, p: float, gamma: float | None = ...
) -> dict[str, Any]: ...
def noise_variables(boundary_spins: list[int], noise: str) -> list[int]: ...
def noise_eigenvalues_all(sites: int, noise: str) -> list[list[int]]: ...
def observable_eigenvalues(
    sites: int,
    family: str,
    origin: int = ...,
    separation: int | None = ...,
) -> list[int]: ...
def standard_normals(seed: int, domain: str, global_id: int, count: int) -> list[float]: ...
def stream_uniforms(seed: int, domain: str, global_id: int, count: int) -> list[float]: ...
def generate_record(
    boundary_spins: list[int],
    noise: str,
    measurement: str,
    p: float,
    seed: int,
    global_id: int,
    gamma: float | None = ...,
) -> dict[str, Any]: ...
def lattice_couplings(regularization: str, delta_tau: float | None) -> tuple[float, float]: ...
def clean_configurations(
    lx: int,
    lt: int,
    kx: float,
    kt: float,
    seed: int,
    thermalization_sweeps: int,
    saving_interval: int,
    count: int,
) -> list[list[int]]: ...
def boundary_from_packed(lx: int, lt: int, packed: list[int]) -> list[int]: ...
def configuration_observables(
    lx: int,
    lt: int,
    kx: float,
    kt: float,
    packed: list[int],
    separations: list[int],
) -> dict[str, Any]: ...
def posterior_observables(
    lx: int,
    lt: int,
    kx: float,
    kt: float,
    noise: str,
    record_couplings: list[float],
    planted_configuration: list[int],
    initial_configuration: list[int],
    update: str,
    seed: int,
    global_id: int,
    stream_label: str,
    decorrelation_gap: int,
    measurements: int,
    saving_interval: int,
    separations: list[int],
    retain_trace: bool,
    tnmc_bond_dimension: int,
) -> dict[str, Any]: ...
def posterior_observables_batch(
    lx: int,
    lt: int,
    kx: float,
    kt: float,
    noise: str,
    record_couplings: list[list[float]],
    planted_configurations: list[list[int]],
    initial_configurations: list[list[int]],
    update: str,
    seed: int,
    global_ids: list[int],
    stream_labels: list[str],
    decorrelation_gap: int,
    measurements: list[int],
    saving_interval: int,
    separations: list[int],
    retain_traces: list[bool],
    tnmc_bond_dimension: int,
    workers: int,
) -> list[dict[str, Any]]: ...
def benchmark_update_method(
    lx: int,
    lt: int,
    kx: float,
    kt: float,
    noise: str,
    record_couplings: list[float],
    planted_configuration: list[int],
    update: str,
    seed: int,
    global_id: int,
    warmup_sweeps: int,
    speed_sweeps: int,
    probes: int,
    probe_interval: int,
    thermalization_sweeps: int,
    thermalization_measurements: int,
    chains: int,
    tnmc_bond_dimension: int,
    workers: int,
) -> dict[str, Any]: ...
