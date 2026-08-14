mod error;
pub mod mc;
pub mod observables;
pub mod physics;
pub mod rng;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::error::DcftError;
use crate::mc::{Couplings, Lattice, Update};
use crate::physics::{Measurement, Noise, ProtocolParameters};

fn python_error(error: DcftError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn rng_contract() -> (&'static str, &'static str) {
    (rng::RNG_ALGORITHM, rng::STREAM_KEY_VERSION)
}

#[pyfunction]
fn measurement_registry() -> Vec<&'static str> {
    vec!["heterodyne", "homodyne", "gaussian", "local-x"]
}

#[pyfunction]
fn noise_registry() -> Vec<&'static str> {
    vec!["z", "zz"]
}

#[pyfunction]
fn update_registry() -> Vec<&'static str> {
    vec![
        "metropolis",
        "sequential-metropolis",
        "metropolis-global",
        "corrected-wolff",
        "tnmc",
    ]
}

#[pyfunction]
fn observable_registry() -> Vec<&'static str> {
    vec![
        "energy",
        "magnetization",
        "boundary-magnetization",
        "spin-profile",
        "bond-profile",
        "spin-correlator",
        "bond-correlator",
    ]
}

#[pyfunction(signature = (measurement, p, gamma=None))]
fn protocol_parameters(
    py: Python<'_>,
    measurement: &str,
    p: f64,
    gamma: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let kind = Measurement::from_name(measurement, gamma).map_err(python_error)?;
    let parameters = ProtocolParameters::new(kind, p).map_err(python_error)?;
    let output = PyDict::new(py);
    output.set_item("measurement", kind.as_str())?;
    output.set_item("p", parameters.p)?;
    output.set_item("lambda", parameters.lambda)?;
    output.set_item("gamma", parameters.gamma)?;
    output.set_item("kappa", parameters.kappa)?;
    output.set_item("coupling", parameters.coupling)?;
    output.set_item("error_probability", parameters.error_probability)?;
    Ok(output.unbind())
}

#[pyfunction]
fn noise_variables(boundary_spins: Vec<i8>, noise: &str) -> PyResult<Vec<i8>> {
    let noise = Noise::parse(noise).map_err(python_error)?;
    physics::noise_variables(&boundary_spins, noise).map_err(python_error)
}

#[pyfunction]
fn noise_eigenvalues_all(sites: usize, noise: &str) -> PyResult<Vec<Vec<i8>>> {
    let noise = Noise::parse(noise).map_err(python_error)?;
    physics::all_noise_eigenvalues(sites, noise).map_err(python_error)
}

#[pyfunction(signature = (sites, family, origin=0, separation=None))]
fn observable_eigenvalues(
    sites: usize,
    family: &str,
    origin: usize,
    separation: Option<usize>,
) -> PyResult<Vec<i8>> {
    physics::observable_eigenvalues(sites, family, origin, separation).map_err(python_error)
}

#[pyfunction]
fn standard_normals(seed: u64, domain: &str, global_id: u64, count: usize) -> Vec<f64> {
    rng::standard_normals(seed, domain, global_id, count)
}

#[pyfunction]
fn stream_uniforms(seed: u64, domain: &str, global_id: u64, count: usize) -> Vec<f64> {
    rng::uniforms(seed, domain, global_id, count)
}

#[pyfunction(signature = (boundary_spins, noise, measurement, p, seed, global_id, gamma=None))]
fn generate_record(
    py: Python<'_>,
    boundary_spins: Vec<i8>,
    noise: &str,
    measurement: &str,
    p: f64,
    seed: u64,
    global_id: u64,
    gamma: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let noise_kind = Noise::parse(noise).map_err(python_error)?;
    let measurement_kind = Measurement::from_name(measurement, gamma).map_err(python_error)?;
    let parameters = ProtocolParameters::new(measurement_kind, p).map_err(python_error)?;
    let variables = physics::noise_variables(&boundary_spins, noise_kind).map_err(python_error)?;
    let output = PyDict::new(py);
    output.set_item("variables", variables.clone())?;
    output.set_item("measurement", measurement_kind.as_str())?;
    output.set_item("noise", noise_kind.as_str())?;

    if let Some(strength) = parameters.gamma {
        let domain = format!("gaussian-normal/{}", noise_kind.as_str());
        let normal = rng::standard_normals(seed, &domain, global_id, variables.len());
        let record =
            physics::gaussian_record(&variables, strength, &normal).map_err(python_error)?;
        output.set_item("raw_record", record.clone())?;
        output.set_item("record_couplings", record)?;
        output.set_item("standard_variates", normal)?;
        output.set_item("gamma", strength)?;
        output.set_item("kappa", py.None())?;
        output.set_item("coupling", py.None())?;
    } else {
        let domain = format!("local-x/{}", noise_kind.as_str());
        let uniform = rng::uniforms(seed, &domain, global_id, variables.len());
        let (outcomes, fields) =
            physics::local_x_record(&variables, parameters, &uniform).map_err(python_error)?;
        output.set_item("raw_record", outcomes)?;
        output.set_item("record_couplings", fields)?;
        output.set_item("standard_variates", uniform)?;
        output.set_item("gamma", py.None())?;
        output.set_item("kappa", parameters.kappa)?;
        output.set_item("coupling", parameters.coupling)?;
    }
    Ok(output.unbind())
}

#[pyfunction]
fn lattice_couplings(regularization: &str, delta_tau: Option<f64>) -> PyResult<(f64, f64)> {
    let couplings = match regularization {
        "isotropic" => Couplings::isotropic(),
        "trotter" => Couplings::trotter(
            delta_tau.ok_or_else(|| PyValueError::new_err("trotter requires delta_tau"))?,
        )
        .map_err(python_error)?,
        _ => {
            return Err(PyValueError::new_err(
                "regularization must be isotropic or trotter",
            ));
        }
    };
    Ok((couplings.kx, couplings.kt))
}

#[pyfunction]
fn clean_configurations(
    lx: usize,
    lt: usize,
    kx: f64,
    kt: f64,
    seed: u64,
    thermalization_sweeps: usize,
    saving_interval: usize,
    count: usize,
) -> PyResult<Vec<Vec<u8>>> {
    let couplings = Couplings::new(kx, kt).map_err(python_error)?;
    mc::generate_clean_configurations(
        lx,
        lt,
        couplings,
        seed,
        thermalization_sweeps,
        saving_interval,
        count,
    )
    .map_err(python_error)
}

#[pyfunction]
fn boundary_from_packed(lx: usize, lt: usize, packed: Vec<u8>) -> PyResult<Vec<i8>> {
    let lattice = Lattice::unpack(lx, lt, &packed).map_err(python_error)?;
    Ok(lattice.boundary().to_vec())
}

#[pyfunction]
fn configuration_observables(
    py: Python<'_>,
    lx: usize,
    lt: usize,
    kx: f64,
    kt: f64,
    packed: Vec<u8>,
    separations: Vec<usize>,
) -> PyResult<Py<PyDict>> {
    let couplings = Couplings::new(kx, kt).map_err(python_error)?;
    let lattice = Lattice::unpack(lx, lt, &packed).map_err(python_error)?;
    let model = mc::Model::clean(lx, lt, couplings).map_err(python_error)?;
    let mut accumulator =
        observables::Accumulator::new(lx, lt, &separations).map_err(python_error)?;
    accumulator.observe(&model, &lattice);
    let values = accumulator.finish().map_err(python_error)?;
    let output = PyDict::new(py);
    output.set_item("energy", values.energy)?;
    output.set_item("magnetization", values.magnetization)?;
    output.set_item("boundary_magnetization", values.boundary_magnetization)?;
    output.set_item("spin_profile", values.spin_profile)?;
    output.set_item("bond_profile", values.bond_profile)?;
    output.set_item("spin_correlator_profile", values.spin_correlator_profile)?;
    output.set_item("bond_correlator_profile", values.bond_correlator_profile)?;
    Ok(output.unbind())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn posterior_observables(
    py: Python<'_>,
    lx: usize,
    lt: usize,
    kx: f64,
    kt: f64,
    noise: &str,
    record_couplings: Vec<f64>,
    planted_configuration: Vec<u8>,
    update: &str,
    seed: u64,
    global_id: u64,
    stream_label: &str,
    decorrelation_gap: usize,
    measurements: usize,
    saving_interval: usize,
    separations: Vec<usize>,
    retain_trace: bool,
    tnmc_bond_dimension: usize,
) -> PyResult<Py<PyDict>> {
    let couplings = Couplings::new(kx, kt).map_err(python_error)?;
    let noise = Noise::parse(noise).map_err(python_error)?;
    let update = Update::parse_with_tnmc_bond_dimension(update, tnmc_bond_dimension)
        .map_err(python_error)?;
    let result = mc::sample_posterior(
        lx,
        lt,
        couplings,
        noise,
        record_couplings,
        &planted_configuration,
        update,
        seed,
        global_id,
        stream_label,
        decorrelation_gap,
        measurements,
        saving_interval,
        &separations,
        retain_trace,
    )
    .map_err(python_error)?;
    posterior_result_dictionary(py, result)
}

fn set_update_statistics(output: &Bound<'_, PyDict>, statistics: mc::UpdateStats) -> PyResult<()> {
    output.set_item("sweeps", statistics.sweeps)?;
    output.set_item("local_proposed", statistics.local_proposed)?;
    output.set_item("local_accepted", statistics.local_accepted)?;
    output.set_item("cluster_proposed", statistics.cluster_proposed)?;
    output.set_item("cluster_accepted", statistics.cluster_accepted)?;
    output.set_item("cluster_sites_proposed", statistics.cluster_sites_proposed)?;
    output.set_item("global_proposed", statistics.global_proposed)?;
    output.set_item("global_attempted", statistics.global_attempted)?;
    output.set_item("global_accepted", statistics.global_accepted)?;
    output.set_item("tnmc_proposed", statistics.tnmc_proposed)?;
    output.set_item("tnmc_accepted", statistics.tnmc_accepted)?;
    output.set_item("tnmc_sites_proposed", statistics.tnmc_sites_proposed)?;
    output.set_item(
        "tnmc_conditionals_regularized",
        statistics.tnmc_conditionals_regularized,
    )?;
    Ok(())
}

fn posterior_result_dictionary(
    py: Python<'_>,
    result: mc::PosteriorResult,
) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    let observables = result.observables;
    output.set_item("samples", observables.samples)?;
    output.set_item("energy", observables.energy)?;
    output.set_item("magnetization", observables.magnetization)?;
    output.set_item("boundary_magnetization", observables.boundary_magnetization)?;
    output.set_item("spin_profile", observables.spin_profile)?;
    output.set_item("bond_profile", observables.bond_profile)?;
    output.set_item("separations", observables.separations)?;
    output.set_item(
        "spin_correlator_profile",
        observables.spin_correlator_profile,
    )?;
    output.set_item(
        "bond_correlator_profile",
        observables.bond_correlator_profile,
    )?;
    set_update_statistics(&output, result.updates)?;
    output.set_item("final_configuration", result.final_configuration)?;
    output.set_item("energy_trace", result.trace.energy)?;
    output.set_item("magnetization_trace", result.trace.magnetization)?;
    output.set_item(
        "boundary_magnetization_trace",
        result.trace.boundary_magnetization,
    )?;
    output.set_item(
        "planted_spin_overlap_trace",
        result.trace.planted_spin_overlap,
    )?;
    output.set_item(
        "planted_bond_overlap_trace",
        result.trace.planted_bond_overlap,
    )?;
    Ok(output.unbind())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn posterior_observables_batch(
    py: Python<'_>,
    lx: usize,
    lt: usize,
    kx: f64,
    kt: f64,
    noise: &str,
    record_couplings: Vec<Vec<f64>>,
    planted_configurations: Vec<Vec<u8>>,
    update: &str,
    seed: u64,
    global_ids: Vec<u64>,
    stream_labels: Vec<String>,
    decorrelation_gap: usize,
    measurements: Vec<usize>,
    saving_interval: usize,
    separations: Vec<usize>,
    retain_traces: Vec<bool>,
    tnmc_bond_dimension: usize,
    workers: usize,
) -> PyResult<Vec<Py<PyDict>>> {
    let count = record_couplings.len();
    if count == 0
        || [
            planted_configurations.len(),
            global_ids.len(),
            stream_labels.len(),
            measurements.len(),
            retain_traces.len(),
        ]
        .into_iter()
        .any(|length| length != count)
    {
        return Err(PyValueError::new_err(
            "posterior batch arrays must have the same positive length",
        ));
    }
    let couplings = Couplings::new(kx, kt).map_err(python_error)?;
    let noise = Noise::parse(noise).map_err(python_error)?;
    let update = Update::parse_with_tnmc_bond_dimension(update, tnmc_bond_dimension)
        .map_err(python_error)?;
    let jobs = record_couplings
        .into_iter()
        .zip(planted_configurations)
        .zip(global_ids)
        .zip(stream_labels)
        .zip(measurements)
        .zip(retain_traces)
        .map(
            |(((((record, planted), global_id), stream_label), measurements), retain_trace)| {
                mc::PosteriorJob {
                    record_couplings: record,
                    planted_configuration: planted,
                    global_id,
                    stream_label,
                    measurements,
                    retain_trace,
                }
            },
        )
        .collect();
    mc::sample_posteriors_parallel(
        lx,
        lt,
        couplings,
        noise,
        update,
        seed,
        decorrelation_gap,
        saving_interval,
        &separations,
        jobs,
        workers,
    )
    .map_err(python_error)?
    .into_iter()
    .map(|result| posterior_result_dictionary(py, result))
    .collect()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn benchmark_update_method(
    py: Python<'_>,
    lx: usize,
    lt: usize,
    kx: f64,
    kt: f64,
    noise: &str,
    record_couplings: Vec<f64>,
    planted_configuration: Vec<u8>,
    update: &str,
    seed: u64,
    global_id: u64,
    warmup_sweeps: usize,
    speed_sweeps: usize,
    probes: usize,
    probe_interval: usize,
    thermalization_sweeps: usize,
    thermalization_measurements: usize,
    chains: usize,
    tnmc_bond_dimension: usize,
    workers: usize,
) -> PyResult<Py<PyDict>> {
    let couplings = Couplings::new(kx, kt).map_err(python_error)?;
    let noise = Noise::parse(noise).map_err(python_error)?;
    let update = Update::parse_with_tnmc_bond_dimension(update, tnmc_bond_dimension)
        .map_err(python_error)?;
    let report = mc::benchmark_update(
        lx,
        lt,
        couplings,
        noise,
        record_couplings,
        &planted_configuration,
        update,
        seed,
        global_id,
        warmup_sweeps,
        speed_sweeps,
        probes,
        probe_interval,
        thermalization_sweeps,
        thermalization_measurements,
        chains,
        workers,
    )
    .map_err(python_error)?;
    let output = PyDict::new(py);
    output.set_item("update", report.update)?;
    output.set_item("workers", report.workers)?;
    output.set_item("speed_sweeps", report.speed_sweeps)?;
    output.set_item("speed_elapsed_seconds", report.speed_elapsed_seconds)?;
    output.set_item("energy_trace", report.energy_trace)?;
    output.set_item(
        "boundary_magnetization_trace",
        report.boundary_magnetization_trace,
    )?;
    output.set_item("planted_overlap_trace", report.planted_overlap_trace)?;
    output.set_item("thermalization_energy", report.thermalization_energy)?;
    output.set_item(
        "thermalization_boundary_magnetization",
        report.thermalization_boundary_magnetization,
    )?;
    output.set_item(
        "thermalization_planted_overlap",
        report.thermalization_planted_overlap,
    )?;
    output.set_item(
        "thermalization_serial_elapsed_seconds",
        report.thermalization_serial_elapsed_seconds,
    )?;
    output.set_item(
        "thermalization_elapsed_seconds",
        report.thermalization_elapsed_seconds,
    )?;
    set_update_statistics(&output, report.updates)?;
    Ok(output.unbind())
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(rng_contract, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_registry, module)?)?;
    module.add_function(wrap_pyfunction!(noise_registry, module)?)?;
    module.add_function(wrap_pyfunction!(update_registry, module)?)?;
    module.add_function(wrap_pyfunction!(observable_registry, module)?)?;
    module.add_function(wrap_pyfunction!(protocol_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(noise_variables, module)?)?;
    module.add_function(wrap_pyfunction!(noise_eigenvalues_all, module)?)?;
    module.add_function(wrap_pyfunction!(observable_eigenvalues, module)?)?;
    module.add_function(wrap_pyfunction!(standard_normals, module)?)?;
    module.add_function(wrap_pyfunction!(stream_uniforms, module)?)?;
    module.add_function(wrap_pyfunction!(generate_record, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_couplings, module)?)?;
    module.add_function(wrap_pyfunction!(clean_configurations, module)?)?;
    module.add_function(wrap_pyfunction!(boundary_from_packed, module)?)?;
    module.add_function(wrap_pyfunction!(configuration_observables, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_observables, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_observables_batch, module)?)?;
    module.add_function(wrap_pyfunction!(benchmark_update_method, module)?)?;
    Ok(())
}
