mod benchmark;
mod lattice;
mod model;
mod tnmc;
mod update;

pub use benchmark::{UpdateBenchmark, benchmark_update};
pub use lattice::Lattice;
pub use model::{Couplings, Model};
pub use update::{Update, UpdateStats};

use crate::error::{DcftError, Result};
use crate::observables::{Accumulator, ObservableSummary};
use crate::physics::Noise;
use crate::rng::Rng64;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorResult {
    pub observables: ObservableSummary,
    pub updates: UpdateStats,
    pub final_configuration: Vec<u8>,
    pub trace: DiagnosticTrace,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DiagnosticTrace {
    pub energy: Vec<f64>,
    pub magnetization: Vec<f64>,
    pub boundary_magnetization: Vec<f64>,
    pub planted_spin_overlap: Vec<f64>,
    pub planted_bond_overlap: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorJob {
    pub record_couplings: Vec<f64>,
    pub planted_configuration: Vec<u8>,
    pub initial_configuration: Vec<u8>,
    pub global_id: u64,
    pub stream_label: String,
    pub measurements: usize,
    pub retain_trace: bool,
}

pub fn generate_clean_configurations(
    lx: usize,
    lt: usize,
    couplings: Couplings,
    seed: u64,
    thermalization_sweeps: usize,
    saving_interval: usize,
    count: usize,
) -> Result<Vec<Vec<u8>>> {
    if count == 0 {
        return Err(DcftError::invalid("clean sample count must be positive"));
    }
    if saving_interval == 0 {
        return Err(DcftError::invalid("clean saving interval must be positive"));
    }
    let mut rng = Rng64::stream(seed, "clean-chain", 0);
    let mut lattice = Lattice::random(lx, lt, &mut rng)?;
    let clean = Model::clean(lx, lt, couplings)?;
    let mut statistics = UpdateStats::default();
    for _ in 0..thermalization_sweeps {
        update::clean_wolff(&clean, &mut lattice, &mut rng, &mut statistics)?;
    }

    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        for _ in 0..saving_interval {
            update::clean_wolff(&clean, &mut lattice, &mut rng, &mut statistics)?;
        }
        output.push(lattice.pack());
    }
    Ok(output)
}

pub fn sample_posterior(
    lx: usize,
    lt: usize,
    couplings: Couplings,
    noise: Noise,
    record_couplings: Vec<f64>,
    planted_configuration: &[u8],
    initial_configuration: &[u8],
    update_method: Update,
    seed: u64,
    global_id: u64,
    stream_label: &str,
    decorrelation_gap: usize,
    measurements: usize,
    saving_interval: usize,
    separations: &[usize],
    retain_trace: bool,
) -> Result<PosteriorResult> {
    if decorrelation_gap == 0 {
        return Err(DcftError::invalid(
            "a positive posterior decorrelation or convergence-thermalization gap is mandatory",
        ));
    }
    if measurements == 0 || saving_interval == 0 {
        return Err(DcftError::invalid(
            "posterior measurements and saving interval must be positive",
        ));
    }
    if stream_label.is_empty() {
        return Err(DcftError::invalid("posterior stream label cannot be empty"));
    }
    let planted = Lattice::unpack(lx, lt, planted_configuration)?;
    let planted_boundary = planted.boundary().to_vec();
    let planted_bonds: Vec<i8> = planted_boundary
        .iter()
        .zip(planted_boundary.iter().cycle().skip(1))
        .take(lx)
        .map(|(left, right)| left * right)
        .collect();
    let model = Model::posterior(lx, lt, couplings, noise, record_couplings)?;
    let mut lattice = Lattice::unpack(lx, lt, initial_configuration)?;
    let domain = format!("posterior/{}/{stream_label}", update_method.as_str());
    let mut rng = Rng64::stream(seed, &domain, global_id);
    let mut statistics = UpdateStats::default();

    for _ in 0..decorrelation_gap {
        update_method.apply(&model, &mut lattice, &mut rng, &mut statistics)?;
    }
    let mut accumulator = Accumulator::new(lx, lt, separations)?;
    let mut trace = DiagnosticTrace::default();
    for _ in 0..measurements {
        for _ in 0..saving_interval {
            update_method.apply(&model, &mut lattice, &mut rng, &mut statistics)?;
        }
        if retain_trace {
            trace.energy.push(model.energy(&lattice));
            trace.magnetization.push(
                lattice
                    .spins()
                    .iter()
                    .map(|spin| f64::from(*spin))
                    .sum::<f64>()
                    / lattice.site_count() as f64,
            );
            trace.boundary_magnetization.push(
                lattice
                    .boundary()
                    .iter()
                    .map(|spin| f64::from(*spin))
                    .sum::<f64>()
                    / lx as f64,
            );
            trace.planted_spin_overlap.push(
                planted_boundary
                    .iter()
                    .zip(lattice.boundary())
                    .map(|(planted, current)| f64::from(planted * current))
                    .sum::<f64>()
                    / lx as f64,
            );
            trace.planted_bond_overlap.push(
                planted_bonds
                    .iter()
                    .enumerate()
                    .map(|(x, planted)| {
                        f64::from(planted * lattice.get(x, 0) * lattice.get((x + 1) % lx, 0))
                    })
                    .sum::<f64>()
                    / lx as f64,
            );
        }
        accumulator.observe(&model, &lattice);
    }
    Ok(PosteriorResult {
        observables: accumulator.finish()?,
        updates: statistics,
        final_configuration: lattice.pack(),
        trace,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn sample_posteriors_parallel(
    lx: usize,
    lt: usize,
    couplings: Couplings,
    noise: Noise,
    update_method: Update,
    seed: u64,
    decorrelation_gap: usize,
    saving_interval: usize,
    separations: &[usize],
    jobs: Vec<PosteriorJob>,
    workers: usize,
) -> Result<Vec<PosteriorResult>> {
    if jobs.is_empty() {
        return Err(DcftError::invalid("posterior batch cannot be empty"));
    }
    if workers == 0 {
        return Err(DcftError::invalid(
            "posterior batch workers must be positive",
        ));
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|error| DcftError::invalid(format!("cannot build Rayon pool: {error}")))?;
    pool.install(|| {
        jobs.into_par_iter()
            .map(|job| {
                sample_posterior(
                    lx,
                    lt,
                    couplings,
                    noise,
                    job.record_couplings,
                    &job.planted_configuration,
                    &job.initial_configuration,
                    update_method,
                    seed,
                    job.global_id,
                    &job.stream_label,
                    decorrelation_gap,
                    job.measurements,
                    saving_interval,
                    separations,
                    job.retain_trace,
                )
            })
            .collect()
    })
}
