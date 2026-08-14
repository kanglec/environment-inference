use std::time::Instant;

use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use crate::error::{DcftError, Result};
use crate::physics::Noise;
use crate::rng::Rng64;

use super::{Couplings, Lattice, Model, Update, UpdateStats};

#[derive(Debug, Clone, PartialEq)]
pub struct UpdateBenchmark {
    pub update: &'static str,
    pub workers: usize,
    pub speed_sweeps: usize,
    pub speed_elapsed_seconds: f64,
    pub energy_trace: Vec<f64>,
    pub boundary_magnetization_trace: Vec<f64>,
    pub planted_overlap_trace: Vec<f64>,
    pub thermalization_energy: Vec<Vec<f64>>,
    pub thermalization_boundary_magnetization: Vec<Vec<f64>>,
    pub thermalization_planted_overlap: Vec<Vec<f64>>,
    pub thermalization_serial_elapsed_seconds: f64,
    pub thermalization_elapsed_seconds: f64,
    pub updates: UpdateStats,
}

#[derive(Debug, Clone, PartialEq)]
struct ChainTrace {
    energy: Vec<f64>,
    boundary_magnetization: Vec<f64>,
    planted_overlap: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn benchmark_update(
    lx: usize,
    lt: usize,
    couplings: Couplings,
    noise: Noise,
    record_couplings: Vec<f64>,
    planted_configuration: &[u8],
    update: Update,
    seed: u64,
    global_id: u64,
    warmup_sweeps: usize,
    speed_sweeps: usize,
    probes: usize,
    probe_interval: usize,
    thermalization_sweeps: usize,
    thermalization_measurements: usize,
    chains: usize,
    workers: usize,
) -> Result<UpdateBenchmark> {
    if speed_sweeps == 0 || probes < 2 || probe_interval == 0 {
        return Err(DcftError::invalid(
            "speed sweeps and probe interval must be positive and probes must be at least two",
        ));
    }
    if thermalization_measurements < 4 || chains < 2 || workers == 0 {
        return Err(DcftError::invalid(
            "thermalization needs at least two chains, four measurements, and one worker",
        ));
    }
    let model = Model::posterior(lx, lt, couplings, noise, record_couplings)?;
    let planted = Lattice::unpack(lx, lt, planted_configuration)?;
    let planted_boundary = planted.boundary().to_vec();
    let mut lattice = planted;
    let trace_domain = format!("benchmark-trace/{}", update.as_str());
    let mut rng = Rng64::stream(seed, &trace_domain, global_id);
    let mut updates = UpdateStats::default();
    apply_sweeps(
        update,
        &model,
        &mut lattice,
        &mut rng,
        &mut updates,
        warmup_sweeps,
    )?;

    let speed_start = Instant::now();
    apply_sweeps(
        update,
        &model,
        &mut lattice,
        &mut rng,
        &mut updates,
        speed_sweeps,
    )?;
    let speed_elapsed_seconds = speed_start.elapsed().as_secs_f64();

    let mut energy_trace = Vec::with_capacity(probes);
    let mut boundary_magnetization_trace = Vec::with_capacity(probes);
    let mut planted_overlap_trace = Vec::with_capacity(probes);
    for _ in 0..probes {
        apply_sweeps(
            update,
            &model,
            &mut lattice,
            &mut rng,
            &mut updates,
            probe_interval,
        )?;
        let (energy, boundary_magnetization, planted_overlap) =
            observables(&model, &lattice, noise, &planted_boundary);
        energy_trace.push(energy);
        boundary_magnetization_trace.push(boundary_magnetization);
        planted_overlap_trace.push(planted_overlap);
    }

    let serial_start = Instant::now();
    let serial_thermalization = (0..chains)
        .map(|chain| {
            thermalization_chain(
                &model,
                noise,
                &planted_boundary,
                update,
                seed,
                global_id,
                chain,
                thermalization_sweeps,
                thermalization_measurements,
                probe_interval,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let thermalization_serial_elapsed_seconds = serial_start.elapsed().as_secs_f64();
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|error| DcftError::invalid(format!("cannot build Rayon pool: {error}")))?;
    let thermalization_start = Instant::now();
    let thermalization = pool.install(|| {
        (0..chains)
            .into_par_iter()
            .map(|chain| {
                thermalization_chain(
                    &model,
                    noise,
                    &planted_boundary,
                    update,
                    seed,
                    global_id,
                    chain,
                    thermalization_sweeps,
                    thermalization_measurements,
                    probe_interval,
                )
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let thermalization_elapsed_seconds = thermalization_start.elapsed().as_secs_f64();
    if thermalization != serial_thermalization {
        return Err(DcftError::invalid(
            "parallel thermalization changed deterministic chain results",
        ));
    }

    Ok(UpdateBenchmark {
        update: update.as_str(),
        workers,
        speed_sweeps,
        speed_elapsed_seconds,
        energy_trace,
        boundary_magnetization_trace,
        planted_overlap_trace,
        thermalization_energy: thermalization
            .iter()
            .map(|trace| trace.energy.clone())
            .collect(),
        thermalization_boundary_magnetization: thermalization
            .iter()
            .map(|trace| trace.boundary_magnetization.clone())
            .collect(),
        thermalization_planted_overlap: thermalization
            .iter()
            .map(|trace| trace.planted_overlap.clone())
            .collect(),
        thermalization_serial_elapsed_seconds,
        thermalization_elapsed_seconds,
        updates,
    })
}

fn apply_sweeps(
    update: Update,
    model: &Model,
    lattice: &mut Lattice,
    rng: &mut Rng64,
    statistics: &mut UpdateStats,
    sweeps: usize,
) -> Result<()> {
    for _ in 0..sweeps {
        update.apply(model, lattice, rng, statistics)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn thermalization_chain(
    model: &Model,
    noise: Noise,
    planted_boundary: &[i8],
    update: Update,
    seed: u64,
    global_id: u64,
    chain: usize,
    thermalization_sweeps: usize,
    measurements: usize,
    saving_interval: usize,
) -> Result<ChainTrace> {
    let domain = format!(
        "benchmark-thermalization/{}/record={global_id}",
        update.as_str()
    );
    let mut rng = Rng64::stream(seed, &domain, chain as u64);
    let mut lattice = match chain {
        0 => Lattice::new(model.lx(), model.lt(), vec![1; model.lx() * model.lt()])?,
        1 => Lattice::new(model.lx(), model.lt(), vec![-1; model.lx() * model.lt()])?,
        _ => Lattice::random(model.lx(), model.lt(), &mut rng)?,
    };
    let mut statistics = UpdateStats::default();
    apply_sweeps(
        update,
        model,
        &mut lattice,
        &mut rng,
        &mut statistics,
        thermalization_sweeps,
    )?;
    let mut trace = ChainTrace {
        energy: Vec::with_capacity(measurements),
        boundary_magnetization: Vec::with_capacity(measurements),
        planted_overlap: Vec::with_capacity(measurements),
    };
    for _ in 0..measurements {
        apply_sweeps(
            update,
            model,
            &mut lattice,
            &mut rng,
            &mut statistics,
            saving_interval,
        )?;
        let (energy, boundary_magnetization, planted_overlap) =
            observables(model, &lattice, noise, planted_boundary);
        trace.energy.push(energy);
        trace.boundary_magnetization.push(boundary_magnetization);
        trace.planted_overlap.push(planted_overlap);
    }
    Ok(trace)
}

fn observables(
    model: &Model,
    lattice: &Lattice,
    noise: Noise,
    planted_boundary: &[i8],
) -> (f64, f64, f64) {
    let lx = lattice.lx();
    let boundary_magnetization = lattice
        .boundary()
        .iter()
        .map(|spin| f64::from(*spin))
        .sum::<f64>()
        / lx as f64;
    let planted_overlap = match noise {
        Noise::Z => {
            planted_boundary
                .iter()
                .zip(lattice.boundary())
                .map(|(planted, current)| f64::from(planted * current))
                .sum::<f64>()
                / lx as f64
        }
        Noise::Zz => {
            (0..lx)
                .map(|x| {
                    f64::from(
                        planted_boundary[x]
                            * planted_boundary[(x + 1) % lx]
                            * lattice.get(x, 0)
                            * lattice.get((x + 1) % lx, 0),
                    )
                })
                .sum::<f64>()
                / lx as f64
        }
    };
    (
        model.energy(lattice),
        boundary_magnetization,
        planted_overlap,
    )
}
