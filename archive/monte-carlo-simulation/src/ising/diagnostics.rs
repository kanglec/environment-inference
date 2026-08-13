use std::time::Instant;

use rayon::prelude::*;

use crate::error::{DcftError, Result, require_at_least};
use crate::rng::{Rng64, RngDomain, StreamId};
use crate::throughput;

use super::disorder::BoundaryRandomIsingModel;
use super::lattice::SpinLattice;
use super::model::{CleanIsingModel, IsingModel};
use super::params::IsingContext;
use super::update::{FixedDisorderSampler, FixedDisorderUpdate, IsingSampler, IsingUpdateMethod};
use super::workflow::{
    NoiseKind, build_boundary_disorder, context_from_input, mu_from_noise_probability,
};

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerBenchmarkInput {
    pub lx: usize,
    pub lt: usize,
    pub seed: u64,
    pub noise: NoiseKind,
    pub p: f64,
    pub clean_therm_sweeps: usize,
    pub disorder_therm_sweeps: usize,
    pub disorder_updates: Vec<FixedDisorderUpdate>,
    pub delta_tau: Option<f64>,
    pub parallel_chains: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerBenchmarkReport {
    pub lx: usize,
    pub lt: usize,
    pub kx: f64,
    pub kt: f64,
    pub seed: u64,
    pub noise: NoiseKind,
    pub p: f64,
    pub mu: f64,
    pub clean: SpeedBenchmarkReport,
    pub disorder: Vec<SpeedBenchmarkReport>,
    pub parallel: Vec<ParallelBenchmarkReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AutocorrelationDiagnosticInput {
    pub lx: usize,
    pub lt: usize,
    pub seed: u64,
    pub noise: NoiseKind,
    pub p: f64,
    pub clean_therm_sweeps: usize,
    pub disorder_therm_sweeps: usize,
    pub disorder_updates: Vec<FixedDisorderUpdate>,
    pub delta_tau: Option<f64>,
    pub probes: usize,
    pub probe_interval_sweeps: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AutocorrelationDiagnosticReport {
    pub lx: usize,
    pub lt: usize,
    pub kx: f64,
    pub kt: f64,
    pub seed: u64,
    pub noise: NoiseKind,
    pub p: f64,
    pub mu: f64,
    pub clean: StageBenchmarkReport,
    pub disorder: Vec<StageBenchmarkReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParallelBenchmarkReport {
    pub stage: &'static str,
    pub update: &'static str,
    pub rayon_threads: usize,
    pub chains: usize,
    pub sweeps_per_chain: usize,
    pub serial_elapsed_seconds: f64,
    pub parallel_elapsed_seconds: f64,
    pub serial_chains_per_second: f64,
    pub parallel_chains_per_second: f64,
    pub speedup: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalizationDiagnosticInput {
    pub lx: usize,
    pub lt: usize,
    pub seed: u64,
    pub noise: NoiseKind,
    pub p: f64,
    pub clean_therm_sweeps: usize,
    pub disorder_therm_sweeps: usize,
    pub disorder_updates: Vec<FixedDisorderUpdate>,
    pub delta_tau: Option<f64>,
    pub chains: usize,
    pub measurements: usize,
    pub skip_sweeps: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalizationDiagnosticReport {
    pub lx: usize,
    pub lt: usize,
    pub kx: f64,
    pub kt: f64,
    pub seed: u64,
    pub noise: NoiseKind,
    pub p: f64,
    pub mu: f64,
    pub clean: MultiChainDiagnosticReport,
    pub disorder: Vec<MultiChainDiagnosticReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageBenchmarkReport {
    pub stage: &'static str,
    pub update: &'static str,
    pub measured_sweeps: usize,
    pub autocorrelation_sweeps: usize,
    pub elapsed_seconds: f64,
    pub probe_elapsed_seconds: f64,
    pub analysis_elapsed_seconds: f64,
    pub total_elapsed_seconds: f64,
    pub sweeps_per_second: f64,
    pub spin_updates_per_second: f64,
    pub probe_interval_sweeps: usize,
    pub probes: usize,
    pub cutoff_lag: Option<usize>,
    pub tau_int_sweeps: f64,
    pub autocorrelation: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeedBenchmarkReport {
    pub stage: &'static str,
    pub update: &'static str,
    pub measured_sweeps: usize,
    pub elapsed_seconds: f64,
    pub sweeps_per_second: f64,
    pub spin_updates_per_second: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiChainDiagnosticReport {
    pub stage: &'static str,
    pub update: &'static str,
    pub disorder_sample_id: Option<u64>,
    pub chains: usize,
    pub therm_sweeps: usize,
    pub measurements: usize,
    pub skip_sweeps: usize,
    pub elapsed_seconds: f64,
    pub measurement_records_per_second: f64,
    pub observables: Vec<ObservableDiagnosticReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservableDiagnosticReport {
    pub observable: &'static str,
    pub mean: f64,
    pub r_hat: f64,
    pub chain_mean_min: f64,
    pub chain_mean_max: f64,
    pub chain_mean_std: f64,
    pub mean_within_chain_std: f64,
}

pub fn benchmark_sampler(input: &SamplerBenchmarkInput) -> Result<SamplerBenchmarkReport> {
    validate_benchmark_input(input)?;
    let context = context_from_input(input.lx, input.lt, input.delta_tau)?;
    let mu = mu_from_noise_probability(input.p)?;

    let clean_result = benchmark_clean_speed(input, &context)?;
    let mut disorder = Vec::with_capacity(input.disorder_updates.len());
    let mut parallel = Vec::with_capacity(input.disorder_updates.len());
    for update in &input.disorder_updates {
        disorder.push(benchmark_disorder_speed(
            input,
            &context,
            &clean_result.boundary_spins,
            mu,
            *update,
        )?);
        parallel.push(benchmark_parallel_disorder(
            input,
            &context,
            &clean_result.boundary_spins,
            mu,
            *update,
        )?);
    }

    let couplings = context.couplings();
    Ok(SamplerBenchmarkReport {
        lx: input.lx,
        lt: input.lt,
        kx: couplings.kx(),
        kt: couplings.kt(),
        seed: input.seed,
        noise: input.noise,
        p: input.p,
        mu,
        clean: clean_result.report,
        disorder,
        parallel,
    })
}

pub fn diagnose_autocorrelation(
    input: &AutocorrelationDiagnosticInput,
) -> Result<AutocorrelationDiagnosticReport> {
    validate_autocorrelation_input(input)?;
    let context = context_from_input(input.lx, input.lt, input.delta_tau)?;
    let mu = mu_from_noise_probability(input.p)?;

    let clean_result = diagnose_clean_autocorrelation(input, &context)?;
    let mut disorder = Vec::with_capacity(input.disorder_updates.len());
    for update in &input.disorder_updates {
        disorder.push(diagnose_disorder_autocorrelation(
            input,
            &context,
            &clean_result.boundary_spins,
            mu,
            *update,
        )?);
    }
    let couplings = context.couplings();
    Ok(AutocorrelationDiagnosticReport {
        lx: input.lx,
        lt: input.lt,
        kx: couplings.kx(),
        kt: couplings.kt(),
        seed: input.seed,
        noise: input.noise,
        p: input.p,
        mu,
        clean: clean_result.report,
        disorder,
    })
}

pub fn diagnose_thermalization(
    input: &ThermalizationDiagnosticInput,
) -> Result<ThermalizationDiagnosticReport> {
    validate_thermalization_input(input)?;
    let context = context_from_input(input.lx, input.lt, input.delta_tau)?;
    let mu = mu_from_noise_probability(input.p)?;
    let clean_result = diagnose_clean_chains(input, &context)?;
    let mut disorder = Vec::with_capacity(input.disorder_updates.len());
    for update in &input.disorder_updates {
        disorder.push(diagnose_disorder_chains(
            input,
            &context,
            &clean_result.boundary_spins,
            mu,
            *update,
        )?);
    }
    let couplings = context.couplings();

    Ok(ThermalizationDiagnosticReport {
        lx: input.lx,
        lt: input.lt,
        kx: couplings.kx(),
        kt: couplings.kt(),
        seed: input.seed,
        noise: input.noise,
        p: input.p,
        mu,
        clean: clean_result.report,
        disorder,
    })
}

pub fn estimate_spin_autocorrelation(
    configurations: &[Vec<i8>],
    interval_sweeps: usize,
) -> Result<(Vec<f64>, Option<usize>, f64)> {
    require_at_least("configuration count", configurations.len(), 2)?;
    require_at_least("probe_interval_sweeps", interval_sweeps, 1)?;
    let spin_count = configurations[0].len();
    let mut packed = PackedSpinConfigurations::with_capacity(configurations.len(), spin_count)?;
    for configuration in configurations {
        packed.push(configuration)?;
    }
    packed.estimate_autocorrelation(interval_sweeps)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedSpinConfigurations {
    spin_count: usize,
    word_count: usize,
    last_word_mask: u64,
    samples: Vec<Vec<u64>>,
    spin_sums: Vec<i64>,
}

impl PackedSpinConfigurations {
    fn with_capacity(capacity: usize, spin_count: usize) -> Result<Self> {
        require_at_least("spin count", spin_count, 1)?;
        let word_count = spin_count.div_ceil(64);
        let last_word_bits = spin_count % 64;
        let last_word_mask = if last_word_bits == 0 {
            u64::MAX
        } else {
            (1_u64 << last_word_bits) - 1
        };
        Ok(Self {
            spin_count,
            word_count,
            last_word_mask,
            samples: Vec::with_capacity(capacity),
            spin_sums: vec![0; spin_count],
        })
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn push(&mut self, spins: &[i8]) -> Result<()> {
        if spins.len() != self.spin_count {
            return Err(DcftError::invalid_parameter(
                "all configurations must have the same spin count",
            ));
        }

        let mut packed = vec![0_u64; self.word_count];
        for (index, spin) in spins.iter().copied().enumerate() {
            match spin {
                1 => {
                    packed[index / 64] |= 1_u64 << (index % 64);
                    self.spin_sums[index] += 1;
                }
                -1 => {
                    self.spin_sums[index] -= 1;
                }
                _ => {
                    return Err(DcftError::invalid_parameter(
                        "spins must be Ising values -1 or 1",
                    ));
                }
            }
        }
        self.samples.push(packed);
        Ok(())
    }

    fn estimate_autocorrelation(
        &self,
        interval_sweeps: usize,
    ) -> Result<(Vec<f64>, Option<usize>, f64)> {
        require_at_least("configuration count", self.samples.len(), 2)?;
        require_at_least("probe_interval_sweeps", interval_sweeps, 1)?;

        let sample_count = self.samples.len();
        let means = self
            .spin_sums
            .iter()
            .map(|sum| *sum as f64 / sample_count as f64)
            .collect::<Vec<_>>();
        let mean_square_sum = means.iter().map(|mean| mean * mean).sum::<f64>();
        let variance = (self.spin_count as f64 - mean_square_sum) / self.spin_count as f64;
        if variance <= f64::EPSILON {
            return Ok((vec![1.0], None, 0.5 * interval_sweeps as f64));
        }

        let mut rho = Vec::with_capacity(sample_count);
        rho.push(1.0);
        let mut sum = 0.5;
        let mut cutoff_lag = None;
        let mut first_lag_spin_sums = vec![0_i64; self.spin_count];
        let mut last_lag_spin_sums = vec![0_i64; self.spin_count];
        for lag in 1..sample_count {
            self.add_sample_to_spin_sums(&mut first_lag_spin_sums, lag - 1);
            self.add_sample_to_spin_sums(&mut last_lag_spin_sums, sample_count - lag);

            let pair_count = sample_count - lag;
            let dot_sum = (0..pair_count)
                .map(|start| self.spin_dot(start, start + lag))
                .sum::<i64>();
            let endpoint_mean_dot = means
                .iter()
                .enumerate()
                .map(|(index, mean)| {
                    let paired_spin_sum = 2 * self.spin_sums[index]
                        - first_lag_spin_sums[index]
                        - last_lag_spin_sums[index];
                    mean * paired_spin_sum as f64
                })
                .sum::<f64>();
            let covariance = (dot_sum as f64 - endpoint_mean_dot
                + pair_count as f64 * mean_square_sum)
                / (pair_count * self.spin_count) as f64;
            let autocorrelation = covariance / variance;
            rho.push(autocorrelation);
            if autocorrelation <= 0.0 {
                cutoff_lag = Some(lag);
                break;
            }
            sum += autocorrelation;
        }
        Ok((rho, cutoff_lag, sum * interval_sweeps as f64))
    }

    fn add_sample_to_spin_sums(&self, sums: &mut [i64], sample_index: usize) {
        let sample = &self.samples[sample_index];
        for (index, sum) in sums.iter_mut().enumerate() {
            if sample[index / 64] & (1_u64 << (index % 64)) == 0 {
                *sum -= 1;
            } else {
                *sum += 1;
            }
        }
    }

    fn spin_dot(&self, first_index: usize, second_index: usize) -> i64 {
        let first = &self.samples[first_index];
        let second = &self.samples[second_index];
        let mut differences = 0_u32;
        for word_index in 0..self.word_count {
            let mut difference = first[word_index] ^ second[word_index];
            if word_index + 1 == self.word_count {
                difference &= self.last_word_mask;
            }
            differences += difference.count_ones();
        }
        self.spin_count as i64 - 2 * i64::from(differences)
    }
}

struct CleanDiagnosticResult {
    report: MultiChainDiagnosticReport,
    boundary_spins: Vec<i8>,
}

fn diagnose_clean_chains(
    input: &ThermalizationDiagnosticInput,
    context: &IsingContext,
) -> Result<CleanDiagnosticResult> {
    let model = CleanIsingModel::new(context.clone());
    let started = Instant::now();
    let chains = (0..input.chains)
        .into_par_iter()
        .map(|chain_id| {
            let mut rng = Rng64::stream(
                input.seed,
                StreamId::for_sample(RngDomain::CleanChain, chain_id as u64),
            );
            let mut lattice = SpinLattice::random(context.spec().clone(), &mut rng);
            IsingSampler::new(IsingUpdateMethod::Wolff, input.clean_therm_sweeps).apply(
                &model,
                &mut lattice,
                &mut rng,
            )?;

            let skip = IsingSampler::new(IsingUpdateMethod::Wolff, input.skip_sweeps);
            let mut measurements = ChainMeasurements::with_capacity(input.measurements);
            for _ in 0..input.measurements {
                skip.apply(&model, &mut lattice, &mut rng)?;
                record_observables(&model, &lattice, context, &mut measurements);
            }
            Ok((measurements, lattice.boundary_spins()))
        })
        .collect::<Result<Vec<_>>>()?;
    let boundary_spins = chains[0].1.clone();
    let chains = chains
        .into_iter()
        .map(|(measurements, _boundary_spins)| measurements)
        .collect::<Vec<_>>();
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let report = multi_chain_report(
        MultiChainRun {
            stage: "clean",
            update: IsingUpdateMethod::Wolff.as_str(),
            disorder_sample_id: None,
            chains: input.chains,
            therm_sweeps: input.clean_therm_sweeps,
            measurements: input.measurements,
            skip_sweeps: input.skip_sweeps,
            elapsed_seconds,
        },
        &chains,
    )?;

    Ok(CleanDiagnosticResult {
        report,
        boundary_spins,
    })
}

fn diagnose_disorder_chains(
    input: &ThermalizationDiagnosticInput,
    context: &IsingContext,
    boundary_spins: &[i8],
    mu: f64,
    update: FixedDisorderUpdate,
) -> Result<MultiChainDiagnosticReport> {
    let disorder_sample_id = 0;
    let disorder = build_boundary_disorder(
        input.noise,
        boundary_spins,
        mu,
        input.seed,
        disorder_sample_id,
    );
    let model = match input.noise {
        NoiseKind::Z => BoundaryRandomIsingModel::boundary_fields(context.clone(), disorder)?,
        NoiseKind::Zz => BoundaryRandomIsingModel::boundary_bonds_x(context.clone(), disorder)?,
    };
    let started = Instant::now();
    let chains = (0..input.chains)
        .into_par_iter()
        .map(|chain_id| {
            run_disorder_measurement_chain(
                &model,
                context,
                DisorderMeasurementChainInput {
                    seed: input.seed,
                    noise: input.noise,
                    update,
                    chain_id: chain_id as u64,
                    therm_sweeps: input.disorder_therm_sweeps,
                    measurements: input.measurements,
                    skip_sweeps: input.skip_sweeps,
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    multi_chain_report(
        MultiChainRun {
            stage: "disorder",
            update: update.as_str(),
            disorder_sample_id: Some(disorder_sample_id),
            chains: input.chains,
            therm_sweeps: input.disorder_therm_sweeps,
            measurements: input.measurements,
            skip_sweeps: input.skip_sweeps,
            elapsed_seconds,
        },
        &chains,
    )
}

fn benchmark_parallel_disorder(
    input: &SamplerBenchmarkInput,
    context: &IsingContext,
    boundary_spins: &[i8],
    mu: f64,
    update: FixedDisorderUpdate,
) -> Result<ParallelBenchmarkReport> {
    let chains = input
        .parallel_chains
        .unwrap_or_else(rayon::current_num_threads);
    require_at_least("parallel_chains", chains, 1)?;
    let disorder = build_boundary_disorder(input.noise, boundary_spins, mu, input.seed, 0);
    let model = match input.noise {
        NoiseKind::Z => BoundaryRandomIsingModel::boundary_fields(context.clone(), disorder)?,
        NoiseKind::Zz => BoundaryRandomIsingModel::boundary_bonds_x(context.clone(), disorder)?,
    };

    let serial_started = Instant::now();
    for chain_id in 0..chains {
        run_disorder_thermalization_chain(
            &model,
            context,
            input.seed,
            input.noise,
            update,
            chain_id as u64,
            input.disorder_therm_sweeps,
        )?;
    }
    let serial_elapsed_seconds = serial_started.elapsed().as_secs_f64();

    let parallel_started = Instant::now();
    (0..chains)
        .into_par_iter()
        .map(|chain_id| {
            run_disorder_thermalization_chain(
                &model,
                context,
                input.seed,
                input.noise,
                update,
                chain_id as u64,
                input.disorder_therm_sweeps,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let parallel_elapsed_seconds = parallel_started.elapsed().as_secs_f64();

    let serial_chains_per_second = throughput(chains, serial_elapsed_seconds);
    let parallel_chains_per_second = throughput(chains, parallel_elapsed_seconds);
    Ok(ParallelBenchmarkReport {
        stage: "disorder",
        update: update.as_str(),
        rayon_threads: rayon::current_num_threads(),
        chains,
        sweeps_per_chain: input.disorder_therm_sweeps,
        serial_elapsed_seconds,
        parallel_elapsed_seconds,
        serial_chains_per_second,
        parallel_chains_per_second,
        speedup: speedup_ratio(serial_elapsed_seconds, parallel_elapsed_seconds),
    })
}

fn run_disorder_thermalization_chain(
    model: &BoundaryRandomIsingModel,
    context: &IsingContext,
    seed: u64,
    noise: NoiseKind,
    update: FixedDisorderUpdate,
    chain_id: u64,
    therm_sweeps: usize,
) -> Result<()> {
    let mut rng = Rng64::stream(
        seed,
        StreamId::for_sample(update.rng_domain(noise), chain_id),
    );
    let mut lattice = SpinLattice::random(context.spec().clone(), &mut rng);
    let mut therm_sampler =
        FixedDisorderSampler::new(update, therm_sweeps, context.spec().site_count());
    therm_sampler
        .apply(model, &mut lattice, &mut rng)
        .map(|_| ())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DisorderMeasurementChainInput {
    seed: u64,
    noise: NoiseKind,
    update: FixedDisorderUpdate,
    chain_id: u64,
    therm_sweeps: usize,
    measurements: usize,
    skip_sweeps: usize,
}

fn run_disorder_measurement_chain(
    model: &BoundaryRandomIsingModel,
    context: &IsingContext,
    input: DisorderMeasurementChainInput,
) -> Result<ChainMeasurements> {
    let mut rng = Rng64::stream(
        input.seed,
        StreamId::for_sample(input.update.rng_domain(input.noise), input.chain_id),
    );
    let mut lattice = SpinLattice::random(context.spec().clone(), &mut rng);
    let mut therm_sampler = FixedDisorderSampler::new(
        input.update,
        input.therm_sweeps,
        context.spec().site_count(),
    );
    therm_sampler.apply(model, &mut lattice, &mut rng)?;

    let mut records = ChainMeasurements::with_capacity(input.measurements);
    let mut skip_sampler =
        FixedDisorderSampler::new(input.update, input.skip_sweeps, context.spec().site_count());
    for _ in 0..input.measurements {
        skip_sampler.apply(model, &mut lattice, &mut rng)?;
        record_observables(model, &lattice, context, &mut records);
    }
    Ok(records)
}

#[derive(Debug, Clone, PartialEq)]
struct ChainMeasurements {
    energy: Vec<f64>,
    magnetization: Vec<f64>,
    boundary_magnetization: Vec<f64>,
}

impl ChainMeasurements {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            energy: Vec::with_capacity(capacity),
            magnetization: Vec::with_capacity(capacity),
            boundary_magnetization: Vec::with_capacity(capacity),
        }
    }
}

fn record_observables<M: IsingModel>(
    model: &M,
    lattice: &SpinLattice,
    context: &IsingContext,
    measurements: &mut ChainMeasurements,
) {
    measurements
        .energy
        .push(model.energy(lattice) / context.spec().site_count() as f64);
    measurements.magnetization.push(lattice.magnetization());
    measurements
        .boundary_magnetization
        .push(lattice.boundary_magnetization());
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MultiChainRun {
    stage: &'static str,
    update: &'static str,
    disorder_sample_id: Option<u64>,
    chains: usize,
    therm_sweeps: usize,
    measurements: usize,
    skip_sweeps: usize,
    elapsed_seconds: f64,
}

fn multi_chain_report(
    run: MultiChainRun,
    chain_measurements: &[ChainMeasurements],
) -> Result<MultiChainDiagnosticReport> {
    let measurement_records = run
        .chains
        .checked_mul(run.measurements)
        .ok_or_else(|| DcftError::invalid_parameter("diagnostic measurement count overflows"))?;

    Ok(MultiChainDiagnosticReport {
        stage: run.stage,
        update: run.update,
        disorder_sample_id: run.disorder_sample_id,
        chains: run.chains,
        therm_sweeps: run.therm_sweeps,
        measurements: run.measurements,
        skip_sweeps: run.skip_sweeps,
        elapsed_seconds: run.elapsed_seconds,
        measurement_records_per_second: throughput(measurement_records, run.elapsed_seconds),
        observables: vec![
            diagnose_observable("energy", chain_measurements, |chain| &chain.energy),
            diagnose_observable("magnetization", chain_measurements, |chain| {
                &chain.magnetization
            }),
            diagnose_observable("boundary_magnetization", chain_measurements, |chain| {
                &chain.boundary_magnetization
            }),
        ],
    })
}

fn diagnose_observable<F>(
    observable: &'static str,
    chains: &[ChainMeasurements],
    values: F,
) -> ObservableDiagnosticReport
where
    F: Fn(&ChainMeasurements) -> &[f64],
{
    let chain_values = chains.iter().map(values).collect::<Vec<_>>();
    let chain_means = chain_values
        .iter()
        .map(|values| mean(values))
        .collect::<Vec<_>>();
    let chain_variances = chain_values
        .iter()
        .map(|values| sample_variance(values, mean(values)))
        .collect::<Vec<_>>();
    let mean = mean(&chain_means);
    let chain_mean_std = sample_variance(&chain_means, mean).sqrt();
    let mean_within_chain_std = chain_variances
        .iter()
        .map(|variance| variance.sqrt())
        .sum::<f64>()
        / chain_variances.len() as f64;

    ObservableDiagnosticReport {
        observable,
        mean,
        r_hat: potential_scale_reduction(&chain_means, &chain_variances, chain_values[0].len()),
        chain_mean_min: chain_means.iter().copied().fold(f64::INFINITY, f64::min),
        chain_mean_max: chain_means
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        chain_mean_std,
        mean_within_chain_std,
    }
}

fn potential_scale_reduction(
    chain_means: &[f64],
    chain_variances: &[f64],
    measurements: usize,
) -> f64 {
    let within = mean(chain_variances);
    let between = measurements as f64 * sample_variance(chain_means, mean(chain_means));
    if within <= f64::EPSILON {
        return if between <= f64::EPSILON {
            1.0
        } else {
            f64::INFINITY
        };
    }
    let measurements = measurements as f64;
    let variance_estimate = ((measurements - 1.0) / measurements) * within + between / measurements;
    (variance_estimate / within).sqrt().max(1.0)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64
}

struct CleanAutocorrelationResult {
    report: StageBenchmarkReport,
    boundary_spins: Vec<i8>,
}

struct CleanSpeedResult {
    report: SpeedBenchmarkReport,
    boundary_spins: Vec<i8>,
}

fn benchmark_clean_speed(
    input: &SamplerBenchmarkInput,
    context: &IsingContext,
) -> Result<CleanSpeedResult> {
    let model = CleanIsingModel::new(context.clone());
    let mut rng = Rng64::stream(input.seed, StreamId::for_sample(RngDomain::CleanChain, 0));
    let mut lattice = SpinLattice::random(context.spec().clone(), &mut rng);
    let report = run_stage_speed_benchmark(
        "clean",
        IsingUpdateMethod::Wolff,
        input.clean_therm_sweeps,
        &model,
        &mut lattice,
        &mut rng,
    )?;
    Ok(CleanSpeedResult {
        report,
        boundary_spins: lattice.boundary_spins(),
    })
}

fn benchmark_disorder_speed(
    input: &SamplerBenchmarkInput,
    context: &IsingContext,
    boundary_spins: &[i8],
    mu: f64,
    update: FixedDisorderUpdate,
) -> Result<SpeedBenchmarkReport> {
    let disorder = build_boundary_disorder(input.noise, boundary_spins, mu, input.seed, 0);
    let model = match input.noise {
        NoiseKind::Z => BoundaryRandomIsingModel::boundary_fields(context.clone(), disorder)?,
        NoiseKind::Zz => BoundaryRandomIsingModel::boundary_bonds_x(context.clone(), disorder)?,
    };
    let mut rng = Rng64::stream(
        input.seed,
        StreamId::for_sample(update.rng_domain(input.noise), 0),
    );
    let mut lattice = SpinLattice::random(context.spec().clone(), &mut rng);
    run_fixed_disorder_speed_benchmark(
        update,
        input.disorder_therm_sweeps,
        &model,
        &mut lattice,
        &mut rng,
    )
}

fn diagnose_clean_autocorrelation(
    input: &AutocorrelationDiagnosticInput,
    context: &IsingContext,
) -> Result<CleanAutocorrelationResult> {
    let model = CleanIsingModel::new(context.clone());
    let mut rng = Rng64::stream(input.seed, StreamId::for_sample(RngDomain::CleanChain, 0));
    let mut lattice = SpinLattice::random(context.spec().clone(), &mut rng);
    let report = run_stage_autocorrelation_diagnostic(
        StageAutocorrelationRun {
            stage: "clean",
            update: IsingUpdateMethod::Wolff,
            sweeps: input.clean_therm_sweeps,
        },
        input,
        &model,
        &mut lattice,
        &mut rng,
    )?;
    Ok(CleanAutocorrelationResult {
        report,
        boundary_spins: lattice.boundary_spins(),
    })
}

fn diagnose_disorder_autocorrelation(
    input: &AutocorrelationDiagnosticInput,
    context: &IsingContext,
    boundary_spins: &[i8],
    mu: f64,
    update: FixedDisorderUpdate,
) -> Result<StageBenchmarkReport> {
    let disorder = build_boundary_disorder(input.noise, boundary_spins, mu, input.seed, 0);
    let model = match input.noise {
        NoiseKind::Z => BoundaryRandomIsingModel::boundary_fields(context.clone(), disorder)?,
        NoiseKind::Zz => BoundaryRandomIsingModel::boundary_bonds_x(context.clone(), disorder)?,
    };
    let mut rng = Rng64::stream(
        input.seed,
        StreamId::for_sample(update.rng_domain(input.noise), 0),
    );
    let mut lattice = SpinLattice::random(context.spec().clone(), &mut rng);
    run_fixed_disorder_autocorrelation_diagnostic(
        FixedDisorderAutocorrelationRun {
            stage: "disorder",
            update,
            sweeps: input.disorder_therm_sweeps,
        },
        input,
        &model,
        &mut lattice,
        &mut rng,
    )
}

fn run_stage_speed_benchmark<M: IsingModel>(
    stage: &'static str,
    update: IsingUpdateMethod,
    sweeps: usize,
    model: &M,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
) -> Result<SpeedBenchmarkReport> {
    require_at_least("sweeps", sweeps, 1)?;
    let started = Instant::now();
    IsingSampler::new(update, sweeps).apply(model, lattice, rng)?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let sweeps_per_second = throughput(sweeps, elapsed_seconds);
    let spin_updates_per_second = throughput(sweeps * model.spec().site_count(), elapsed_seconds);

    Ok(SpeedBenchmarkReport {
        stage,
        update: update.as_str(),
        measured_sweeps: sweeps,
        elapsed_seconds,
        sweeps_per_second,
        spin_updates_per_second,
    })
}

fn run_fixed_disorder_speed_benchmark(
    update: FixedDisorderUpdate,
    sweeps: usize,
    model: &BoundaryRandomIsingModel,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
) -> Result<SpeedBenchmarkReport> {
    require_at_least("sweeps", sweeps, 1)?;
    let started = Instant::now();
    let mut sampler = FixedDisorderSampler::new(update, sweeps, model.spec().site_count());
    sampler.apply(model, lattice, rng)?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let sweeps_per_second = throughput(sweeps, elapsed_seconds);
    let spin_updates_per_second = throughput(sweeps * model.spec().site_count(), elapsed_seconds);

    Ok(SpeedBenchmarkReport {
        stage: "disorder",
        update: update.as_str(),
        measured_sweeps: sweeps,
        elapsed_seconds,
        sweeps_per_second,
        spin_updates_per_second,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StageAutocorrelationRun {
    stage: &'static str,
    update: IsingUpdateMethod,
    sweeps: usize,
}

fn run_stage_autocorrelation_diagnostic<M>(
    run: StageAutocorrelationRun,
    input: &AutocorrelationDiagnosticInput,
    model: &M,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
) -> Result<StageBenchmarkReport>
where
    M: IsingModel,
{
    require_at_least("therm_sweeps", run.sweeps, 1)?;
    let interval = input.probe_interval_sweeps;
    let started = Instant::now();
    IsingSampler::new(run.update, run.sweeps).apply(model, lattice, rng)?;
    let elapsed_seconds = started.elapsed().as_secs_f64();

    let autocorrelation_sweeps = (input.probes - 1)
        .checked_mul(interval)
        .ok_or_else(|| DcftError::invalid_parameter("autocorrelation sweep count overflows"))?;
    let sampler = IsingSampler::new(run.update, interval);
    let mut configurations =
        PackedSpinConfigurations::with_capacity(input.probes, model.spec().site_count())?;
    let probe_started = Instant::now();
    configurations.push(lattice.spins())?;
    for _ in 1..input.probes {
        sampler.apply(model, lattice, rng)?;
        configurations.push(lattice.spins())?;
    }
    let probe_elapsed_seconds = probe_started.elapsed().as_secs_f64();

    let analysis_started = Instant::now();
    let (autocorrelation, cutoff_lag, tau_int_sweeps) =
        configurations.estimate_autocorrelation(interval)?;
    let analysis_elapsed_seconds = analysis_started.elapsed().as_secs_f64();
    let total_elapsed_seconds = elapsed_seconds + probe_elapsed_seconds + analysis_elapsed_seconds;
    let sweeps_per_second = throughput(run.sweeps, elapsed_seconds);
    let spin_updates_per_second =
        throughput(run.sweeps * model.spec().site_count(), elapsed_seconds);

    Ok(StageBenchmarkReport {
        stage: run.stage,
        update: run.update.as_str(),
        measured_sweeps: run.sweeps,
        autocorrelation_sweeps,
        elapsed_seconds,
        probe_elapsed_seconds,
        analysis_elapsed_seconds,
        total_elapsed_seconds,
        sweeps_per_second,
        spin_updates_per_second,
        probe_interval_sweeps: interval,
        probes: configurations.len(),
        cutoff_lag,
        tau_int_sweeps,
        autocorrelation,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedDisorderAutocorrelationRun {
    stage: &'static str,
    update: FixedDisorderUpdate,
    sweeps: usize,
}

fn run_fixed_disorder_autocorrelation_diagnostic(
    run: FixedDisorderAutocorrelationRun,
    input: &AutocorrelationDiagnosticInput,
    model: &BoundaryRandomIsingModel,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
) -> Result<StageBenchmarkReport> {
    require_at_least("therm_sweeps", run.sweeps, 1)?;
    let interval = input.probe_interval_sweeps;
    let started = Instant::now();
    let mut therm_sampler =
        FixedDisorderSampler::new(run.update, run.sweeps, model.spec().site_count());
    therm_sampler.apply(model, lattice, rng)?;
    let elapsed_seconds = started.elapsed().as_secs_f64();

    let autocorrelation_sweeps = (input.probes - 1)
        .checked_mul(interval)
        .ok_or_else(|| DcftError::invalid_parameter("autocorrelation sweep count overflows"))?;
    let mut configurations =
        PackedSpinConfigurations::with_capacity(input.probes, model.spec().site_count())?;
    let probe_started = Instant::now();
    configurations.push(lattice.spins())?;
    let mut interval_sampler =
        FixedDisorderSampler::new(run.update, interval, model.spec().site_count());
    for _ in 1..input.probes {
        interval_sampler.apply(model, lattice, rng)?;
        configurations.push(lattice.spins())?;
    }
    let probe_elapsed_seconds = probe_started.elapsed().as_secs_f64();

    let analysis_started = Instant::now();
    let (autocorrelation, cutoff_lag, tau_int_sweeps) =
        configurations.estimate_autocorrelation(interval)?;
    let analysis_elapsed_seconds = analysis_started.elapsed().as_secs_f64();
    let total_elapsed_seconds = elapsed_seconds + probe_elapsed_seconds + analysis_elapsed_seconds;
    let sweeps_per_second = throughput(run.sweeps, elapsed_seconds);
    let spin_updates_per_second =
        throughput(run.sweeps * model.spec().site_count(), elapsed_seconds);

    Ok(StageBenchmarkReport {
        stage: run.stage,
        update: run.update.as_str(),
        measured_sweeps: run.sweeps,
        autocorrelation_sweeps,
        elapsed_seconds,
        probe_elapsed_seconds,
        analysis_elapsed_seconds,
        total_elapsed_seconds,
        sweeps_per_second,
        spin_updates_per_second,
        probe_interval_sweeps: interval,
        probes: configurations.len(),
        cutoff_lag,
        tau_int_sweeps,
        autocorrelation,
    })
}

fn validate_benchmark_input(input: &SamplerBenchmarkInput) -> Result<()> {
    require_at_least("clean_therm_sweeps", input.clean_therm_sweeps, 1)?;
    require_at_least("disorder_therm_sweeps", input.disorder_therm_sweeps, 1)?;
    require_at_least("disorder_updates", input.disorder_updates.len(), 1)?;
    if let Some(chains) = input.parallel_chains {
        require_at_least("parallel_chains", chains, 1)?;
    }
    Ok(())
}

fn validate_autocorrelation_input(input: &AutocorrelationDiagnosticInput) -> Result<()> {
    require_at_least("clean_therm_sweeps", input.clean_therm_sweeps, 1)?;
    require_at_least("disorder_therm_sweeps", input.disorder_therm_sweeps, 1)?;
    require_at_least("disorder_updates", input.disorder_updates.len(), 1)?;
    require_at_least("probes", input.probes, 2)?;
    require_at_least("probe_interval_sweeps", input.probe_interval_sweeps, 1)?;
    Ok(())
}

fn validate_thermalization_input(input: &ThermalizationDiagnosticInput) -> Result<()> {
    require_at_least("clean_therm_sweeps", input.clean_therm_sweeps, 1)?;
    require_at_least("disorder_therm_sweeps", input.disorder_therm_sweeps, 1)?;
    require_at_least("disorder_updates", input.disorder_updates.len(), 1)?;
    require_at_least("chains", input.chains, 2)?;
    require_at_least("measurements", input.measurements, 2)?;
    require_at_least("skip_sweeps", input.skip_sweeps, 1)?;
    Ok(())
}

fn speedup_ratio(serial_seconds: f64, parallel_seconds: f64) -> f64 {
    if serial_seconds == 0.0 && parallel_seconds == 0.0 {
        1.0
    } else if parallel_seconds == 0.0 {
        f64::INFINITY
    } else {
        serial_seconds / parallel_seconds
    }
}
