use std::fs;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{DcftError, Result, require_at_least, require_finite};
use crate::rng::{Rng64, RngDomain, StreamId};

use super::lattice::SpinLattice;
use super::model::CleanIsingModel;
use super::observable::{FixedDisorderMeasureInput, measure_fixed_disorder_sample};
use super::params::{IsingContext, IsingCouplings, LatticeSpec};
use super::storage::{
    AggregateFileHeader, AggregateRecord, CleanBoundarySample, CleanSampleFile,
    CleanSampleFileHeader, write_aggregate_binary, write_clean_boundary_sample,
};
use super::update::{FixedDisorderUpdate, IsingSampler, IsingUpdateMethod};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseKind {
    Z,
    Zz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementKind {
    Heterodyne,
    Homodyne,
    LocalX,
}

impl MeasurementKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Heterodyne => "heterodyne",
            Self::Homodyne => "homodyne",
            Self::LocalX => "local-x",
        }
    }

    fn rng_domain(self, noise: NoiseKind) -> RngDomain {
        match (self, noise) {
            (Self::Heterodyne, NoiseKind::Z) => RngDomain::ZDisorderNormals,
            (Self::Heterodyne, NoiseKind::Zz) => RngDomain::ZzDisorderNormals,
            (Self::Homodyne, NoiseKind::Z) => RngDomain::ZHomodyneDisorder,
            (Self::Homodyne, NoiseKind::Zz) => RngDomain::ZzHomodyneDisorder,
            (Self::LocalX, NoiseKind::Z) => RngDomain::ZLocalXDisorder,
            (Self::LocalX, NoiseKind::Zz) => RngDomain::ZzLocalXDisorder,
        }
    }
}

impl NoiseKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Z => "z",
            Self::Zz => "zz",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryStageInput {
    pub lx: usize,
    pub lt: usize,
    pub seed: u64,
    pub sample_count: usize,
    pub clean_therm_sweeps: usize,
    pub clean_skip_sweeps: usize,
    pub delta_tau: Option<f64>,
    pub out_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasureStageInput {
    pub clean_path: PathBuf,
    pub noise: NoiseKind,
    pub measurement: MeasurementKind,
    pub p: f64,
    pub sample_start: u64,
    pub sample_count: usize,
    pub disorder_update: FixedDisorderUpdate,
    pub disorder_therm_sweeps: usize,
    pub measurements: usize,
    pub skip_sweeps: usize,
    pub out_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRange {
    pub task_id: usize,
    pub sample_start: u64,
    pub sample_count: usize,
}

pub fn mu_from_noise_probability(p: f64) -> Result<f64> {
    validate_noise_probability(p)?;
    Ok(-(1.0 - 2.0 * p).ln())
}

pub fn measurement_parameter(measurement: MeasurementKind, p: f64) -> Result<f64> {
    let heterodyne_mu = mu_from_noise_probability(p)?;
    Ok(match measurement {
        MeasurementKind::Heterodyne => heterodyne_mu,
        MeasurementKind::Homodyne => 2.0 * heterodyne_mu,
        MeasurementKind::LocalX => {
            let kappa = 2.0 * (p * (1.0 - p)).sqrt();
            kappa.atanh()
        }
    })
}

pub fn generate_clean_stage(input: &BoundaryStageInput) -> Result<()> {
    require_at_least("sample_count", input.sample_count, 1)?;
    let context = context_from_input(input.lx, input.lt, input.delta_tau)?;
    let couplings = context.couplings();
    let header = CleanSampleFileHeader {
        lx: input.lx,
        lt: input.lt,
        kx: couplings.kx(),
        kt: couplings.kt(),
        delta_tau: input.delta_tau,
        seed: input.seed,
        sample_count: input.sample_count,
        clean_therm_sweeps: input.clean_therm_sweeps,
        clean_skip_sweeps: input.clean_skip_sweeps,
    };

    let clean_model = CleanIsingModel::new(context.clone());
    let mut rng = Rng64::stream(input.seed, StreamId::for_sample(RngDomain::CleanChain, 0));
    let mut lattice = SpinLattice::random(context.spec().clone(), &mut rng);
    IsingSampler::new(IsingUpdateMethod::Wolff, input.clean_therm_sweeps).apply(
        &clean_model,
        &mut lattice,
        &mut rng,
    )?;

    let mut writer = CleanSampleFile::create(&input.out_path, &header)?;
    let skip = IsingSampler::new(IsingUpdateMethod::Wolff, input.clean_skip_sweeps);
    for sample_id in 0..input.sample_count as u64 {
        skip.apply(&clean_model, &mut lattice, &mut rng)?;
        let sample = CleanBoundarySample {
            sample_id,
            boundary_spins: lattice.boundary_spins(),
        };
        write_clean_boundary_sample(&mut writer, &sample, context.spec().lx())?;
    }
    Ok(())
}

pub fn measure_stage(input: &MeasureStageInput) -> Result<Vec<AggregateRecord>> {
    validate_measure_input(input)?;
    let mut clean_file = CleanSampleFile::open(&input.clean_path)?;
    let clean_header = clean_file.header().clone();
    checked_sample_range(
        input.sample_start,
        input.sample_count,
        clean_header.sample_count,
    )?;
    let context = clean_header.context()?;
    let samples = (input.sample_start..input.sample_start + input.sample_count as u64)
        .map(|sample_id| clean_file.read_sample(sample_id))
        .collect::<Result<Vec<_>>>()?;
    let records = samples
        .par_iter()
        .map(|sample| {
            measure_fixed_disorder_sample(
                &context,
                sample,
                FixedDisorderMeasureInput {
                    noise: input.noise,
                    measurement: input.measurement,
                    p: input.p,
                    seed: clean_header.seed,
                    update: input.disorder_update,
                    therm_sweeps: input.disorder_therm_sweeps,
                    measurements: input.measurements,
                    skip_sweeps: input.skip_sweeps,
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(records)
}

pub fn measure_stage_to_file(input: &MeasureStageInput) -> Result<usize> {
    validate_measure_input(input)?;
    let clean_file = CleanSampleFile::open(&input.clean_path)?;
    let clean_header = clean_file.header().clone();
    drop(clean_file);
    let records = measure_stage(input)?;
    let header = AggregateFileHeader::from_measurement_input(input, &clean_header)?;
    write_aggregate_binary(&input.out_path, &header, &records)?;
    Ok(records.len())
}

pub(super) fn build_measurement_disorder(
    noise: NoiseKind,
    measurement: MeasurementKind,
    boundary_spins: &[i8],
    p: f64,
    seed: u64,
    sample_id: u64,
) -> Result<Vec<f64>> {
    let mut rng = Rng64::stream(
        seed,
        StreamId::for_sample(measurement.rng_domain(noise), sample_id),
    );
    let parameter = measurement_parameter(measurement, p)?;
    let variables = match noise {
        NoiseKind::Z => boundary_spins.to_vec(),
        NoiseKind::Zz => (0..boundary_spins.len())
            .map(|x| {
                let right = if x + 1 < boundary_spins.len() {
                    x + 1
                } else {
                    0
                };
                boundary_spins[x] * boundary_spins[right]
            })
            .collect(),
    };
    Ok(match measurement {
        MeasurementKind::Heterodyne | MeasurementKind::Homodyne => {
            let sigma = parameter.sqrt();
            variables
                .iter()
                .map(|variable| parameter.mul_add(f64::from(*variable), sigma * rng.normal()))
                .collect()
        }
        MeasurementKind::LocalX => {
            let kappa = 2.0 * (p * (1.0 - p)).sqrt();
            variables
                .iter()
                .map(|variable| {
                    let probability_plus = 0.5 * (1.0 + kappa * f64::from(*variable));
                    if rng.uniform() < probability_plus {
                        parameter
                    } else {
                        -parameter
                    }
                })
                .collect()
        }
    })
}

pub(super) fn build_boundary_disorder(
    noise: NoiseKind,
    boundary_spins: &[i8],
    mu: f64,
    seed: u64,
    sample_id: u64,
) -> Vec<f64> {
    let mut rng = Rng64::stream(
        seed,
        StreamId::for_sample(MeasurementKind::Heterodyne.rng_domain(noise), sample_id),
    );
    let sigma = mu.sqrt();
    let variables = match noise {
        NoiseKind::Z => boundary_spins.to_vec(),
        NoiseKind::Zz => (0..boundary_spins.len())
            .map(|x| boundary_spins[x] * boundary_spins[(x + 1) % boundary_spins.len()])
            .collect(),
    };
    variables
        .iter()
        .map(|variable| mu.mul_add(f64::from(*variable), sigma * rng.normal()))
        .collect()
}

pub(super) fn context_from_input(
    lx: usize,
    lt: usize,
    delta_tau: Option<f64>,
) -> Result<IsingContext> {
    let spec = LatticeSpec::new(lx, lt)?;
    let couplings = match delta_tau {
        Some(delta_tau) => IsingCouplings::from_critical_tfim_delta_tau(delta_tau)?,
        None => IsingCouplings::from_critical_tfim_isotropic(),
    };
    Ok(IsingContext::new(spec, couplings))
}

pub fn chunk_plan(sample_count: usize, chunks: usize) -> Result<Vec<ChunkRange>> {
    require_at_least("sample_count", sample_count, 1)?;
    require_at_least("chunks", chunks, 1)?;
    if chunks > sample_count {
        return Err(DcftError::invalid_parameter(
            "chunks must be <= sample_count so every array task has work",
        ));
    }
    let base = sample_count / chunks;
    let rem = sample_count % chunks;
    let mut ranges = Vec::with_capacity(chunks);
    for task_id in 0..chunks {
        let task_count = if task_id < rem { base + 1 } else { base };
        let start = if task_id < rem {
            task_id * (base + 1)
        } else {
            rem * (base + 1) + (task_id - rem) * base
        };
        ranges.push(ChunkRange {
            task_id,
            sample_start: start as u64,
            sample_count: task_count,
        });
    }
    Ok(ranges)
}

pub fn read_run_spec_chunk_plan(path: impl AsRef<Path>) -> Result<Vec<ChunkRange>> {
    let text = fs::read_to_string(path)?;
    let sample_count = parse_toml_usize(&text, "samples")?;
    let chunks = parse_toml_usize(&text, "chunks")?;
    chunk_plan(sample_count, chunks)
}

fn parse_toml_usize(text: &str, key: &str) -> Result<usize> {
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some((left, right)) = line.split_once('=')
            && left.trim() == key
        {
            return right.trim().parse::<usize>().map_err(|_| {
                DcftError::invalid_parameter(format!("could not parse `{key}` as usize"))
            });
        }
    }
    Err(DcftError::invalid_parameter(format!(
        "run spec is missing `{key}`"
    )))
}

fn validate_measure_input(input: &MeasureStageInput) -> Result<()> {
    validate_noise_probability(input.p)?;
    require_at_least("sample_count", input.sample_count, 1)?;
    require_at_least("measurements", input.measurements, 1)?;
    Ok(())
}

fn validate_noise_probability(p: f64) -> Result<()> {
    require_finite("p", p)?;
    if !(0.0..0.5).contains(&p) {
        return Err(DcftError::invalid_parameter("p must satisfy 0 <= p < 0.5"));
    }
    Ok(())
}

fn checked_sample_range(start: u64, count: usize, available: usize) -> Result<()> {
    let end = start
        .checked_add(count as u64)
        .ok_or_else(|| DcftError::invalid_parameter("sample id range overflows u64"))?;
    if end > available as u64 {
        return Err(DcftError::invalid_parameter(format!(
            "requested sample range {start}..{end} exceeds clean sample count {available}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_measurement_disorder_has_the_requested_law() {
        let p = 0.2;
        let variables = vec![1; 100_000];

        for measurement in [MeasurementKind::Heterodyne, MeasurementKind::Homodyne] {
            let values =
                build_measurement_disorder(NoiseKind::Z, measurement, &variables, p, 91, 3)
                    .unwrap();
            let parameter = measurement_parameter(measurement, p).unwrap();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f64>()
                / values.len() as f64;
            assert!((mean - parameter).abs() < 0.02);
            assert!((variance - parameter).abs() < 0.02);
        }

        let values =
            build_measurement_disorder(NoiseKind::Z, MeasurementKind::LocalX, &variables, p, 91, 3)
                .unwrap();
        let parameter = measurement_parameter(MeasurementKind::LocalX, p).unwrap();
        assert!(values.iter().all(|value| value.abs() == parameter));
        let observed_plus =
            values.iter().filter(|value| **value > 0.0).count() as f64 / values.len() as f64;
        let expected_plus = 0.5 * (1.0 + 2.0 * (p * (1.0 - p)).sqrt());
        assert!((observed_plus - expected_plus).abs() < 0.01);
    }

    #[test]
    fn all_measurements_are_uninformative_at_zero_noise() {
        for measurement in [
            MeasurementKind::Heterodyne,
            MeasurementKind::Homodyne,
            MeasurementKind::LocalX,
        ] {
            let values =
                build_measurement_disorder(NoiseKind::Z, measurement, &[1, -1, 1, -1], 0.0, 3, 7)
                    .unwrap();
            assert_eq!(values, vec![0.0; 4]);
        }
    }
}
