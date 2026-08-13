use crate::error::{DcftError, Result, require_at_least};
use crate::rng::{Rng64, StreamId};

use super::super::disorder::BoundaryRandomIsingModel;
use super::super::lattice::SpinLattice;
use super::super::model::IsingModel;
use super::super::params::IsingContext;
use super::super::storage::{AggregateRecord, CleanBoundarySample};
use super::super::update::{FixedDisorderSampler, FixedDisorderUpdate};
use super::super::workflow::{MeasurementKind, NoiseKind, build_measurement_disorder};

#[derive(Debug, Clone)]
pub struct AggregateAccumulator {
    lx: usize,
    r_count: usize,
    samples: usize,
    energy_sum: f64,
    bulk_magnetization_sum: f64,
    boundary_spin_sum: Vec<f64>,
    boundary_bond_sum: Vec<f64>,
    spin_pair_sum: Vec<f64>,
    bond_pair_sum: Vec<f64>,
    boundary_bonds: Vec<i8>,
}

impl AggregateAccumulator {
    pub fn new(lx: usize) -> Result<Self> {
        require_at_least("lx", lx, 2)?;
        let r_count = lx / 2 + 1;
        let pair_count = r_count
            .checked_mul(lx)
            .ok_or_else(|| DcftError::invalid_parameter("correlator buffer size overflows"))?;
        Ok(Self {
            lx,
            r_count,
            samples: 0,
            energy_sum: 0.0,
            bulk_magnetization_sum: 0.0,
            boundary_spin_sum: vec![0.0; lx],
            boundary_bond_sum: vec![0.0; lx],
            spin_pair_sum: vec![0.0; pair_count],
            bond_pair_sum: vec![0.0; pair_count],
            boundary_bonds: vec![0; lx],
        })
    }

    pub fn record(
        &mut self,
        model: &BoundaryRandomIsingModel,
        lattice: &SpinLattice,
    ) -> Result<()> {
        if lattice.spec() != model.spec() {
            return Err(DcftError::invalid_parameter(
                "lattice spec does not match model lattice spec",
            ));
        }
        let boundary_spins = &lattice.spins()[..self.lx];
        self.samples = self
            .samples
            .checked_add(1)
            .ok_or_else(|| DcftError::invalid_parameter("measurement count overflows usize"))?;
        self.energy_sum += model.energy(lattice);
        self.bulk_magnetization_sum += lattice.spin_sum() as f64;
        for x in 0..self.lx {
            self.boundary_spin_sum[x] += f64::from(boundary_spins[x]);
            let right = if x + 1 < self.lx { x + 1 } else { 0 };
            let bond = boundary_spins[x] * boundary_spins[right];
            self.boundary_bonds[x] = bond;
            self.boundary_bond_sum[x] += f64::from(bond);
        }
        for r in 0..self.r_count {
            let offset = r * self.lx;
            for x in 0..self.lx {
                let y = x + r;
                let y = if y < self.lx { y } else { y - self.lx };
                self.spin_pair_sum[offset + x] += f64::from(boundary_spins[x] * boundary_spins[y]);
                self.bond_pair_sum[offset + x] +=
                    f64::from(self.boundary_bonds[x] * self.boundary_bonds[y]);
            }
        }
        Ok(())
    }

    pub fn finish(self, disorder_id: u64) -> Result<AggregateRecord> {
        require_at_least("measurements", self.samples, 1)?;
        let norm = 1.0 / self.samples as f64;
        let boundary_spin_mean = self
            .boundary_spin_sum
            .into_iter()
            .map(|value| value * norm)
            .collect::<Vec<_>>();
        let boundary_bond_mean = self
            .boundary_bond_sum
            .into_iter()
            .map(|value| value * norm)
            .collect::<Vec<_>>();
        let mut spin_corr_signed = vec![0.0; self.r_count];
        let mut bond_corr_signed = vec![0.0; self.r_count];
        let mut spin_corr_abs = vec![0.0; self.r_count];
        let mut bond_corr_abs = vec![0.0; self.r_count];
        let x_norm = 1.0 / self.lx as f64;
        for r in 0..self.r_count {
            let offset = r * self.lx;
            for x in 0..self.lx {
                let spin_mean = self.spin_pair_sum[offset + x] * norm;
                let bond_mean = self.bond_pair_sum[offset + x] * norm;
                spin_corr_signed[r] += spin_mean * x_norm;
                bond_corr_signed[r] += bond_mean * x_norm;
                spin_corr_abs[r] += spin_mean.abs() * x_norm;
                bond_corr_abs[r] += bond_mean.abs() * x_norm;
            }
        }
        Ok(AggregateRecord {
            disorder_id,
            energy_mean: self.energy_sum * norm,
            bulk_magnetization_sum_mean: self.bulk_magnetization_sum * norm,
            boundary_spin_mean,
            boundary_bond_mean,
            spin_corr_signed,
            bond_corr_signed,
            spin_corr_abs,
            bond_corr_abs,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FixedDisorderMeasureInput {
    pub noise: NoiseKind,
    pub measurement: MeasurementKind,
    pub p: f64,
    pub seed: u64,
    pub update: FixedDisorderUpdate,
    pub therm_sweeps: usize,
    pub measurements: usize,
    pub skip_sweeps: usize,
}

pub fn measure_fixed_disorder_sample(
    context: &IsingContext,
    sample: &CleanBoundarySample,
    input: FixedDisorderMeasureInput,
) -> Result<AggregateRecord> {
    require_at_least("measurements", input.measurements, 1)?;
    let disorder = build_measurement_disorder(
        input.noise,
        input.measurement,
        &sample.boundary_spins,
        input.p,
        input.seed,
        sample.sample_id,
    )?;
    let model = match input.noise {
        NoiseKind::Z => BoundaryRandomIsingModel::boundary_fields(context.clone(), disorder)?,
        NoiseKind::Zz => BoundaryRandomIsingModel::boundary_bonds_x(context.clone(), disorder)?,
    };
    let mut rng = Rng64::stream(
        input.seed,
        StreamId::for_sample(input.update.rng_domain(input.noise), sample.sample_id),
    );
    let mut lattice = SpinLattice::random(context.spec().clone(), &mut rng);
    let mut sampler = FixedDisorderSampler::new(
        input.update,
        input.therm_sweeps,
        context.spec().site_count(),
    );
    sampler.apply(&model, &mut lattice, &mut rng)?;
    let mut accumulator = AggregateAccumulator::new(context.spec().lx())?;
    for _ in 0..input.measurements {
        sampler.apply_sweeps(&model, &mut lattice, &mut rng, input.skip_sweeps)?;
        accumulator.record(&model, &lattice)?;
    }
    accumulator.finish(sample.sample_id)
}
