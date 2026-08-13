mod corrected_wolff;
mod metropolis;
mod wolff;

use crate::error::{DcftError, Result, require_nonnegative_finite};
use crate::rng::{Rng64, RngDomain};

use self::corrected_wolff::{
    CorrectedWolffScratch, corrected_wolff_sweep, corrected_wolff_sweep_with_scratch,
};
use self::metropolis::{metropolis_global_sweep, metropolis_sweep, sequential_metropolis_sweep};
use self::wolff::wolff_sweep;
use super::disorder::BoundaryRandomIsingModel;
use super::lattice::SpinLattice;
use super::model::IsingModel;
use super::workflow::NoiseKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsingUpdateMethod {
    Metropolis,
    Wolff,
}

impl IsingUpdateMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metropolis => "metropolis",
            Self::Wolff => "wolff",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedDisorderUpdate {
    Metropolis,
    MetropolisGlobal,
    SequentialMetropolis,
    CorrectedWolff,
}

impl FixedDisorderUpdate {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metropolis => "metropolis",
            Self::MetropolisGlobal => "metropolis-global",
            Self::SequentialMetropolis => "sequential-metropolis",
            Self::CorrectedWolff => "corrected-wolff",
        }
    }

    pub fn rng_domain(self, noise: NoiseKind) -> RngDomain {
        match (self, noise) {
            (Self::Metropolis, NoiseKind::Z) => RngDomain::ZMetropolis,
            (Self::Metropolis, NoiseKind::Zz) => RngDomain::ZzMetropolis,
            (Self::MetropolisGlobal, NoiseKind::Z) => RngDomain::ZMetropolisGlobal,
            (Self::MetropolisGlobal, NoiseKind::Zz) => RngDomain::ZzMetropolisGlobal,
            (Self::SequentialMetropolis, NoiseKind::Z) => RngDomain::ZSequentialMetropolis,
            (Self::SequentialMetropolis, NoiseKind::Zz) => RngDomain::ZzSequentialMetropolis,
            (Self::CorrectedWolff, NoiseKind::Z) => RngDomain::ZCorrectedWolff,
            (Self::CorrectedWolff, NoiseKind::Zz) => RngDomain::ZzCorrectedWolff,
        }
    }

    pub fn apply(
        self,
        model: &BoundaryRandomIsingModel,
        lattice: &mut SpinLattice,
        rng: &mut Rng64,
        sweeps: usize,
    ) -> Result<()> {
        if lattice.spec() != model.spec() {
            return Err(DcftError::invalid_parameter(
                "lattice spec does not match model lattice spec",
            ));
        }
        match self {
            Self::Metropolis => {
                metropolis_sweep(model, lattice, rng, sweeps);
                Ok(())
            }
            Self::MetropolisGlobal => {
                metropolis_global_sweep(model, lattice, rng, sweeps);
                Ok(())
            }
            Self::SequentialMetropolis => {
                sequential_metropolis_sweep(model, lattice, rng, sweeps);
                Ok(())
            }
            Self::CorrectedWolff => {
                require_nonnegative_finite("kx", model.couplings().kx())?;
                require_nonnegative_finite("kt", model.couplings().kt())?;
                corrected_wolff_sweep(model, lattice, rng, sweeps);
                Ok(())
            }
        }
    }
}

pub struct FixedDisorderSampler {
    update: FixedDisorderUpdate,
    sweeps: usize,
    corrected_wolff_scratch: Option<CorrectedWolffScratch>,
}

impl FixedDisorderSampler {
    pub fn new(update: FixedDisorderUpdate, sweeps: usize, site_count: usize) -> Self {
        let corrected_wolff_scratch = match update {
            FixedDisorderUpdate::Metropolis
            | FixedDisorderUpdate::MetropolisGlobal
            | FixedDisorderUpdate::SequentialMetropolis => None,
            FixedDisorderUpdate::CorrectedWolff => Some(CorrectedWolffScratch::new(site_count)),
        };
        Self {
            update,
            sweeps,
            corrected_wolff_scratch,
        }
    }

    pub fn apply(
        &mut self,
        model: &BoundaryRandomIsingModel,
        lattice: &mut SpinLattice,
        rng: &mut Rng64,
    ) -> Result<()> {
        self.apply_sweeps(model, lattice, rng, self.sweeps)
    }

    pub fn apply_sweeps(
        &mut self,
        model: &BoundaryRandomIsingModel,
        lattice: &mut SpinLattice,
        rng: &mut Rng64,
        sweeps: usize,
    ) -> Result<()> {
        if lattice.spec() != model.spec() {
            return Err(DcftError::invalid_parameter(
                "lattice spec does not match model lattice spec",
            ));
        }
        match self.update {
            FixedDisorderUpdate::Metropolis => {
                metropolis_sweep(model, lattice, rng, sweeps);
                Ok(())
            }
            FixedDisorderUpdate::MetropolisGlobal => {
                metropolis_global_sweep(model, lattice, rng, sweeps);
                Ok(())
            }
            FixedDisorderUpdate::SequentialMetropolis => {
                sequential_metropolis_sweep(model, lattice, rng, sweeps);
                Ok(())
            }
            FixedDisorderUpdate::CorrectedWolff => {
                require_nonnegative_finite("kx", model.couplings().kx())?;
                require_nonnegative_finite("kt", model.couplings().kt())?;
                let scratch = self
                    .corrected_wolff_scratch
                    .as_mut()
                    .expect("corrected Wolff scratch must be initialized");
                corrected_wolff_sweep_with_scratch(model, lattice, rng, sweeps, scratch);
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsingSampler {
    update_method: IsingUpdateMethod,
    sweeps: usize,
}

impl IsingSampler {
    pub fn new(update_method: IsingUpdateMethod, sweeps: usize) -> Self {
        Self {
            update_method,
            sweeps,
        }
    }

    pub fn apply<M: IsingModel>(
        &self,
        model: &M,
        lattice: &mut SpinLattice,
        rng: &mut Rng64,
    ) -> Result<()> {
        validate_update_method(model, self.update_method)?;
        if lattice.spec() != model.spec() {
            return Err(DcftError::invalid_parameter(
                "lattice spec does not match model lattice spec",
            ));
        }
        match self.update_method {
            IsingUpdateMethod::Metropolis => metropolis_sweep(model, lattice, rng, self.sweeps),
            IsingUpdateMethod::Wolff => wolff_sweep(model.couplings(), lattice, rng, self.sweeps),
        }
        Ok(())
    }
}

fn validate_update_method<M: IsingModel>(
    model: &M,
    update_method: IsingUpdateMethod,
) -> Result<()> {
    match update_method {
        IsingUpdateMethod::Metropolis => {}
        IsingUpdateMethod::Wolff => {
            if !model.supports_wolff() {
                return Err(DcftError::invalid_parameter(
                    "Wolff updates are only supported for clean Ising models",
                ));
            }
            require_nonnegative_finite("kx", model.couplings().kx())?;
            require_nonnegative_finite("kt", model.couplings().kt())?;
        }
    }
    Ok(())
}
