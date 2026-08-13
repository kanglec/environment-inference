use crate::error::{DcftError, Result};
use crate::rng::Rng64;

use super::{Lattice, Model};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Update {
    Metropolis,
    SequentialMetropolis,
    MetropolisGlobal,
    CorrectedWolff,
    Tnmc { maximum_bond_dimension: usize },
}

impl Update {
    pub fn parse(name: &str) -> Result<Self> {
        Self::parse_with_tnmc_bond_dimension(name, 16)
    }

    pub fn parse_with_tnmc_bond_dimension(
        name: &str,
        maximum_bond_dimension: usize,
    ) -> Result<Self> {
        match name {
            "metropolis" => Ok(Self::Metropolis),
            "sequential-metropolis" => Ok(Self::SequentialMetropolis),
            "metropolis-global" => Ok(Self::MetropolisGlobal),
            "corrected-wolff" => Ok(Self::CorrectedWolff),
            "tnmc" if maximum_bond_dimension > 0 => Ok(Self::Tnmc {
                maximum_bond_dimension,
            }),
            "tnmc" => Err(DcftError::invalid(
                "TNMC maximum bond dimension must be positive",
            )),
            _ => Err(DcftError::invalid(
                "unknown update; expected metropolis, sequential-metropolis, metropolis-global, corrected-wolff, or tnmc",
            )),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metropolis => "metropolis",
            Self::SequentialMetropolis => "sequential-metropolis",
            Self::MetropolisGlobal => "metropolis-global",
            Self::CorrectedWolff => "corrected-wolff",
            Self::Tnmc { .. } => "tnmc",
        }
    }

    pub fn apply(
        self,
        model: &Model,
        lattice: &mut Lattice,
        rng: &mut Rng64,
        statistics: &mut UpdateStats,
    ) -> Result<()> {
        match self {
            Self::Metropolis => random_metropolis(model, lattice, rng, statistics),
            Self::SequentialMetropolis => {
                sequential_metropolis(model, lattice, rng, statistics);
                Ok(())
            }
            Self::MetropolisGlobal => {
                random_metropolis(model, lattice, rng, statistics)?;
                statistics.global_proposed += 1;
                if rng.bool() {
                    statistics.global_attempted += 1;
                    if accept(model.global_flip_delta(lattice), rng) {
                        lattice.flip_all();
                        statistics.global_accepted += 1;
                    }
                }
                Ok(())
            }
            Self::CorrectedWolff => corrected_wolff(model, lattice, rng, statistics),
            Self::Tnmc {
                maximum_bond_dimension,
            } => super::tnmc::update(model, lattice, rng, statistics, maximum_bond_dimension),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UpdateStats {
    pub sweeps: u64,
    pub local_proposed: u64,
    pub local_accepted: u64,
    pub cluster_proposed: u64,
    pub cluster_accepted: u64,
    pub cluster_sites_proposed: u64,
    pub global_proposed: u64,
    pub global_attempted: u64,
    pub global_accepted: u64,
    pub tnmc_proposed: u64,
    pub tnmc_accepted: u64,
    pub tnmc_sites_proposed: u64,
    pub tnmc_conditionals_regularized: u64,
}

pub fn clean_wolff(
    model: &Model,
    lattice: &mut Lattice,
    rng: &mut Rng64,
    statistics: &mut UpdateStats,
) -> Result<()> {
    let cluster = grow_clean_cluster(model, lattice, rng)?;
    for (index, included) in cluster.iter().enumerate() {
        if *included {
            lattice.flip(index);
        }
    }
    let size = cluster.iter().filter(|included| **included).count();
    statistics.sweeps += 1;
    statistics.cluster_proposed += 1;
    statistics.cluster_accepted += 1;
    statistics.cluster_sites_proposed += size as u64;
    Ok(())
}

fn random_metropolis(
    model: &Model,
    lattice: &mut Lattice,
    rng: &mut Rng64,
    statistics: &mut UpdateStats,
) -> Result<()> {
    for _ in 0..lattice.site_count() {
        let index = rng.index(lattice.site_count())?;
        statistics.local_proposed += 1;
        if accept(model.flip_delta(lattice, index), rng) {
            lattice.flip(index);
            statistics.local_accepted += 1;
        }
    }
    statistics.sweeps += 1;
    Ok(())
}

fn sequential_metropolis(
    model: &Model,
    lattice: &mut Lattice,
    rng: &mut Rng64,
    statistics: &mut UpdateStats,
) {
    for index in 0..lattice.site_count() {
        statistics.local_proposed += 1;
        if accept(model.flip_delta(lattice, index), rng) {
            lattice.flip(index);
            statistics.local_accepted += 1;
        }
    }
    statistics.sweeps += 1;
}

fn corrected_wolff(
    model: &Model,
    lattice: &mut Lattice,
    rng: &mut Rng64,
    statistics: &mut UpdateStats,
) -> Result<()> {
    let cluster = grow_clean_cluster(model, lattice, rng)?;
    let size = cluster.iter().filter(|included| **included).count();
    let accepted = accept(model.cluster_disorder_delta(lattice, &cluster), rng);
    statistics.sweeps += 1;
    statistics.cluster_proposed += 1;
    statistics.cluster_sites_proposed += size as u64;
    if accepted {
        for (index, included) in cluster.iter().enumerate() {
            if *included {
                lattice.flip(index);
            }
        }
        statistics.cluster_accepted += 1;
    }
    Ok(())
}

fn grow_clean_cluster(model: &Model, lattice: &Lattice, rng: &mut Rng64) -> Result<Vec<bool>> {
    let seed = rng.index(lattice.site_count())?;
    let seed_spin = lattice.get_index(seed);
    let probability_x = 1.0 - (-2.0 * model.couplings().kx).exp();
    let probability_t = 1.0 - (-2.0 * model.couplings().kt).exp();
    let mut included = vec![false; lattice.site_count()];
    let mut stack = vec![seed];
    included[seed] = true;
    while let Some(index) = stack.pop() {
        let [left, right, down, up] = lattice.neighbors(index);
        for (neighbor, probability) in [
            (left, probability_x),
            (right, probability_x),
            (down, probability_t),
            (up, probability_t),
        ] {
            if !included[neighbor]
                && lattice.get_index(neighbor) == seed_spin
                && rng.uniform() < probability
            {
                included[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }
    Ok(included)
}

fn accept(delta: f64, rng: &mut Rng64) -> bool {
    delta <= 0.0 || rng.log_uniform() < -delta
}
