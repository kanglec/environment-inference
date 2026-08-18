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
    TnmcGlobal { maximum_bond_dimension: usize },
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
            "tnmc" | "tnmc-global" if maximum_bond_dimension > 0 => {
                if name == "tnmc" {
                    Ok(Self::Tnmc {
                        maximum_bond_dimension,
                    })
                } else {
                    Ok(Self::TnmcGlobal {
                        maximum_bond_dimension,
                    })
                }
            }
            "tnmc" | "tnmc-global" => Err(DcftError::invalid(
                "TNMC maximum bond dimension must be positive",
            )),
            _ => Err(DcftError::invalid(
                "unknown update; expected metropolis, sequential-metropolis, metropolis-global, corrected-wolff, tnmc, or tnmc-global",
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
            Self::TnmcGlobal { .. } => "tnmc-global",
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
                lazy_global_flip(model, lattice, rng, statistics);
                Ok(())
            }
            Self::CorrectedWolff => corrected_wolff(model, lattice, rng, statistics),
            Self::Tnmc {
                maximum_bond_dimension,
            } => super::tnmc::update(model, lattice, rng, statistics, maximum_bond_dimension),
            Self::TnmcGlobal {
                maximum_bond_dimension,
            } => {
                super::tnmc::update(model, lattice, rng, statistics, maximum_bond_dimension)?;
                lazy_global_flip(model, lattice, rng, statistics);
                Ok(())
            }
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

fn lazy_global_flip(
    model: &Model,
    lattice: &mut Lattice,
    rng: &mut Rng64,
    statistics: &mut UpdateStats,
) {
    statistics.global_proposed += 1;
    if rng.bool() {
        statistics.global_attempted += 1;
        if accept(model.global_flip_delta(lattice), rng) {
            lattice.flip_all();
            statistics.global_accepted += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::Couplings;
    use crate::physics::Noise;

    #[test]
    fn tnmc_names_distinguish_pure_and_composite_kernels() {
        assert!(matches!(
            Update::parse_with_tnmc_bond_dimension("tnmc", 3).expect("pure TNMC"),
            Update::Tnmc {
                maximum_bond_dimension: 3
            }
        ));
        assert!(matches!(
            Update::parse_with_tnmc_bond_dimension("tnmc-global", 3).expect("composite TNMC"),
            Update::TnmcGlobal {
                maximum_bond_dimension: 3
            }
        ));
    }

    #[test]
    fn pure_tnmc_does_not_add_a_global_flip() {
        let couplings = Couplings::new(0.3, 0.7).expect("valid couplings");
        let model = Model::clean(2, 2, couplings).expect("valid clean model");
        let mut lattice = Lattice::new(2, 2, vec![1; 4]).expect("valid lattice");
        let mut rng = Rng64::seeded(0x7075_7265);
        let mut statistics = UpdateStats::default();
        for _ in 0..100 {
            Update::Tnmc {
                maximum_bond_dimension: 2,
            }
            .apply(&model, &mut lattice, &mut rng, &mut statistics)
            .expect("valid TNMC update");
        }

        assert_eq!(statistics.sweeps, 100);
        assert_eq!(statistics.tnmc_proposed, 100);
        assert_eq!(statistics.global_proposed, 0);
    }

    #[test]
    fn tnmc_global_adds_lazy_always_accepted_flips_for_exact_symmetries() {
        let couplings = Couplings::new(0.3, 0.7).expect("valid couplings");
        let models = [
            Model::clean(2, 2, couplings).expect("valid clean model"),
            Model::posterior(2, 2, couplings, Noise::Zz, vec![0.4, -0.8]).expect("valid ZZ model"),
        ];

        for (case, model) in models.iter().enumerate() {
            let mut lattice = Lattice::new(2, 2, vec![1; 4]).expect("valid lattice");
            let mut rng = Rng64::seeded(0x6c61_7a79 + case as u64);
            let mut statistics = UpdateStats::default();
            for _ in 0..2_000 {
                Update::TnmcGlobal {
                    maximum_bond_dimension: 2,
                }
                .apply(model, &mut lattice, &mut rng, &mut statistics)
                .expect("valid TNMC update");
            }

            assert_eq!(statistics.sweeps, 2_000);
            assert_eq!(statistics.tnmc_proposed, 2_000);
            assert_eq!(statistics.global_proposed, 2_000);
            assert_eq!(statistics.global_accepted, statistics.global_attempted);
            assert!((900..=1_100).contains(&statistics.global_attempted));
        }
    }

    #[test]
    fn lazy_global_flip_uses_the_exact_metropolis_probability_for_z_fields() {
        let couplings = Couplings::new(0.3, 0.7).expect("valid couplings");
        let field = std::f64::consts::LN_2 / 4.0;
        let model =
            Model::posterior(2, 2, couplings, Noise::Z, vec![field; 2]).expect("valid Z model");
        let initial = Lattice::new(2, 2, vec![1; 4]).expect("valid lattice");
        assert!((model.global_flip_delta(&initial) - std::f64::consts::LN_2).abs() < 1.0e-12);

        let trials = 100_000;
        let mut rng = Rng64::seeded(0x5a66_1e1d);
        let mut statistics = UpdateStats::default();
        for _ in 0..trials {
            let mut lattice = initial.clone();
            lazy_global_flip(&model, &mut lattice, &mut rng, &mut statistics);
        }

        assert_eq!(statistics.global_proposed, trials);
        let attempt_fraction = statistics.global_attempted as f64 / trials as f64;
        let acceptance = statistics.global_accepted as f64 / statistics.global_attempted as f64;
        assert!((attempt_fraction - 0.5).abs() < 0.01);
        assert!((acceptance - 0.5).abs() < 0.01);
    }
}
