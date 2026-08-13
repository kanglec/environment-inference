use crate::error::{DcftError, Result, require_positive};
use crate::physics::Noise;

use super::Lattice;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Couplings {
    pub kx: f64,
    pub kt: f64,
}

impl Couplings {
    pub fn new(kx: f64, kt: f64) -> Result<Self> {
        require_positive("kx", kx)?;
        require_positive("kt", kt)?;
        Ok(Self { kx, kt })
    }

    #[must_use]
    pub fn isotropic() -> Self {
        let value = 0.5 * (1.0 + 2_f64.sqrt()).ln();
        Self {
            kx: value,
            kt: value,
        }
    }

    pub fn trotter(delta_tau: f64) -> Result<Self> {
        require_positive("delta_tau", delta_tau)?;
        Self::new(delta_tau, -0.5 * delta_tau.tanh().ln())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    lx: usize,
    lt: usize,
    couplings: Couplings,
    noise: Option<Noise>,
    record: Vec<f64>,
}

impl Model {
    pub fn clean(lx: usize, lt: usize, couplings: Couplings) -> Result<Self> {
        Lattice::new(lx, lt, vec![1; lx.saturating_mul(lt)])?;
        Ok(Self {
            lx,
            lt,
            couplings,
            noise: None,
            record: Vec::new(),
        })
    }

    pub fn posterior(
        lx: usize,
        lt: usize,
        couplings: Couplings,
        noise: Noise,
        record: Vec<f64>,
    ) -> Result<Self> {
        if record.len() != lx {
            return Err(DcftError::invalid("record length must equal lx"));
        }
        if record.iter().any(|value| !value.is_finite()) {
            return Err(DcftError::invalid("record couplings must be finite"));
        }
        Lattice::new(lx, lt, vec![1; lx.saturating_mul(lt)])?;
        Ok(Self {
            lx,
            lt,
            couplings,
            noise: Some(noise),
            record,
        })
    }

    #[must_use]
    pub const fn lx(&self) -> usize {
        self.lx
    }

    #[must_use]
    pub const fn lt(&self) -> usize {
        self.lt
    }

    #[must_use]
    pub const fn couplings(&self) -> Couplings {
        self.couplings
    }

    #[must_use]
    pub const fn noise(&self) -> Option<Noise> {
        self.noise
    }

    #[must_use]
    pub fn record(&self) -> &[f64] {
        &self.record
    }

    #[must_use]
    pub fn clean_energy(&self, lattice: &Lattice) -> f64 {
        let mut total = 0.0;
        for t in 0..self.lt {
            for x in 0..self.lx {
                let spin = f64::from(lattice.get(x, t));
                total -= self.couplings.kx * spin * f64::from(lattice.get((x + 1) % self.lx, t));
                total -= self.couplings.kt * spin * f64::from(lattice.get(x, (t + 1) % self.lt));
            }
        }
        total
    }

    #[must_use]
    pub fn disorder_energy(&self, lattice: &Lattice) -> f64 {
        match self.noise {
            None => 0.0,
            Some(Noise::Z) => self
                .record
                .iter()
                .zip(lattice.boundary())
                .map(|(field, spin)| -field * f64::from(*spin))
                .sum(),
            Some(Noise::Zz) => self
                .record
                .iter()
                .enumerate()
                .map(|(x, bond)| {
                    -bond * f64::from(lattice.get(x, 0) * lattice.get((x + 1) % self.lx, 0))
                })
                .sum(),
        }
    }

    #[must_use]
    pub fn energy(&self, lattice: &Lattice) -> f64 {
        self.clean_energy(lattice) + self.disorder_energy(lattice)
    }

    #[must_use]
    pub fn flip_delta(&self, lattice: &Lattice, index: usize) -> f64 {
        let spin = f64::from(lattice.get_index(index));
        let [left, right, down, up] = lattice.neighbors(index);
        let clean_field = self.couplings.kx
            * f64::from(lattice.get_index(left) + lattice.get_index(right))
            + self.couplings.kt * f64::from(lattice.get_index(down) + lattice.get_index(up));
        let mut delta = 2.0 * spin * clean_field;
        let (x, t) = lattice.coordinates(index);
        if t == 0 {
            match self.noise {
                Some(Noise::Z) => delta += 2.0 * spin * self.record[x],
                Some(Noise::Zz) => {
                    let left_x = (x + self.lx - 1) % self.lx;
                    let disorder_field = self.record[left_x] * f64::from(lattice.get(left_x, 0))
                        + self.record[x] * f64::from(lattice.get((x + 1) % self.lx, 0));
                    delta += 2.0 * spin * disorder_field;
                }
                None => {}
            }
        }
        delta
    }

    #[must_use]
    pub fn global_flip_delta(&self, lattice: &Lattice) -> f64 {
        if self.noise != Some(Noise::Z) {
            return 0.0;
        }
        2.0 * self
            .record
            .iter()
            .zip(lattice.boundary())
            .map(|(field, spin)| field * f64::from(*spin))
            .sum::<f64>()
    }

    #[must_use]
    pub fn cluster_disorder_delta(&self, lattice: &Lattice, in_cluster: &[bool]) -> f64 {
        match self.noise {
            None => 0.0,
            Some(Noise::Z) => {
                2.0 * (0..self.lx)
                    .filter(|x| in_cluster[*x])
                    .map(|x| self.record[x] * f64::from(lattice.get(x, 0)))
                    .sum::<f64>()
            }
            Some(Noise::Zz) => {
                2.0 * (0..self.lx)
                    .filter(|x| in_cluster[*x] != in_cluster[(*x + 1) % self.lx])
                    .map(|x| {
                        self.record[x]
                            * f64::from(lattice.get(x, 0) * lattice.get((x + 1) % self.lx, 0))
                    })
                    .sum::<f64>()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_flip_delta_matches_full_energy_difference() {
        let couplings = Couplings::new(0.3, 0.7).expect("valid");
        for noise in [Noise::Z, Noise::Zz] {
            let model =
                Model::posterior(3, 2, couplings, noise, vec![0.2, -0.4, 0.1]).expect("valid");
            for bits in 0..1_usize << 6 {
                let spins = (0..6)
                    .map(|site| if bits & (1 << site) == 0 { 1 } else { -1 })
                    .collect();
                let lattice = Lattice::new(3, 2, spins).expect("valid");
                for index in 0..6 {
                    let expected = model.flip_delta(&lattice, index);
                    let mut flipped = lattice.clone();
                    flipped.flip(index);
                    let actual = model.energy(&flipped) - model.energy(&lattice);
                    assert!((expected - actual).abs() < 1.0e-12);
                }
            }
        }
    }

    #[test]
    fn every_cluster_and_global_delta_matches_full_disorder_difference() {
        let couplings = Couplings::new(0.3, 0.7).expect("valid");
        for noise in [Noise::Z, Noise::Zz] {
            let model =
                Model::posterior(3, 2, couplings, noise, vec![0.2, -0.4, 0.1]).expect("valid");
            for bits in 0..1_usize << 6 {
                let spins = (0..6)
                    .map(|site| if bits & (1 << site) == 0 { 1 } else { -1 })
                    .collect();
                let lattice = Lattice::new(3, 2, spins).expect("valid");
                for cluster_bits in 0..1_usize << 6 {
                    let cluster: Vec<bool> =
                        (0..6).map(|site| cluster_bits & (1 << site) != 0).collect();
                    let expected = model.cluster_disorder_delta(&lattice, &cluster);
                    let mut flipped = lattice.clone();
                    for (index, included) in cluster.iter().enumerate() {
                        if *included {
                            flipped.flip(index);
                        }
                    }
                    let actual = model.disorder_energy(&flipped) - model.disorder_energy(&lattice);
                    assert!((expected - actual).abs() < 1.0e-12);
                }

                let expected = model.global_flip_delta(&lattice);
                let mut flipped = lattice.clone();
                flipped.flip_all();
                let actual = model.energy(&flipped) - model.energy(&lattice);
                assert!((expected - actual).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn planted_full_configuration_conditionals_equal_posterior_weights() {
        let lx = 2;
        let lt = 2;
        let couplings = Couplings::new(0.31, 0.67).expect("valid");
        let clean = Model::clean(lx, lt, couplings).expect("valid");
        let configurations: Vec<Lattice> = (0..1_usize << (lx * lt))
            .map(|bits| {
                let spins = (0..lx * lt)
                    .map(|site| if bits & (1 << site) == 0 { 1 } else { -1 })
                    .collect();
                Lattice::new(lx, lt, spins).expect("valid")
            })
            .collect();
        let clean_weights: Vec<f64> = configurations
            .iter()
            .map(|lattice| (-clean.energy(lattice)).exp())
            .collect();

        for noise in [Noise::Z, Noise::Zz] {
            let parameters =
                crate::physics::ProtocolParameters::new(crate::physics::Measurement::LocalX, 0.2)
                    .expect("valid protocol");
            let kappa = parameters.kappa.expect("local-X kappa");
            let coupling = parameters.coupling.expect("local-X coupling");
            for outcome_bits in 0..1_usize << lx {
                let outcomes: Vec<i8> = (0..lx)
                    .map(|site| {
                        if outcome_bits & (1 << site) == 0 {
                            1
                        } else {
                            -1
                        }
                    })
                    .collect();
                let joint: Vec<f64> = configurations
                    .iter()
                    .zip(&clean_weights)
                    .map(|(lattice, clean_weight)| {
                        let variables = crate::physics::noise_variables(lattice.boundary(), noise)
                            .expect("valid boundary");
                        let likelihood: f64 = variables
                            .iter()
                            .zip(&outcomes)
                            .map(|(variable, outcome)| {
                                0.5 * (1.0 + kappa * f64::from(variable * outcome))
                            })
                            .product();
                        clean_weight * likelihood
                    })
                    .collect();
                let posterior = Model::posterior(
                    lx,
                    lt,
                    couplings,
                    noise,
                    outcomes
                        .iter()
                        .map(|outcome| coupling * f64::from(*outcome))
                        .collect(),
                )
                .expect("valid posterior");
                let posterior_weights: Vec<f64> = configurations
                    .iter()
                    .map(|lattice| (-posterior.energy(lattice)).exp())
                    .collect();
                let joint_total: f64 = joint.iter().sum();
                let posterior_total: f64 = posterior_weights.iter().sum();
                for (conditional, posterior_weight) in joint.iter().zip(&posterior_weights) {
                    assert!(
                        (conditional / joint_total - posterior_weight / posterior_total).abs()
                            < 2.0e-14
                    );
                }
            }
        }
    }
}
