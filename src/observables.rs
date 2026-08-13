use crate::error::{DcftError, Result};
use crate::mc::{Lattice, Model};

#[derive(Debug, Clone, PartialEq)]
pub struct ObservableSummary {
    pub samples: usize,
    pub energy: f64,
    pub magnetization: f64,
    pub boundary_magnetization: f64,
    pub spin_profile: Vec<f64>,
    pub bond_profile: Vec<f64>,
    pub separations: Vec<usize>,
    pub spin_correlator_profile: Vec<Vec<f64>>,
    pub bond_correlator_profile: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub struct Accumulator {
    lx: usize,
    site_count: usize,
    separations: Vec<usize>,
    samples: usize,
    energy: f64,
    magnetization: f64,
    boundary_magnetization: f64,
    spin_profile: Vec<f64>,
    bond_profile: Vec<f64>,
    spin_correlator_profile: Vec<Vec<f64>>,
    bond_correlator_profile: Vec<Vec<f64>>,
}

impl Accumulator {
    pub fn new(lx: usize, lt: usize, separations: &[usize]) -> Result<Self> {
        if lx < 2 || lt < 2 {
            return Err(DcftError::invalid("invalid accumulator lattice"));
        }
        if separations.is_empty() {
            return Err(DcftError::invalid(
                "at least one correlator separation is required",
            ));
        }
        if separations.iter().any(|separation| *separation >= lx) {
            return Err(DcftError::invalid(
                "correlator separations must be smaller than lx",
            ));
        }
        let rows = vec![vec![0.0; lx]; separations.len()];
        Ok(Self {
            lx,
            site_count: lx * lt,
            separations: separations.to_vec(),
            samples: 0,
            energy: 0.0,
            magnetization: 0.0,
            boundary_magnetization: 0.0,
            spin_profile: vec![0.0; lx],
            bond_profile: vec![0.0; lx],
            spin_correlator_profile: rows.clone(),
            bond_correlator_profile: rows,
        })
    }

    pub fn observe(&mut self, model: &Model, lattice: &Lattice) {
        self.samples += 1;
        self.energy += model.energy(lattice);
        self.magnetization += lattice
            .spins()
            .iter()
            .map(|spin| f64::from(*spin))
            .sum::<f64>()
            / self.site_count as f64;
        self.boundary_magnetization += lattice
            .boundary()
            .iter()
            .map(|spin| f64::from(*spin))
            .sum::<f64>()
            / self.lx as f64;

        for x in 0..self.lx {
            let spin = lattice.get(x, 0);
            let bond = spin * lattice.get((x + 1) % self.lx, 0);
            self.spin_profile[x] += f64::from(spin);
            self.bond_profile[x] += f64::from(bond);
            for (row, separation) in self.separations.iter().enumerate() {
                let displaced = (x + *separation) % self.lx;
                let displaced_spin = lattice.get(displaced, 0);
                let displaced_bond = displaced_spin * lattice.get((displaced + 1) % self.lx, 0);
                self.spin_correlator_profile[row][x] += f64::from(spin * displaced_spin);
                self.bond_correlator_profile[row][x] += f64::from(bond * displaced_bond);
            }
        }
    }

    pub fn finish(mut self) -> Result<ObservableSummary> {
        if self.samples == 0 {
            return Err(DcftError::invalid("cannot finish an empty accumulator"));
        }
        let inverse = 1.0 / self.samples as f64;
        for value in self
            .spin_profile
            .iter_mut()
            .chain(&mut self.bond_profile)
            .chain(self.spin_correlator_profile.iter_mut().flatten())
            .chain(self.bond_correlator_profile.iter_mut().flatten())
        {
            *value *= inverse;
        }
        Ok(ObservableSummary {
            samples: self.samples,
            energy: self.energy * inverse,
            magnetization: self.magnetization * inverse,
            boundary_magnetization: self.boundary_magnetization * inverse,
            spin_profile: self.spin_profile,
            bond_profile: self.bond_profile,
            separations: self.separations,
            spin_correlator_profile: self.spin_correlator_profile,
            bond_correlator_profile: self.bond_correlator_profile,
        })
    }
}
