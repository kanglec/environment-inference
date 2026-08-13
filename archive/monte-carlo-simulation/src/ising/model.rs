use crate::error::Result;

use super::lattice::SpinLattice;
use super::params::{IsingContext, IsingCouplings, LatticeSpec};

pub trait IsingModel {
    fn context(&self) -> &IsingContext;

    fn spec(&self) -> &LatticeSpec {
        self.context().spec()
    }

    fn couplings(&self) -> IsingCouplings {
        self.context().couplings()
    }

    fn energy(&self, lattice: &SpinLattice) -> f64;

    fn flip_energy_delta(&self, lattice: &SpinLattice, index: usize) -> f64;

    fn supports_wolff(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CleanIsingModel {
    context: IsingContext,
}

impl CleanIsingModel {
    pub fn new(context: IsingContext) -> Self {
        Self { context }
    }

    pub fn from_critical_tfim_isotropic(lx: usize, lt: usize) -> Result<Self> {
        Ok(Self::new(IsingContext::from_critical_tfim_isotropic(
            lx, lt,
        )?))
    }

    pub fn from_critical_tfim_delta_tau(lx: usize, lt: usize, delta_tau: f64) -> Result<Self> {
        Ok(Self::new(IsingContext::from_critical_tfim_delta_tau(
            lx, lt, delta_tau,
        )?))
    }

    pub fn energy(&self, lattice: &SpinLattice) -> f64 {
        <Self as IsingModel>::energy(self, lattice)
    }

    pub fn flip_energy_delta(&self, lattice: &SpinLattice, index: usize) -> f64 {
        <Self as IsingModel>::flip_energy_delta(self, lattice, index)
    }
}

impl IsingModel for CleanIsingModel {
    fn context(&self) -> &IsingContext {
        &self.context
    }

    fn energy(&self, lattice: &SpinLattice) -> f64 {
        debug_assert_eq!(self.spec(), lattice.spec());
        clean_energy(&self.context, lattice)
    }

    fn flip_energy_delta(&self, lattice: &SpinLattice, index: usize) -> f64 {
        debug_assert_eq!(self.spec(), lattice.spec());
        assert!(index < self.spec().site_count(), "site index out of bounds");
        // SAFETY: the assertion above proves `index` is in range; the lattice
        // spec constructs only in-range neighbor indices.
        unsafe { clean_flip_energy_delta_unchecked(&self.context, lattice, index) }
    }

    fn supports_wolff(&self) -> bool {
        true
    }
}

pub(super) fn clean_energy(context: &IsingContext, lattice: &SpinLattice) -> f64 {
    let spec = context.spec();
    let kx = context.couplings().kx();
    let kt = context.couplings().kt();
    let mut total = 0.0;
    for index in 0..spec.site_count() {
        // SAFETY: `index` is generated from `0..site_count`; neighbor indices
        // are precomputed by `LatticeSpec::new` and are always in range.
        let (spin, right_spin, up_spin) = unsafe {
            let [_left, right, up, _down] = spec.neighbors_unchecked(index);
            (
                lattice.get_index_unchecked(index),
                lattice.get_index_unchecked(right),
                lattice.get_index_unchecked(up),
            )
        };
        let spin = f64::from(spin);
        total -= kx * spin * f64::from(right_spin);
        total -= kt * spin * f64::from(up_spin);
    }
    total
}

pub(super) unsafe fn clean_flip_energy_delta_unchecked(
    context: &IsingContext,
    lattice: &SpinLattice,
    index: usize,
) -> f64 {
    let spec = context.spec();
    debug_assert!(index < spec.site_count());
    // SAFETY: callers guarantee `index` is in range; neighbor indices are
    // precomputed by `LatticeSpec::new` and are always in range.
    let neighbors = unsafe { spec.neighbors_unchecked(index) };
    // SAFETY: callers guarantee `index` is in range; `neighbors` came from
    // `spec.neighbors_unchecked(index)`, so each neighbor is in range.
    unsafe { clean_flip_energy_delta_with_neighbors_unchecked(context, lattice, index, neighbors) }
}

pub(super) unsafe fn clean_flip_energy_delta_with_neighbors_unchecked(
    context: &IsingContext,
    lattice: &SpinLattice,
    index: usize,
    neighbors: [usize; 4],
) -> f64 {
    let spec = context.spec();
    let couplings = context.couplings();
    debug_assert!(index < spec.site_count());
    let [left, right, up, down] = neighbors;
    // SAFETY: `index` and its precomputed neighbors are in range.
    let (spin, x_sum, t_sum) = unsafe {
        (
            lattice.get_index_unchecked(index),
            lattice.get_index_unchecked(left) + lattice.get_index_unchecked(right),
            lattice.get_index_unchecked(up) + lattice.get_index_unchecked(down),
        )
    };
    let local_field = couplings.kx() * f64::from(x_sum) + couplings.kt() * f64::from(t_sum);
    2.0 * f64::from(spin) * local_field
}
