use std::num::NonZeroUsize;

use crate::rng::Rng64;

use super::super::disorder::{BoundaryDisorder, BoundaryRandomIsingModel};
use super::super::lattice::SpinLattice;
use super::super::model::IsingModel;
use super::wolff::{WolffScratch, bond_probability, flip_cluster, grow_wolff_cluster};

pub(super) struct CorrectedWolffScratch {
    wolff: WolffScratch,
}

impl CorrectedWolffScratch {
    pub(super) fn new(site_count: usize) -> Self {
        Self {
            wolff: WolffScratch::new(site_count),
        }
    }
}

pub(super) fn corrected_wolff_sweep(
    model: &BoundaryRandomIsingModel,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
    sweeps: usize,
) {
    let mut scratch = CorrectedWolffScratch::new(lattice.spec().site_count());
    corrected_wolff_sweep_with_scratch(model, lattice, rng, sweeps, &mut scratch);
}

pub(super) fn corrected_wolff_sweep_with_scratch(
    model: &BoundaryRandomIsingModel,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
    sweeps: usize,
    scratch: &mut CorrectedWolffScratch,
) {
    debug_assert_eq!(model.spec(), lattice.spec());
    let site_count = lattice.spec().site_count();
    let site_count_nonzero =
        NonZeroUsize::new(site_count).expect("lattice site count must be positive");
    let p_add_x = bond_probability(model.couplings().kx());
    let p_add_t = bond_probability(model.couplings().kt());
    for _ in 0..sweeps {
        grow_wolff_cluster(
            lattice,
            rng,
            site_count_nonzero,
            &mut scratch.wolff,
            p_add_x,
            p_add_t,
        );
        let delta = disorder_delta(model.disorder(), lattice, &scratch.wolff);
        if delta <= 0.0 || rng.log_uniform() < -delta {
            flip_cluster(lattice, &scratch.wolff);
        }
    }
}

#[inline(always)]
fn disorder_delta(
    disorder: &BoundaryDisorder,
    lattice: &SpinLattice,
    scratch: &WolffScratch,
) -> f64 {
    match disorder {
        BoundaryDisorder::Fields(fields) => field_delta(fields, lattice, scratch),
        BoundaryDisorder::BondsX(bonds) => bond_x_delta(bonds, lattice, scratch),
    }
}

#[inline(always)]
fn field_delta(fields: &[f64], lattice: &SpinLattice, scratch: &WolffScratch) -> f64 {
    let lx = fields.len();
    let mut delta = 0.0;
    for index in scratch.cluster.iter().copied() {
        if index < lx {
            // SAFETY: boundary indices are valid lattice indices.
            let spin = unsafe { lattice.get_index_unchecked(index) };
            // SAFETY: `index < lx == fields.len()`.
            let field = unsafe { *fields.get_unchecked(index) };
            delta += field * f64::from(spin);
        }
    }
    2.0 * delta
}

#[inline(always)]
fn bond_x_delta(bonds: &[f64], lattice: &SpinLattice, scratch: &WolffScratch) -> f64 {
    let lx = bonds.len();
    let mut delta = 0.0;
    for x in 0..lx {
        let right = if x + 1 < lx { x + 1 } else { 0 };
        if scratch.contains(x) == scratch.contains(right) {
            continue;
        }
        // SAFETY: `x` and `right` are boundary indices and therefore valid
        // lattice indices; `x < lx == bonds.len()`.
        let contribution = unsafe {
            *bonds.get_unchecked(x)
                * f64::from(lattice.get_index_unchecked(x) * lattice.get_index_unchecked(right))
        };
        delta += contribution;
    }
    2.0 * delta
}
