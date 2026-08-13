use std::num::NonZeroUsize;

use crate::rng::Rng64;

use super::super::disorder::BoundaryRandomIsingModel;
use super::super::lattice::SpinLattice;
use super::super::model::IsingModel;

pub(super) fn metropolis_sweep<M: IsingModel>(
    model: &M,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
    sweeps: usize,
) {
    debug_assert_eq!(model.spec(), lattice.spec());
    let site_count = model.spec().site_count();
    let site_count_nonzero =
        NonZeroUsize::new(site_count).expect("lattice site count must be positive");
    for _ in 0..sweeps {
        for _ in 0..site_count {
            let index = rng.usize_nonzero(site_count_nonzero);
            let delta = model.flip_energy_delta(lattice, index);
            if delta <= 0.0 || rng.log_uniform() < -delta {
                // SAFETY: `rng.usize_nonzero(site_count)` returns an index in
                // `0..site_count`, and the lattice spec matches the model.
                unsafe { lattice.flip_index_unchecked(index) };
            }
        }
    }
}

/// Random single-site Metropolis sweeps mixed with one global-flip
/// Metropolis proposal per sweep.
///
/// The global move is especially useful for weak boundary fields, where the
/// two critical Ising magnetization sectors mix slowly under local proposals.
pub(super) fn metropolis_global_sweep(
    model: &BoundaryRandomIsingModel,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
    sweeps: usize,
) {
    debug_assert_eq!(model.spec(), lattice.spec());
    for _ in 0..sweeps {
        metropolis_sweep(model, lattice, rng, 1);
        // Laziness avoids a deterministic period-two alternation when the
        // proposal has zero energy cost (notably at p=0).
        if rng.bool() {
            let delta = model.global_flip_energy_delta(lattice);
            if delta <= 0.0 || rng.log_uniform() < -delta {
                lattice.flip_all();
            }
        }
    }
}

pub(super) fn sequential_metropolis_sweep<M: IsingModel>(
    model: &M,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
    sweeps: usize,
) {
    debug_assert_eq!(model.spec(), lattice.spec());
    let site_count = model.spec().site_count();
    for _ in 0..sweeps {
        for index in 0..site_count {
            let delta = model.flip_energy_delta(lattice, index);
            if delta <= 0.0 || rng.log_uniform() < -delta {
                // SAFETY: `index` is generated from `0..site_count`, and the
                // lattice spec matches the model.
                unsafe { lattice.flip_index_unchecked(index) };
            }
        }
    }
}
