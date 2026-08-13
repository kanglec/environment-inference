use std::num::NonZeroUsize;

use crate::rng::Rng64;

use super::super::lattice::SpinLattice;
use super::super::params::IsingCouplings;

pub(super) fn wolff_sweep(
    couplings: IsingCouplings,
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
    sweeps: usize,
) {
    let site_count = lattice.spec().site_count();
    let site_count_nonzero =
        NonZeroUsize::new(site_count).expect("lattice site count must be positive");
    let p_add_x = bond_probability(couplings.kx());
    let p_add_t = bond_probability(couplings.kt());
    let mut scratch = WolffScratch::new(site_count);
    for _ in 0..sweeps {
        wolff_update(
            lattice,
            rng,
            site_count_nonzero,
            &mut scratch,
            p_add_x,
            p_add_t,
        );
    }
}

fn wolff_update(
    lattice: &mut SpinLattice,
    rng: &mut Rng64,
    site_count: NonZeroUsize,
    scratch: &mut WolffScratch,
    p_add_x: f64,
    p_add_t: f64,
) {
    grow_wolff_cluster(lattice, rng, site_count, scratch, p_add_x, p_add_t);
    flip_cluster(lattice, scratch);
}

pub(super) fn grow_wolff_cluster(
    lattice: &SpinLattice,
    rng: &mut Rng64,
    site_count: NonZeroUsize,
    scratch: &mut WolffScratch,
    p_add_x: f64,
    p_add_t: f64,
) {
    let spec = lattice.spec();
    debug_assert_eq!(site_count.get(), spec.site_count());
    let seed_index = rng.usize_nonzero(site_count);
    // SAFETY: `rng.usize_nonzero(site_count)` returns an index in `0..site_count`.
    let seed_spin = unsafe { lattice.get_index_unchecked(seed_index) };

    scratch.begin_cluster();
    scratch.add(seed_index);

    while let Some(index) = scratch.stack.pop() {
        // SAFETY: all stack entries are created from either the seed index or
        // precomputed neighbor indices, all of which are in range.
        let [left, right, up, down] = unsafe { spec.neighbors_unchecked(index) };
        try_add_neighbor(left, p_add_x, seed_spin, lattice, scratch, rng);
        try_add_neighbor(right, p_add_x, seed_spin, lattice, scratch, rng);
        try_add_neighbor(up, p_add_t, seed_spin, lattice, scratch, rng);
        try_add_neighbor(down, p_add_t, seed_spin, lattice, scratch, rng);
    }
}

pub(super) fn flip_cluster(lattice: &mut SpinLattice, scratch: &WolffScratch) {
    for index in scratch.cluster.iter().copied() {
        // SAFETY: cluster entries are either the seed index or precomputed
        // neighbor indices, all of which are in range.
        unsafe { lattice.flip_index_unchecked(index) };
    }
}

pub(super) fn bond_probability(coupling: f64) -> f64 {
    1.0 - (-2.0 * coupling).exp()
}

fn try_add_neighbor(
    index: usize,
    probability: f64,
    seed_spin: i8,
    lattice: &SpinLattice,
    scratch: &mut WolffScratch,
    rng: &mut Rng64,
) {
    // SAFETY: callers pass only precomputed neighbor indices.
    let spin = unsafe { lattice.get_index_unchecked(index) };
    if scratch.contains(index) || spin != seed_spin {
        return;
    }
    if rng.uniform() < probability {
        scratch.add(index);
    }
}

#[derive(Debug)]
pub(super) struct WolffScratch {
    marks: Vec<usize>,
    active_mark: usize,
    pub(super) cluster: Vec<usize>,
    stack: Vec<usize>,
}

impl WolffScratch {
    pub(super) fn new(site_count: usize) -> Self {
        Self {
            marks: vec![0; site_count],
            active_mark: 0,
            cluster: Vec::new(),
            stack: Vec::new(),
        }
    }

    fn begin_cluster(&mut self) {
        if self.active_mark == usize::MAX {
            self.marks.fill(0);
            self.active_mark = 0;
        }
        self.active_mark += 1;
        self.cluster.clear();
        self.stack.clear();
    }

    pub(super) fn contains(&self, index: usize) -> bool {
        self.marks[index] == self.active_mark
    }

    fn add(&mut self, index: usize) {
        self.marks[index] = self.active_mark;
        self.cluster.push(index);
        self.stack.push(index);
    }
}
