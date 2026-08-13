use crate::error::{DcftError, Result};
use crate::rng::Rng64;

use super::params::LatticeSpec;

#[derive(Debug, Clone, PartialEq)]
pub struct SpinLattice {
    spec: LatticeSpec,
    spins: Vec<i8>,
}

impl SpinLattice {
    pub fn new(spec: LatticeSpec, spins: Vec<i8>) -> Result<Self> {
        if spins.len() != spec.site_count() {
            return Err(DcftError::invalid_parameter(format!(
                "spin count {} does not match lx * lt {}",
                spins.len(),
                spec.site_count()
            )));
        }
        if spins.iter().any(|spin| *spin != -1 && *spin != 1) {
            return Err(DcftError::invalid_parameter("spins must be -1 or 1"));
        }
        Ok(Self { spec, spins })
    }

    pub fn cold(spec: LatticeSpec, spin: i8) -> Result<Self> {
        if spin != -1 && spin != 1 {
            return Err(DcftError::invalid_parameter("spins must be -1 or 1"));
        }
        let site_count = spec.site_count();
        Ok(Self {
            spec,
            spins: vec![spin; site_count],
        })
    }

    pub fn random(spec: LatticeSpec, rng: &mut Rng64) -> Self {
        let spins = (0..spec.site_count())
            .map(|_| if rng.bool() { 1 } else { -1 })
            .collect();
        Self { spec, spins }
    }

    pub fn spec(&self) -> &LatticeSpec {
        &self.spec
    }

    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    #[inline(always)]
    pub fn get(&self, x: usize, t: usize) -> i8 {
        self.spins[self.spec.index(x, t)]
    }

    #[inline(always)]
    pub fn get_index(&self, index: usize) -> i8 {
        debug_assert!(index < self.spins.len());
        self.spins[index]
    }

    #[inline(always)]
    pub(super) unsafe fn get_index_unchecked(&self, index: usize) -> i8 {
        debug_assert!(index < self.spins.len());
        // SAFETY: callers must guarantee `index < spins.len()`.
        unsafe { *self.spins.get_unchecked(index) }
    }

    #[inline(always)]
    pub fn flip_index(&mut self, index: usize) {
        debug_assert!(index < self.spins.len());
        self.spins[index] = -self.spins[index];
    }

    /// Flip every spin.  This is used by the optional global Metropolis proposal.
    pub fn flip_all(&mut self) {
        for spin in &mut self.spins {
            *spin = -*spin;
        }
    }

    #[inline(always)]
    pub(super) unsafe fn flip_index_unchecked(&mut self, index: usize) {
        debug_assert!(index < self.spins.len());
        // SAFETY: callers must guarantee `index < spins.len()`.
        let spin = unsafe { self.spins.get_unchecked_mut(index) };
        *spin = -*spin;
    }

    pub fn boundary_spins(&self) -> Vec<i8> {
        self.spins[..self.spec.lx()].to_vec()
    }

    pub fn boundary_bonds_x(&self) -> Vec<i8> {
        (0..self.spec.lx())
            .map(|x| {
                let right = if x + 1 < self.spec.lx() { x + 1 } else { 0 };
                self.spins[x] * self.spins[right]
            })
            .collect()
    }

    pub fn spin_sum(&self) -> i64 {
        self.spins.iter().map(|spin| i64::from(*spin)).sum()
    }

    pub fn magnetization(&self) -> f64 {
        self.spin_sum() as f64 / self.spec.site_count() as f64
    }

    pub fn boundary_magnetization(&self) -> f64 {
        let sum: i64 = self.spins[..self.spec.lx()]
            .iter()
            .map(|spin| i64::from(*spin))
            .sum();
        sum as f64 / self.spec.lx() as f64
    }
}
