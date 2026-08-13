use crate::error::{DcftError, Result, require_finite};

use super::lattice::SpinLattice;
use super::model::{IsingModel, clean_energy, clean_flip_energy_delta_with_neighbors_unchecked};
use super::params::{IsingContext, LatticeSpec};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum BoundaryDisorder {
    Fields(Vec<f64>),
    BondsX(Vec<f64>),
}

impl BoundaryDisorder {
    pub fn values(&self) -> &[f64] {
        match self {
            Self::Fields(values) | Self::BondsX(values) => values,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryRandomIsingModel {
    context: IsingContext,
    disorder: BoundaryDisorder,
}

impl BoundaryRandomIsingModel {
    fn new(context: IsingContext, disorder: BoundaryDisorder) -> Result<Self> {
        validate_boundary_disorder(context.spec(), &disorder)?;
        Ok(Self { context, disorder })
    }

    pub fn boundary_fields(context: IsingContext, fields: Vec<f64>) -> Result<Self> {
        Self::new(context, BoundaryDisorder::Fields(fields))
    }

    pub fn boundary_bonds_x(context: IsingContext, bonds: Vec<f64>) -> Result<Self> {
        Self::new(context, BoundaryDisorder::BondsX(bonds))
    }

    pub fn energy(&self, lattice: &SpinLattice) -> f64 {
        <Self as IsingModel>::energy(self, lattice)
    }

    pub fn flip_energy_delta(&self, lattice: &SpinLattice, index: usize) -> f64 {
        <Self as IsingModel>::flip_energy_delta(self, lattice, index)
    }

    /// Energy change for flipping every spin in the lattice.
    ///
    /// The clean Ising energy and boundary-bond disorder are invariant.  For
    /// boundary fields only the field energy changes sign.
    pub fn global_flip_energy_delta(&self, lattice: &SpinLattice) -> f64 {
        debug_assert_eq!(self.spec(), lattice.spec());
        match &self.disorder {
            BoundaryDisorder::Fields(fields) => {
                2.0 * fields
                    .iter()
                    .zip(&lattice.spins()[..self.spec().lx()])
                    .map(|(field, spin)| field * f64::from(*spin))
                    .sum::<f64>()
            }
            BoundaryDisorder::BondsX(_) => 0.0,
        }
    }

    pub(super) fn disorder(&self) -> &BoundaryDisorder {
        &self.disorder
    }

    fn boundary_energy(&self, lattice: &SpinLattice) -> f64 {
        let spec = self.context.spec();
        match &self.disorder {
            BoundaryDisorder::Fields(fields) => fields
                .iter()
                .enumerate()
                .map(|(x, field)| {
                    // SAFETY: `fields.len() == lx`, so enumerated `x` lies on
                    // the boundary and is a valid lattice index.
                    -field * f64::from(unsafe { lattice.get_index_unchecked(x) })
                })
                .sum(),
            BoundaryDisorder::BondsX(bonds) => bonds
                .iter()
                .enumerate()
                .map(|(x, bond)| {
                    // SAFETY: `bonds.len() == lx`, so `x` is a valid boundary
                    // lattice index; neighbor indices are precomputed in range.
                    let [_left, right, _up, _down] = unsafe { spec.neighbors_unchecked(x) };
                    -bond
                        * f64::from(unsafe {
                            lattice.get_index_unchecked(x) * lattice.get_index_unchecked(right)
                        })
                })
                .sum(),
        }
    }

    unsafe fn boundary_flip_energy_delta_unchecked(
        &self,
        lattice: &SpinLattice,
        index: usize,
        neighbors: [usize; 4],
    ) -> f64 {
        let spec = self.context.spec();
        debug_assert!(index < spec.site_count());
        if !spec.is_boundary(index) {
            return 0.0;
        }
        let x = index;
        // SAFETY: callers guarantee `index` is in range.
        let spin = f64::from(unsafe { lattice.get_index_unchecked(index) });
        match &self.disorder {
            BoundaryDisorder::Fields(fields) => {
                // SAFETY: `is_boundary(index)` ensures `x < lx == fields.len()`.
                2.0 * spin * unsafe { *fields.get_unchecked(x) }
            }
            BoundaryDisorder::BondsX(bonds) => {
                let [left, right, _up, _down] = neighbors;
                // SAFETY: `is_boundary(index)` ensures `x < lx == bonds.len()`.
                // For boundary site `x`, its left neighbor is also a boundary site,
                // so `left < lx == bonds.len()`.
                let field = unsafe {
                    *bonds.get_unchecked(x) * f64::from(lattice.get_index_unchecked(right))
                        + *bonds.get_unchecked(left) * f64::from(lattice.get_index_unchecked(left))
                };
                2.0 * spin * field
            }
        }
    }
}

impl IsingModel for BoundaryRandomIsingModel {
    fn context(&self) -> &IsingContext {
        &self.context
    }

    fn energy(&self, lattice: &SpinLattice) -> f64 {
        debug_assert_eq!(self.spec(), lattice.spec());
        clean_energy(&self.context, lattice) + self.boundary_energy(lattice)
    }

    fn flip_energy_delta(&self, lattice: &SpinLattice, index: usize) -> f64 {
        debug_assert_eq!(self.spec(), lattice.spec());
        assert!(index < self.spec().site_count(), "site index out of bounds");
        // SAFETY: the assertion above proves `index` is in range; the lattice
        // spec constructs only in-range neighbor indices.
        unsafe {
            let neighbors = self.spec().neighbors_unchecked(index);
            clean_flip_energy_delta_with_neighbors_unchecked(
                &self.context,
                lattice,
                index,
                neighbors,
            ) + self.boundary_flip_energy_delta_unchecked(lattice, index, neighbors)
        }
    }
}

fn validate_boundary_disorder(spec: &LatticeSpec, disorder: &BoundaryDisorder) -> Result<()> {
    if disorder.values().len() != spec.lx() {
        return Err(DcftError::invalid_parameter(format!(
            "boundary disorder count {} does not match lx {}",
            disorder.values().len(),
            spec.lx()
        )));
    }
    for value in disorder.values() {
        require_finite("boundary disorder", *value)?;
    }
    Ok(())
}
