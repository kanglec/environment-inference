use std::sync::Arc;

use crate::error::{DcftError, Result, require_at_least, require_finite, require_positive_finite};

#[derive(Debug, Clone)]
pub struct LatticeSpec {
    inner: Arc<LatticeSpecInner>,
}

#[derive(Debug)]
struct LatticeSpecInner {
    lx: usize,
    lt: usize,
    site_count: usize,
    nn: Vec<[usize; 4]>,
}

impl LatticeSpec {
    pub fn new(lx: usize, lt: usize) -> Result<Self> {
        require_at_least("lx", lx, 2)?;
        require_at_least("lt", lt, 2)?;
        let site_count = lx
            .checked_mul(lt)
            .ok_or_else(|| DcftError::invalid_parameter("lattice dimensions overflow"))?;
        let mut nn = vec![[0; 4]; site_count];

        for t in 0..lt {
            for x in 0..lx {
                let index = x + lx * t;
                let left = (if x > 0 { x - 1 } else { lx - 1 }) + lx * t;
                let right = (if x + 1 < lx { x + 1 } else { 0 }) + lx * t;
                let up = x + lx * if t + 1 < lt { t + 1 } else { 0 };
                let down = x + lx * if t > 0 { t - 1 } else { lt - 1 };
                nn[index] = [left, right, up, down];
            }
        }

        Ok(Self {
            inner: Arc::new(LatticeSpecInner {
                lx,
                lt,
                site_count,
                nn,
            }),
        })
    }

    pub fn lx(&self) -> usize {
        self.inner.lx
    }

    pub fn lt(&self) -> usize {
        self.inner.lt
    }

    pub fn site_count(&self) -> usize {
        self.inner.site_count
    }

    #[inline(always)]
    pub fn index(&self, x: usize, t: usize) -> usize {
        debug_assert!(x < self.lx());
        debug_assert!(t < self.lt());
        x + self.lx() * t
    }

    #[inline(always)]
    pub fn neighbors(&self, index: usize) -> [usize; 4] {
        debug_assert!(index < self.site_count());
        self.inner.nn[index]
    }

    #[inline(always)]
    pub(super) unsafe fn neighbors_unchecked(&self, index: usize) -> [usize; 4] {
        debug_assert!(index < self.site_count());
        // SAFETY: callers must guarantee `index < site_count`.
        unsafe { *self.inner.nn.get_unchecked(index) }
    }

    #[inline(always)]
    pub fn is_boundary(&self, index: usize) -> bool {
        debug_assert!(index < self.site_count());
        index < self.lx()
    }
}

impl PartialEq for LatticeSpec {
    fn eq(&self, other: &Self) -> bool {
        self.lx() == other.lx() && self.lt() == other.lt()
    }
}

impl Eq for LatticeSpec {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsingCouplings {
    kx: f64,
    kt: f64,
}

impl IsingCouplings {
    pub fn new(kx: f64, kt: f64) -> Result<Self> {
        require_finite("kx", kx)?;
        require_finite("kt", kt)?;
        Ok(Self { kx, kt })
    }

    pub fn from_critical_tfim_delta_tau(delta_tau: f64) -> Result<Self> {
        require_positive_finite("delta_tau", delta_tau)?;
        let kt = -0.5 * delta_tau.tanh().ln();
        Self::new(delta_tau, if kt == -0.0 { 0.0 } else { kt })
    }

    pub fn from_critical_tfim_isotropic() -> Self {
        let coupling = 0.5 * (1.0 + 2_f64.sqrt()).ln();
        Self {
            kx: coupling,
            kt: coupling,
        }
    }

    pub fn kx(&self) -> f64 {
        self.kx
    }

    pub fn kt(&self) -> f64 {
        self.kt
    }
}

#[derive(Debug, Clone)]
pub struct IsingContext {
    spec: LatticeSpec,
    couplings: IsingCouplings,
}

impl IsingContext {
    pub fn new(spec: LatticeSpec, couplings: IsingCouplings) -> Self {
        Self { spec, couplings }
    }

    pub fn from_critical_tfim_isotropic(lx: usize, lt: usize) -> Result<Self> {
        Ok(Self::new(
            LatticeSpec::new(lx, lt)?,
            IsingCouplings::from_critical_tfim_isotropic(),
        ))
    }

    pub fn from_critical_tfim_delta_tau(lx: usize, lt: usize, delta_tau: f64) -> Result<Self> {
        Ok(Self::new(
            LatticeSpec::new(lx, lt)?,
            IsingCouplings::from_critical_tfim_delta_tau(delta_tau)?,
        ))
    }

    pub fn spec(&self) -> &LatticeSpec {
        &self.spec
    }

    pub fn couplings(&self) -> IsingCouplings {
        self.couplings
    }
}

impl PartialEq for IsingContext {
    fn eq(&self, other: &Self) -> bool {
        self.spec() == other.spec() && self.couplings() == other.couplings()
    }
}
