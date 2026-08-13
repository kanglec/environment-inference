use crate::error::{DcftError, Result};
use crate::rng::Rng64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lattice {
    lx: usize,
    lt: usize,
    spins: Vec<i8>,
}

impl Lattice {
    pub fn new(lx: usize, lt: usize, spins: Vec<i8>) -> Result<Self> {
        validate_dimensions(lx, lt)?;
        if spins.len() != lx * lt {
            return Err(DcftError::invalid("spin count does not match lx * lt"));
        }
        if spins.iter().any(|spin| !matches!(spin, -1 | 1)) {
            return Err(DcftError::invalid("spins must equal -1 or +1"));
        }
        Ok(Self { lx, lt, spins })
    }

    pub fn random(lx: usize, lt: usize, rng: &mut Rng64) -> Result<Self> {
        validate_dimensions(lx, lt)?;
        let spins = (0..lx * lt)
            .map(|_| if rng.bool() { 1 } else { -1 })
            .collect();
        Ok(Self { lx, lt, spins })
    }

    pub fn unpack(lx: usize, lt: usize, packed: &[u8]) -> Result<Self> {
        validate_dimensions(lx, lt)?;
        let count = lx * lt;
        let expected_bytes = count.div_ceil(8);
        if packed.len() != expected_bytes {
            return Err(DcftError::invalid(format!(
                "packed configuration has {} bytes; expected {expected_bytes}",
                packed.len()
            )));
        }
        let spins = (0..count)
            .map(|index| {
                if packed[index / 8] & (1 << (index % 8)) == 0 {
                    1
                } else {
                    -1
                }
            })
            .collect();
        Self::new(lx, lt, spins)
    }

    #[must_use]
    pub fn pack(&self) -> Vec<u8> {
        let mut output = vec![0_u8; self.spins.len().div_ceil(8)];
        for (index, spin) in self.spins.iter().enumerate() {
            if *spin == -1 {
                output[index / 8] |= 1 << (index % 8);
            }
        }
        output
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
    pub fn site_count(&self) -> usize {
        self.spins.len()
    }

    #[must_use]
    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    #[must_use]
    pub fn boundary(&self) -> &[i8] {
        &self.spins[..self.lx]
    }

    #[must_use]
    pub const fn index(&self, x: usize, t: usize) -> usize {
        x + self.lx * t
    }

    #[must_use]
    pub fn coordinates(&self, index: usize) -> (usize, usize) {
        (index % self.lx, index / self.lx)
    }

    #[must_use]
    pub fn get(&self, x: usize, t: usize) -> i8 {
        self.spins[self.index(x, t)]
    }

    #[must_use]
    pub fn get_index(&self, index: usize) -> i8 {
        self.spins[index]
    }

    pub fn flip(&mut self, index: usize) {
        self.spins[index] = -self.spins[index];
    }

    pub fn flip_all(&mut self) {
        for spin in &mut self.spins {
            *spin = -*spin;
        }
    }

    #[must_use]
    pub fn neighbors(&self, index: usize) -> [usize; 4] {
        let (x, t) = self.coordinates(index);
        let left = self.index((x + self.lx - 1) % self.lx, t);
        let right = self.index((x + 1) % self.lx, t);
        let down = self.index(x, (t + self.lt - 1) % self.lt);
        let up = self.index(x, (t + 1) % self.lt);
        [left, right, down, up]
    }
}

fn validate_dimensions(lx: usize, lt: usize) -> Result<()> {
    if lx < 2 || lt < 2 {
        return Err(DcftError::invalid("lx and lt must both be at least two"));
    }
    lx.checked_mul(lt)
        .ok_or_else(|| DcftError::invalid("lattice dimensions overflow"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_packing_uses_one_for_negative_spin() {
        let lattice = Lattice::new(3, 2, vec![1, -1, 1, -1, -1, 1]).expect("valid");
        assert_eq!(lattice.pack(), [0b0001_1010]);
        assert_eq!(
            Lattice::unpack(3, 2, &lattice.pack()).expect("valid"),
            lattice
        );
    }

    #[test]
    fn neighbors_are_periodic() {
        let lattice = Lattice::new(3, 2, vec![1; 6]).expect("valid");
        assert_eq!(lattice.neighbors(0), [2, 1, 3, 3]);
    }
}
