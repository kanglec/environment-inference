//! Stable pseudorandom streams for reproducible campaigns.
//!
//! The contract is xoshiro256++ 1.0, seeded by `SplitMix64` after hashing the
//! base seed, a UTF-8 domain label, and the global outer id. Stream keys never
//! contain a thread, chunk, process, or scheduler id.

use crate::error::{DcftError, Result};

pub const RNG_ALGORITHM: &str = "xoshiro256++/splitmix64";
pub const STREAM_KEY_VERSION: &str = "dcft-stream-v1";

#[derive(Debug, Clone)]
pub struct Rng64 {
    state: [u64; 4],
    cached_normal: Option<f64>,
}

impl Rng64 {
    #[must_use]
    pub fn seeded(seed: u64) -> Self {
        let mut splitmix = SplitMix64(seed);
        let mut state = [0_u64; 4];
        for item in &mut state {
            *item = splitmix.next_u64();
        }
        if state == [0; 4] {
            state[0] = 1;
        }
        Self {
            state,
            cached_normal: None,
        }
    }

    #[must_use]
    pub fn stream(base_seed: u64, domain: &str, global_id: u64) -> Self {
        Self::seeded(derive_stream_seed(base_seed, domain, global_id))
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);

        let temporary = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= temporary;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    #[must_use]
    pub fn uniform(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64 + 0.5) * (1.0 / (1_u64 << 53) as f64)
    }

    #[must_use]
    pub fn log_uniform(&mut self) -> f64 {
        self.uniform().ln()
    }

    #[must_use]
    pub fn bool(&mut self) -> bool {
        self.next_u64() >> 63 == 1
    }

    pub fn index(&mut self, upper: usize) -> Result<usize> {
        if upper == 0 {
            return Err(DcftError::invalid(
                "random index upper bound must be positive",
            ));
        }
        let upper_u64 = upper as u64;
        let mut product = u128::from(self.next_u64()) * u128::from(upper_u64);
        let mut low = product as u64;
        if low < upper_u64 {
            let threshold = upper_u64.wrapping_neg() % upper_u64;
            while low < threshold {
                product = u128::from(self.next_u64()) * u128::from(upper_u64);
                low = product as u64;
            }
        }
        Ok((product >> 64) as usize)
    }

    #[must_use]
    pub fn normal(&mut self) -> f64 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }
        let first = self.uniform();
        let second = self.uniform();
        let radius = (-2.0 * first.ln()).sqrt();
        let angle = std::f64::consts::TAU * second;
        let (sine, cosine) = angle.sin_cos();
        self.cached_normal = Some(radius * sine);
        radius * cosine
    }
}

#[must_use]
pub fn derive_stream_seed(base_seed: u64, domain: &str, global_id: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in STREAM_KEY_VERSION.bytes().chain([0]).chain(domain.bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash = combine(hash, base_seed);
    combine(hash, global_id)
}

#[must_use]
pub fn standard_normals(base_seed: u64, domain: &str, global_id: u64, count: usize) -> Vec<f64> {
    let mut rng = Rng64::stream(base_seed, domain, global_id);
    (0..count).map(|_| rng.normal()).collect()
}

#[must_use]
pub fn uniforms(base_seed: u64, domain: &str, global_id: u64, count: usize) -> Vec<f64> {
    let mut rng = Rng64::stream(base_seed, domain, global_id);
    (0..count).map(|_| rng.uniform()).collect()
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        finalize(self.0)
    }
}

fn combine(hash: u64, value: u64) -> u64 {
    finalize(hash ^ finalize(value))
}

fn finalize(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyed_streams_are_stable_and_distinct() {
        let a = derive_stream_seed(7, "gaussian-normal/z", 3);
        let b = derive_stream_seed(7, "gaussian-normal/zz", 3);
        let c = derive_stream_seed(7, "gaussian-normal/z", 4);
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn xoshiro_golden_vector() {
        let mut rng = Rng64::stream(0x1234_5678_9abc_def0, "golden", 42);
        let actual: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
        // This vector is part of DCFT's RNG file-format contract.
        let expected = [
            8_122_992_829_048_943_606,
            14_887_561_678_065_885_490,
            12_213_390_394_398_493_239,
            4_458_159_480_994_923_238,
            1_552_814_934_267_694_593,
            12_016_362_651_054_279_812,
            2_870_092_926_268_116_656,
            13_511_114_224_061_842_929,
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn random_index_is_bounded() {
        let mut rng = Rng64::seeded(1);
        for upper in 1..64 {
            for _ in 0..100 {
                assert!(rng.index(upper).expect("valid upper bound") < upper);
            }
        }
    }
}
