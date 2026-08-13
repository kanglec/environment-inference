//! Reproducible pseudo-random number generation for Monte Carlo work.
//!
//! The generator is xoshiro256++ seeded by SplitMix64.

use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngDomain {
    CleanChain,
    ZDisorderNormals,
    ZzDisorderNormals,
    ZMetropolis,
    ZzMetropolis,
    ZCorrectedWolff,
    ZzCorrectedWolff,
    ZSequentialMetropolis,
    ZzSequentialMetropolis,
    ZMetropolisGlobal,
    ZzMetropolisGlobal,
    ZHomodyneDisorder,
    ZzHomodyneDisorder,
    ZLocalXDisorder,
    ZzLocalXDisorder,
}

impl RngDomain {
    fn tag(self) -> u64 {
        match self {
            Self::CleanChain => 1,
            Self::ZDisorderNormals => 2,
            Self::ZzDisorderNormals => 3,
            Self::ZMetropolis => 4,
            Self::ZzMetropolis => 5,
            Self::ZCorrectedWolff => 6,
            Self::ZzCorrectedWolff => 7,
            Self::ZSequentialMetropolis => 8,
            Self::ZzSequentialMetropolis => 9,
            Self::ZMetropolisGlobal => 10,
            Self::ZzMetropolisGlobal => 11,
            Self::ZHomodyneDisorder => 12,
            Self::ZzHomodyneDisorder => 13,
            Self::ZLocalXDisorder => 14,
            Self::ZzLocalXDisorder => 15,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamId {
    domain: RngDomain,
    sample_id: u64,
}

impl StreamId {
    pub fn for_sample(domain: RngDomain, sample_id: u64) -> Self {
        Self { domain, sample_id }
    }

    pub fn domain(&self) -> RngDomain {
        self.domain
    }

    pub fn sample_id(&self) -> u64 {
        self.sample_id
    }

    fn seed(&self, base_seed: u64) -> u64 {
        derive_stream_seed(base_seed, self.domain.tag(), self.sample_id)
    }
}

#[derive(Debug)]
pub struct Rng64 {
    state: [u64; 4],
    cached_normal: Option<f64>,
}

impl Rng64 {
    pub fn seeded(seed: u64) -> Self {
        let mut splitmix = SplitMix64::new(seed);
        let mut state = [0_u64; 4];
        for slot in &mut state {
            *slot = splitmix.next_u64();
        }
        if state.iter().all(|value| *value == 0) {
            state[0] = 1;
        }
        Self {
            state,
            cached_normal: None,
        }
    }

    pub fn stream(base_seed: u64, stream: StreamId) -> Self {
        Self::seeded(stream.seed(base_seed))
    }
}

impl Rng64 {
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);

        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    pub fn uniform(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        ((value as f64) + 0.5) * (1.0 / ((1_u64 << 53) as f64))
    }

    pub fn log_uniform(&mut self) -> f64 {
        self.uniform().ln()
    }

    pub fn bool(&mut self) -> bool {
        self.next_u64() >> 63 == 1
    }

    pub fn usize(&mut self, upper: usize) -> usize {
        let upper = NonZeroUsize::new(upper).expect("upper bound must be positive");
        self.usize_nonzero(upper)
    }

    pub fn usize_nonzero(&mut self, upper: NonZeroUsize) -> usize {
        let upper = upper.get() as u64;
        let mut m = (self.next_u64() as u128) * (upper as u128);
        let mut l = m as u64;
        if l < upper {
            let t = upper.wrapping_neg() % upper;
            while l < t {
                m = (self.next_u64() as u128) * (upper as u128);
                l = m as u64;
            }
        }
        (m >> 64) as usize
    }

    pub fn normal(&mut self) -> f64 {
        if let Some(cached) = self.cached_normal.take() {
            return cached;
        }

        let u1 = self.uniform();
        let u2 = self.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        let z0 = radius * angle.cos();
        let z1 = radius * angle.sin();
        self.cached_normal = Some(z1);
        z0
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        splitmix64_finalize(self.state)
    }
}

const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn derive_stream_seed(base_seed: u64, domain_tag: u64, sample_id: u64) -> u64 {
    let mut hash = 0x243f_6a88_85a3_08d3;
    hash = hash_combine(hash, base_seed);
    hash = hash_combine(hash, domain_tag);
    hash_combine(hash, sample_id)
}

fn hash_combine(hash: u64, value: u64) -> u64 {
    splitmix64_finalize(hash ^ splitmix64_finalize(value))
}

fn splitmix64_finalize(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
