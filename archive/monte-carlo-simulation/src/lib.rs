pub mod error;
pub mod ising;
pub mod rng;

pub use error::{DcftError, Result};
pub use rng::{Rng64, RngDomain, StreamId};

pub fn throughput(items: usize, seconds: f64) -> f64 {
    if seconds == 0.0 {
        f64::INFINITY
    } else {
        items as f64 / seconds
    }
}
