use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, DcftError>;

#[derive(Debug)]
pub enum DcftError {
    InvalidParameter(String),
    Io(io::Error),
}

impl DcftError {
    pub fn invalid_parameter(message: impl Into<String>) -> Self {
        Self::InvalidParameter(message.into())
    }
}

impl fmt::Display for DcftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(message) => write!(formatter, "invalid parameter: {message}"),
            Self::Io(error) => write!(formatter, "io error: {error}"),
        }
    }
}

impl std::error::Error for DcftError {}

impl From<io::Error> for DcftError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn require_at_least(name: &str, value: usize, minimum: usize) -> Result<()> {
    if value < minimum {
        return Err(DcftError::invalid_parameter(format!(
            "{name} must be >= {minimum}"
        )));
    }
    Ok(())
}

pub fn require_finite(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() {
        return Err(DcftError::invalid_parameter(format!(
            "{name} must be finite"
        )));
    }
    Ok(())
}

pub fn require_positive_finite(name: &str, value: f64) -> Result<()> {
    require_finite(name, value)?;
    if value <= 0.0 {
        return Err(DcftError::invalid_parameter(format!(
            "{name} must be finite and > 0"
        )));
    }
    Ok(())
}

pub fn require_nonnegative_finite(name: &str, value: f64) -> Result<()> {
    require_finite(name, value)?;
    if value < 0.0 {
        return Err(DcftError::invalid_parameter(format!(
            "{name} must be finite and >= 0"
        )));
    }
    Ok(())
}
