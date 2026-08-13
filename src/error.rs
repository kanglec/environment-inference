use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcftError(String);

impl DcftError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for DcftError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for DcftError {}

pub type Result<T> = std::result::Result<T, DcftError>;

pub fn require_finite(name: &str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DcftError::invalid(format!("{name} must be finite")))
    }
}

pub fn require_positive(name: &str, value: f64) -> Result<()> {
    require_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(DcftError::invalid(format!("{name} must be positive")))
    }
}
