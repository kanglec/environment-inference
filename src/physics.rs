//! Shared numerical definitions from `notes/main.tex` and
//! `notes/simulation.tex`.

use crate::error::{DcftError, Result, require_finite};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Noise {
    Z,
    Zz,
}

impl Noise {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "z" | "Z" => Ok(Self::Z),
            "zz" | "ZZ" => Ok(Self::Zz),
            _ => Err(DcftError::invalid("noise must be 'z' or 'zz'")),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Z => "z",
            Self::Zz => "zz",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Measurement {
    Heterodyne,
    Homodyne,
    Gaussian(f64),
    LocalX,
}

impl Measurement {
    pub fn from_name(name: &str, gamma: Option<f64>) -> Result<Self> {
        match name {
            "heterodyne" => Ok(Self::Heterodyne),
            "homodyne" => Ok(Self::Homodyne),
            "gaussian" => gamma
                .map(Self::Gaussian)
                .ok_or_else(|| DcftError::invalid("gaussian measurement requires gamma")),
            "local-x" | "local_x" => Ok(Self::LocalX),
            _ => Err(DcftError::invalid(
                "measurement must be heterodyne, homodyne, gaussian, or local-x",
            )),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Heterodyne => "heterodyne",
            Self::Homodyne => "homodyne",
            Self::Gaussian(_) => "gaussian",
            Self::LocalX => "local-x",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProtocolParameters {
    pub p: f64,
    pub lambda: f64,
    pub gamma: Option<f64>,
    pub kappa: Option<f64>,
    pub coupling: Option<f64>,
    pub error_probability: Option<f64>,
}

impl ProtocolParameters {
    pub fn new(measurement: Measurement, p: f64) -> Result<Self> {
        let lambda = lambda_from_probability(p)?;
        match measurement {
            Measurement::Heterodyne => Ok(Self::gaussian(p, lambda, 2.0 * lambda)),
            Measurement::Homodyne => Ok(Self::gaussian(p, lambda, 4.0 * lambda)),
            Measurement::Gaussian(gamma) => {
                require_finite("gamma", gamma)?;
                if !(0.0..=4.0 * lambda).contains(&gamma) {
                    return Err(DcftError::invalid(format!(
                        "gamma must satisfy 0 <= gamma <= 4 lambda ({})",
                        4.0 * lambda
                    )));
                }
                Ok(Self::gaussian(p, lambda, gamma))
            }
            Measurement::LocalX => {
                // The direct forms ``(1-kappa)/2`` and ``atanh(kappa)`` lose
                // all mismatch probability when p approaches 1/2.  Express
                // the two quantities through the channel amplitudes instead.
                let correct_amplitude = (1.0 - p).sqrt();
                let error_amplitude = p.sqrt();
                let amplitude_difference = correct_amplitude - error_amplitude;
                let error_probability = 0.5 * amplitude_difference * amplitude_difference;
                let coupling = ((correct_amplitude + error_amplitude) / amplitude_difference).ln();
                let kappa = 2.0 * correct_amplitude * error_amplitude;
                require_finite("local-X error probability", error_probability)?;
                require_finite("local-X coupling", coupling)?;
                require_finite("local-X kappa", kappa)?;
                Ok(Self {
                    p,
                    lambda,
                    gamma: None,
                    kappa: Some(kappa),
                    coupling: Some(coupling),
                    error_probability: Some(error_probability),
                })
            }
        }
    }

    const fn gaussian(p: f64, lambda: f64, gamma: f64) -> Self {
        Self {
            p,
            lambda,
            gamma: Some(gamma),
            kappa: None,
            coupling: None,
            error_probability: None,
        }
    }
}

pub fn lambda_from_probability(p: f64) -> Result<f64> {
    require_finite("p", p)?;
    if !(0.0..0.5).contains(&p) {
        return Err(DcftError::invalid("p must satisfy 0 <= p < 1/2"));
    }
    Ok(-0.5 * (-2.0 * p).ln_1p())
}

pub fn noise_variables(boundary_spins: &[i8], noise: Noise) -> Result<Vec<i8>> {
    if boundary_spins.len() < 2 {
        return Err(DcftError::invalid(
            "boundary must contain at least two spins",
        ));
    }
    if boundary_spins.iter().any(|spin| !matches!(spin, -1 | 1)) {
        return Err(DcftError::invalid("spins must equal -1 or +1"));
    }
    Ok(match noise {
        Noise::Z => boundary_spins.to_vec(),
        Noise::Zz => boundary_spins
            .iter()
            .zip(boundary_spins.iter().cycle().skip(1))
            .take(boundary_spins.len())
            .map(|(left, right)| left * right)
            .collect(),
    })
}

pub fn gaussian_record(variables: &[i8], gamma: f64, normals: &[f64]) -> Result<Vec<f64>> {
    require_finite("gamma", gamma)?;
    if gamma < 0.0 {
        return Err(DcftError::invalid("gamma must be nonnegative"));
    }
    if variables.len() != normals.len() {
        return Err(DcftError::invalid(
            "noise variables and normal vector lengths differ",
        ));
    }
    if normals.iter().any(|value| !value.is_finite()) {
        return Err(DcftError::invalid("normal variates must be finite"));
    }
    let standard_deviation = gamma.sqrt();
    Ok(variables
        .iter()
        .zip(normals)
        .map(|(variable, normal)| gamma * f64::from(*variable) + standard_deviation * normal)
        .collect())
}

pub fn local_x_record(
    variables: &[i8],
    parameters: ProtocolParameters,
    uniforms: &[f64],
) -> Result<(Vec<i8>, Vec<f64>)> {
    if variables.len() != uniforms.len() {
        return Err(DcftError::invalid(
            "noise variables and uniform vector lengths differ",
        ));
    }
    if uniforms
        .iter()
        .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
    {
        return Err(DcftError::invalid("uniform variates must lie in [0, 1)"));
    }
    let error_probability = parameters
        .error_probability
        .ok_or_else(|| DcftError::invalid("local-X parameters required"))?;
    let coupling = parameters
        .coupling
        .ok_or_else(|| DcftError::invalid("local-X parameters required"))?;
    let outcomes: Vec<i8> = variables
        .iter()
        .zip(uniforms)
        .map(|(variable, uniform)| {
            if *uniform < error_probability {
                -*variable
            } else {
                *variable
            }
        })
        .collect();
    let fields = outcomes
        .iter()
        .map(|outcome| coupling * f64::from(*outcome))
        .collect();
    Ok((outcomes, fields))
}

#[must_use]
pub fn basis_spins(state: usize, sites: usize) -> Vec<i8> {
    (0..sites)
        .map(|site| if state & (1 << site) == 0 { 1 } else { -1 })
        .collect()
}

pub fn all_noise_eigenvalues(sites: usize, noise: Noise) -> Result<Vec<Vec<i8>>> {
    if !(2..usize::BITS as usize).contains(&sites) {
        return Err(DcftError::invalid(
            "sites must be between 2 and word_bits - 1",
        ));
    }
    (0..1_usize << sites)
        .map(|state| noise_variables(&basis_spins(state, sites), noise))
        .collect()
}

pub fn observable_eigenvalues(
    sites: usize,
    family: &str,
    origin: usize,
    separation: Option<usize>,
) -> Result<Vec<i8>> {
    if sites < 2 || origin >= sites {
        return Err(DcftError::invalid("invalid sites or observable origin"));
    }
    let separation = separation.unwrap_or(0) % sites;
    (0..1_usize << sites)
        .map(|state| {
            let spins = basis_spins(state, sites);
            let right = (origin + 1) % sites;
            let displaced = (origin + separation) % sites;
            let displaced_right = (displaced + 1) % sites;
            match family {
                "spin" => Ok(spins[origin]),
                "bond" => Ok(spins[origin] * spins[right]),
                "spin-pair" => Ok(spins[origin] * spins[displaced]),
                "bond-pair" => {
                    Ok(spins[origin] * spins[right] * spins[displaced] * spins[displaced_right])
                }
                _ => Err(DcftError::invalid(
                    "observable family must be spin, bond, spin-pair, or bond-pair",
                )),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_protocol_strengths_match_contract() {
        let p = 0.2;
        let lambda = lambda_from_probability(p).expect("valid p");
        let heterodyne = ProtocolParameters::new(Measurement::Heterodyne, p).expect("valid");
        let homodyne = ProtocolParameters::new(Measurement::Homodyne, p).expect("valid");
        assert_eq!(heterodyne.gamma, Some(2.0 * lambda));
        assert_eq!(homodyne.gamma, Some(4.0 * lambda));
    }

    #[test]
    fn arbitrary_gaussian_is_bounded() {
        let p = 0.1;
        let maximum = 4.0 * lambda_from_probability(p).expect("valid p");
        assert!(ProtocolParameters::new(Measurement::Gaussian(maximum), p).is_ok());
        assert!(ProtocolParameters::new(Measurement::Gaussian(maximum + 1.0e-12), p).is_err());
    }

    #[test]
    fn zz_variables_are_periodic_and_globally_invariant() {
        let spins = [1, -1, -1, 1];
        let flipped = spins.map(|spin| -spin);
        let expected = [-1, 1, -1, 1];
        assert_eq!(noise_variables(&spins, Noise::Zz).expect("valid"), expected);
        assert_eq!(
            noise_variables(&flipped, Noise::Zz).expect("valid"),
            expected
        );
    }

    #[test]
    fn records_have_correct_zero_strength_limit() {
        let record = gaussian_record(&[1, -1], 0.0, &[2.0, -3.0]).expect("valid");
        assert_eq!(record, [0.0, 0.0]);
    }

    #[test]
    fn local_x_parameters_remain_finite_near_the_projective_endpoint() {
        let p = 0.499_999_999_999;
        let parameters = ProtocolParameters::new(Measurement::LocalX, p).expect("valid p");
        let mismatch = parameters.error_probability.expect("mismatch probability");
        let coupling = parameters.coupling.expect("local-X coupling");
        assert!(mismatch > 0.0 && mismatch < 1.0e-20);
        assert!(coupling.is_finite() && coupling > 20.0);
        let (_outcomes, fields) =
            local_x_record(&[1, -1], parameters, &[0.25, 0.75]).expect("finite record");
        assert!(fields.iter().all(|field| field.is_finite()));
    }
}
