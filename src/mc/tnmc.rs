use nalgebra::DMatrix;

use crate::error::{DcftError, Result};
use crate::physics::Noise;
use crate::rng::Rng64;

use super::{Lattice, Model, UpdateStats};

const PROBABILITY_FLOOR: f64 = 1.0e-12;

#[derive(Debug, Clone)]
struct OpenRectangle {
    width: usize,
    height: usize,
    x_coordinates: Vec<usize>,
    t_coordinates: Vec<usize>,
    fields: Vec<f64>,
    horizontal: Vec<f64>,
    vertical: Vec<f64>,
}

impl OpenRectangle {
    fn from_periodic_cut(
        model: &Model,
        lattice: &Lattice,
        frozen_x: usize,
        frozen_t: usize,
    ) -> Result<Self> {
        if model.lx() != lattice.lx() || model.lt() != lattice.lt() {
            return Err(DcftError::invalid(
                "TNMC model and lattice dimensions must agree",
            ));
        }
        if frozen_x >= model.lx() || frozen_t >= model.lt() {
            return Err(DcftError::invalid("TNMC cut line is out of bounds"));
        }

        let width = model.lx() - 1;
        let height = model.lt() - 1;
        let x_coordinates = (1..model.lx())
            .map(|offset| (frozen_x + offset) % model.lx())
            .collect::<Vec<_>>();
        let t_coordinates = (1..model.lt())
            .map(|offset| (frozen_t + offset) % model.lt())
            .collect::<Vec<_>>();
        let mut output = Self {
            width,
            height,
            x_coordinates,
            t_coordinates,
            fields: vec![0.0; width * height],
            horizontal: vec![0.0; height * width.saturating_sub(1)],
            vertical: vec![0.0; height.saturating_sub(1) * width],
        };

        let couplings = model.couplings();
        for row in 0..height {
            let t = output.t_coordinates[row];
            for x_index in 0..width.saturating_sub(1) {
                let x = output.x_coordinates[x_index];
                let mut coupling = couplings.kx;
                if model.noise() == Some(Noise::Zz) && t == 0 {
                    coupling += model.record()[x];
                }
                output.horizontal[row * width.saturating_sub(1) + x_index] = coupling;
            }
            for x_index in 0..width {
                let x = output.x_coordinates[x_index];
                if model.noise() == Some(Noise::Z) && t == 0 {
                    output.fields[row * width + x_index] += model.record()[x];
                }
            }

            // The two bonds incident on the removed column become fields.  They
            // are distinct bonds even for Lx=2, where they join the same pair
            // of sites in opposite periodic directions.
            let fixed_spin = f64::from(lattice.get(frozen_x, t));
            output.fields[row * width] += couplings.kx * fixed_spin;
            output.fields[row * width + width - 1] += couplings.kx * fixed_spin;
            if model.noise() == Some(Noise::Zz) && t == 0 {
                let right_x = output.x_coordinates[width - 1];
                output.fields[row * width] += model.record()[frozen_x] * fixed_spin;
                output.fields[row * width + width - 1] += model.record()[right_x] * fixed_spin;
            }
        }

        for row in 0..height.saturating_sub(1) {
            for x_index in 0..width {
                output.vertical[row * width + x_index] = couplings.kt;
            }
        }

        // Likewise, the two bonds incident on the removed row become fields.
        // Both contributions are required for Lt=2.
        for x_index in 0..width {
            let x = output.x_coordinates[x_index];
            let fixed_spin = f64::from(lattice.get(x, frozen_t));
            output.fields[x_index] += couplings.kt * fixed_spin;
            output.fields[(height - 1) * width + x_index] += couplings.kt * fixed_spin;
        }

        if output
            .fields
            .iter()
            .chain(&output.horizontal)
            .chain(&output.vertical)
            .any(|value| !value.is_finite())
        {
            return Err(DcftError::invalid(
                "TNMC effective fields and couplings must be finite",
            ));
        }
        Ok(output)
    }

    fn active_spins(&self, lattice: &Lattice) -> Vec<i8> {
        let mut output = Vec::with_capacity(self.width * self.height);
        for &t in &self.t_coordinates {
            for &x in &self.x_coordinates {
                output.push(lattice.get(x, t));
            }
        }
        output
    }

    fn with_active_spins(&self, lattice: &Lattice, active: &[i8]) -> Result<Lattice> {
        self.validate_active_spins(active)?;
        let mut output = lattice.clone();
        for (row, &t) in self.t_coordinates.iter().enumerate() {
            for (column, &x) in self.x_coordinates.iter().enumerate() {
                let spin = active[row * self.width + column];
                let index = output.index(x, t);
                if output.get_index(index) != spin {
                    output.flip(index);
                }
            }
        }
        Ok(output)
    }

    fn validate_active_spins(&self, active: &[i8]) -> Result<()> {
        if active.len() != self.width * self.height {
            return Err(DcftError::invalid(
                "TNMC active spin count does not match the cut rectangle",
            ));
        }
        if active.iter().any(|spin| !matches!(spin, -1 | 1)) {
            return Err(DcftError::invalid("TNMC spins must equal -1 or +1"));
        }
        Ok(())
    }

    fn field(&self, row: usize, column: usize) -> f64 {
        self.fields[row * self.width + column]
    }

    fn horizontal(&self, row: usize, column: usize) -> f64 {
        self.horizontal[row * (self.width - 1) + column]
    }

    fn vertical(&self, row: usize, column: usize) -> f64 {
        self.vertical[row * self.width + column]
    }

    #[cfg(test)]
    fn log_weight(&self, active: &[i8]) -> f64 {
        let mut total = 0.0;
        for row in 0..self.height {
            for column in 0..self.width {
                total += self.field(row, column) * f64::from(active[row * self.width + column]);
                if column + 1 < self.width {
                    total += self.horizontal(row, column)
                        * f64::from(
                            active[row * self.width + column]
                                * active[row * self.width + column + 1],
                        );
                }
                if row + 1 < self.height {
                    total += self.vertical(row, column)
                        * f64::from(
                            active[row * self.width + column]
                                * active[(row + 1) * self.width + column],
                        );
                }
            }
        }
        total
    }
}

#[derive(Debug, Clone)]
struct SiteTensor {
    left: usize,
    right: usize,
    data: Vec<f64>,
}

impl SiteTensor {
    fn zeros(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            data: vec![0.0; left * right * 2],
        }
    }

    fn get(&self, left: usize, right: usize, physical: usize) -> f64 {
        self.data[(left * self.right + right) * 2 + physical]
    }

    fn set(&mut self, left: usize, right: usize, physical: usize, value: f64) {
        self.data[(left * self.right + right) * 2 + physical] = value;
    }
}

#[derive(Debug, Clone)]
struct MatrixProductState {
    sites: Vec<SiteTensor>,
}

impl MatrixProductState {
    fn ones(width: usize) -> Self {
        Self {
            sites: (0..width)
                .map(|_| SiteTensor {
                    left: 1,
                    right: 1,
                    data: vec![1.0, 1.0],
                })
                .collect(),
        }
    }

    // Keeping the left-edge, bulk, right-edge, and width-one tensor layouts
    // together makes their virtual-index convention auditable.
    #[allow(clippy::too_many_lines)]
    fn transfer_up(
        &self,
        rectangle: &OpenRectangle,
        next_row: usize,
        maximum_bond_dimension: usize,
    ) -> Result<Self> {
        let width = rectangle.width;
        let mut sites = Vec::with_capacity(width);
        if width == 1 {
            let input = &self.sites[0];
            let mut output = SiteTensor::zeros(input.left, input.right);
            let field = rectangle.field(next_row, 0);
            let vertical = rectangle.vertical(next_row - 1, 0);
            let shift = maximum_exponent(field, vertical, None)?;
            for left in 0..input.left {
                for right in 0..input.right {
                    for upper in 0..2 {
                        let upper_spin = spin_value(upper);
                        let mut value = 0.0;
                        for lower in 0..2 {
                            let lower_spin = spin_value(lower);
                            let exponent = field * lower_spin + vertical * upper_spin * lower_spin;
                            value += input.get(left, right, lower) * (exponent - shift).exp();
                        }
                        output.set(left, right, upper, value);
                    }
                }
            }
            sites.push(output);
        } else {
            for column in 0..width {
                let input = &self.sites[column];
                let field = rectangle.field(next_row, column);
                let vertical = rectangle.vertical(next_row - 1, column);
                let preceding_horizontal = if column == 0 {
                    None
                } else {
                    Some(rectangle.horizontal(next_row, column - 1))
                };
                let shift = maximum_exponent(field, vertical, preceding_horizontal)?;
                if column == 0 {
                    let mut output = SiteTensor::zeros(input.left, input.right * 2);
                    for left in 0..input.left {
                        for right in 0..input.right {
                            for lower in 0..2 {
                                let lower_spin = spin_value(lower);
                                for upper in 0..2 {
                                    let upper_spin = spin_value(upper);
                                    let exponent =
                                        field * lower_spin + vertical * upper_spin * lower_spin;
                                    output.set(
                                        left,
                                        right * 2 + lower,
                                        upper,
                                        input.get(left, right, lower) * (exponent - shift).exp(),
                                    );
                                }
                            }
                        }
                    }
                    sites.push(output);
                } else if column + 1 == width {
                    let mut output = SiteTensor::zeros(input.left * 2, input.right);
                    let horizontal = preceding_horizontal.expect("last site has a left bond");
                    for left in 0..input.left {
                        for previous in 0..2 {
                            let previous_spin = spin_value(previous);
                            for right in 0..input.right {
                                for upper in 0..2 {
                                    let upper_spin = spin_value(upper);
                                    let mut value = 0.0;
                                    for lower in 0..2 {
                                        let lower_spin = spin_value(lower);
                                        let exponent = field * lower_spin
                                            + vertical * upper_spin * lower_spin
                                            + horizontal * previous_spin * lower_spin;
                                        value += input.get(left, right, lower)
                                            * (exponent - shift).exp();
                                    }
                                    output.set(left * 2 + previous, right, upper, value);
                                }
                            }
                        }
                    }
                    sites.push(output);
                } else {
                    let mut output = SiteTensor::zeros(input.left * 2, input.right * 2);
                    let horizontal = preceding_horizontal.expect("interior site has a left bond");
                    for left in 0..input.left {
                        for previous in 0..2 {
                            let previous_spin = spin_value(previous);
                            for right in 0..input.right {
                                for lower in 0..2 {
                                    let lower_spin = spin_value(lower);
                                    for upper in 0..2 {
                                        let upper_spin = spin_value(upper);
                                        let exponent = field * lower_spin
                                            + vertical * upper_spin * lower_spin
                                            + horizontal * previous_spin * lower_spin;
                                        output.set(
                                            left * 2 + previous,
                                            right * 2 + lower,
                                            upper,
                                            input.get(left, right, lower)
                                                * (exponent - shift).exp(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    sites.push(output);
                }
            }
        }
        let mut output = Self { sites };
        output.compress(maximum_bond_dimension)?;
        Ok(output)
    }

    fn compress(&mut self, maximum_bond_dimension: usize) -> Result<()> {
        if maximum_bond_dimension == 0 {
            return Err(DcftError::invalid(
                "TNMC maximum bond dimension must be positive",
            ));
        }
        if self.sites.len() == 1 {
            normalize_slice(&mut self.sites[0].data)?;
            return Ok(());
        }

        // Right-canonicalize without truncation.  The subsequent left-to-right
        // SVD then truncates true Schmidt bonds of the represented row message.
        for site_index in (1..self.sites.len()).rev() {
            let site = self.sites[site_index].clone();
            let matrix = DMatrix::from_fn(site.left, site.right * 2, |left, column| {
                let physical = column / site.right;
                let right = column % site.right;
                site.get(left, right, physical)
            });
            let (u, singular, v_transpose) = decompose(matrix)?;
            let rank = singular.len();
            let mut canonical = SiteTensor::zeros(rank, site.right);
            for bond in 0..rank {
                for physical in 0..2 {
                    for right in 0..site.right {
                        canonical.set(
                            bond,
                            right,
                            physical,
                            v_transpose[(bond, physical * site.right + right)],
                        );
                    }
                }
            }
            let mut factor = DMatrix::from_fn(site.left, rank, |left, bond| {
                u[(left, bond)] * singular[bond]
            });
            normalize_matrix(&mut factor)?;
            let previous = self.sites[site_index - 1].clone();
            if previous.right != site.left {
                return Err(DcftError::invalid("TNMC MPS bond mismatch"));
            }
            let mut absorbed = SiteTensor::zeros(previous.left, rank);
            for left in 0..previous.left {
                for bond in 0..rank {
                    for physical in 0..2 {
                        let value = (0..previous.right)
                            .map(|old| previous.get(left, old, physical) * factor[(old, bond)])
                            .sum();
                        absorbed.set(left, bond, physical, value);
                    }
                }
            }
            self.sites[site_index] = canonical;
            self.sites[site_index - 1] = absorbed;
        }

        for site_index in 0..self.sites.len() - 1 {
            let site = self.sites[site_index].clone();
            let matrix = DMatrix::from_fn(site.left * 2, site.right, |row, right| {
                let left = row / 2;
                let physical = row % 2;
                site.get(left, right, physical)
            });
            let (u, singular, v_transpose) = decompose(matrix)?;
            let rank = maximum_bond_dimension.min(singular.len());
            let mut truncated = SiteTensor::zeros(site.left, rank);
            for left in 0..site.left {
                for bond in 0..rank {
                    for physical in 0..2 {
                        truncated.set(left, bond, physical, u[(left * 2 + physical, bond)]);
                    }
                }
            }
            let mut remainder = DMatrix::from_fn(rank, site.right, |bond, old| {
                singular[bond] * v_transpose[(bond, old)]
            });
            normalize_matrix(&mut remainder)?;
            let next = self.sites[site_index + 1].clone();
            if next.left != site.right {
                return Err(DcftError::invalid("TNMC MPS bond mismatch"));
            }
            let mut absorbed = SiteTensor::zeros(rank, next.right);
            for bond in 0..rank {
                for right in 0..next.right {
                    for physical in 0..2 {
                        let value = (0..next.left)
                            .map(|old| remainder[(bond, old)] * next.get(old, right, physical))
                            .sum();
                        absorbed.set(bond, right, physical, value);
                    }
                }
            }
            self.sites[site_index] = truncated;
            self.sites[site_index + 1] = absorbed;
        }
        normalize_slice(&mut self.sites.last_mut().expect("nonempty MPS").data)?;
        Ok(())
    }
}

fn decompose(matrix: DMatrix<f64>) -> Result<(DMatrix<f64>, Vec<f64>, DMatrix<f64>)> {
    let decomposition = matrix.svd(true, true);
    let u = decomposition
        .u
        .ok_or_else(|| DcftError::invalid("TNMC SVD did not return left vectors"))?;
    let v_transpose = decomposition
        .v_t
        .ok_or_else(|| DcftError::invalid("TNMC SVD did not return right vectors"))?;
    let singular = decomposition.singular_values.as_slice().to_vec();
    if singular.is_empty() || singular.iter().any(|value| !value.is_finite()) {
        return Err(DcftError::invalid(
            "TNMC SVD produced invalid singular values",
        ));
    }
    Ok((u, singular, v_transpose))
}

fn normalize_matrix(matrix: &mut DMatrix<f64>) -> Result<()> {
    normalize_slice(matrix.as_mut_slice())
}

fn normalize_slice(values: &mut [f64]) -> Result<()> {
    let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return Err(DcftError::invalid(
            "TNMC contraction produced a zero or non-finite tensor",
        ));
    }
    for value in values {
        *value /= scale;
    }
    Ok(())
}

fn maximum_exponent(field: f64, vertical: f64, horizontal: Option<f64>) -> Result<f64> {
    let mut maximum = f64::NEG_INFINITY;
    for upper in [-1.0, 1.0] {
        for lower in [-1.0, 1.0] {
            if let Some(coupling) = horizontal {
                for previous in [-1.0, 1.0] {
                    maximum = maximum.max(
                        field * lower + vertical * upper * lower + coupling * previous * lower,
                    );
                }
            } else {
                maximum = maximum.max(field * lower + vertical * upper * lower);
            }
        }
    }
    if maximum.is_finite() {
        Ok(maximum)
    } else {
        Err(DcftError::invalid("TNMC local exponent is not finite"))
    }
}

struct ConditionalProposal {
    suffix_messages: Vec<MatrixProductState>,
    maximum_bond_dimension: usize,
}

impl ConditionalProposal {
    fn new(rectangle: &OpenRectangle, maximum_bond_dimension: usize) -> Result<Self> {
        if maximum_bond_dimension == 0 {
            return Err(DcftError::invalid(
                "TNMC maximum bond dimension must be positive",
            ));
        }
        let mut suffix_messages = vec![None; rectangle.height];
        suffix_messages[rectangle.height - 1] = Some(MatrixProductState::ones(rectangle.width));
        for row in (0..rectangle.height - 1).rev() {
            let next = suffix_messages[row + 1]
                .as_ref()
                .expect("lower suffix was constructed")
                .transfer_up(rectangle, row + 1, maximum_bond_dimension)?;
            suffix_messages[row] = Some(next);
        }
        Ok(Self {
            suffix_messages: suffix_messages
                .into_iter()
                .map(|message| message.expect("every suffix was constructed"))
                .collect(),
            maximum_bond_dimension,
        })
    }

    fn log_probability(&self, rectangle: &OpenRectangle, active: &[i8]) -> Result<(f64, u64)> {
        rectangle.validate_active_spins(active)?;
        let (_, log_probability, regularized) = self.traverse(rectangle, Some(active), None)?;
        Ok((log_probability, regularized))
    }

    fn sample(&self, rectangle: &OpenRectangle, rng: &mut Rng64) -> Result<(Vec<i8>, f64, u64)> {
        self.traverse(rectangle, None, Some(rng))
    }

    fn traverse(
        &self,
        rectangle: &OpenRectangle,
        fixed: Option<&[i8]>,
        mut rng: Option<&mut Rng64>,
    ) -> Result<(Vec<i8>, f64, u64)> {
        debug_assert!(self.maximum_bond_dimension > 0);
        let mut active = Vec::with_capacity(rectangle.width * rectangle.height);
        let mut log_probability = 0.0;
        let mut regularized = 0_u64;
        for row in 0..rectangle.height {
            let previous_row = if row == 0 {
                None
            } else {
                Some(&active[(row - 1) * rectangle.width..row * rectangle.width])
            };
            let fields = (0..rectangle.width)
                .map(|column| {
                    rectangle.field(row, column)
                        + previous_row.map_or(0.0, |previous| {
                            rectangle.vertical(row - 1, column) * f64::from(previous[column])
                        })
                })
                .collect::<Vec<_>>();
            let message = &self.suffix_messages[row];
            let right = right_environments(rectangle, row, &fields, message)?;
            let mut left = vec![1.0];
            let mut preceding_spin = None;
            for column in 0..rectangle.width {
                let tensor = &message.sites[column];
                if left.len() != tensor.left {
                    return Err(DcftError::invalid("TNMC left environment bond mismatch"));
                }
                let local_log = |physical: usize| {
                    let spin = spin_value(physical);
                    fields[column] * spin
                        + preceding_spin.map_or(0.0, |previous| {
                            rectangle.horizontal(row, column - 1) * previous * spin
                        })
                };
                let mut contractions = [0.0; 2];
                for physical in 0..2 {
                    for (left_bond, left_value) in left.iter().copied().enumerate() {
                        for right_bond in 0..tensor.right {
                            contractions[physical] += left_value
                                * tensor.get(left_bond, right_bond, physical)
                                * right[column][right_bond * 2 + physical];
                        }
                    }
                }
                let mut probability_plus = if contractions
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
                {
                    let log_minus = local_log(0) + contractions[0].ln();
                    let log_plus = local_log(1) + contractions[1].ln();
                    logistic(log_plus - log_minus)
                } else {
                    regularized += 1;
                    0.5
                };
                if !(PROBABILITY_FLOOR..=1.0 - PROBABILITY_FLOOR).contains(&probability_plus) {
                    regularized += 1;
                    probability_plus =
                        probability_plus.clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR);
                }

                let physical = if let Some(configuration) = fixed {
                    spin_index(configuration[row * rectangle.width + column])
                } else {
                    usize::from(
                        rng.as_deref_mut()
                            .expect("sampling traversal has an RNG")
                            .uniform()
                            < probability_plus,
                    )
                };
                let selected_probability = if physical == 1 {
                    probability_plus
                } else {
                    1.0 - probability_plus
                };
                log_probability += selected_probability.ln();
                let selected_spin = spin_value_i8(physical);
                active.push(selected_spin);

                let local_shift = local_log(0).max(local_log(1));
                let local_factor = (local_log(physical) - local_shift).exp();
                let mut next_left = vec![0.0; tensor.right];
                for (left_bond, left_value) in left.iter().copied().enumerate() {
                    for (right_bond, value) in next_left.iter_mut().enumerate() {
                        *value +=
                            left_value * tensor.get(left_bond, right_bond, physical) * local_factor;
                    }
                }
                normalize_environment(&mut next_left);
                left = next_left;
                preceding_spin = Some(f64::from(selected_spin));
            }
        }
        Ok((active, log_probability, regularized))
    }
}

fn right_environments(
    rectangle: &OpenRectangle,
    row: usize,
    fields: &[f64],
    message: &MatrixProductState,
) -> Result<Vec<Vec<f64>>> {
    let width = rectangle.width;
    let mut output = vec![Vec::new(); width];
    let last = &message.sites[width - 1];
    if last.right != 1 {
        return Err(DcftError::invalid(
            "TNMC MPS has a nontrivial right boundary",
        ));
    }
    output[width - 1] = vec![1.0; 2];
    for column in (0..width - 1).rev() {
        let current = &message.sites[column];
        let next = &message.sites[column + 1];
        if current.right != next.left {
            return Err(DcftError::invalid("TNMC right environment bond mismatch"));
        }
        let coupling = rectangle.horizontal(row, column);
        let shift = maximum_pair_exponent(fields[column + 1], coupling)?;
        let mut environment = vec![0.0; current.right * 2];
        for bond in 0..current.right {
            for physical in 0..2 {
                let spin = spin_value(physical);
                for next_physical in 0..2 {
                    let next_spin = spin_value(next_physical);
                    let factor = (fields[column + 1] * next_spin + coupling * spin * next_spin
                        - shift)
                        .exp();
                    for right_bond in 0..next.right {
                        environment[bond * 2 + physical] +=
                            next.get(bond, right_bond, next_physical)
                                * factor
                                * output[column + 1][right_bond * 2 + next_physical];
                    }
                }
            }
        }
        normalize_environment(&mut environment);
        output[column] = environment;
    }
    Ok(output)
}

fn maximum_pair_exponent(field: f64, coupling: f64) -> Result<f64> {
    let maximum = [-1.0, 1.0]
        .into_iter()
        .flat_map(|left| {
            [-1.0, 1.0]
                .into_iter()
                .map(move |right| field * right + coupling * left * right)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    if maximum.is_finite() {
        Ok(maximum)
    } else {
        Err(DcftError::invalid("TNMC row exponent is not finite"))
    }
}

fn normalize_environment(values: &mut [f64]) {
    let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale.is_finite() && scale > 0.0 {
        for value in values {
            *value /= scale;
        }
    }
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

const fn spin_value(physical: usize) -> f64 {
    if physical == 0 { -1.0 } else { 1.0 }
}

const fn spin_value_i8(physical: usize) -> i8 {
    if physical == 0 { -1 } else { 1 }
}

const fn spin_index(spin: i8) -> usize {
    if spin == -1 { 0 } else { 1 }
}

pub(super) fn update(
    model: &Model,
    lattice: &mut Lattice,
    rng: &mut Rng64,
    statistics: &mut UpdateStats,
    maximum_bond_dimension: usize,
) -> Result<()> {
    if maximum_bond_dimension == 0 {
        return Err(DcftError::invalid(
            "TNMC maximum bond dimension must be positive",
        ));
    }
    let frozen_x = rng.index(lattice.lx())?;
    let frozen_t = rng.index(lattice.lt())?;
    let rectangle = OpenRectangle::from_periodic_cut(model, lattice, frozen_x, frozen_t)?;
    let proposal = ConditionalProposal::new(&rectangle, maximum_bond_dimension)?;
    let old_active = rectangle.active_spins(lattice);
    let (old_log_probability, old_regularized) =
        proposal.log_probability(&rectangle, &old_active)?;
    let (new_active, new_log_probability, new_regularized) = proposal.sample(&rectangle, rng)?;
    let candidate = rectangle.with_active_spins(lattice, &new_active)?;
    let log_acceptance = model.energy(lattice) - model.energy(&candidate) + old_log_probability
        - new_log_probability;
    if log_acceptance.is_nan() {
        return Err(DcftError::invalid("TNMC acceptance ratio is NaN"));
    }

    statistics.sweeps += 1;
    statistics.tnmc_proposed += 1;
    statistics.tnmc_sites_proposed += (rectangle.width * rectangle.height) as u64;
    statistics.tnmc_conditionals_regularized += old_regularized + new_regularized;
    if log_acceptance >= 0.0 || rng.log_uniform() < log_acceptance {
        *lattice = candidate;
        statistics.tnmc_accepted += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::Couplings;

    fn lattice_from_bits(lx: usize, lt: usize, bits: usize) -> Lattice {
        let spins = (0..lx * lt)
            .map(|site| if bits & (1 << site) == 0 { 1 } else { -1 })
            .collect();
        Lattice::new(lx, lt, spins).expect("valid enumerated lattice")
    }

    fn state_index(lattice: &Lattice) -> usize {
        lattice
            .spins()
            .iter()
            .enumerate()
            .fold(0, |bits, (site, spin)| {
                if *spin == -1 {
                    bits | (1 << site)
                } else {
                    bits
                }
            })
    }

    fn active_from_bits(count: usize, bits: usize) -> Vec<i8> {
        (0..count)
            .map(|site| if bits & (1 << site) == 0 { 1 } else { -1 })
            .collect()
    }

    #[test]
    fn periodic_cut_weight_matches_full_energy_for_fields_and_bonds() {
        let couplings = Couplings::new(0.31, 0.67).expect("valid couplings");
        for (lx, lt) in [(2, 2), (3, 2), (3, 3), (4, 3)] {
            let record = (0..lx)
                .map(|x| [0.73, -1.19, 0.41, -0.27][x])
                .collect::<Vec<_>>();
            for noise in [Noise::Z, Noise::Zz] {
                let model = Model::posterior(lx, lt, couplings, noise, record.clone())
                    .expect("valid disordered model");
                for base_bits in [0, (1_usize << (lx * lt)) / 3, (1_usize << (lx * lt)) - 1] {
                    let base = lattice_from_bits(lx, lt, base_bits);
                    for frozen_x in 0..lx {
                        for frozen_t in 0..lt {
                            let rectangle =
                                OpenRectangle::from_periodic_cut(&model, &base, frozen_x, frozen_t)
                                    .expect("valid cut");
                            let base_active = rectangle.active_spins(&base);
                            let constant =
                                -model.energy(&base) - rectangle.log_weight(&base_active);
                            let active_count = rectangle.width * rectangle.height;
                            for active_bits in 0..1_usize << active_count {
                                let active = active_from_bits(active_count, active_bits);
                                let candidate = rectangle
                                    .with_active_spins(&base, &active)
                                    .expect("valid active configuration");
                                let actual =
                                    -model.energy(&candidate) - rectangle.log_weight(&active);
                                assert!(
                                    (actual - constant).abs() < 2.0e-12,
                                    "cut mismatch for {lx}x{lt}, {noise:?}, x={frozen_x}, t={frozen_t}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn untruncated_proposal_equals_exact_disordered_conditional() {
        let lx = 4;
        let lt = 3;
        let couplings = Couplings::new(0.23, 0.81).expect("valid couplings");
        let base = lattice_from_bits(lx, lt, 0b1011_0010_1101);
        for noise in [Noise::Z, Noise::Zz] {
            let model = Model::posterior(lx, lt, couplings, noise, vec![0.8, -1.1, 0.37, -0.2])
                .expect("valid disordered model");
            let rectangle =
                OpenRectangle::from_periodic_cut(&model, &base, 1, 2).expect("valid cut");
            let proposal = ConditionalProposal::new(&rectangle, 64).expect("exact contraction");
            let count = rectangle.width * rectangle.height;
            let mut log_targets = Vec::new();
            let mut log_proposals = Vec::new();
            for bits in 0..1_usize << count {
                let active = active_from_bits(count, bits);
                log_targets.push(rectangle.log_weight(&active));
                let (log_probability, regularized) = proposal
                    .log_probability(&rectangle, &active)
                    .expect("valid proposal probability");
                assert_eq!(regularized, 0);
                log_proposals.push(log_probability);
            }
            let target_normalizer = log_targets.iter().copied().fold(f64::NEG_INFINITY, log_add);
            let proposal_total: f64 = log_proposals.iter().map(|value| value.exp()).sum();
            assert!((proposal_total - 1.0).abs() < 2.0e-12);
            let mut maximum_difference = 0.0_f64;
            for (target, proposal) in log_targets.iter().zip(&log_proposals) {
                let exact = (target - target_normalizer).exp();
                maximum_difference = maximum_difference.max((proposal.exp() - exact).abs());
                assert!(
                    (proposal.exp() - exact).abs() < 2.0e-11,
                    "exact conditional mismatch for {noise:?}: proposal={}, exact={exact}",
                    proposal.exp()
                );
            }
            eprintln!(
                "untruncated TNMC {noise:?}: normalization error={:.3e}, max probability error={maximum_difference:.3e}",
                (proposal_total - 1.0).abs()
            );
        }
    }

    #[test]
    fn exhaustive_truncated_kernel_obeys_detailed_balance_with_disorder() {
        let lx = 3;
        let lt = 3;
        let state_count = 1_usize << (lx * lt);
        let line_probability = 1.0 / (lx * lt) as f64;
        let couplings = Couplings::new(0.31, 0.67).expect("valid couplings");
        for noise in [Noise::Z, Noise::Zz] {
            // The middle value makes one effective ZZ boundary coupling
            // antiferromagnetic, so this is also a frustrated test case.
            let model = Model::posterior(lx, lt, couplings, noise, vec![0.9, -1.2, 0.4])
                .expect("valid disordered model");
            let states = (0..state_count)
                .map(|bits| lattice_from_bits(lx, lt, bits))
                .collect::<Vec<_>>();
            let weights = states
                .iter()
                .map(|state| (-model.energy(state)).exp())
                .collect::<Vec<_>>();
            let mut transition = vec![0.0; state_count * state_count];
            let mut maximum_row_error = 0.0_f64;

            for (old_index, old) in states.iter().enumerate() {
                for frozen_x in 0..lx {
                    for frozen_t in 0..lt {
                        let rectangle =
                            OpenRectangle::from_periodic_cut(&model, old, frozen_x, frozen_t)
                                .expect("valid cut");
                        let proposal =
                            ConditionalProposal::new(&rectangle, 1).expect("chi=1 proposal");
                        let old_active = rectangle.active_spins(old);
                        let (old_log_probability, _) = proposal
                            .log_probability(&rectangle, &old_active)
                            .expect("old proposal probability");
                        let active_count = rectangle.width * rectangle.height;
                        let mut line_total = 0.0;
                        for bits in 0..1_usize << active_count {
                            let active = active_from_bits(active_count, bits);
                            let (new_log_probability, _) = proposal
                                .log_probability(&rectangle, &active)
                                .expect("new proposal probability");
                            let candidate = rectangle
                                .with_active_spins(old, &active)
                                .expect("candidate");
                            let new_index = state_index(&candidate);
                            let log_acceptance = model.energy(old) - model.energy(&candidate)
                                + old_log_probability
                                - new_log_probability;
                            let acceptance = log_acceptance.min(0.0).exp();
                            let proposal_probability = new_log_probability.exp();
                            transition[old_index * state_count + new_index] +=
                                line_probability * proposal_probability * acceptance;
                            transition[old_index * state_count + old_index] +=
                                line_probability * proposal_probability * (1.0 - acceptance);
                            line_total += proposal_probability;
                        }
                        assert!((line_total - 1.0).abs() < 3.0e-12);
                    }
                }
                let row_total: f64 = transition
                    [old_index * state_count..(old_index + 1) * state_count]
                    .iter()
                    .sum();
                maximum_row_error = maximum_row_error.max((row_total - 1.0).abs());
                assert!((row_total - 1.0).abs() < 2.0e-11);
            }

            let mut maximum_relative_balance_error = 0.0_f64;
            for left in 0..state_count {
                for right in 0..state_count {
                    let forward = weights[left] * transition[left * state_count + right];
                    let reverse = weights[right] * transition[right * state_count + left];
                    let scale = forward.abs().max(reverse.abs()).max(1.0e-14);
                    maximum_relative_balance_error =
                        maximum_relative_balance_error.max((forward - reverse).abs() / scale);
                    assert!(
                        (forward - reverse).abs() < 3.0e-10 * scale,
                        "detailed balance failed for {noise:?}, {left}->{right}: {forward} != {reverse}"
                    );
                }
            }
            eprintln!(
                "truncated TNMC {noise:?}: max row error={maximum_row_error:.3e}, max relative detailed-balance error={maximum_relative_balance_error:.3e}"
            );
        }
    }

    fn log_add(left: f64, right: f64) -> f64 {
        let maximum = left.max(right);
        if maximum.is_infinite() {
            maximum
        } else {
            maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
        }
    }
}
