use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{DcftError, Result, require_at_least};

use super::super::storage::{AggregateFileHeader, AggregateRecord, read_aggregate_binary};

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisReport {
    pub samples: usize,
    pub lx: usize,
    pub lt: usize,
    pub energy_density: f64,
    pub magnetization_density: f64,
    pub boundary_magnetization: f64,
    pub local_spin_fidelity: f64,
    pub local_bond_fidelity: f64,
    pub spin_linear_corr: Vec<f64>,
    pub bond_linear_corr: Vec<f64>,
    pub spin_fidelity_corr: Vec<f64>,
    pub bond_fidelity_corr: Vec<f64>,
    pub spin_ea_corr: Vec<f64>,
    pub bond_ea_corr: Vec<f64>,
}

pub fn analyze_aggregate_file(path: impl AsRef<Path>) -> Result<AnalysisReport> {
    let (header, records) = read_aggregate_binary(path)?;
    analyze_records(&header, &records)
}

/// Write every measured separation and every derived correlator in a
/// self-contained, machine-readable table. Scalar run results are repeated on
/// each row so a single CSV remains sufficient for downstream analysis.
pub fn write_analysis_csv(path: impl AsRef<Path>, report: &AnalysisReport) -> Result<()> {
    let r_count = report.lx / 2 + 1;
    let correlators = [
        &report.spin_linear_corr,
        &report.spin_fidelity_corr,
        &report.spin_ea_corr,
        &report.bond_linear_corr,
        &report.bond_fidelity_corr,
        &report.bond_ea_corr,
    ];
    if correlators.iter().any(|values| values.len() != r_count) {
        return Err(DcftError::invalid_parameter(
            "analysis correlator length does not match lx",
        ));
    }
    if let Some(parent) = path.as_ref().parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "samples,lx,lt,energy_density,magnetization_density,boundary_magnetization,local_spin_fidelity,local_bond_fidelity,r,spin_linear_corr,spin_fidelity_corr,spin_ea_corr,bond_linear_corr,bond_fidelity_corr,bond_ea_corr"
    )?;
    for r in 0..r_count {
        writeln!(
            writer,
            "{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            report.samples,
            report.lx,
            report.lt,
            report.energy_density,
            report.magnetization_density,
            report.boundary_magnetization,
            report.local_spin_fidelity,
            report.local_bond_fidelity,
            r,
            report.spin_linear_corr[r],
            report.spin_fidelity_corr[r],
            report.spin_ea_corr[r],
            report.bond_linear_corr[r],
            report.bond_fidelity_corr[r],
            report.bond_ea_corr[r],
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Write one row per disorder realization and separation.
///
/// Keeping the independent outer-disorder records makes standard errors and
/// bootstrap intervals available without changing the stable aggregate CSV
/// schema consumed by `data-analysis`.
pub fn write_disorder_records_csv(
    path: impl AsRef<Path>,
    header: &AggregateFileHeader,
    records: &[AggregateRecord],
) -> Result<()> {
    require_at_least("aggregate record count", records.len(), 1)?;
    if let Some(parent) = path.as_ref().parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "disorder_id,lx,lt,p,local_spin_linear,local_spin_fidelity,local_bond_linear,local_bond_fidelity,r,spin_linear_corr,spin_fidelity_corr,bond_linear_corr,bond_fidelity_corr"
    )?;
    for record in records {
        record.validate(header)?;
        let local_spin_linear = mean(&record.boundary_spin_mean)?;
        let local_spin_fidelity = mean_abs(&record.boundary_spin_mean)?;
        let local_bond_linear = mean(&record.boundary_bond_mean)?;
        let local_bond_fidelity = mean_abs(&record.boundary_bond_mean)?;
        for r in 0..header.r_count() {
            writeln!(
                writer,
                "{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{:.17},{:.17},{:.17},{:.17}",
                record.disorder_id,
                header.lx,
                header.lt,
                header.p,
                local_spin_linear,
                local_spin_fidelity,
                local_bond_linear,
                local_bond_fidelity,
                r,
                record.spin_corr_signed[r],
                record.spin_corr_abs[r],
                record.bond_corr_signed[r],
                record.bond_corr_abs[r],
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

pub fn analyze_records(
    header: &AggregateFileHeader,
    records: &[AggregateRecord],
) -> Result<AnalysisReport> {
    require_at_least("aggregate record count", records.len(), 1)?;
    let lx = header.lx;
    let r_count = header.r_count();
    let disorder_norm = 1.0 / records.len() as f64;
    let mut energy = 0.0;
    let mut magnetization = 0.0;
    let mut boundary_magnetization = 0.0;
    let mut local_spin_fidelity = 0.0;
    let mut local_bond_fidelity = 0.0;
    let mut spin_linear_corr = vec![0.0; r_count];
    let mut bond_linear_corr = vec![0.0; r_count];
    let mut spin_fidelity_corr = vec![0.0; r_count];
    let mut bond_fidelity_corr = vec![0.0; r_count];
    let mut spin_ea_corr = vec![0.0; r_count];
    let mut bond_ea_corr = vec![0.0; r_count];

    for record in records {
        record.validate(header)?;
        energy += record.energy_mean * disorder_norm;
        magnetization += record.bulk_magnetization_sum_mean * disorder_norm;
        boundary_magnetization += mean(&record.boundary_spin_mean)? * disorder_norm;
        local_spin_fidelity += mean_abs(&record.boundary_spin_mean)? * disorder_norm;
        local_bond_fidelity += mean_abs(&record.boundary_bond_mean)? * disorder_norm;
        for r in 0..r_count {
            spin_linear_corr[r] += record.spin_corr_signed[r] * disorder_norm;
            bond_linear_corr[r] += record.bond_corr_signed[r] * disorder_norm;
            spin_fidelity_corr[r] += record.spin_corr_abs[r] * disorder_norm;
            bond_fidelity_corr[r] += record.bond_corr_abs[r] * disorder_norm;
        }
        for r in 0..r_count {
            let mut spin_sum = 0.0;
            let mut bond_sum = 0.0;
            for x in 0..lx {
                let y = (x + r) % lx;
                spin_sum += record.boundary_spin_mean[x] * record.boundary_spin_mean[y];
                bond_sum += record.boundary_bond_mean[x] * record.boundary_bond_mean[y];
            }
            spin_ea_corr[r] += spin_sum / lx as f64 * disorder_norm;
            bond_ea_corr[r] += bond_sum / lx as f64 * disorder_norm;
        }
    }

    Ok(AnalysisReport {
        samples: records.len(),
        lx,
        lt: header.lt,
        energy_density: energy / (header.lx * header.lt) as f64,
        magnetization_density: magnetization / (header.lx * header.lt) as f64,
        boundary_magnetization,
        local_spin_fidelity,
        local_bond_fidelity,
        spin_linear_corr,
        bond_linear_corr,
        spin_fidelity_corr,
        bond_fidelity_corr,
        spin_ea_corr,
        bond_ea_corr,
    })
}

fn mean(values: &[f64]) -> Result<f64> {
    require_at_least("value count", values.len(), 1)?;
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_abs(values: &[f64]) -> Result<f64> {
    require_at_least("value count", values.len(), 1)?;
    let value = values.iter().map(|value| value.abs()).sum::<f64>() / values.len() as f64;
    if !value.is_finite() {
        return Err(DcftError::invalid_parameter(
            "mean absolute value is not finite",
        ));
    }
    Ok(value)
}
