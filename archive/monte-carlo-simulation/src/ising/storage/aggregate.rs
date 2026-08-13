use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{DcftError, Result, require_at_least, require_finite};

use super::super::update::FixedDisorderUpdate;
use super::super::workflow::{MeasurementKind, NoiseKind, measurement_parameter};
use super::clean::CleanSampleFileHeader;

const MAGIC_V2: &[u8; 11] = b"DCFT_AGG_V2";
const MAGIC_V3: &[u8; 11] = b"DCFT_AGG_V3";
const HEADER_LEN: u64 = 11 + 8 * 18;

#[derive(Debug, Clone, PartialEq)]
pub struct AggregateFileHeader {
    pub clean_sample_count: usize,
    pub lx: usize,
    pub lt: usize,
    pub kx: f64,
    pub kt: f64,
    pub seed: u64,
    pub noise: NoiseKind,
    pub measurement: MeasurementKind,
    pub p: f64,
    pub mu: f64,
    pub sample_start: u64,
    pub sample_count: usize,
    pub disorder_update: FixedDisorderUpdate,
    pub disorder_therm_sweeps: usize,
    pub measurements: usize,
    pub skip_sweeps: usize,
}

impl AggregateFileHeader {
    pub fn from_measurement_input(
        input: &super::super::workflow::MeasureStageInput,
        clean_header: &CleanSampleFileHeader,
    ) -> Result<Self> {
        Ok(Self {
            clean_sample_count: clean_header.sample_count,
            lx: clean_header.lx,
            lt: clean_header.lt,
            kx: clean_header.kx,
            kt: clean_header.kt,
            seed: clean_header.seed,
            noise: input.noise,
            measurement: input.measurement,
            p: input.p,
            mu: measurement_parameter(input.measurement, input.p)?,
            sample_start: input.sample_start,
            sample_count: input.sample_count,
            disorder_update: input.disorder_update,
            disorder_therm_sweeps: input.disorder_therm_sweeps,
            measurements: input.measurements,
            skip_sweeps: input.skip_sweeps,
        })
    }

    pub fn validate(&self) -> Result<()> {
        require_at_least("lx", self.lx, 2)?;
        require_at_least("lt", self.lt, 2)?;
        require_at_least("sample_count", self.sample_count, 1)?;
        require_at_least("measurements", self.measurements, 1)?;
        require_finite("kx", self.kx)?;
        require_finite("kt", self.kt)?;
        require_finite("p", self.p)?;
        require_finite("mu", self.mu)?;
        let expected_mu = measurement_parameter(self.measurement, self.p)?;
        if self.mu.to_bits() != expected_mu.to_bits() {
            return Err(DcftError::invalid_parameter(
                "aggregate measurement parameter is inconsistent with measurement and p",
            ));
        }
        let end = self
            .sample_start
            .checked_add(self.sample_count as u64)
            .ok_or_else(|| DcftError::invalid_parameter("aggregate sample range overflows u64"))?;
        if end > self.clean_sample_count as u64 {
            return Err(DcftError::invalid_parameter(
                "aggregate sample range exceeds clean sample count",
            ));
        }
        Ok(())
    }

    pub fn r_count(&self) -> usize {
        self.lx / 2 + 1
    }

    pub fn record_len(&self) -> u64 {
        let f64_count = 2 + 2 * self.lx + 4 * self.r_count();
        8 + 8 * f64_count as u64
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AggregateRecord {
    pub disorder_id: u64,
    pub energy_mean: f64,
    pub bulk_magnetization_sum_mean: f64,
    pub boundary_spin_mean: Vec<f64>,
    pub boundary_bond_mean: Vec<f64>,
    pub spin_corr_signed: Vec<f64>,
    pub bond_corr_signed: Vec<f64>,
    pub spin_corr_abs: Vec<f64>,
    pub bond_corr_abs: Vec<f64>,
}

impl AggregateRecord {
    pub fn validate(&self, header: &AggregateFileHeader) -> Result<()> {
        if self.boundary_spin_mean.len() != header.lx
            || self.boundary_bond_mean.len() != header.lx
            || self.spin_corr_signed.len() != header.r_count()
            || self.bond_corr_signed.len() != header.r_count()
            || self.spin_corr_abs.len() != header.r_count()
            || self.bond_corr_abs.len() != header.r_count()
        {
            return Err(DcftError::invalid_parameter(
                "aggregate record vector length does not match header",
            ));
        }
        for value in self
            .boundary_spin_mean
            .iter()
            .chain(&self.boundary_bond_mean)
            .chain(&self.spin_corr_signed)
            .chain(&self.bond_corr_signed)
            .chain(&self.spin_corr_abs)
            .chain(&self.bond_corr_abs)
        {
            require_finite("aggregate record value", *value)?;
        }
        require_finite("energy_mean", self.energy_mean)?;
        require_finite(
            "bulk_magnetization_sum_mean",
            self.bulk_magnetization_sum_mean,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AggregateMetadata {
    pub header: AggregateFileHeader,
    pub record_count: usize,
}

pub fn write_aggregate_binary(
    path: impl AsRef<Path>,
    header: &AggregateFileHeader,
    records: &[AggregateRecord],
) -> Result<()> {
    header.validate()?;
    if records.len() != header.sample_count {
        return Err(DcftError::invalid_parameter(
            "aggregate record count does not match header sample_count",
        ));
    }
    if let Some(parent) = path.as_ref().parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    write_header(&mut writer, header)?;
    let mut sorted = records.to_vec();
    sorted.sort_by_key(|record| record.disorder_id);
    validate_record_ids(header.sample_start, header.sample_count, &sorted)?;
    for record in &sorted {
        write_record(&mut writer, header, record)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_aggregate_binary(
    path: impl AsRef<Path>,
) -> Result<(AggregateFileHeader, Vec<AggregateRecord>)> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let header = read_header(&mut reader)?;
    header.validate()?;
    validate_file_len(file_len, &header)?;
    let mut records = Vec::with_capacity(header.sample_count);
    for _ in 0..header.sample_count {
        records.push(read_record(&mut reader, &header)?);
    }
    validate_record_ids(header.sample_start, header.sample_count, &records)?;
    Ok((header, records))
}

pub fn read_aggregate_metadata(path: impl AsRef<Path>) -> Result<AggregateMetadata> {
    let path = path.as_ref();
    let mut reader = BufReader::new(File::open(path)?);
    let header = read_header(&mut reader)?;
    header.validate()?;
    let len = path.metadata()?.len();
    if len < HEADER_LEN {
        return Err(DcftError::invalid_parameter(
            "aggregate file is shorter than its header",
        ));
    }
    let payload_len = len - HEADER_LEN;
    if !payload_len.is_multiple_of(header.record_len()) {
        return Err(DcftError::invalid_parameter(
            "aggregate file has a partial trailing record",
        ));
    }
    let record_count = (payload_len / header.record_len()) as usize;
    if record_count != header.sample_count {
        return Err(DcftError::invalid_parameter(
            "aggregate file record count does not match header sample_count",
        ));
    }
    Ok(AggregateMetadata {
        record_count,
        header,
    })
}

fn validate_file_len(file_len: u64, header: &AggregateFileHeader) -> Result<()> {
    let payload_len = header
        .record_len()
        .checked_mul(header.sample_count as u64)
        .ok_or_else(|| DcftError::invalid_parameter("aggregate file size overflows u64"))?;
    let expected_len = HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| DcftError::invalid_parameter("aggregate file size overflows u64"))?;
    if file_len != expected_len {
        return Err(DcftError::invalid_parameter(format!(
            "aggregate file length {file_len} does not match header-declared length {expected_len}"
        )));
    }
    Ok(())
}

pub fn merge_aggregate_binaries(
    out_path: impl AsRef<Path>,
    input_paths: &[PathBuf],
) -> Result<usize> {
    require_at_least("input file count", input_paths.len(), 1)?;
    let mut chunks = input_paths
        .iter()
        .map(|path| read_aggregate_binary(path).map(|(header, records)| (path, header, records)))
        .collect::<Result<Vec<_>>>()?;
    chunks.sort_by_key(|(_, header, _)| header.sample_start);
    let first = chunks
        .first()
        .ok_or_else(|| DcftError::invalid_parameter("input file count must be >= 1"))?;
    for (_, header, _) in chunks.iter().skip(1) {
        ensure_merge_compatible(&first.1, header)?;
    }

    let mut records = Vec::new();
    for (_, _, chunk_records) in &chunks {
        records.extend(chunk_records.iter().cloned());
    }
    records.sort_by_key(|record| record.disorder_id);
    let sample_start = records
        .first()
        .ok_or_else(|| DcftError::invalid_parameter("cannot merge empty aggregate records"))?
        .disorder_id;
    validate_record_ids(sample_start, records.len(), &records)?;
    let mut merged_header = first.1.clone();
    merged_header.sample_start = sample_start;
    merged_header.sample_count = records.len();
    write_aggregate_binary(out_path, &merged_header, &records)?;
    Ok(records.len())
}

fn ensure_merge_compatible(first: &AggregateFileHeader, other: &AggregateFileHeader) -> Result<()> {
    if first.clean_sample_count != other.clean_sample_count
        || first.lx != other.lx
        || first.lt != other.lt
        || first.kx != other.kx
        || first.kt != other.kt
        || first.seed != other.seed
        || first.noise != other.noise
        || first.measurement != other.measurement
        || first.p != other.p
        || first.mu != other.mu
        || first.disorder_update != other.disorder_update
        || first.disorder_therm_sweeps != other.disorder_therm_sweeps
        || first.measurements != other.measurements
        || first.skip_sweeps != other.skip_sweeps
    {
        return Err(DcftError::invalid_parameter(
            "aggregate chunk metadata mismatch",
        ));
    }
    Ok(())
}

fn validate_record_ids(
    sample_start: u64,
    sample_count: usize,
    records: &[AggregateRecord],
) -> Result<()> {
    if records.len() != sample_count {
        return Err(DcftError::invalid_parameter(
            "aggregate record count does not match expected sample count",
        ));
    }
    for (offset, record) in records.iter().enumerate() {
        let expected = sample_start + offset as u64;
        if record.disorder_id != expected {
            return Err(DcftError::invalid_parameter(format!(
                "aggregate records must be contiguous by disorder_id: expected {expected}, got {}",
                record.disorder_id
            )));
        }
    }
    Ok(())
}

fn write_header(writer: &mut impl Write, header: &AggregateFileHeader) -> Result<()> {
    writer.write_all(MAGIC_V3)?;
    write_u64(writer, header.clean_sample_count as u64)?;
    write_u64(writer, header.lx as u64)?;
    write_u64(writer, header.lt as u64)?;
    write_f64(writer, header.kx)?;
    write_f64(writer, header.kt)?;
    write_u64(writer, header.seed)?;
    write_u64(writer, encode_noise(header.noise))?;
    write_f64(writer, header.p)?;
    write_f64(writer, header.mu)?;
    write_u64(writer, header.sample_start)?;
    write_u64(writer, header.sample_count as u64)?;
    write_u64(writer, encode_update(header.disorder_update))?;
    write_u64(writer, header.disorder_therm_sweeps as u64)?;
    write_u64(writer, header.measurements as u64)?;
    write_u64(writer, header.skip_sweeps as u64)?;
    write_u64(writer, header.r_count() as u64)?;
    write_u64(writer, header.record_len())?;
    write_u64(writer, encode_measurement(header.measurement))?;
    Ok(())
}

fn read_header(reader: &mut impl Read) -> Result<AggregateFileHeader> {
    let mut magic = [0_u8; 11];
    reader.read_exact(&mut magic)?;
    let is_v2 = &magic == MAGIC_V2;
    if !is_v2 && &magic != MAGIC_V3 {
        return Err(DcftError::invalid_parameter(
            "not a supported DCFT aggregate file",
        ));
    }
    let header = AggregateFileHeader {
        clean_sample_count: read_u64(reader)? as usize,
        lx: read_u64(reader)? as usize,
        lt: read_u64(reader)? as usize,
        kx: read_f64(reader)?,
        kt: read_f64(reader)?,
        seed: read_u64(reader)?,
        noise: decode_noise(read_u64(reader)?)?,
        measurement: MeasurementKind::Heterodyne,
        p: read_f64(reader)?,
        mu: read_f64(reader)?,
        sample_start: read_u64(reader)?,
        sample_count: read_u64(reader)? as usize,
        disorder_update: decode_update(read_u64(reader)?)?,
        disorder_therm_sweeps: read_u64(reader)? as usize,
        measurements: read_u64(reader)? as usize,
        skip_sweeps: read_u64(reader)? as usize,
    };
    let r_count = read_u64(reader)? as usize;
    let record_len = read_u64(reader)?;
    let measurement_code = read_u64(reader)?;
    let header = AggregateFileHeader {
        measurement: if is_v2 {
            MeasurementKind::Heterodyne
        } else {
            decode_measurement(measurement_code)?
        },
        ..header
    };
    if r_count != header.r_count() || record_len != header.record_len() {
        return Err(DcftError::invalid_parameter(
            "aggregate header shape fields do not match dimensions",
        ));
    }
    Ok(header)
}

fn write_record(
    writer: &mut impl Write,
    header: &AggregateFileHeader,
    record: &AggregateRecord,
) -> Result<()> {
    record.validate(header)?;
    write_u64(writer, record.disorder_id)?;
    write_f64(writer, record.energy_mean)?;
    write_f64(writer, record.bulk_magnetization_sum_mean)?;
    write_f64_slice(writer, &record.boundary_spin_mean)?;
    write_f64_slice(writer, &record.boundary_bond_mean)?;
    write_f64_slice(writer, &record.spin_corr_signed)?;
    write_f64_slice(writer, &record.bond_corr_signed)?;
    write_f64_slice(writer, &record.spin_corr_abs)?;
    write_f64_slice(writer, &record.bond_corr_abs)?;
    Ok(())
}

fn read_record(reader: &mut impl Read, header: &AggregateFileHeader) -> Result<AggregateRecord> {
    let record = AggregateRecord {
        disorder_id: read_u64(reader)?,
        energy_mean: read_f64(reader)?,
        bulk_magnetization_sum_mean: read_f64(reader)?,
        boundary_spin_mean: read_f64_vec(reader, header.lx)?,
        boundary_bond_mean: read_f64_vec(reader, header.lx)?,
        spin_corr_signed: read_f64_vec(reader, header.r_count())?,
        bond_corr_signed: read_f64_vec(reader, header.r_count())?,
        spin_corr_abs: read_f64_vec(reader, header.r_count())?,
        bond_corr_abs: read_f64_vec(reader, header.r_count())?,
    };
    record.validate(header)?;
    Ok(record)
}

fn encode_noise(noise: NoiseKind) -> u64 {
    match noise {
        NoiseKind::Z => 1,
        NoiseKind::Zz => 2,
    }
}

fn decode_noise(value: u64) -> Result<NoiseKind> {
    match value {
        1 => Ok(NoiseKind::Z),
        2 => Ok(NoiseKind::Zz),
        _ => Err(DcftError::invalid_parameter("unknown aggregate noise code")),
    }
}

fn encode_measurement(measurement: MeasurementKind) -> u64 {
    match measurement {
        MeasurementKind::Heterodyne => 1,
        MeasurementKind::Homodyne => 2,
        MeasurementKind::LocalX => 3,
    }
}

fn decode_measurement(value: u64) -> Result<MeasurementKind> {
    match value {
        1 => Ok(MeasurementKind::Heterodyne),
        2 => Ok(MeasurementKind::Homodyne),
        3 => Ok(MeasurementKind::LocalX),
        _ => Err(DcftError::invalid_parameter(
            "unknown aggregate measurement code",
        )),
    }
}

fn encode_update(update: FixedDisorderUpdate) -> u64 {
    match update {
        FixedDisorderUpdate::Metropolis => 1,
        FixedDisorderUpdate::SequentialMetropolis => 2,
        FixedDisorderUpdate::CorrectedWolff => 3,
        FixedDisorderUpdate::MetropolisGlobal => 4,
    }
}

fn decode_update(value: u64) -> Result<FixedDisorderUpdate> {
    match value {
        1 => Ok(FixedDisorderUpdate::Metropolis),
        2 => Ok(FixedDisorderUpdate::SequentialMetropolis),
        3 => Ok(FixedDisorderUpdate::CorrectedWolff),
        4 => Ok(FixedDisorderUpdate::MetropolisGlobal),
        _ => Err(DcftError::invalid_parameter(
            "unknown aggregate update code",
        )),
    }
}

fn write_f64_slice(writer: &mut impl Write, values: &[f64]) -> Result<()> {
    for value in values {
        write_f64(writer, *value)?;
    }
    Ok(())
}

fn read_f64_vec(reader: &mut impl Read, len: usize) -> Result<Vec<f64>> {
    (0..len).map(|_| read_f64(reader)).collect()
}

fn write_u64(writer: &mut impl Write, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f64(writer: &mut impl Write, value: f64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f64(reader: &mut impl Read) -> Result<f64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}
