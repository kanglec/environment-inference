use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{DcftError, Result, require_at_least, require_finite, require_positive_finite};

use super::super::params::{IsingContext, IsingCouplings, LatticeSpec};

const MAGIC: &[u8; 13] = b"DCFT_CLEAN_V2";
const HEADER_LEN: u64 = 13 + 8 * 9;
const NO_DELTA_TAU: f64 = -1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct CleanSampleFileHeader {
    pub lx: usize,
    pub lt: usize,
    pub kx: f64,
    pub kt: f64,
    pub delta_tau: Option<f64>,
    pub seed: u64,
    pub sample_count: usize,
    pub clean_therm_sweeps: usize,
    pub clean_skip_sweeps: usize,
}

impl CleanSampleFileHeader {
    pub fn context(&self) -> Result<IsingContext> {
        Ok(IsingContext::new(
            LatticeSpec::new(self.lx, self.lt)?,
            IsingCouplings::new(self.kx, self.kt)?,
        ))
    }

    pub fn validate(&self) -> Result<()> {
        require_at_least("lx", self.lx, 2)?;
        require_at_least("lt", self.lt, 2)?;
        require_at_least("sample_count", self.sample_count, 1)?;
        require_finite("kx", self.kx)?;
        require_finite("kt", self.kt)?;
        if let Some(delta_tau) = self.delta_tau {
            require_positive_finite("delta_tau", delta_tau)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanBoundarySample {
    pub sample_id: u64,
    pub boundary_spins: Vec<i8>,
}

#[derive(Debug)]
pub struct CleanSampleFile {
    header: CleanSampleFileHeader,
    reader: BufReader<File>,
}

impl CleanSampleFile {
    pub fn create(
        path: impl AsRef<Path>,
        header: &CleanSampleFileHeader,
    ) -> Result<BufWriter<File>> {
        header.validate()?;
        if let Some(parent) = path.as_ref().parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(File::create(path)?);
        write_header(&mut writer, header)?;
        Ok(writer)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        let header = read_header(&mut reader)?;
        header.validate()?;
        let payload_len = (header.lx as u64)
            .checked_mul(header.sample_count as u64)
            .ok_or_else(|| DcftError::invalid_parameter("clean sample file size overflows u64"))?;
        let expected_len = HEADER_LEN
            .checked_add(payload_len)
            .ok_or_else(|| DcftError::invalid_parameter("clean sample file size overflows u64"))?;
        if file_len != expected_len {
            return Err(DcftError::invalid_parameter(format!(
                "clean sample file length {file_len} does not match header-declared length {expected_len}"
            )));
        }
        Ok(Self { header, reader })
    }

    pub fn header(&self) -> &CleanSampleFileHeader {
        &self.header
    }

    pub fn read_sample(&mut self, sample_id: u64) -> Result<CleanBoundarySample> {
        let index = usize::try_from(sample_id).map_err(|_| {
            DcftError::invalid_parameter("sample_id cannot be represented as usize")
        })?;
        if index >= self.header.sample_count {
            return Err(DcftError::invalid_parameter(format!(
                "sample_id {sample_id} is outside clean sample range 0..{}",
                self.header.sample_count
            )));
        }
        let offset = HEADER_LEN + sample_id * self.header.lx as u64;
        self.reader.seek(SeekFrom::Start(offset))?;
        let mut raw = vec![0_u8; self.header.lx];
        self.reader.read_exact(&mut raw)?;
        let mut boundary_spins = Vec::with_capacity(raw.len());
        for byte in raw {
            let spin = byte as i8;
            if spin != -1 && spin != 1 {
                return Err(DcftError::invalid_parameter(
                    "clean sample file contains a non-Ising boundary spin",
                ));
            }
            boundary_spins.push(spin);
        }
        Ok(CleanBoundarySample {
            sample_id,
            boundary_spins,
        })
    }
}

pub(crate) fn write_clean_boundary_sample(
    writer: &mut impl Write,
    sample: &CleanBoundarySample,
    lx: usize,
) -> Result<()> {
    if sample.boundary_spins.len() != lx {
        return Err(DcftError::invalid_parameter("boundary spin count mismatch"));
    }
    for spin in &sample.boundary_spins {
        if *spin != -1 && *spin != 1 {
            return Err(DcftError::invalid_parameter(
                "boundary spins must be -1 or 1",
            ));
        }
        writer.write_all(&[*spin as u8])?;
    }
    Ok(())
}

fn write_header(writer: &mut impl Write, header: &CleanSampleFileHeader) -> Result<()> {
    writer.write_all(MAGIC)?;
    write_u64(writer, header.lx as u64)?;
    write_u64(writer, header.lt as u64)?;
    write_f64(writer, header.kx)?;
    write_f64(writer, header.kt)?;
    write_f64(writer, header.delta_tau.unwrap_or(NO_DELTA_TAU))?;
    write_u64(writer, header.seed)?;
    write_u64(writer, header.sample_count as u64)?;
    write_u64(writer, header.clean_therm_sweeps as u64)?;
    write_u64(writer, header.clean_skip_sweeps as u64)?;
    Ok(())
}

fn read_header(reader: &mut impl Read) -> Result<CleanSampleFileHeader> {
    let mut magic = [0_u8; 13];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(DcftError::invalid_parameter(
            "not a DCFT_CLEAN_V2 clean sample file",
        ));
    }
    let lx = read_u64(reader)? as usize;
    let lt = read_u64(reader)? as usize;
    let kx = read_f64(reader)?;
    let kt = read_f64(reader)?;
    let raw_delta_tau = read_f64(reader)?;
    let delta_tau = if raw_delta_tau < 0.0 {
        None
    } else {
        Some(raw_delta_tau)
    };
    Ok(CleanSampleFileHeader {
        lx,
        lt,
        kx,
        kt,
        delta_tau,
        seed: read_u64(reader)?,
        sample_count: read_u64(reader)? as usize,
        clean_therm_sweeps: read_u64(reader)? as usize,
        clean_skip_sweeps: read_u64(reader)? as usize,
    })
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
