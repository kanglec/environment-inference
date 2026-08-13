use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use clap::{ArgGroup, Args, CommandFactory, Parser, Subcommand};
use decohered_cft::error::{DcftError, Result};
use decohered_cft::ising::diagnostics::{
    AutocorrelationDiagnosticInput, MultiChainDiagnosticReport, ParallelBenchmarkReport,
    SamplerBenchmarkInput, SpeedBenchmarkReport, StageBenchmarkReport,
    ThermalizationDiagnosticInput, benchmark_sampler, diagnose_autocorrelation,
    diagnose_thermalization,
};
use decohered_cft::ising::{
    AggregateFileHeader, BoundaryStageInput, CleanSampleFile, FixedDisorderUpdate,
    MeasureStageInput, MeasurementKind, NoiseKind, analyze_aggregate_file, generate_clean_stage,
    measure_stage_to_file, merge_aggregate_binaries, read_aggregate_binary,
    read_aggregate_metadata, read_run_spec_chunk_plan, write_analysis_csv,
    write_disorder_records_csv,
};
use decohered_cft::throughput;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => {
            Cli::command().print_help()?;
            println!();
            Ok(())
        }
        Some(CliCommand::GenerateClean(args)) => run_generate_clean(args.into_input()?),
        Some(CliCommand::Measure(args)) => {
            let threads = args.threads;
            run_measure(args.into_input()?, threads)
        }
        Some(CliCommand::InspectClean(args)) => run_inspect_clean(args.clean_path),
        Some(CliCommand::InspectAggregate(args)) => run_inspect_aggregate(args.path),
        Some(CliCommand::MergeAggregates(args)) => {
            let (out_path, inputs) = args.into_parts();
            run_merge_aggregates(out_path, inputs)
        }
        Some(CliCommand::Analyze(args)) => {
            run_analyze(args.path, args.csv, args.disorder_records_csv)
        }
        Some(CliCommand::RunSpecDryRun(args)) => run_run_spec_dry_run(args.path),
        Some(CliCommand::BenchmarkSampler(args)) => {
            let threads = args.threads;
            run_benchmark_sampler(args.into_input()?, threads)
        }
        Some(CliCommand::DiagnoseAutocorrelation(args)) => {
            run_diagnose_autocorrelation(args.into_input()?)
        }
        Some(CliCommand::DiagnoseThermalization(args)) => {
            let threads = args.threads;
            run_diagnose_thermalization(args.into_input()?, threads)
        }
    }
}

#[derive(Debug, Parser)]
#[command(version, about = "Decohered CFT Monte Carlo tools")]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Generate clean boundary samples used to draw tilted disorder
    GenerateClean(GenerateCleanArgs),
    /// Run fixed-disorder chains and write aggregate nonlinear-diagnostic records
    Measure(MeasureArgs),
    /// Print metadata from a clean binary file
    InspectClean(InspectCleanArgs),
    /// Print metadata from an aggregate binary file
    InspectAggregate(InspectAggregateArgs),
    /// Merge aggregate chunks into one deterministic aggregate file
    MergeAggregates(MergeAggregatesArgs),
    /// Analyze aggregate records into disorder-averaged observables
    Analyze(AnalyzeArgs),
    /// Print array chunk ranges from a minimal TOML run spec
    RunSpecDryRun(RunSpecDryRunArgs),
    /// Benchmark sampler sweep speeds
    BenchmarkSampler(BenchmarkSamplerArgs),
    /// Diagnose post-thermal spin autocorrelation
    DiagnoseAutocorrelation(DiagnoseAutocorrelationArgs),
    /// Diagnose clean and fixed-disorder thermalization
    DiagnoseThermalization(DiagnoseThermalizationArgs),
}

#[derive(Debug, Args)]
#[command(group(ArgGroup::new("output").required(true).args(["out_path", "out_dir"])))]
struct GenerateCleanArgs {
    #[arg(long)]
    lx: usize,
    #[arg(long)]
    lt: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long = "samples")]
    sample_count: usize,
    #[arg(long)]
    clean_therm_sweeps: usize,
    #[arg(long)]
    clean_skip_sweeps: usize,
    #[arg(long = "out", conflicts_with = "out_dir")]
    out_path: Option<PathBuf>,
    #[arg(long = "out-dir")]
    out_dir: Option<PathBuf>,
    #[arg(long)]
    delta_tau: Option<f64>,
}

#[derive(Debug, Args)]
#[command(group(ArgGroup::new("output").required(true).args(["out_path", "out_dir"])))]
struct MeasureArgs {
    #[arg(long = "clean")]
    clean_path: PathBuf,
    #[arg(long, value_parser = parse_noise)]
    noise: NoiseKind,
    #[arg(long, value_parser = parse_measurement, default_value = "heterodyne")]
    measurement: MeasurementKind,
    #[arg(long)]
    p: f64,
    #[arg(long)]
    sample_start: u64,
    #[arg(long = "samples")]
    sample_count: usize,
    #[arg(long, value_parser = parse_fixed_disorder_update, default_value = "metropolis")]
    disorder_update: FixedDisorderUpdate,
    #[arg(long)]
    disorder_therm_sweeps: usize,
    #[arg(long)]
    measurements: usize,
    #[arg(long)]
    skip_sweeps: usize,
    #[arg(long = "out", conflicts_with = "out_dir")]
    out_path: Option<PathBuf>,
    #[arg(long = "out-dir")]
    out_dir: Option<PathBuf>,
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Debug, Args)]
struct InspectCleanArgs {
    #[arg(long = "clean")]
    clean_path: PathBuf,
}

#[derive(Debug, Args)]
struct InspectAggregateArgs {
    #[arg(long = "path")]
    path: PathBuf,
}

#[derive(Debug, Args)]
struct MergeAggregatesArgs {
    #[arg(long = "out")]
    out_path: PathBuf,
    #[arg(long = "input")]
    input_flags: Vec<PathBuf>,
    inputs: Vec<PathBuf>,
}

#[derive(Debug, Args)]
struct AnalyzeArgs {
    #[arg(long = "path")]
    path: PathBuf,
    /// Write the complete scalar and correlator results as CSV
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Write one row per independent disorder realization and separation
    #[arg(long = "disorder-records-csv")]
    disorder_records_csv: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct RunSpecDryRunArgs {
    #[arg(long = "path")]
    path: PathBuf,
}

#[derive(Debug, Args)]
struct BenchmarkSamplerArgs {
    #[arg(long)]
    lx: usize,
    #[arg(long)]
    lt: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, value_parser = parse_noise)]
    noise: NoiseKind,
    #[arg(long)]
    p: f64,
    #[arg(long)]
    clean_therm_sweeps: usize,
    /// Comma-separated updates: metropolis, metropolis-global, sequential-metropolis, corrected-wolff, or all
    #[arg(long, default_value = "metropolis", value_delimiter = ',')]
    disorder_update: Vec<String>,
    #[arg(long)]
    disorder_therm_sweeps: usize,
    #[arg(long)]
    delta_tau: Option<f64>,
    #[arg(long)]
    parallel_chains: Option<usize>,
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Debug, Args)]
struct DiagnoseAutocorrelationArgs {
    #[arg(long)]
    lx: usize,
    #[arg(long)]
    lt: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, value_parser = parse_noise)]
    noise: NoiseKind,
    #[arg(long)]
    p: f64,
    #[arg(long)]
    clean_therm_sweeps: usize,
    /// Comma-separated updates: metropolis, metropolis-global, sequential-metropolis, corrected-wolff, or all
    #[arg(long, default_value = "metropolis", value_delimiter = ',')]
    disorder_update: Vec<String>,
    #[arg(long)]
    disorder_therm_sweeps: usize,
    #[arg(long)]
    delta_tau: Option<f64>,
    #[arg(long)]
    probes: usize,
    #[arg(long)]
    probe_interval_sweeps: usize,
}

#[derive(Debug, Args)]
struct DiagnoseThermalizationArgs {
    #[arg(long)]
    lx: usize,
    #[arg(long)]
    lt: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, value_parser = parse_noise)]
    noise: NoiseKind,
    #[arg(long)]
    p: f64,
    #[arg(long)]
    clean_therm_sweeps: usize,
    /// Comma-separated updates: metropolis, metropolis-global, sequential-metropolis, corrected-wolff, or all
    #[arg(long, default_value = "metropolis", value_delimiter = ',')]
    disorder_update: Vec<String>,
    #[arg(long)]
    disorder_therm_sweeps: usize,
    #[arg(long)]
    delta_tau: Option<f64>,
    #[arg(long)]
    chains: usize,
    #[arg(long)]
    measurements: usize,
    #[arg(long)]
    skip_sweeps: usize,
    #[arg(long)]
    threads: Option<usize>,
}

impl GenerateCleanArgs {
    fn into_input(self) -> Result<BoundaryStageInput> {
        let Self {
            lx,
            lt,
            seed,
            sample_count,
            clean_therm_sweeps,
            clean_skip_sweeps,
            out_path,
            out_dir,
            delta_tau,
        } = self;

        Ok(BoundaryStageInput {
            lx,
            lt,
            seed,
            sample_count,
            clean_therm_sweeps,
            clean_skip_sweeps,
            delta_tau,
            out_path: resolve_output_path(
                out_path,
                out_dir,
                default_clean_filename(lx, lt, seed, sample_count, delta_tau),
            ),
        })
    }
}

impl MeasureArgs {
    fn into_input(self) -> Result<MeasureStageInput> {
        let Self {
            clean_path,
            noise,
            measurement,
            p,
            sample_start,
            sample_count,
            disorder_update,
            disorder_therm_sweeps,
            measurements,
            skip_sweeps,
            out_path,
            out_dir,
            threads: _,
        } = self;

        Ok(MeasureStageInput {
            out_path: resolve_output_path(
                out_path,
                out_dir,
                default_aggregate_filename(DefaultAggregateFilename {
                    clean_path: &clean_path,
                    noise,
                    measurement,
                    p,
                    sample_start,
                    sample_count,
                    disorder_update,
                    measurements,
                }),
            ),
            clean_path,
            noise,
            measurement,
            p,
            sample_start,
            sample_count,
            disorder_update,
            disorder_therm_sweeps,
            measurements,
            skip_sweeps,
        })
    }
}

impl MergeAggregatesArgs {
    fn into_parts(self) -> (PathBuf, Vec<PathBuf>) {
        let inputs = self.input_flags.into_iter().chain(self.inputs).collect();
        (self.out_path, inputs)
    }
}

impl BenchmarkSamplerArgs {
    fn into_input(self) -> Result<SamplerBenchmarkInput> {
        Ok(SamplerBenchmarkInput {
            lx: self.lx,
            lt: self.lt,
            seed: self.seed,
            noise: self.noise,
            p: self.p,
            clean_therm_sweeps: self.clean_therm_sweeps,
            disorder_therm_sweeps: self.disorder_therm_sweeps,
            disorder_updates: parse_fixed_disorder_update_selection(&self.disorder_update)?,
            delta_tau: self.delta_tau,
            parallel_chains: self.parallel_chains,
        })
    }
}

impl DiagnoseAutocorrelationArgs {
    fn into_input(self) -> Result<AutocorrelationDiagnosticInput> {
        Ok(AutocorrelationDiagnosticInput {
            lx: self.lx,
            lt: self.lt,
            seed: self.seed,
            noise: self.noise,
            p: self.p,
            clean_therm_sweeps: self.clean_therm_sweeps,
            disorder_therm_sweeps: self.disorder_therm_sweeps,
            disorder_updates: parse_fixed_disorder_update_selection(&self.disorder_update)?,
            delta_tau: self.delta_tau,
            probes: self.probes,
            probe_interval_sweeps: self.probe_interval_sweeps,
        })
    }
}

impl DiagnoseThermalizationArgs {
    fn into_input(self) -> Result<ThermalizationDiagnosticInput> {
        Ok(ThermalizationDiagnosticInput {
            lx: self.lx,
            lt: self.lt,
            seed: self.seed,
            noise: self.noise,
            p: self.p,
            clean_therm_sweeps: self.clean_therm_sweeps,
            disorder_therm_sweeps: self.disorder_therm_sweeps,
            disorder_updates: parse_fixed_disorder_update_selection(&self.disorder_update)?,
            delta_tau: self.delta_tau,
            chains: self.chains,
            measurements: self.measurements,
            skip_sweeps: self.skip_sweeps,
        })
    }
}

fn run_generate_clean(input: BoundaryStageInput) -> Result<()> {
    let started = Instant::now();
    generate_clean_stage(&input)?;
    let seconds = started.elapsed().as_secs_f64();

    println!("Output");
    println!("  clean:   {}", input.out_path.display());
    print_timing("generate-clean", input.sample_count, "samples", seconds);
    Ok(())
}

fn run_measure(input: MeasureStageInput, threads: Option<usize>) -> Result<()> {
    configure_rayon_threads(threads)?;
    let started = Instant::now();
    let records = measure_stage_to_file(&input)?;
    let seconds = started.elapsed().as_secs_f64();

    println!("Output");
    println!("  format:  aggregate bin");
    println!("  out:     {}", input.out_path.display());
    print_timing("measure", records, "aggregate records", seconds);
    Ok(())
}

fn run_inspect_clean(clean_path: PathBuf) -> Result<()> {
    let header = CleanSampleFile::open(&clean_path)?.header().clone();
    println!("Clean boundary file");
    println!("  path:                 {}", clean_path.display());
    println!("  lx:                   {}", header.lx);
    println!("  lt:                   {}", header.lt);
    println!("  kx:                   {:.17}", header.kx);
    println!("  kt:                   {:.17}", header.kt);
    match header.delta_tau {
        Some(delta_tau) => println!("  delta_tau:            {:.17}", delta_tau),
        None => println!("  delta_tau:            isotropic"),
    }
    println!("  seed:                 {}", header.seed);
    println!("  samples:              {}", header.sample_count);
    println!("  clean_therm_sweeps:   {}", header.clean_therm_sweeps);
    println!("  clean_skip_sweeps:    {}", header.clean_skip_sweeps);
    Ok(())
}

fn run_inspect_aggregate(path: PathBuf) -> Result<()> {
    let metadata = read_aggregate_metadata(&path)?;
    let header = metadata.header;
    println!("Aggregate binary");
    println!("  path:                 {}", path.display());
    print_header(&header);
    println!("  records:              {}", metadata.record_count);
    println!("  record_len:           {}", header.record_len());
    Ok(())
}

fn run_merge_aggregates(out_path: PathBuf, inputs: Vec<PathBuf>) -> Result<()> {
    let started = Instant::now();
    let records = merge_aggregate_binaries(&out_path, &inputs)?;
    let seconds = started.elapsed().as_secs_f64();

    println!("Merged aggregate records");
    println!("  inputs:  {}", inputs.len());
    println!("  records: {records}");
    println!("  out:     {}", out_path.display());
    print_timing("merge-aggregates", records, "records", seconds);
    Ok(())
}

fn run_analyze(
    path: PathBuf,
    csv: Option<PathBuf>,
    disorder_records_csv: Option<PathBuf>,
) -> Result<()> {
    let report = analyze_aggregate_file(&path)?;
    println!("Analysis");
    println!("  path:                  {}", path.display());
    println!("  samples:               {}", report.samples);
    println!("  lx:                    {}", report.lx);
    println!("  lt:                    {}", report.lt);
    println!("  energy_density:        {:.12}", report.energy_density);
    println!(
        "  magnetization_density: {:.12}",
        report.magnetization_density
    );
    println!(
        "  boundary_magnetization:{:>13.12}",
        report.boundary_magnetization
    );
    println!(
        "  local_spin_fidelity:   {:.12}",
        report.local_spin_fidelity
    );
    println!(
        "  local_bond_fidelity:   {:.12}",
        report.local_bond_fidelity
    );
    println!(
        "  spin_linear_corr:      {}",
        format_preview(&report.spin_linear_corr)
    );
    println!(
        "  spin_fidelity_corr:    {}",
        format_preview(&report.spin_fidelity_corr)
    );
    println!(
        "  spin_ea_corr:          {}",
        format_preview(&report.spin_ea_corr)
    );
    println!(
        "  bond_linear_corr:      {}",
        format_preview(&report.bond_linear_corr)
    );
    println!(
        "  bond_fidelity_corr:    {}",
        format_preview(&report.bond_fidelity_corr)
    );
    println!(
        "  bond_ea_corr:          {}",
        format_preview(&report.bond_ea_corr)
    );
    if let Some(csv) = csv {
        write_analysis_csv(&csv, &report)?;
        println!("  csv:                   {}", csv.display());
    }
    if let Some(csv) = disorder_records_csv {
        let (header, records) = read_aggregate_binary(&path)?;
        write_disorder_records_csv(&csv, &header, &records)?;
        println!("  disorder_records_csv:  {}", csv.display());
    }
    Ok(())
}

fn run_run_spec_dry_run(path: PathBuf) -> Result<()> {
    let ranges = read_run_spec_chunk_plan(path)?;
    println!("Chunk plan");
    for range in ranges {
        println!(
            "  task={} sample_start={} samples={}",
            range.task_id, range.sample_start, range.sample_count
        );
    }
    Ok(())
}

fn run_benchmark_sampler(input: SamplerBenchmarkInput, threads: Option<usize>) -> Result<()> {
    configure_rayon_threads(threads)?;
    let report = benchmark_sampler(&input)?;
    print_diagnostic_header(
        "Sampler benchmark",
        report.lx,
        report.lt,
        report.kx,
        report.kt,
        report.seed,
        report.noise,
        report.p,
        report.mu,
    );
    print_speed_benchmark(&report.clean);
    for disorder in &report.disorder {
        print_speed_benchmark(disorder);
    }
    for parallel in &report.parallel {
        print_parallel_benchmark(parallel);
    }
    Ok(())
}

fn run_diagnose_autocorrelation(input: AutocorrelationDiagnosticInput) -> Result<()> {
    let report = diagnose_autocorrelation(&input)?;
    print_diagnostic_header(
        "Autocorrelation diagnosis",
        report.lx,
        report.lt,
        report.kx,
        report.kt,
        report.seed,
        report.noise,
        report.p,
        report.mu,
    );
    print_stage_benchmark(&report.clean);
    for disorder in &report.disorder {
        print_stage_benchmark(disorder);
    }
    Ok(())
}

fn run_diagnose_thermalization(
    input: ThermalizationDiagnosticInput,
    threads: Option<usize>,
) -> Result<()> {
    configure_rayon_threads(threads)?;
    let report = diagnose_thermalization(&input)?;
    print_diagnostic_header(
        "Thermalization diagnosis",
        report.lx,
        report.lt,
        report.kx,
        report.kt,
        report.seed,
        report.noise,
        report.p,
        report.mu,
    );
    print_multi_chain_diagnostic(&report.clean);
    for disorder in &report.disorder {
        print_multi_chain_diagnostic(disorder);
    }
    Ok(())
}

fn print_diagnostic_header(
    title: &str,
    lx: usize,
    lt: usize,
    kx: f64,
    kt: f64,
    seed: u64,
    noise: NoiseKind,
    p: f64,
    mu: f64,
) {
    println!("{title}");
    println!("  lx:                   {lx}");
    println!("  lt:                   {lt}");
    println!("  kx:                   {kx:.17}");
    println!("  kt:                   {kt:.17}");
    println!("  seed:                 {seed}");
    println!("  noise:                {}", noise.as_str());
    println!("  p:                    {p:.17}");
    println!("  mu:                   {mu:.17}");
}

fn print_stage_benchmark(report: &StageBenchmarkReport) {
    println!(
        "\nResult\n  stage:                {}\n  update:               {}\n  measured_sweeps:      {}\n  therm_elapsed:        {:.3} s\n  sweep_speed:          {:.3} sweeps/s\n  spin_update_speed:    {:.3} spins/s\n  probe_interval:       {} sweeps\n  probe_sweeps:         {}\n  probe_elapsed:        {:.3} s\n  analysis_elapsed:     {:.3} s\n  total_elapsed:        {:.3} s\n  probes:               {}",
        report.stage,
        report.update,
        report.measured_sweeps,
        report.elapsed_seconds,
        report.sweeps_per_second,
        report.spin_updates_per_second,
        report.probe_interval_sweeps,
        report.autocorrelation_sweeps,
        report.probe_elapsed_seconds,
        report.analysis_elapsed_seconds,
        report.total_elapsed_seconds,
        report.probes
    );
    match report.cutoff_lag {
        Some(lag) => println!("  cutoff_lag:           {lag}"),
        None => println!("  cutoff_lag:           none"),
    }
    println!(
        "  tau_int:              {:.3} sweeps\n  rho_lags:             {}",
        report.tau_int_sweeps,
        format_autocorrelation_preview(&report.autocorrelation)
    );
}

fn print_speed_benchmark(report: &SpeedBenchmarkReport) {
    println!(
        "\nSingle-thread benchmark\n  stage:                {}\n  update:               {}\n  measured_sweeps:      {}\n  elapsed:              {:.3} s\n  sweep_speed:          {:.3} sweeps/s\n  spin_update_speed:    {:.3} spins/s",
        report.stage,
        report.update,
        report.measured_sweeps,
        report.elapsed_seconds,
        report.sweeps_per_second,
        report.spin_updates_per_second
    );
}

fn print_multi_chain_diagnostic(report: &MultiChainDiagnosticReport) {
    println!(
        "\nResult\n  stage:                {}\n  update:               {}",
        report.stage, report.update
    );
    if let Some(sample_id) = report.disorder_sample_id {
        println!("  disorder_sample_id:   {sample_id}");
    }
    println!(
        "  chains:               {}\n  therm_sweeps:         {}\n  measurements:         {}\n  skip_sweeps:          {}\n  elapsed:              {:.3} s\n  measurement_speed:    {:.3} records/s",
        report.chains,
        report.therm_sweeps,
        report.measurements,
        report.skip_sweeps,
        report.elapsed_seconds,
        report.measurement_records_per_second
    );
    for observable in &report.observables {
        println!(
            "\nObservable\n  name:                 {}\n  mean:                 {:.8}\n  r_hat:                {:.6}\n  chain_mean_min:       {:.8}\n  chain_mean_max:       {:.8}\n  chain_mean_std:       {:.8}\n  mean_within_chain_std:{:>12.8}",
            observable.observable,
            observable.mean,
            observable.r_hat,
            observable.chain_mean_min,
            observable.chain_mean_max,
            observable.chain_mean_std,
            observable.mean_within_chain_std
        );
    }
}

fn print_parallel_benchmark(report: &ParallelBenchmarkReport) {
    println!(
        "\nParallel benchmark\n  stage:                {}\n  update:               {}\n  rayon_threads:        {}\n  chains:               {}\n  sweeps_per_chain:     {}\n  serial_elapsed:       {:.3} s\n  parallel_elapsed:     {:.3} s\n  serial_speed:         {:.3} chains/s\n  parallel_speed:       {:.3} chains/s\n  speedup:              {:.3}x",
        report.stage,
        report.update,
        report.rayon_threads,
        report.chains,
        report.sweeps_per_chain,
        report.serial_elapsed_seconds,
        report.parallel_elapsed_seconds,
        report.serial_chains_per_second,
        report.parallel_chains_per_second,
        report.speedup
    );
}

fn format_autocorrelation_preview(values: &[f64]) -> String {
    let mut preview = values
        .iter()
        .take(8)
        .enumerate()
        .map(|(lag, value)| format!("{lag}:{value:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    if values.len() > 8 {
        preview.push_str(", ...");
    }
    preview
}

fn print_header(header: &AggregateFileHeader) {
    println!("  clean_sample_count:   {}", header.clean_sample_count);
    println!("  lx:                   {}", header.lx);
    println!("  lt:                   {}", header.lt);
    println!("  kx:                   {:.17}", header.kx);
    println!("  kt:                   {:.17}", header.kt);
    println!("  seed:                 {}", header.seed);
    println!("  noise:                {}", header.noise.as_str());
    println!("  measurement:          {}", header.measurement.as_str());
    println!("  p:                    {:.17}", header.p);
    println!("  mu:                   {:.17}", header.mu);
    println!("  sample_start:         {}", header.sample_start);
    println!("  samples:              {}", header.sample_count);
    println!(
        "  disorder_update:      {}",
        header.disorder_update.as_str()
    );
    println!("  disorder_therm_sweeps:{}", header.disorder_therm_sweeps);
    println!("  measurements:         {}", header.measurements);
    println!("  skip_sweeps:          {}", header.skip_sweeps);
    println!("  r_count:              {}", header.r_count());
}

fn resolve_output_path(
    out_path: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    default_filename: String,
) -> PathBuf {
    if let Some(path) = out_path {
        return path;
    }
    if let Some(dir) = out_dir {
        return dir.join(default_filename);
    }
    unreachable!("Clap ensures exactly one output path is provided")
}

fn default_clean_filename(
    lx: usize,
    lt: usize,
    seed: u64,
    sample_count: usize,
    delta_tau: Option<f64>,
) -> String {
    let tau = match delta_tau {
        Some(delta_tau) => format!("dt{}", filename_f64(delta_tau)),
        None => "iso".to_string(),
    };
    format!("clean_lx{lx}_lt{lt}_{tau}_seed{seed}_samples{sample_count}.bin")
}

struct DefaultAggregateFilename<'a> {
    clean_path: &'a Path,
    noise: NoiseKind,
    measurement: MeasurementKind,
    p: f64,
    sample_start: u64,
    sample_count: usize,
    disorder_update: FixedDisorderUpdate,
    measurements: usize,
}

fn default_aggregate_filename(input: DefaultAggregateFilename<'_>) -> String {
    let clean_stem = input
        .clean_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("clean");
    format!(
        "agg_{clean_stem}_noise{}_measurement{}_p{}_update{}_start{}_samples{}_meas{}.bin",
        input.noise.as_str(),
        input.measurement.as_str(),
        filename_f64(input.p),
        input.disorder_update.as_str(),
        input.sample_start,
        input.sample_count,
        input.measurements,
    )
}

fn filename_f64(value: f64) -> String {
    value.to_string().replace('-', "m").replace('.', "p")
}

fn print_timing(stage: &str, items: usize, item_label: &str, seconds: f64) {
    println!("Timing");
    println!("  stage:   {stage}");
    println!("  elapsed: {seconds:.3} s");
    println!("  work:    {items} {item_label}");
    println!(
        "  speed:   {:.3} {item_label}/s",
        throughput(items, seconds)
    );
}

fn format_preview(values: &[f64]) -> String {
    let mut preview = values
        .iter()
        .take(8)
        .enumerate()
        .map(|(index, value)| format!("{index}:{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    if values.len() > 8 {
        preview.push_str(", ...");
    }
    preview
}

fn configure_rayon_threads(cli_threads: Option<usize>) -> Result<()> {
    let threads = match cli_threads {
        Some(threads) => Some(threads),
        None if env::var_os("RAYON_NUM_THREADS").is_some() => None,
        None => env::var("SLURM_CPUS_PER_TASK")
            .ok()
            .map(|value| {
                value.parse().map_err(|_| {
                    DcftError::invalid_parameter(format!(
                        "could not parse `{value}` as value for SLURM_CPUS_PER_TASK"
                    ))
                })
            })
            .transpose()?,
    };
    if let Some(threads) = threads {
        if threads == 0 {
            return Err(DcftError::invalid_parameter(
                "--threads must be greater than zero",
            ));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| {
                DcftError::invalid_parameter(format!("failed to configure Rayon: {error}"))
            })?;
    }
    Ok(())
}

fn parse_fixed_disorder_update(value: &str) -> Result<FixedDisorderUpdate> {
    match value {
        "metropolis" => Ok(FixedDisorderUpdate::Metropolis),
        "metropolis-global" | "metropolis_global" | "mg" => {
            Ok(FixedDisorderUpdate::MetropolisGlobal)
        }
        "sequential-metropolis" | "sequential_metropolis" | "seq-metropolis" | "seq_metropolis" => {
            Ok(FixedDisorderUpdate::SequentialMetropolis)
        }
        "corrected-wolff" | "corrected_wolff" | "cw" => Ok(FixedDisorderUpdate::CorrectedWolff),
        _ => Err(DcftError::invalid_parameter(
            "fixed-disorder update must be `metropolis`, `sequential-metropolis`, or `corrected-wolff`",
        )),
    }
}

fn parse_fixed_disorder_update_selection(values: &[String]) -> Result<Vec<FixedDisorderUpdate>> {
    let mut updates = Vec::new();
    for value in values {
        if value == "all" {
            for update in [
                FixedDisorderUpdate::Metropolis,
                FixedDisorderUpdate::MetropolisGlobal,
                FixedDisorderUpdate::SequentialMetropolis,
                FixedDisorderUpdate::CorrectedWolff,
            ] {
                push_unique(&mut updates, update);
            }
        } else {
            push_unique(&mut updates, parse_fixed_disorder_update(value)?);
        }
    }
    if updates.is_empty() {
        return Err(DcftError::invalid_parameter(
            "--disorder-update must select at least one update",
        ));
    }
    Ok(updates)
}

fn push_unique(updates: &mut Vec<FixedDisorderUpdate>, update: FixedDisorderUpdate) {
    if !updates.contains(&update) {
        updates.push(update);
    }
}

fn parse_noise(value: &str) -> Result<NoiseKind> {
    match value.to_ascii_lowercase().as_str() {
        "z" => Ok(NoiseKind::Z),
        "zz" => Ok(NoiseKind::Zz),
        _ => Err(DcftError::invalid_parameter(
            "--noise must be either `z` or `zz`",
        )),
    }
}

fn parse_measurement(value: &str) -> Result<MeasurementKind> {
    match value.to_ascii_lowercase().as_str() {
        "heterodyne" | "het" => Ok(MeasurementKind::Heterodyne),
        "homodyne" | "hom" => Ok(MeasurementKind::Homodyne),
        "local-x" | "local_x" | "x" => Ok(MeasurementKind::LocalX),
        _ => Err(DcftError::invalid_parameter(
            "--measurement must be `heterodyne`, `homodyne`, or `local-x`",
        )),
    }
}
