use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use decohered_cft::Rng64;
use decohered_cft::ising::{
    AggregateAccumulator, AggregateFileHeader, AggregateRecord, BoundaryRandomIsingModel,
    BoundaryStageInput, CleanIsingModel, CleanSampleFile, FixedDisorderUpdate, IsingContext,
    IsingCouplings, IsingSampler, IsingUpdateMethod, LatticeSpec, MeasureStageInput,
    MeasurementKind, NoiseKind, SpinLattice, analyze_aggregate_file, chunk_plan,
    generate_clean_stage, measure_stage, measure_stage_to_file, measurement_parameter,
    merge_aggregate_binaries, mu_from_noise_probability, read_aggregate_binary,
    read_aggregate_metadata, read_run_spec_chunk_plan, write_aggregate_binary, write_analysis_csv,
    write_disorder_records_csv,
};

fn temp_path(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "decohered-cft-{label}-{}-{}.tmp",
        std::process::id(),
        Rng64::seeded(label.len() as u64).next_u64()
    ));
    path
}

#[test]
fn lattice_uses_periodic_row_major_indices() {
    let spec = LatticeSpec::new(3, 2).unwrap();

    assert_eq!(spec.index(2, 1), 5);
    assert_eq!(spec.neighbors(spec.index(0, 0)), [2, 1, 3, 3]);
    assert!(spec.is_boundary(2));
    assert!(!spec.is_boundary(3));
}

#[test]
fn clean_energy_and_flip_delta_use_periodic_bonds() {
    let spec = LatticeSpec::new(2, 2).unwrap();
    let couplings = IsingCouplings::new(0.3, 0.4).unwrap();
    let context = IsingContext::new(spec.clone(), couplings);
    let model = CleanIsingModel::new(context);
    let lattice = SpinLattice::cold(spec, 1).unwrap();

    assert!((model.energy(&lattice) + 2.8).abs() < 1e-12);
    assert!((model.flip_energy_delta(&lattice, 0) - 2.8).abs() < 1e-12);
}

#[test]
fn disorder_models_use_expected_boundary_energy_terms() {
    let spec = LatticeSpec::new(3, 2).unwrap();
    let context = IsingContext::new(spec.clone(), IsingCouplings::new(0.1, 0.2).unwrap());
    let lattice = SpinLattice::new(spec.clone(), vec![1, -1, 1, 1, 1, -1]).unwrap();
    let clean_energy = CleanIsingModel::new(context.clone()).energy(&lattice);

    let fields = vec![0.5, -0.25, 0.75];
    let field_model = BoundaryRandomIsingModel::boundary_fields(context.clone(), fields).unwrap();
    assert!((field_model.energy(&lattice) - (clean_energy - 1.5)).abs() < 1e-12);

    let bonds = vec![0.5, -0.25, 0.75];
    let bond_model = BoundaryRandomIsingModel::boundary_bonds_x(context, bonds).unwrap();
    assert!((bond_model.energy(&lattice) - (clean_energy - 0.5)).abs() < 1e-12);
}

#[test]
fn global_flip_delta_matches_exact_disordered_energy_difference() {
    let context = IsingContext::from_critical_tfim_delta_tau(4, 5, 0.2).unwrap();
    let model =
        BoundaryRandomIsingModel::boundary_fields(context.clone(), vec![0.1, -0.4, 0.7, 0.2])
            .unwrap();
    let mut lattice = SpinLattice::cold(context.spec().clone(), 1).unwrap();
    lattice.flip_index(1);
    let before = model.energy(&lattice);
    let delta = model.global_flip_energy_delta(&lattice);
    lattice.flip_all();
    let after = model.energy(&lattice);
    assert!((after - before - delta).abs() < 1e-12);
}

#[test]
fn every_local_energy_delta_matches_exact_energy_difference() {
    let spec = LatticeSpec::new(3, 2).unwrap();
    let context = IsingContext::new(spec.clone(), IsingCouplings::new(0.37, 0.23).unwrap());
    let field_model =
        BoundaryRandomIsingModel::boundary_fields(context.clone(), vec![0.41, -0.19, 0.07])
            .unwrap();
    let bond_model =
        BoundaryRandomIsingModel::boundary_bonds_x(context, vec![0.31, -0.17, 0.29]).unwrap();

    for model in [&field_model, &bond_model] {
        for mask in 0..(1_usize << spec.site_count()) {
            let spins = (0..spec.site_count())
                .map(|bit| if (mask >> bit) & 1 == 0 { -1 } else { 1 })
                .collect();
            let lattice = SpinLattice::new(spec.clone(), spins).unwrap();
            let energy = model.energy(&lattice);
            for index in 0..spec.site_count() {
                let predicted = model.flip_energy_delta(&lattice, index);
                let mut flipped = lattice.clone();
                flipped.flip_index(index);
                let exact = model.energy(&flipped) - energy;
                assert!(
                    (predicted - exact).abs() < 1.0e-12,
                    "mask={mask}, index={index}, predicted={predicted}, exact={exact}"
                );
            }
        }
    }
}

#[test]
fn noise_probability_maps_to_mu_and_validates_range() {
    assert_eq!(mu_from_noise_probability(0.0).unwrap(), 0.0);
    let p = 0.1;
    let expected = -(1.0_f64 - 2.0 * p).ln();

    assert!((mu_from_noise_probability(p).unwrap() - expected).abs() < 1e-12);
    assert!(mu_from_noise_probability(0.5).is_err());
    assert!(mu_from_noise_probability(-0.1).is_err());
    assert!(mu_from_noise_probability(f64::NAN).is_err());

    assert_eq!(
        measurement_parameter(MeasurementKind::Heterodyne, p).unwrap(),
        expected
    );
    assert_eq!(
        measurement_parameter(MeasurementKind::Homodyne, p).unwrap(),
        2.0 * expected
    );
    let kappa = 2.0 * (p * (1.0 - p)).sqrt();
    assert!(
        (measurement_parameter(MeasurementKind::LocalX, p).unwrap() - kappa.atanh()).abs() < 1e-12
    );
}

#[test]
fn generated_clean_files_are_byte_reproducible() {
    let first = temp_path("clean-repro-a");
    let second = temp_path("clean-repro-b");
    let base = BoundaryStageInput {
        lx: 4,
        lt: 4,
        seed: 123,
        sample_count: 3,
        clean_therm_sweeps: 1,
        clean_skip_sweeps: 1,
        delta_tau: None,
        out_path: first.clone(),
    };
    generate_clean_stage(&base).unwrap();
    generate_clean_stage(&BoundaryStageInput {
        out_path: second.clone(),
        ..base
    })
    .unwrap();

    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
    let _ = fs::remove_file(first);
    let _ = fs::remove_file(second);
}

#[test]
fn one_clean_wolff_sweep_is_exactly_one_cluster_flip() {
    let spec = LatticeSpec::new(3, 2).unwrap();
    let context = IsingContext::new(spec.clone(), IsingCouplings::new(0.0, 0.0).unwrap());
    let model = CleanIsingModel::new(context);
    let mut lattice = SpinLattice::cold(spec, 1).unwrap();
    let mut rng = Rng64::seeded(17);

    IsingSampler::new(IsingUpdateMethod::Wolff, 1)
        .apply(&model, &mut lattice, &mut rng)
        .unwrap();

    assert_eq!(
        lattice.spins().iter().filter(|spin| **spin == -1).count(),
        1
    );
}

#[test]
fn one_corrected_wolff_sweep_consumes_one_rejected_proposal() {
    let spec = LatticeSpec::new(3, 2).unwrap();
    let site_count = NonZeroUsize::new(spec.site_count()).unwrap();
    let seed = (0_u64..)
        .find(|seed| Rng64::seeded(*seed).usize_nonzero(site_count) < spec.lx())
        .unwrap();
    let mut reference_rng = Rng64::seeded(seed);
    let proposal_site = reference_rng.usize_nonzero(site_count);
    for _ in 0..4 {
        reference_rng.uniform();
    }
    reference_rng.log_uniform();

    let context = IsingContext::new(spec.clone(), IsingCouplings::new(0.0, 0.0).unwrap());
    let mut fields = vec![0.0; spec.lx()];
    fields[proposal_site] = 100.0;
    let model = BoundaryRandomIsingModel::boundary_fields(context, fields).unwrap();
    let mut lattice = SpinLattice::cold(spec, 1).unwrap();
    let before = lattice.clone();
    let mut rng = Rng64::seeded(seed);

    FixedDisorderUpdate::CorrectedWolff
        .apply(&model, &mut lattice, &mut rng, 1)
        .unwrap();

    assert_eq!(
        lattice, before,
        "the deliberately unfavorable proposal must be rejected"
    );
    assert_eq!(rng.next_u64(), reference_rng.next_u64());
}

#[test]
fn public_clean_workflow_matches_exact_small_lattice_observable() {
    let path = temp_path("clean-exact-observable");
    let lx = 2;
    let lt = 3;
    generate_clean_stage(&BoundaryStageInput {
        lx,
        lt,
        seed: 4242,
        sample_count: 60_000,
        clean_therm_sweeps: 1_000,
        clean_skip_sweeps: 1,
        delta_tau: None,
        out_path: path.clone(),
    })
    .unwrap();

    let mut file = CleanSampleFile::open(&path).unwrap();
    let sampled = (0..file.header().sample_count as u64)
        .map(|sample_id| {
            let spins = file.read_sample(sample_id).unwrap().boundary_spins;
            f64::from(spins[0] * spins[1])
        })
        .sum::<f64>()
        / file.header().sample_count as f64;

    let spec = LatticeSpec::new(lx, lt).unwrap();
    let model = CleanIsingModel::from_critical_tfim_isotropic(lx, lt).unwrap();
    let mut partition = 0.0;
    let mut weighted_observable = 0.0;
    for mask in 0..(1_usize << spec.site_count()) {
        let spins = (0..spec.site_count())
            .map(|bit| if (mask >> bit) & 1 == 0 { -1 } else { 1 })
            .collect::<Vec<_>>();
        let observable = f64::from(spins[0] * spins[1]);
        let lattice = SpinLattice::new(spec.clone(), spins).unwrap();
        let weight = (-model.energy(&lattice)).exp();
        partition += weight;
        weighted_observable += weight * observable;
    }
    let exact = weighted_observable / partition;

    assert!(
        (sampled - exact).abs() < 0.02,
        "sampled boundary correlation {sampled} differs from exact {exact}"
    );
    let _ = fs::remove_file(path);
}

#[test]
fn clean_sample_file_round_trips_indexed_boundary_samples() {
    let path = temp_path("clean-round-trip");
    generate_clean_stage(&BoundaryStageInput {
        lx: 3,
        lt: 4,
        seed: 99,
        sample_count: 2,
        clean_therm_sweeps: 1,
        clean_skip_sweeps: 1,
        delta_tau: Some(0.3),
        out_path: path.clone(),
    })
    .unwrap();

    let mut file = CleanSampleFile::open(&path).unwrap();
    assert_eq!(file.header().lx, 3);
    assert_eq!(file.header().lt, 4);
    assert_eq!(file.header().sample_count, 2);
    assert_eq!(file.read_sample(1).unwrap().boundary_spins.len(), 3);
    assert!(file.read_sample(2).is_err());
    let _ = fs::remove_file(path);
}

#[test]
fn clean_sample_reader_rejects_trailing_or_missing_data() {
    let path = temp_path("clean-length");
    generate_clean_stage(&BoundaryStageInput {
        lx: 3,
        lt: 2,
        seed: 9,
        sample_count: 1,
        clean_therm_sweeps: 0,
        clean_skip_sweeps: 0,
        delta_tau: None,
        out_path: path.clone(),
    })
    .unwrap();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[1])
        .unwrap();

    assert!(CleanSampleFile::open(&path).is_err());
    let _ = fs::remove_file(path);
}

#[test]
fn aggregate_accumulator_computes_all_r_and_absolute_thermal_means() {
    let spec = LatticeSpec::new(4, 2).unwrap();
    let context = IsingContext::new(spec.clone(), IsingCouplings::new(0.0, 0.0).unwrap());
    let model = BoundaryRandomIsingModel::boundary_fields(context, vec![0.0; 4]).unwrap();
    let first = SpinLattice::new(spec.clone(), vec![1, 1, -1, -1, 1, 1, 1, 1]).unwrap();
    let second = SpinLattice::new(spec, vec![1, 1, -1, -1, 1, 1, 1, 1]).unwrap();
    let mut accumulator = AggregateAccumulator::new(4).unwrap();

    accumulator.record(&model, &first).unwrap();
    accumulator.record(&model, &second).unwrap();
    let record = accumulator.finish(7).unwrap();

    assert_eq!(record.spin_corr_signed.len(), 3);
    assert_eq!(record.boundary_spin_mean, vec![1.0, 1.0, -1.0, -1.0]);
    assert_eq!(record.spin_corr_signed[0], 1.0);
    assert!(record.spin_corr_signed[1].abs() < 1e-12);
    assert!((record.spin_corr_abs[1] - 1.0).abs() < 1e-12);
    assert!(record.spin_corr_abs[1] >= record.spin_corr_signed[1].abs());
}

#[test]
fn aggregate_file_round_trips_records() {
    let path = temp_path("aggregate-round-trip");
    let header = aggregate_header(0, 2);
    let records = vec![aggregate_record(0), aggregate_record(1)];

    write_aggregate_binary(&path, &header, &records).unwrap();
    let metadata = read_aggregate_metadata(&path).unwrap();
    let (read_header, read_records) = read_aggregate_binary(&path).unwrap();

    assert_eq!(metadata.record_count, 2);
    assert_eq!(read_header, header);
    assert_eq!(read_records, records);
    let _ = fs::remove_file(path);
}

#[test]
fn aggregate_file_round_trips_new_measurements() {
    for measurement in [MeasurementKind::Homodyne, MeasurementKind::LocalX] {
        let path = temp_path(measurement.as_str());
        let mut header = aggregate_header(0, 1);
        header.measurement = measurement;
        header.mu = measurement_parameter(measurement, header.p).unwrap();
        write_aggregate_binary(&path, &header, &[aggregate_record(0)]).unwrap();
        let (read_header, _) = read_aggregate_binary(&path).unwrap();
        assert_eq!(read_header.measurement, measurement);
        let _ = fs::remove_file(path);
    }
}

#[test]
fn aggregate_reader_rejects_trailing_records() {
    let path = temp_path("aggregate-length");
    write_aggregate_binary(&path, &aggregate_header(0, 1), &[aggregate_record(0)]).unwrap();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[0; 8])
        .unwrap();

    assert!(read_aggregate_binary(&path).is_err());
    assert!(read_aggregate_metadata(&path).is_err());
    let _ = fs::remove_file(path);
}

#[test]
fn analysis_csv_contains_every_periodic_separation() {
    let aggregate = temp_path("analysis-csv-input");
    let csv = temp_path("analysis-csv-output");
    write_aggregate_binary(&aggregate, &aggregate_header(0, 1), &[aggregate_record(0)]).unwrap();
    let report = analyze_aggregate_file(&aggregate).unwrap();
    write_analysis_csv(&csv, &report).unwrap();
    let text = fs::read_to_string(&csv).unwrap();
    let lines = text.lines().collect::<Vec<_>>();

    assert_eq!(lines.len(), report.lx / 2 + 2);
    assert!(lines[0].contains("spin_fidelity_corr"));
    assert!(lines.last().unwrap().split(',').nth(8) == Some("2"));
    let _ = fs::remove_file(aggregate);
    let _ = fs::remove_file(csv);
}

#[test]
fn disorder_record_csv_preserves_independent_outer_samples() {
    let csv = temp_path("disorder-records-csv");
    let header = aggregate_header(0, 2);
    let records = vec![aggregate_record(0), aggregate_record(1)];
    write_disorder_records_csv(&csv, &header, &records).unwrap();
    let text = fs::read_to_string(&csv).unwrap();
    let lines = text.lines().collect::<Vec<_>>();

    assert_eq!(lines.len(), 1 + records.len() * header.r_count());
    assert!(lines[0].contains("local_spin_fidelity"));
    assert!(lines[0].contains("spin_linear_corr"));
    let _ = fs::remove_file(csv);
}

#[test]
fn aggregate_merge_sorts_and_rejects_missing_or_duplicate_ids() {
    let first_path = temp_path("aggregate-merge-first");
    let second_path = temp_path("aggregate-merge-second");
    let merged_path = temp_path("aggregate-merge-out");
    write_aggregate_binary(&first_path, &aggregate_header(2, 1), &[aggregate_record(2)]).unwrap();
    write_aggregate_binary(
        &second_path,
        &aggregate_header(0, 2),
        &[aggregate_record(0), aggregate_record(1)],
    )
    .unwrap();

    let merged =
        merge_aggregate_binaries(&merged_path, &[first_path.clone(), second_path.clone()]).unwrap();
    let (_, records) = read_aggregate_binary(&merged_path).unwrap();
    assert_eq!(merged, 3);
    assert_eq!(
        records
            .iter()
            .map(|record| record.disorder_id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );

    let bad_path = temp_path("aggregate-merge-bad");
    write_aggregate_binary(&bad_path, &aggregate_header(4, 1), &[aggregate_record(4)]).unwrap();
    assert!(merge_aggregate_binaries(temp_path("aggregate-merge-missing"), &[bad_path]).is_ok());
    let _ = fs::remove_file(first_path);
    let _ = fs::remove_file(second_path);
    let _ = fs::remove_file(merged_path);
}

#[test]
fn aggregate_merge_rejects_duplicate_or_gap_across_chunks() {
    let first_path = temp_path("aggregate-gap-first");
    let second_path = temp_path("aggregate-gap-second");
    write_aggregate_binary(&first_path, &aggregate_header(0, 1), &[aggregate_record(0)]).unwrap();
    write_aggregate_binary(
        &second_path,
        &aggregate_header(2, 1),
        &[aggregate_record(2)],
    )
    .unwrap();

    assert!(
        merge_aggregate_binaries(
            temp_path("aggregate-gap-out"),
            &[first_path.clone(), second_path.clone()]
        )
        .is_err()
    );
    let _ = fs::remove_file(first_path);
    let _ = fs::remove_file(second_path);
}

#[test]
fn measurement_stage_writes_aggregate_records_for_remaining_updates() {
    let clean = temp_path("measure-clean");
    let out = temp_path("measure-agg");
    generate_clean_stage(&BoundaryStageInput {
        lx: 4,
        lt: 4,
        seed: 321,
        sample_count: 2,
        clean_therm_sweeps: 1,
        clean_skip_sweeps: 1,
        delta_tau: None,
        out_path: clean.clone(),
    })
    .unwrap();

    let records = measure_stage_to_file(&MeasureStageInput {
        clean_path: clean.clone(),
        noise: NoiseKind::Zz,
        measurement: MeasurementKind::Heterodyne,
        p: 0.1,
        sample_start: 0,
        sample_count: 2,
        disorder_update: FixedDisorderUpdate::SequentialMetropolis,
        disorder_therm_sweeps: 1,
        measurements: 2,
        skip_sweeps: 1,
        out_path: out.clone(),
    })
    .unwrap();
    let report = analyze_aggregate_file(&out).unwrap();

    assert_eq!(records, 2);
    assert_eq!(report.samples, 2);
    assert!(report.energy_density.is_finite());
    let _ = fs::remove_file(clean);
    let _ = fs::remove_file(out);
}

#[test]
fn measurement_is_reproducible_and_chunk_deterministic() {
    let clean = temp_path("measure-repro-clean");
    generate_clean_stage(&BoundaryStageInput {
        lx: 4,
        lt: 4,
        seed: 777,
        sample_count: 4,
        clean_therm_sweeps: 1,
        clean_skip_sweeps: 1,
        delta_tau: None,
        out_path: clean.clone(),
    })
    .unwrap();
    let base = MeasureStageInput {
        clean_path: clean.clone(),
        noise: NoiseKind::Z,
        measurement: MeasurementKind::Heterodyne,
        p: 0.1,
        sample_start: 0,
        sample_count: 4,
        disorder_update: FixedDisorderUpdate::Metropolis,
        disorder_therm_sweeps: 1,
        measurements: 2,
        skip_sweeps: 1,
        out_path: temp_path("unused"),
    };
    let full = measure_stage(&base).unwrap();
    let first = measure_stage(&MeasureStageInput {
        sample_count: 2,
        ..base.clone()
    })
    .unwrap();
    let second = measure_stage(&MeasureStageInput {
        sample_start: 2,
        sample_count: 2,
        ..base
    })
    .unwrap();
    let mut chunked = first;
    chunked.extend(second);
    chunked.sort_by_key(|record| record.disorder_id);
    assert_eq!(full, chunked);
    let _ = fs::remove_file(clean);
}

#[test]
fn chunk_plan_and_run_spec_dry_run_partition_samples() {
    let ranges = chunk_plan(10, 3).unwrap();
    assert_eq!(
        ranges
            .iter()
            .map(|range| (range.sample_start, range.sample_count))
            .collect::<Vec<_>>(),
        vec![(0, 4), (4, 3), (7, 3)]
    );

    let path = temp_path("run-spec");
    fs::write(&path, "samples = 10\nchunks = 3\n").unwrap();
    assert_eq!(read_run_spec_chunk_plan(&path).unwrap(), ranges);
    let _ = fs::remove_file(path);
}

fn aggregate_header(sample_start: u64, sample_count: usize) -> AggregateFileHeader {
    AggregateFileHeader {
        clean_sample_count: 8,
        lx: 4,
        lt: 4,
        kx: 0.1,
        kt: 0.2,
        seed: 11,
        noise: NoiseKind::Z,
        measurement: MeasurementKind::Heterodyne,
        p: 0.1,
        mu: mu_from_noise_probability(0.1).unwrap(),
        sample_start,
        sample_count,
        disorder_update: FixedDisorderUpdate::Metropolis,
        disorder_therm_sweeps: 1,
        measurements: 2,
        skip_sweeps: 1,
    }
}

fn aggregate_record(disorder_id: u64) -> AggregateRecord {
    AggregateRecord {
        disorder_id,
        energy_mean: -1.0,
        bulk_magnetization_sum_mean: 2.0,
        boundary_spin_mean: vec![1.0, 0.0, -1.0, 0.0],
        boundary_bond_mean: vec![0.0, 0.0, 0.0, 0.0],
        spin_corr_signed: vec![1.0, 0.0, -1.0],
        bond_corr_signed: vec![1.0, 0.5, 0.0],
        spin_corr_abs: vec![1.0, 0.5, 1.0],
        bond_corr_abs: vec![1.0, 0.5, 0.0],
    }
}
