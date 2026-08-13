use _core::mc::{Couplings, Lattice, Model, Update, UpdateStats};
use _core::physics::{
    Measurement, Noise, ProtocolParameters, gaussian_record, local_x_record, noise_variables,
};
use _core::rng::Rng64;

#[test]
fn property_bit_packing_round_trips_many_lattices() {
    let mut rng = Rng64::seeded(0x5eed);
    for lx in 2..10 {
        for lt in 2..8 {
            for _ in 0..32 {
                let lattice = Lattice::random(lx, lt, &mut rng).expect("valid lattice");
                let recovered =
                    Lattice::unpack(lx, lt, &lattice.pack()).expect("valid packed data");
                assert_eq!(recovered, lattice);
            }
        }
    }
}

#[test]
fn property_energy_deltas_match_recomputation() {
    let mut rng = Rng64::seeded(0xdeca_fbad);
    for noise in [Noise::Z, Noise::Zz] {
        for lx in 2..8 {
            let lt = lx + 1;
            let couplings = Couplings::new(0.13 * lx as f64, 0.37).expect("positive");
            for _ in 0..20 {
                let lattice = Lattice::random(lx, lt, &mut rng).expect("valid lattice");
                let record: Vec<f64> = (0..lx).map(|_| 2.0 * rng.uniform() - 1.0).collect();
                let model =
                    Model::posterior(lx, lt, couplings, noise, record).expect("valid model");
                for index in 0..lattice.site_count() {
                    let mut flipped = lattice.clone();
                    flipped.flip(index);
                    let recomputed = model.energy(&flipped) - model.energy(&lattice);
                    assert!((model.flip_delta(&lattice, index) - recomputed).abs() < 2.0e-12);
                }
            }
        }
    }
}

#[test]
fn property_protocol_records_obey_the_shared_formulas() {
    let mut rng = Rng64::seeded(9);
    for sites in 2..32 {
        let boundary: Vec<i8> = (0..sites)
            .map(|_| if rng.bool() { 1 } else { -1 })
            .collect();
        for noise in [Noise::Z, Noise::Zz] {
            let variables = noise_variables(&boundary, noise).expect("valid boundary");
            let normals: Vec<f64> = (0..sites).map(|_| rng.normal()).collect();
            let gamma = 0.713;
            let record = gaussian_record(&variables, gamma, &normals).expect("valid record");
            for index in 0..sites {
                let expected = gamma * f64::from(variables[index]) + gamma.sqrt() * normals[index];
                assert!((record[index] - expected).abs() <= f64::EPSILON);
            }

            let parameters = ProtocolParameters::new(Measurement::LocalX, 0.23).expect("valid p");
            let uniforms: Vec<f64> = (0..sites).map(|_| rng.uniform()).collect();
            let (outcomes, fields) =
                local_x_record(&variables, parameters, &uniforms).expect("valid record");
            let coupling = parameters.coupling.expect("local-X coupling");
            for index in 0..sites {
                assert!(matches!(outcomes[index], -1 | 1));
                assert!(
                    (fields[index] - coupling * f64::from(outcomes[index])).abs() <= f64::EPSILON
                );
            }
        }
    }
}

fn exact_energy_and_boundary_order(model: &Model, noise: Noise) -> (f64, f64) {
    let site_count = model.lx() * model.lt();
    let mut normalizer = 0.0;
    let mut energy = 0.0;
    let mut order = 0.0;
    for bits in 0..1_usize << site_count {
        let spins = (0..site_count)
            .map(|site| if bits & (1 << site) == 0 { 1 } else { -1 })
            .collect();
        let lattice = Lattice::new(model.lx(), model.lt(), spins).expect("valid lattice");
        let weight = (-model.energy(&lattice)).exp();
        let observable = match noise {
            Noise::Z => {
                lattice
                    .boundary()
                    .iter()
                    .map(|spin| f64::from(*spin))
                    .sum::<f64>()
                    / model.lx() as f64
            }
            Noise::Zz => {
                (0..model.lx())
                    .map(|x| f64::from(lattice.get(x, 0) * lattice.get((x + 1) % model.lx(), 0)))
                    .sum::<f64>()
                    / model.lx() as f64
            }
        };
        normalizer += weight;
        energy += weight * model.energy(&lattice) / site_count as f64;
        order += weight * observable;
    }
    (energy / normalizer, order / normalizer)
}

fn sampled_energy_and_boundary_order(
    model: &Model,
    noise: Noise,
    update: Update,
    seed: u64,
) -> (f64, f64, UpdateStats) {
    let mut rng = Rng64::seeded(seed);
    let mut lattice = Lattice::random(model.lx(), model.lt(), &mut rng).expect("valid lattice");
    let mut statistics = UpdateStats::default();
    for _ in 0..8_000 {
        update
            .apply(model, &mut lattice, &mut rng, &mut statistics)
            .expect("valid update");
    }
    let samples = 160_000_u32;
    let mut energy = 0.0;
    let mut order = 0.0;
    for _ in 0..samples {
        update
            .apply(model, &mut lattice, &mut rng, &mut statistics)
            .expect("valid update");
        energy += model.energy(&lattice) / lattice.site_count() as f64;
        order += match noise {
            Noise::Z => {
                lattice
                    .boundary()
                    .iter()
                    .map(|spin| f64::from(*spin))
                    .sum::<f64>()
                    / model.lx() as f64
            }
            Noise::Zz => {
                (0..model.lx())
                    .map(|x| f64::from(lattice.get(x, 0) * lattice.get((x + 1) % model.lx(), 0)))
                    .sum::<f64>()
                    / model.lx() as f64
            }
        };
    }
    (
        energy / f64::from(samples),
        order / f64::from(samples),
        statistics,
    )
}

#[test]
fn tnmc_and_metropolis_agree_with_exact_disordered_ensembles() {
    let couplings = Couplings::new(0.29, 0.53).expect("valid couplings");
    for (case, noise) in [Noise::Z, Noise::Zz].into_iter().enumerate() {
        let model = Model::posterior(3, 3, couplings, noise, vec![0.74, -0.91, 0.36])
            .expect("valid disordered model");
        let exact = exact_energy_and_boundary_order(&model, noise);
        let metropolis = sampled_energy_and_boundary_order(
            &model,
            noise,
            Update::SequentialMetropolis,
            0x51a7 + case as u64,
        );
        let tnmc = sampled_energy_and_boundary_order(
            &model,
            noise,
            Update::Tnmc {
                maximum_bond_dimension: 1,
            },
            0x7e11 + case as u64,
        );
        let metropolis_acceptance =
            metropolis.2.local_accepted as f64 / metropolis.2.local_proposed as f64;
        let tnmc_acceptance = tnmc.2.tnmc_accepted as f64 / tnmc.2.tnmc_proposed as f64;
        eprintln!(
            "{noise:?}: exact={exact:?}, sequential Metropolis=({:.9}, {:.9}, acceptance={metropolis_acceptance:.4}), TNMC chi=1=({:.9}, {:.9}, acceptance={tnmc_acceptance:.4})",
            metropolis.0, metropolis.1, tnmc.0, tnmc.1
        );
        assert_eq!(tnmc.2.tnmc_conditionals_regularized, 0);
        for (label, estimate) in [("Metropolis", metropolis), ("TNMC chi=1", tnmc)] {
            assert!(
                (estimate.0 - exact.0).abs() < 0.02,
                "{label} energy disagrees for {noise:?}: {} versus {}",
                estimate.0,
                exact.0
            );
            assert!(
                (estimate.1 - exact.1).abs() < 0.025,
                "{label} boundary observable disagrees for {noise:?}: {} versus {}",
                estimate.1,
                exact.1
            );
        }
        assert!((tnmc.0 - metropolis.0).abs() < 0.025);
        assert!((tnmc.1 - metropolis.1).abs() < 0.03);
    }
}
