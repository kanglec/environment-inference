import unittest

import numpy as np
from decohered_cft_ed.numerics import (
    density_square_root,
    fidelity_average_from_sqrt,
    gaussian_measurement_witness,
    linear_expectation,
    local_x_measurement_witness,
    observable_families,
    tfim_ground_state,
    tfim_hamiltonian,
    transfer_boundary_prior,
    z_noise_density,
)


class NumpyNumericsTests(unittest.TestCase):
    def test_ground_state_solves_dense_problem(self):
        energy, state = tfim_ground_state(4)
        residual = np.linalg.norm(tfim_hamiltonian(4) @ state - energy * state)
        self.assertLess(residual, 1e-11)
        self.assertAlmostEqual(float(state @ state), 1.0)

    def test_zero_noise_fidelity_is_absolute_pure_expectation(self):
        _, state = tfim_ground_state(4)
        rho = z_noise_density(state, 4, 0.0)
        sqrt_rho = density_square_root(rho)
        for family in observable_families(4):
            exact = fidelity_average_from_sqrt(rho, sqrt_rho, family.eigenvalues[0])
            expected = abs(float(family.eigenvalues[0] @ (state * state)))
            self.assertAlmostEqual(exact, expected, places=10)

    def test_gaussian_record_obeys_linear_rule_and_fidelity_bound(self):
        _, state = tfim_ground_state(4)
        prior = state * state
        family = next(f for f in observable_families(4) if f.name == "local_bond")
        rho = z_noise_density(state, 4, 0.25)
        exact = fidelity_average_from_sqrt(rho, density_square_root(rho), family.eigenvalues[0])
        estimate = gaussian_measurement_witness(
            prior, [family], n_sites=4, p=0.25, samples=20_000, seed=7
        )[(family.name, family.separation)]
        self.assertLessEqual(estimate.value, exact + 5 * estimate.standard_error)
        self.assertAlmostEqual(
            estimate.linear_sample_mean,
            linear_expectation(prior, family),
            delta=5 * estimate.linear_standard_error,
        )

    def test_transfer_priors_are_normalized(self):
        finite, ground = transfer_boundary_prior(4, l_tau=64, delta_tau=0.2)
        self.assertAlmostEqual(float(np.sum(finite)), 1.0)
        self.assertAlmostEqual(float(np.sum(ground)), 1.0)
        self.assertTrue(np.all(finite >= 0.0))
        self.assertTrue(np.all(ground >= 0.0))

    def test_homodyne_improves_the_diagonal_witness(self):
        _, state = tfim_ground_state(4)
        prior = state * state
        families = [f for f in observable_families(4) if f.name in {"local_spin", "local_bond"}]
        heterodyne = gaussian_measurement_witness(
            prior,
            families,
            n_sites=4,
            p=0.25,
            samples=50_000,
            seed=19,
            measurement="heterodyne",
        )
        homodyne = gaussian_measurement_witness(
            prior,
            families,
            n_sites=4,
            p=0.25,
            samples=50_000,
            seed=19,
            measurement="homodyne",
        )
        for family in families:
            key = (family.name, family.separation)
            tolerance = 5.0 * np.hypot(heterodyne[key].standard_error, homodyne[key].standard_error)
            self.assertGreaterEqual(homodyne[key].value + tolerance, heterodyne[key].value)

    def test_local_x_enumeration_preserves_linear_average_and_fidelity_bound(self):
        _, state = tfim_ground_state(4)
        prior = state * state
        family = next(f for f in observable_families(4) if f.name == "local_bond")
        estimate = local_x_measurement_witness(prior, [family], n_sites=4, p=0.25)[
            (family.name, family.separation)
        ]
        rho = z_noise_density(state, 4, 0.25)
        exact = fidelity_average_from_sqrt(rho, density_square_root(rho), family.eigenvalues[0])
        self.assertAlmostEqual(
            estimate.linear_sample_mean, linear_expectation(prior, family), places=12
        )
        self.assertLessEqual(estimate.value, exact + 1e-12)
        self.assertEqual(estimate.standard_error, 0.0)


if __name__ == "__main__":
    unittest.main()
