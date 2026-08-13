import math
import unittest

from decohered_cft_ed import (
    TFIMHamiltonian,
    apply_decoherence_channel,
    decoherence_factor,
    ground_state,
    pure_density_matrix,
    trace,
)
from decohered_cft_ed.decoherence import lambda_from_noise_probability


class TFIMHamiltonianTests(unittest.TestCase):
    def test_hamiltonian_is_symmetric(self):
        matrix = TFIMHamiltonian(n_sites=3, j=1.0, h=0.7).to_dense()
        for row in range(len(matrix)):
            for col in range(len(matrix)):
                self.assertAlmostEqual(matrix[row][col], matrix[col][row])

    def test_known_classical_limit_for_two_periodic_sites(self):
        ham = TFIMHamiltonian(n_sites=2, j=1.0, h=0.0)
        diagonal = [ham.to_dense()[i][i] for i in range(ham.dim)]
        self.assertEqual(diagonal, [-2.0, 2.0, 2.0, -2.0])

    def test_known_transverse_limit_ground_energy(self):
        energy, state = ground_state(TFIMHamiltonian(n_sites=3, j=0.0, h=2.0))
        self.assertAlmostEqual(energy, -6.0, places=10)
        self.assertAlmostEqual(sum(value * value for value in state), 1.0, places=10)

    def test_critical_tfim_ground_state_solves_eigenproblem(self):
        ham = TFIMHamiltonian(n_sites=4, j=1.0, h=1.0)
        energy, state = ground_state(ham)
        applied = ham.apply(state)
        residual = math.sqrt(
            sum((applied[i] - energy * state[i]) ** 2 for i in range(ham.dim))
        )
        self.assertLess(residual, 1e-9)


class DecoherenceChannelTests(unittest.TestCase):
    def test_z_decoherence_entry_factor_counts_bit_differences(self):
        # 0b001 and 0b101 differ only at site 2.
        factor = decoherence_factor(
            0b001,
            0b101,
            n_sites=3,
            p=0.2,
            noise="z",
        )
        self.assertAlmostEqual(factor, 1.0 - 2.0 * 0.2)

    def test_zz_decoherence_entry_factor_counts_bond_differences(self):
        factor = decoherence_factor(
            0b000,
            0b001,
            n_sites=3,
            p=0.2,
            noise="zz",
        )
        # For a periodic 3-site chain, flipping one spin changes two ZZ bonds.
        self.assertAlmostEqual(factor, (1.0 - 2.0 * 0.2) ** 2)

    def test_channel_preserves_trace(self):
        _, state = ground_state(TFIMHamiltonian(n_sites=3, j=1.0, h=1.0))
        rho = pure_density_matrix(state)
        decohered = apply_decoherence_channel(
            rho,
            n_sites=3,
            p=0.25,
            noise="zz",
        )
        self.assertAlmostEqual(trace(rho), 1.0)
        self.assertAlmostEqual(trace(decohered), 1.0)

    def test_zero_noise_is_identity_channel(self):
        _, state = ground_state(TFIMHamiltonian(n_sites=3, j=1.0, h=1.0))
        rho = pure_density_matrix(state)
        decohered = apply_decoherence_channel(rho, n_sites=3, p=0.0, noise="z")
        self.assertEqual(decohered, rho)

    def test_lambda_from_noise_probability_matches_notes(self):
        p = 0.2
        self.assertAlmostEqual(lambda_from_noise_probability(p), -0.5 * math.log(1 - 2 * p))


if __name__ == "__main__":
    unittest.main()

