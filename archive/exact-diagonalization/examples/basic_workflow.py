"""Run the first exact-diagonalization workflow.

Usage from ``exact-diagonalization``:

    PYTHONPATH=src python3 examples/basic_workflow.py
"""

from decohered_cft_ed import (
    TFIMHamiltonian,
    apply_decoherence_channel,
    ground_state,
    pure_density_matrix,
    trace,
)


def main() -> None:
    n_sites = 4
    p = 0.25

    hamiltonian = TFIMHamiltonian(n_sites=n_sites, j=1.0, h=1.0)
    energy, state = ground_state(hamiltonian)
    rho0 = pure_density_matrix(state)
    rho_z = apply_decoherence_channel(rho0, n_sites=n_sites, p=p, noise="z")
    rho_zz = apply_decoherence_channel(rho0, n_sites=n_sites, p=p, noise="zz")

    print(f"n_sites = {n_sites}")
    print(f"ground_energy = {energy:.12f}")
    print(f"Tr rho0 = {trace(rho0):.12f}")
    print(f"Tr N_z(rho0) = {trace(rho_z):.12f}")
    print(f"Tr N_zz(rho0) = {trace(rho_zz):.12f}")


if __name__ == "__main__":
    main()

