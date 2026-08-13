# Exact Diagonalization

Small-system Python tools for the calculations documented in `notes/main.tex`.

The implemented quantum model is

```text
H_TFIM = -J sum_x Z_x Z_{x+1} - h sum_x X_x
```

with periodic boundary conditions by default. The computational basis uses
integer bit strings. A bit value `0` has `Z=+1`, and a bit value `1` has
`Z=-1`.

The first implemented workflow is:

1. Build the TFIM Hamiltonian.
2. Solve the finite-size ground state.
3. Build `rho0 = |psi><psi|`.
4. Apply the iid `Z` or `ZZ` decoherence channel

```text
N_x(rho) = (1 - p) rho + p O_x rho O_x
```

where `O_x = Z_x` for `z` noise and `O_x = Z_x Z_{x+1}` for `zz` noise.

The original list-based helpers remain available.  The comparison study uses
NumPy dense linear algebra and is intended for modest Hilbert spaces.

## Numerical comparison campaign

The study driver evaluates exact linear observables and trace-norm fidelity
averages, and independently evaluates heterodyne, homodyne, and local-X
environment-measurement witnesses using exact posterior sums. It does this
both for the finite TFIM ground-state prior
and for the finite-torus transfer-matrix prior matching a Monte Carlo lattice.

```bash
dcft-ed-study \
  --sizes 4,6,8 \
  --p-values 0,0.1,0.2,0.3,0.4,0.49 \
  --delta-tau 0.2 \
  --l-tau-multiplier 16 \
  --witness-samples 100000 \
  --measurements heterodyne,homodyne,local_x \
  --output ../results/exact-diagonalization
```

The posterior over boundary configurations is summed exactly for each record.
Only the outer Gaussian integral is sampled for heterodyne and homodyne;
local-X binary records are completely enumerated. `prior_diagnostics.csv`
separately quantifies finite-imaginary-time and Trotter mismatch.

## Example

```python
from decohered_cft_ed import (
    TFIMHamiltonian,
    apply_decoherence_channel,
    ground_state,
    pure_density_matrix,
)

ham = TFIMHamiltonian(n_sites=4, j=1.0, h=1.0)
energy, psi = ground_state(ham)
rho0 = pure_density_matrix(psi)
rho = apply_decoherence_channel(rho0, n_sites=4, p=0.25, noise="zz")
```

## Tests

From this directory:

```bash
python3 -m unittest discover -s tests
```
