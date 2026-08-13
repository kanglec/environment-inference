# Decohered-CFT campaign analysis

This directory contains the campaign-level analysis tool for the Monte Carlo
`observables.csv` files. It validates the Rust writer's schema and metadata,
combines all simulation points into tidy tables, produces PNG and PDF figures,
and records what it found in a concise manifest.

The analysis follows the definitions in `notes/main.tex`. In particular, it
treats annealed linear correlators as sampling/thermalization checks, labels
fidelity quantities as approximate nonlinear observables from the all-to-all
replica approximation, and labels Edwards–Anderson quantities separately. Finite-size campaign
plots, especially an `L8` pilot, do not by themselves establish a transition
or critical behavior.

## Setup with uv

From this directory:

```sh
uv sync
```

The committed `uv.lock` pins the resolved environment. The package exposes
`dcft-analyze-campaign` for one lattice size and `dcft-analyze-scaling` for
combining completed campaign outputs across square system sizes.

## Input layout

Pass a single lattice directory as `--root`:

```text
ROOT/L8/
  z/
    p000/analysis/observables.csv
    p005/analysis/observables.csv
  zz/
    p000/analysis/observables.csv
    p005/analysis/observables.csv
```

The pipeline uses `L<N>` for square lattices and `L<Lx>x<Lt>` for rectangular
lattices. CSV metadata remains authoritative, and a recognizable lattice tag
is checked against `lx` and `lt`. A probability directory must be exactly
`pNNN`, where `NNN = 100 p`, zero-padded to three digits. Supported campaign
tags are `p000` through `p049`, corresponding to `0.00` through `0.49` in
increments of `0.01`. Supported noise directories are `z` and `zz`.

The tool recursively finds files with the suffix
`{noise}/pNNN/analysis/observables.csv`. It requires the exact 16-column Rust
analysis schema, one unique row for every `r = 0..floor(Lx/2)`, constant scalar
and metadata columns across `r`, and consistent `samples`, `lx`, and `lt`
across all campaign points.

## Commands

Typical strict run:

```sh
uv run dcft-analyze-campaign \
  --root /path/to/data/L8 \
  --out /path/to/data/L8/campaign-analysis
```

Strict validation is the default. The inferred expected grid is the Cartesian
product of discovered noise types and the union of discovered p-tags. This
detects, for example, `zz/p025` missing when `z/p025` exists. To detect values
missing from every noise directory, state the campaign grid explicitly:

```sh
uv run dcft-analyze-campaign \
  --root /path/to/data/L8 \
  --out /path/to/data/L8/campaign-analysis \
  --expected-noise z --expected-noise zz \
  --expected-p 0.00 --expected-p 0.05 --expected-p 0.25 --expected-p 0.49
```

Use `--allow-incomplete` to produce partial-campaign outputs while recording
missing points as warnings. This option does **not** permit malformed p-tags,
unsupported noise types, duplicate points or separations, invalid schemas,
non-finite observables, inconsistent repeated scalars, inconsistent metadata,
or an `r` range that does not match `lx`.

Development checks:

```sh
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

## Outputs

The output directory is organized as follows:

```text
campaign-analysis/
  tables/
    scalar_observables.csv
    correlators.csv
  figures/
    *.png
    *.pdf
  summary.json
  summary.md
```

`scalar_observables.csv` has one row per `(noise, p)`.
`correlators.csv` has one row per `(noise, p, r)`. Every figure is emitted as a
high-resolution PNG and a vector PDF. Figures cover scalar observables, local
fidelities, all six correlator families versus
`r`, short- and largest-separation p-dependence, and the expected
p-independence of annealed linear correlators.

## Column-to-physics mapping

| Rust `observables.csv` column | Tidy output column | Definition/role |
|---|---|---|
| `energy_density` | `E_D` | Disordered energy density $E_D(p)$ |
| `magnetization_density` | `M` | Bulk magnetization density $M(p)$ |
| `boundary_magnetization` | `M_partial` | Boundary magnetization density $M_\partial(p)$ |
| `local_spin_fidelity` | `F_sigma_loc` | Approximate local spin fidelity $F_\sigma^{\rm loc}(p)$ |
| `local_bond_fidelity` | `F_epsilon_loc` | Approximate local bond fidelity $F_\epsilon^{\rm loc}(p)$ |
| `spin_linear_corr` | `C_sigma_lin` | Linear spin correlator $C_\sigma^{\rm lin}(p,r)$; annealed sanity check |
| `bond_linear_corr` | `C_epsilon_lin` | Linear bond correlator $C_\epsilon^{\rm lin}(p,r)$; annealed sanity check |
| `spin_fidelity_corr` | `F_sigma` | Approximate spin fidelity correlator $F_\sigma(p,r)$ |
| `bond_fidelity_corr` | `F_epsilon` | Approximate bond fidelity correlator $F_\epsilon(p,r)$ |
| `spin_ea_corr` | `C_sigma_EA` | Spin Edwards–Anderson correlator $C_\sigma^{\rm EA}(p,r)$ |
| `bond_ea_corr` | `C_epsilon_EA` | Bond Edwards–Anderson correlator $C_\epsilon^{\rm EA}(p,r)$ |

The fidelity and EA observables above inherit the all-to-all replica
approximation described in the notes. The normalized annealed linear
expectations have the expected cancellation of the p-dependence; the
fixed-disorder energy is an exception because it explicitly contains the
disorder term.

## Finite-size-scaling analysis

After running `dcft-analyze-campaign` separately for every square size, place
or leave the outputs anywhere beneath one common root. The scaling command is
directory-name agnostic: it recursively discovers every
`tables/scalar_observables.csv` and pairs it with the sibling
`tables/correlators.csv`. For example, both `L8/analysis/tables/` and
`L12/campaign-analysis/tables/` are accepted.

```text
/path/to/data/
  L8/analysis/tables/
    scalar_observables.csv
    correlators.csv
  L12/campaign-analysis/tables/
    scalar_observables.csv
    correlators.csv
  L16/campaign-analysis/tables/
    scalar_observables.csv
    correlators.csv
```

Run:

```sh
uv run dcft-analyze-scaling \
  --root /path/to/data \
  --out /path/to/data/scaling-analysis
```

Scaling validation requires `lx == lt` and uses the CSV metadata—not directory
names—to identify `L`. It validates the point keys and complete separation
range again, rejects duplicate `(L, noise, p)` inputs, and by default requires
the complete Cartesian product of discovered sizes, noise types, and p-tags.
Use `--allow-incomplete` to retain a partial size grid while recording missing
points in the manifest. Malformed, rectangular, or duplicate data remain
fatal.

The scaling output is:

```text
scaling-analysis/
  tables/
    scalar_observables_by_size.csv
    correlators_by_size.csv
    long_distance_observables.csv
    power_law_fits.csv
  figures/
    *.png
    *.pdf
  summary.json
  summary.md
```

`correlators_by_size.csv` adds

\[
  r/L, \qquad
  d_L(r)=\frac{L}{\pi}\sin\left(\frac{\pi r}{L}\right),
\]

where `d_L(r)` is the periodic chord distance.
`long_distance_observables.csv` selects the largest periodic separation
`r = floor(L/2)`, which is exactly `L/2` for the even sizes normally used by
the pipeline.

For every `(noise, p, observable)`, `power_law_fits.csv` fits the positive
long-distance values to

\[
  C(r_{\max};L)=A L^{-\alpha}=A L^{-2\Delta},
  \qquad r_{\max}=\lfloor L/2\rfloor.
\]

It records `amplitude`, `decay_exponent` $\alpha$, `scaling_dimension`
$\Delta=\alpha/2$, `r_squared`, the fit-size range, and a `fit_status`.
The fit uses unweighted ordinary least squares in log space and requires at
least two positive sizes. These are exploratory point-estimate fits, not
precision exponent determinations.

Scaling figures include:

- scalar and local-fidelity p-dependence with one curve per `L`;
- all six largest-separation correlators versus p;
- log-log largest-separation size dependence for selected p values;
- fitted exploratory $\Delta(p)$ values;
- all six correlator families versus chord distance, with curves for each `L`.

If more than six p values are present, the chord-distance and crowded log-log
figures select six evenly spaced p values, including the endpoints. All p
values remain present in the machine-readable tables and the versus-p plots.

## Current statistical limitation

This version computes no bootstrap or jackknife estimates, confidence
intervals, uncertainty estimates, covariance matrices, or error bars. The
tables, plots, and scaling fits use only the disorder-averaged point estimates
written by the Rust analysis stage. Apparent crossings, collapses, and fitted
exponents therefore do not by themselves establish a transition, critical
point, or universal exponent.
