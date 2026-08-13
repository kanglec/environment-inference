# Monte Carlo Simulation

Rust Monte Carlo tools for the boundary random Ising representation of the
decohered CFT problem.

## Build

```sh
cargo build --release
```

Use the release binary for production jobs:

```sh
./target/release/decohered-cft <subcommand> [args...]
```

Run `decohered-cft --help` or `decohered-cft <subcommand> --help` for the
Clap-generated usage summary.

## CLI Reference

Common parameter rules:

- `--lx` and `--lt` are the spatial and imaginary-time lattice sizes and must
  both be at least 2.
- `--seed` is the base RNG seed and defaults to `0`.
- `--noise` accepts `z` or `zz`.
- `--p` is the decoherence probability and must satisfy `0 <= p < 0.5`.
- With no `--delta-tau`, the simulation uses the isotropic critical Ising
  coupling. With `--delta-tau DT`, it uses `Kx = DT` and
  `Kt = -0.5 ln(tanh(DT))`; `DT` must be positive and finite.
- Commands that write a new simulation file require exactly one of `--out`
  (an exact path) or `--out-dir` (a directory plus an automatically generated
  filename). Missing parent directories are created.
- On commands with `--threads`, an explicit value takes precedence. Otherwise
  `RAYON_NUM_THREADS` is honored, followed by `SLURM_CPUS_PER_TASK`; if none is set, Rayon chooses its default thread count.

### `generate-clean`

Generate clean-model boundary samples with Wolff updates. The initial random
configuration is thermalized once, after which the same chain advances by
`--clean-skip-sweeps` before each saved sample.

```text
decohered-cft generate-clean \
  --lx L --lt T \
  [--seed SEED] \
  --samples N \
  --clean-therm-sweeps N \
  --clean-skip-sweeps N \
  (--out clean.bin | --out-dir DIR) \
  [--delta-tau DT]
```

Arguments:

- `--samples`: number of boundary samples to save; must be at least 1.
- `--clean-therm-sweeps`: clean Wolff sweeps before sample generation.
- `--clean-skip-sweeps`: clean Wolff sweeps before each saved sample.
- For the clean Wolff update, one sweep means exactly one cluster construction
  and flip, independent of the realized cluster size.
- `--out-dir`: generates a name of the form
  `clean_lx<L>_lt<T>_<iso-or-dt>_seed<SEED>_samples<N>.bin`.

### `measure`

Read clean boundary samples, draw tilted disorder, run a fixed-disorder chain
for each requested sample, and write one aggregate record per disorder
realization.

```text
decohered-cft measure \
  --clean clean.bin \
  --noise z|zz \
  [--measurement heterodyne|homodyne|local-x] \
  --p P \
  --sample-start I \
  --samples N \
  [--disorder-update UPDATE] \
  --disorder-therm-sweeps N \
  --measurements N \
  --skip-sweeps N \
  (--out aggregates.bin | --out-dir DIR) \
  [--threads N]
```

Arguments:

- `--clean`: a `DCFT_CLEAN_V2` file produced by `generate-clean`.
- `--sample-start`: first clean sample id to process, starting from 0.
- `--measurement`: environment measurement used to generate the posterior
  boundary disorder; defaults to `heterodyne`. Heterodyne and homodyne draw
  Gaussian disorder with mean and variance `mu`, where
  `mu=-ln(1-2p)` and `mu=-2ln(1-2p)`, respectively. `local-x` draws binary
  disorder `K z`, with `tanh(K)=2sqrt(p(1-p))` and
  `Pr(z|phi)=(1+2sqrt(p(1-p)) z phi)/2`.
- `--samples`: number of consecutive clean samples to process; must be at
  least 1 and the requested range must fit in the clean file.
- `--disorder-update`: fixed-disorder sampler; defaults to `metropolis`.
- `--disorder-therm-sweeps`: update sweeps before measurements for each
  disorder realization.
- `--measurements`: measurements accumulated into each aggregate record; must
  be at least 1.
- `--skip-sweeps`: fixed-disorder update sweeps before each measurement.
- `--out-dir`: generates a filename containing the clean-file stem, noise,
  environment measurement, `p`, update, sample range, and measurement count.

New aggregate files use `DCFT_AGG_V3` and store the environment-measurement
kind in the formerly reserved header word. The reader remains compatible with
`DCFT_AGG_V2`, which is interpreted as heterodyne.

Supported fixed-disorder updates:

- `metropolis`: random single-site Metropolis proposals.
- `metropolis-global`: the same random single-site Metropolis sweep followed
  with probability one half by a detailed-balance global-spin-flip Metropolis
  proposal.  This helps the
  weak-random-field chains move between the two critical magnetization sectors.
- `sequential-metropolis`: one row-major single-site Metropolis scan per
  sweep.
- `corrected-wolff`: clean-model Wolff clusters accepted with the
  boundary-disorder Metropolis correction.

For corrected Wolff, one sweep means exactly one cluster proposal followed by
its accept/reject decision; rejected proposals count. For random Metropolis,
one sweep means `Lx * Lt` attempted flips, while sequential Metropolis means
one row-major pass. Sweep counts therefore have update-specific meanings.

The aliases `sequential_metropolis`, `seq-metropolis`, and `seq_metropolis`
are accepted for `sequential-metropolis`; `corrected_wolff` and `cw` are
accepted for `corrected-wolff`.

### `inspect-clean`

Print the lattice, coupling, seed, sample-count, and sweep metadata from a
clean sample file.

```sh
decohered-cft inspect-clean --clean clean.bin
```

### `inspect-aggregate`

Validate an aggregate file's header and payload size, then print its metadata,
record count, and record length.

```sh
decohered-cft inspect-aggregate --path aggregates.bin
```

### `merge-aggregates`

Merge compatible aggregate chunks. Inputs can be positional, supplied with
repeated `--input PATH` flags, or a mixture of both.

```sh
decohered-cft merge-aggregates \
  --out aggregates.bin \
  chunks/*.bin
```

The command sorts chunks and records by `disorder_id`. It rejects incompatible
metadata and any duplicate, missing, or noncontiguous ids.

### `analyze`

Combine fixed-disorder aggregate records into the disorder-averaged scalar and
correlator observables defined in `notes/main.tex`.

```sh
decohered-cft analyze \
  --path aggregates.bin \
  [--csv observables.csv] \
  [--disorder-records-csv disorder_records.csv]
```

The terminal output previews correlators. `--csv` writes every available
separation together with the scalar observables at full `f64` precision.
`--disorder-records-csv` preserves one row per independent tilted-disorder
realization and separation, allowing downstream standard errors or bootstrap
intervals to be computed without treating inner-chain measurements as
independent disorder samples.

### `run-spec-dry-run`

Read `samples` and `chunks` from a pipeline TOML file and print the balanced,
contiguous sample range assigned to every array task.

```sh
decohered-cft run-spec-dry-run --path scripts/run.example.toml
```

Both values must be at least 1, and `chunks` cannot exceed `samples`.

### `benchmark-sampler`

Time clean Wolff sweeps, the selected fixed-disorder update sweeps, and serial
versus Rayon throughput for independent fixed-disorder chains.

```text
decohered-cft benchmark-sampler \
  --lx L --lt T \
  [--seed SEED] \
  --noise z|zz \
  --p P \
  --clean-therm-sweeps N \
  [--disorder-update UPDATE[,UPDATE...]] \
  --disorder-therm-sweeps N \
  [--parallel-chains N] \
  [--threads N] \
  [--delta-tau DT]
```

`--disorder-update` defaults to `metropolis`; it may be repeated, contain a
comma-separated list, or be `all`. `all` selects `metropolis`,
`sequential-metropolis`, and `corrected-wolff`. Both sweep counts must
be at least 1. `--parallel-chains` defaults to the active Rayon thread count
and must be at least 1.

Output fields include:

- `sweep_speed`: measured sweeps per second.
- `spin_update_speed`: nominal `sweeps * Lx * Lt / elapsed`; for Wolff methods
  this is a lattice-size-normalized throughput, not a count of flipped spins.
- `speedup`: parallel elapsed-time speedup over serial execution of the same
  independent chains.

Example:

```sh
decohered-cft benchmark-sampler \
  --lx 64 --lt 64 --seed 2254 \
  --noise z --p 0.25 \
  --clean-therm-sweeps 100000 \
  --disorder-update all \
  --disorder-therm-sweeps 10000 \
  --parallel-chains 12 \
  --threads 4
```

### `diagnose-autocorrelation`

Estimate post-thermal spin-configuration autocorrelation times for one clean
chain and each selected fixed-disorder update. Each chain first completes its
requested thermalization sweeps and then continues from that state to collect
probes.

```text
decohered-cft diagnose-autocorrelation \
  --lx L --lt T \
  [--seed SEED] \
  --noise z|zz \
  --p P \
  --clean-therm-sweeps N \
  [--disorder-update UPDATE[,UPDATE...]] \
  --disorder-therm-sweeps N \
  --probes N \
  --probe-interval-sweeps N \
  [--delta-tau DT]
```

The update selection has the same syntax as `benchmark-sampler`. Both thermal
sweep counts and the probe interval must be at least 1; at least 2 probes are
required.

Output fields include:

- `probe_interval`: sweep spacing between sampled configurations.
- `cutoff_lag`: first lag with non-positive normalized autocorrelation, or
  `none` if no available lag meets that condition.
- `tau_int`: integrated autocorrelation time in sweeps, summing positive lags
  up to the cutoff or all available lags when there is no cutoff.
- `rho_lags`: a preview where lag `k` represents
  `k * probe_interval` sweeps.

Example:

```sh
decohered-cft diagnose-autocorrelation \
  --lx 64 --lt 64 --seed 2254 \
  --noise z --p 0.25 \
  --clean-therm-sweeps 10000 \
  --disorder-update all \
  --disorder-therm-sweeps 10000 \
  --probes 1000 \
  --probe-interval-sweeps 2
```

### `diagnose-thermalization`

Run independent random-start clean and fixed-disorder chains and compare their
post-thermal measurement histories.

```text
decohered-cft diagnose-thermalization \
  --lx L --lt T \
  [--seed SEED] \
  --noise z|zz \
  --p P \
  --clean-therm-sweeps N \
  [--disorder-update UPDATE[,UPDATE...]] \
  --disorder-therm-sweeps N \
  --chains N \
  --measurements N \
  --skip-sweeps N \
  [--threads N] \
  [--delta-tau DT]
```

The update selection has the same syntax as `benchmark-sampler`. Both thermal
sweep counts and `--skip-sweeps` must be at least 1; at least 2 chains and 2
measurements per chain are required.

For energy, magnetization, and boundary magnetization, the command reports
`r_hat` together with the minimum, maximum, and standard deviation of chain
means. Values near 1 and a small chain-mean spread indicate agreement between
independent chains after thermalization. As rough guidance, `r_hat < 1.01` is
strict, `r_hat < 1.05` is often acceptable for this diagnostic, and
`r_hat > 1.1` is suspicious; interpret it together with the chain-mean spread
and the available chain and measurement counts.

Example:

```sh
decohered-cft diagnose-thermalization \
  --lx 64 --lt 64 --seed 2254 \
  --noise z --p 0.25 \
  --clean-therm-sweeps 10000 \
  --disorder-update all \
  --disorder-therm-sweeps 10000 \
  --chains 100 \
  --measurements 100 \
  --skip-sweeps 20
```

## Workflow

### Generate clean samples

```sh
decohered-cft generate-clean \
  --lx 64 --lt 64 \
  --seed 2254 \
  --samples 10000 \
  --clean-therm-sweeps 10000 \
  --clean-skip-sweeps 100 \
  --out clean.bin
```

This writes `DCFT_CLEAN_V2` boundary samples. Version 1 files are intentionally
rejected because they were produced with the former state-dependent Wolff
sweep boundary. Version 2 samples are used to draw
tilted disorder from the measure `Q(D)`.

### Measure aggregate records

```sh
decohered-cft measure \
  --clean clean.bin \
  --noise z \
  --p 0.25 \
  --sample-start 0 \
  --samples 100 \
  --disorder-update metropolis \
  --disorder-therm-sweeps 2000 \
  --measurements 100 \
  --skip-sweeps 20 \
  --out chunk_0.bin \
  --threads 8
```

Supported fixed-disorder updates:

- `metropolis`
- `metropolis-global`
- `sequential-metropolis`
- `corrected-wolff`

### Merge chunks

```sh
decohered-cft merge-aggregates \
  --out aggregates.bin \
  chunks/*.bin
```

Merge sorts records by `disorder_id` and rejects duplicate or missing ids.

### Analyze

```sh
decohered-cft analyze --path aggregates.bin --csv observables.csv
```

The terminal summary previews the disorder-averaged linear, fidelity, and
Edwards--Anderson correlators. `--csv` writes all separations together with the
scalar observables at full `f64` precision. The Slurm pipeline always produces
`analysis/observables.csv` so no long-distance data is lost from its summary.

## Aggregate Format

Aggregate files use magic bytes `DCFT_AGG_V2`. Version 1 data are intentionally
unsupported because their layout and removed update method are obsolete.

Each record corresponds to one disorder realization and stores:

- `disorder_id`
- `<H_cl^D>_D`
- `<sum_{x,n} s_{x,n}>_D`
- boundary profiles `<s_{x,0}>_D` and `<s_{x,0}s_{x+1,0}>_D`
- all unique periodic separations `r = 0..floor(Lx/2)`
- signed translation-averaged spin and bond correlators
- absolute-value counterparts, computed as translation averages of absolute
  fixed-disorder thermal means

`analyze` combines these fixed-disorder records into the linear, fidelity, and
Edwards--Anderson spin and bond diagnostics defined in `notes/main.tex`.

The record order in a chunk is deterministic on disk because records are sorted
by `disorder_id` before writing.

## Slurm

The Slurm pipeline operates on a campaign containing one or more noise types
and decoherence probabilities. Copy and edit the example:

```sh
cp scripts/run.example.toml scripts/run.my-run.toml
```

Dry-run the chunk plan:

```sh
scripts/slurm-submit-pipeline.sh scripts/run.my-run.toml --dry-run
```

Submit the full pipeline:

```sh
scripts/slurm-submit-pipeline.sh scripts/run.my-run.toml
```

The campaign spec uses whitespace-separated lists:

```toml
output_root = "/path/to/data"
bin = "/path/to/decohered-cft/monte-carlo-simulation/target/release/decohered-cft"
lx = 8
lt = 8
noises = "z zz"
p_values = "0.00 0.05 0.10 0.25 0.49"
```

No manually synchronized run name is needed. For a square lattice, the
pipeline derives the point directories from `lx`, `noise`, and `p`:

```text
/path/to/data/
  L8/
    clean/
      clean_lt8_iso_seed2254_n1000_therm10000_skip100.bin
    z/
      p000/
      p005/
      p010/
      p025/
      p049/
    zz/
      p000/
      p005/
      p010/
      p025/
      p049/
```

The `pNNN` tag is the probability multiplied by 100 and zero-padded to three
digits, so the campaign interface accepts probabilities in increments of
`0.01`. The simulation CLI itself continues to accept arbitrary valid
probabilities.

Each point directory contains:

- `config/`
- `chunks/`
- `merged/`
- `analysis/`
- `manifests/`

Lattice-level `clean/`, `config/`, and `logs/` directories are shared by the
campaign. The clean filename includes every parameter that affects clean
sample generation. Campaigns with matching clean parameters therefore reuse
the same validated clean file across all noise types and probabilities. Clean
jobs with the same key use Slurm's `singleton` dependency and write through a
temporary file, avoiding races when compatible campaigns are submitted at the
same time.

The measurement array spans every `(noise, p, chunk)` tuple, and the final
array merges and analyzes every `(noise, p)` point. The dry run prints this
complete task mapping. Chunks are preserved by default, and submission refuses
to overwrite an existing point dataset.
