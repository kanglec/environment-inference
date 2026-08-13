# Unified DCFT simulation

This directory is the maintained numerical implementation for
`notes/main.tex` and `notes/simulation.tex`. It is one maturin project, one
locked `uv` environment, one Rust library (`dcft._core`), and one `dcft` CLI.
Everything under `archive/` is reference-only; new artifacts intentionally
have no archive-format compatibility reader.

The Git repository is rooted at this directory. The numerical notes remain
local reference material and are not part of source deployment; artifact
source digests cover only maintained files inside this installable project.
Bouchet deployments clone an exact public Git commit into durable project
storage, while generated runtimes and task state remain in scratch.

The implementation is **cluster-qualified** on Yale Bouchet. Qualification
completed on 2026-08-13 with `dcft cluster doctor`, a frozen install from the
committed locks, Linux Rust tests, and the CPU-only `day` smoke campaign in
`configs/bouchet-smoke.toml`. The qualification root job was `22146897`; see
`BOUCHET_QUALIFICATION.md` for the reproducible evidence record.

## Install and verify

```bash
uv sync --frozen --all-extras
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
uv run ruff check python tests
uv run mypy python/dcft
uv run pytest
```

`rust-toolchain.toml`, `Cargo.lock`, `.python-version`, and `uv.lock` define
the development toolchain (Rust 1.96.1 and Python 3.12). Same-toolchain/platform runs are byte-reproducible
after canonical row sorting; other architectures are required to pass the
numerical and statistical equivalence tests instead.

The bounded scientific acceptance sequence is:

```bash
dcft campaign plan --config configs/acceptance.toml
dcft campaign run --executor local --config configs/acceptance.toml
dcft campaign analyze --config configs/acceptance.toml
dcft campaign validate --config configs/acceptance.toml
dcft campaign run --executor local --config configs/mc-only-smoke.toml
dcft campaign analyze --config configs/mc-only-smoke.toml
```

The final two commands are the MC-only beyond-ED smoke path. Its deliberately
small record count is not a promotion-grade statistical validation campaign.

## Commands

```text
dcft campaign plan [--config CONFIG]
dcft campaign run --executor local [--config CONFIG]
dcft campaign validate [--config CONFIG]
dcft campaign analyze [--config CONFIG]
dcft cluster render [--config CONFIG]
dcft cluster doctor [--config CONFIG]
dcft cluster submit [--config CONFIG]
dcft cluster status JOB_ID
dcft cluster resume [--config CONFIG]
dcft inspect PATH
```

The maintained presets are `configs/comparison.toml` and
`configs/scaling.toml`; `configs/local-smoke.toml` is a fast development
campaign, `configs/acceptance.toml` is the bounded `L=4,6,8` local acceptance
matrix, and `configs/mc-only-smoke.toml` exercises `L=10` beyond its configured
ED limit. `comparison` uses `L=4,6,8`, all requested protocols/noises and all
three ED priors, and includes TNMC alongside the earlier update kernels.
`scaling` is MC-only at large sizes, uses TNMC, and has a configurable
separation grid and an explicit Gaussian grid for I--MMSE integration.

## Numerical architecture

The Rust library contains:

- `physics`: Trotter/isotropic couplings, Z/ZZ variables, the heterodyne,
  homodyne, arbitrary `gaussian(gamma)`, and local-X protocols, record
  generation, and observable eigenvalues;
- `rng`: xoshiro256++ with SplitMix64 expansion and the versioned
  `dcft-stream-v1` key contract;
- `mc`: periodic lattices, clean Wolff generation, random/sequential/lazy
  global Metropolis, corrected Wolff, and periodic TNMC. TNMC freezes a random
  row and column, contracts the remaining open rectangle with a truncated
  boundary MPS, and applies the exact Hastings correction;
- `observables`: scalar, boundary-profile, and configured-separation
  accumulators.

Python provides dense TFIM ED, finite-transfer and transfer-ground priors,
generic Z/ZZ density matrices, exact posterior sums, Parquet artifacts,
Geyer autocorrelations, moving-block resampling, I--MMSE integration,
campaign execution, plotting, validation, and Slurm rendering.

For Slurm campaigns, an explicit `environment` stage prepares exactly one
campaign-shared runtime under scratch, keyed by the full source digest. It
installs from `uv.lock` and `Cargo.lock`, verifies the shared Rust registry,
and then makes the runtime read-only. Compute, analysis, and validation jobs
depend on that stage and execute its `dcft` directly: they never run `uv
sync`, Cargo, or a package install. Each task retains only lightweight scratch
and cache directories. A durable-source digest check at task start prevents a
submitted campaign from mixing a prepared runtime with subsequently edited
source.

Every fixed-record chain starts from the saved **full** planted Euclidean
configuration, which is an exact posterior draw. There is no ordinary
production burn-in. A positive measured decorrelation gap is still mandatory
before planted-replica estimators are retained. Gaussian strengths share the
same planted state and normal vector; local-X uses a separately labeled
stream. Random keys never depend on thread, chunk, process, or Slurm order.

The TNMC cutoff is the positive `[mc].tnmc_bond_dimension` value. One TNMC
sweep is one random-cut block proposal, including a rejection. Artifacts store
the block acceptance counts, proposed active-site count, and the number of
conditional probabilities that required the defensive positivity
regularization. A nonzero regularization count is a numerical warning, not an
unreported change of target distribution, because generation and reverse
scoring use the identical regularized proposal.

## Artifacts

Scientific outputs are immutable, content-addressed, Hive-partitioned
Parquet datasets with canonical JSON manifests (`DCFT_PARQUET_V1`). Manifests
record the complete parameter point, source digest, package/core versions,
RNG contract, parent artifacts, global-ID range, Arrow schema, configuration
hash digest, and SHA-256 checksums. Clean datasets retain bit-packed full
Euclidean configurations. MC datasets retain one production row per outer ID
plus explicitly keyed replicated `1x/2x/4x` diagnostic rows. Raw Gaussian ED
records are retained for paired analysis.

Operational `plan.json` and `state.json` files make tasks resumable; they are
not scientific data. Validation failures remain in an immutable validation
report and block promotion. Generated plots are checksummed and marked as
regenerable. Campaign inspection reports only artifacts referenced by the
current resumable state and counts older immutable attempts separately.

## Statistical contract

- Autocorrelations use Geyer's initial-positive monotone sequence.
- Moving-block bootstrap resamples contiguous outer IDs and complete curves.
- Gaussian protocols are compared at paired outer IDs.
- Posterior overlaps default to planted estimators; squared inner means are
  diagnostics only.
- I--MMSE uses a common resample at every gamma and reports the
  Simpson-minus-trapezoid quadrature difference separately.
- MC--exact curves use simultaneous max-t bootstrap bands.
- The 1x/2x/4x ladder contributes a contiguous-outer-ID bootstrap upper
  envelope for finite-inner absolute-value bias.
- Local-X outcomes are enumerated through `N_noise <= 14`; larger runs are
  labeled `sampled-binary-fallback`.
