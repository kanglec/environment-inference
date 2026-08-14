# Agent workflow for DCFT simulations

## Purpose

This is the operating guide for an AI agent that develops, benchmarks, runs,
or interprets this simulation. The user normally supplies a scientific
question and, when relevant, an accuracy or time target. Translate that request
into an inspectable configuration, use the normal program interfaces, and
report the scientific result with its diagnostics.

The program is not a deployment system. Use ordinary Git, SSH, Slurm, and
rsync where needed. Do not add wrappers around short, clear operations.

## Sources of truth

- Physics and TNMC reference:
  `../local/tensor-network-monte-carlo/arXiv-2409.06538v2/main.tex`.
- Simulation contract:
  `../notes/archive/all-to-all-approximation/section/simulation.tex`.
- Maintained implementation: this repository outside `archive/`.
- Historic implementation: `archive/`, read-only reference for ideas and
  checks, never a production pipeline.

Preserve the scientific invariants: periodic lattice conventions, sweep
semantics, planted construction, deterministic global-ID streams, exact
Hastings corrections, paired outer IDs, and the documented estimators. Parallel
shape must never enter an RNG key.

## Start from the scientific request

Choose the nearest maintained configuration:

- `configs/comparison.toml` for MC/ED agreement or small-size physics;
- `configs/scaling.toml` for large MC-only work;
- `configs/local-smoke.toml` for development verification;
- `configs/update-benchmark.toml` for update selection and resource tuning.

Create a request-specific TOML file. Record every choice that affects the
answer:

- lattice sizes, temporal factor, and regularization;
- Z or ZZ noise, probabilities, measurements, and Gaussian grid;
- update method and TNMC bond dimension;
- clean, outer-record, inner-chain, replicated-chain, and bootstrap budgets;
- separations and exact-diagonalization range.

Use a new readable campaign name and output root for a new scientific request.
Ask the user only when plausible choices answer materially different questions.
Otherwise run a conservative pilot and state the assumptions.

## Benchmark before committing production resources

Update costs and mixing depend strongly on size and disorder. For a new regime,
run:

```bash
dcft benchmark updates \
  --config CONFIG \
  --workers N \
  --speed-sweeps 256 \
  --probes 512 \
  --thermalization-sweeps 128 \
  --thermalization-measurements 128 \
  --chains 8 \
  --output PATH.json
```

Repeat representative points if the request spans very different sizes or
noise strengths. Compare:

- sweep rate;
- planted-overlap and energy autocorrelation in sweeps;
- effective samples per wall second;
- split R-hat after the candidate thermalization budget;
- serial-versus-Rayon independent-chain speedup;
- acceptance and TNMC regularization counts.

Select the method by effective independent samples per wall time subject to
acceptable convergence and numerical diagnostics. A faster sweep is not
necessarily a faster simulation.

## Shape parallel work

There are two independent levels of parallelism:

1. `[execution].mc_chunks` divides every parameter point into balanced,
   deterministic outer-ID ranges. On Slurm these become array elements.
2. Rayon processes independent posterior chains inside a chunk. Local runs use
   `[execution].local_workers` unless `--workers` overrides it. Cluster runs use
   `[cluster].cpus_per_task`.

Choose enough chunks to expose scheduler parallelism and enough outer records
per chunk to keep the Rayon workers busy. Very small chunks waste startup and
artifact overhead. A reasonable pilot begins with several records per worker,
then uses measured elapsed time and memory to tune the shape.

Fix the chunk count before planning a production campaign. Changing it changes
task boundaries and requires a new output root; worker count, memory, and time
can be adjusted without replanning.

The program imposes neither an array-concurrency cap nor an aggregate CPU
ceiling. Respect limits explicitly supplied by the user or scheduler/account
policy; do not invent a smaller one.

Every point has a merge task. It verifies complete, non-overlapping production
IDs before analysis, so never bypass the merge stage by manually concatenating
files.

## Local development and demonstration

Prepare and verify once:

```bash
uv sync --frozen
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
uv run ruff check python tests
uv run mypy python/dcft
uv run pytest
```

Run a campaign locally:

```bash
dcft campaign plan --config CONFIG
dcft campaign run --config CONFIG
dcft campaign analyze --config CONFIG
dcft campaign validate --config CONFIG
dcft inspect OUTPUT_ROOT
```

Use `--workers N` for a one-off worker override. Use `--task-id ID` only for
debugging or explicit partial execution; dependencies must already be complete.

The local machine must be able to exercise planning, clean sampling, all MC
updates, chunk merging, ED when enabled, analysis, validation, plotting,
benchmarking, and cluster rendering. It need not carry production statistical
budgets.

## Cluster workflow

The cluster adapter assumes the checkout and its `.venv` already exist. It
does not use environment modules, copy source, build per campaign, or seal a
runtime.

On the cluster checkout, prepare once after code or lock changes:

```bash
cd /path/to/simulation
git pull --ff-only
uv sync --frozen
```

Then, using a config whose `project_root`, `output_root`, and `scratch_root`
resolve correctly on the cluster:

```bash
dcft cluster doctor --config CONFIG
dcft campaign plan --config CONFIG
dcft cluster render --config CONFIG
dcft cluster submit --config CONFIG
```

Inspect `dag.json` and the generated `*.sbatch` files for a materially new
shape. Generated jobs call `<project_root>/.venv/bin/dcft`, set
`RAYON_NUM_THREADS` from `SLURM_CPUS_PER_TASK`, and use `[cluster].memory` as
memory per task.

Do not edit or rebuild the checkout while its jobs run. Artifacts record their
source digest, and downstream stages reject mixed-source dependencies. This
check is the only source-consistency mechanism; no immutable release directory
is required.

## Monitor and recover

Track every submitted stage, not only the final job:

```bash
squeue --jobs JOB_IDS
sacct -j JOB_IDS --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocCPUS
```

The expected order is clean and exact, MC chunks, merge, analysis, validation.
When something fails, read its Slurm log and campaign `state.json` before
resubmitting. Classify it:

- transient scheduler/node failure: `dcft cluster resume --config CONFIG`;
- time or memory failure: tune only `[cluster]` and resume;
- poor throughput: tune workers from benchmark evidence; use a new output root
  if changing chunks or the scientific update method;
- code/runtime failure: fix and rebuild, then rerun stale upstream tasks;
- numerical or validation failure: preserve the evidence and do not promote
  the result.

Completion means all planned tasks are complete, merged ID ranges are valid,
artifacts verify, analysis completes, validation succeeds, and a final resume
reports `nothing-to-resume`.

## Retrieve and analyze

Copy the output root with ordinary resumable rsync when computation is remote:

```bash
rsync -aP --partial HOST:REMOTE_OUTPUT/ LOCAL_OUTPUT/
```

Run `dcft inspect LOCAL_OUTPUT` after transfer. Begin interpretation with:

- the validation report;
- update acceptance, autocorrelation, replicated-chain, finite-inner-budget,
  and TNMC-regularization diagnostics;
- the scalar, curve, I--MMSE, or MC/ED artifact relevant to the question.

For extra analysis, read the Parquet artifacts with PyArrow and keep paired
outer IDs and complete curves together during resampling. Do not replace
moving-block or simultaneous intervals with independent pointwise errors. Mark
exploratory finite-size fits as exploratory.

## Report to the user

Lead with the scientific answer. Include only provenance needed to assess it:

- campaign name and source digest or Git revision;
- essential physical parameters and budgets;
- update method, TNMC bond dimension, and relevant benchmark evidence;
- validation, autocorrelation, R-hat, acceptance, and regularization status;
- estimate, uncertainty, and material systematic errors;
- links to useful local plots, benchmark JSON, or artifacts;
- limitations that could change the conclusion.

Never claim a result from a pending, failed, stale, unmerged, or unvalidated
campaign. Never hide a sampled fallback, poor mixing, finite-inner bias,
regularized TNMC conditional, or inadequate finite-size range.
