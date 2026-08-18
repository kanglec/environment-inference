# TNMC parameter suggestions

## Status and scope

These suggestions summarize local benchmarks completed on 2026-08-13 and
2026-08-14 for the periodic isotropic critical model with

- `L_x = 8, 16, 24, 32` and `L_t = 2 L_x`;
- heterodyne records;
- Z and ZZ noise at `p = 0, 0.05, 0.25, 0.48`;
- four Rayon workers;
- TNMC bond dimensions `chi = 1, 2, 4, 8`, with a light `chi = 16` timing
  pilot.

Clean benchmarks after commit `c281b93` used the composite now named
`tnmc-global`, which includes a lazy global-spin-flip Metropolis substep.
Disordered benchmarks were collected with the pure kernel now named `tnmc`.
Their bond-dimension and cost evidence remains useful, but the two methods must
not be compared or reported under one label, and their decorrelation-gap results
must be rechecked before a new disordered production campaign.

Pre-split artifacts whose `update` field says `tnmc` may contain the old
composite kernel. They must remain under their original source digest and must
not be merged with new `tnmc` data. The maintained presets use new campaign
names and output roots so this provenance boundary is explicit.

## Recommended starting point

For the tested size range, start with:

```toml
[mc]
updates = ["tnmc-global"] # use "tnmc" for the pure reference kernel
posterior_decorrelation_gap = 1 # clean p=0 only; rebenchmark for disorder
inner_measurements = 64
inner_saving_interval = 8
inner_budget_multipliers = [1, 2, 4]
diagnostic_outer_records = 4
replicated_chains = 2
tnmc_bond_dimension = 2
```

The total updates in one posterior chain are

```text
posterior_decorrelation_gap + inner_measurements * inner_saving_interval.
```

The first saved measurement occurs after
`posterior_decorrelation_gap + inner_saving_interval` updates. For the clean
suggestion above, the base chain uses 513 TNMC sweeps and the first measurement
is nine sweeps from the planted initialization.

## Bond dimension

Use `tnmc_bond_dimension = 2` as the default pilot value through `32 x 64`.

At `32 x 64`, the disordered six-point pilot gave approximately:

| chi | Sweeps/s | TNMC acceptance | Mean effective planted-overlap samples/s | Assessment |
|---:|---:|---:|---:|---|
| 1 | 423 | 0.05--0.08 | 106, but with failed convergence | Reject |
| 2 | 203 | 0.75--0.88 | 135 | Select |
| 4 | 77 | 0.99--1.00 | 62 | Too expensive for its acceptance gain |
| 8 | 22 | 1.00 | 19 | Dominated by `chi = 2` |
| 16 | 5 | Light timing pilot only | Not estimated | Stop after timing pilot |

The `chi = 1` chain was fast but barely moved; its apparently favorable
effective-sample values could be artifacts of nearly constant traces. Select
bond dimension using effective samples per wall time subject to acceptable
split R-hat, not acceptance or sweep rate alone.

Increase `chi` only when a representative benchmark shows that the acceptance
gain compensates for the contraction cost. In the tested regime, moving from
`chi = 2` to 4 increased acceptance toward one but usually reduced useful
throughput. Do not increase `chi` solely to obtain nearly perfect acceptance.

All tested runs had zero TNMC positivity regularizations. Any nonzero
regularization count must be reported and investigated before production.

## Clean `tnmc-global` model after lazy global flips

Each `tnmc-global` sweep ends with a global flip proposed with probability one half.
The flip

- always accepts in the clean and ZZ models, where it is an exact symmetry;
- uses the exact Metropolis probability for Z disorder;
- has negligible measured cost.

For clean `tnmc-global` at `chi = 2`, the after-change size grid found:

| Lx | Sweeps/s | TNMC acceptance | tau energy | tau spin overlap | tau bond overlap |
|---:|---:|---:|---:|---:|---:|
| 8 | 5008 | 0.997 | 1.95 | 0.51 | 0.71 |
| 16 | 939 | 0.984 | 2.41 | 0.50 | 0.71 |
| 24 | 378 | 0.884 | 1.13 | 0.50 | 0.59 |
| 32 | 203 | 0.803 | 2.12 | 0.55 | 1.03 |

With one preliminary sweep and 128 subsequent measurements, the maximum split
R-hat across the relevant clean energy, spin-overlap, and bond-overlap checks
was 1.039. Before the global move, spin-overlap split R-hat at `32 x 64`
remained 1.129 even after 2048 preliminary sweeps.

For clean `p = 0`, use `posterior_decorrelation_gap = 1` with `tnmc-global`.
The planted configuration is already an exact posterior draw, so this is not a
thermalization budget. The one update executes the exact global-sector
randomization; the saving interval supplies another eight updates before the
first measurement.

## Saving interval

Use `inner_saving_interval = 8` as a conservative common setting in the tested
range. It exceeds twice the largest reliable clean energy autocorrelation time
and the reliable pre-change disordered autocorrelation times measured in the
pilot.

Thinning is not required for unbiased posterior means. Saving every sweep can
maximize effective samples per compute time when downstream uncertainty
analysis retains autocorrelation. The interval of eight is instead intended to
make a fixed finite inner budget nearly independent and easier to diagnose.

The planted configuration initializes the production estimator and is used as
the independent posterior replica in the default Nishimori estimators for
`q_ea_planted`, posterior correlators, and I--MMSE. Consequently, that inner
chain must be sufficiently separated from the planted draw even though it
starts in the correct marginal distribution. Separate overdispersed chains,
not duplicated planted starts, supply split-R-hat evidence.

## Disordered decorrelation gap

Do not copy the clean gap of one into a Z-disordered production campaign.

Before the lazy global move was added, a 64-sweep gap passed the largest-lattice
energy and relevant planted-overlap checks for

- Z noise at `p = 0.25` and `0.48`;
- ZZ noise at `p = 0.05`, `0.25`, and `0.48`.

For Z noise at `p = 0.05`, three disorder seeds failed the overdispersed
spin-overlap check after 64 sweeps, and one seed still failed after 2048. Higher
bond dimension did not repair that failure. The new exact-Metropolis global
substep may change these results, especially where the global-flip acceptance
is appreciable, so the disordered matrix must be rerun under `c281b93` before
promoting a gap.

For ZZ noise, raw magnetization is odd under an exact global symmetry and is
not a valid convergence requirement. Use energy and planted bond overlap for
the primary check.

## Benchmark protocol

For a materially new regime, begin with a light largest-lattice timing pilot,
then increase the statistical budget only for feasible bond dimensions. A
representative full check is:

```bash
dcft benchmark updates \
  --config REQUEST_CONFIG \
  --update tnmc-global \
  --workers 4 \
  --speed-sweeps 256 \
  --probes 512 \
  --probe-interval 1 \
  --thermalization-sweeps CANDIDATE_GAP \
  --thermalization-measurements 128 \
  --chains 8 \
  --output OUTPUT.json
```

Repeat multiple disorder seeds or records at the largest and most difficult
points. The current benchmark command uses one disorder record per invocation,
so one report is not enough to establish a production gap.

Require all of the following before promotion:

- acceptable energy and physically relevant overlap split R-hat;
- stable integrated autocorrelation estimates from traces long relative to
  their autocorrelation window;
- high effective samples per wall time relative to other `chi` choices;
- adequate TNMC acceptance and, for `tnmc-global`, global-flip acceptance;
- zero, or explicitly explained, positivity regularizations;
- stable conclusions over representative disorder records;
- timing on the actual production machine.

## Configuration hygiene

Do not place one-off benchmark configurations in the maintained `configs/`
directory. Use an ignored campaign artifact/scratch directory or a temporary
path, and preserve the exact request configuration alongside the benchmark
JSON. Add a file to `configs/` only when it is intended to become a maintained
reusable preset.

## Limitations

- Timings are specific to the benchmark machine and four-worker shape.
- The maximum tested lattice was `32 x 64`.
- Only heterodyne records were benchmarked.
- The clean `tnmc-global` result is validated; disordered `tnmc` gaps are
  pre-composite evidence and must be retested for whichever named kernel will
  be used.
- The benchmark's overdispersed split R-hat is stricter than ordinary burn-in
  from the exact planted posterior draw, but it remains useful for detecting
  hidden sector failures.
