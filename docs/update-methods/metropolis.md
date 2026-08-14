# Plain Metropolis parameter suggestions

## Status and scope

These suggestions summarize local benchmarks completed on 2026-08-14 for the
plain random-site `metropolis` updater on the periodic isotropic critical model
with

- `L_x = 8, 16, 24, 32` and `L_t = 2 L_x`;
- heterodyne records;
- Z and ZZ noise at `p = 0, 0.05, 0.25, 0.48`;
- four Rayon workers;
- 131,072-sweep autocorrelation traces at every point, with a 524,288-sweep
  clean-spin trace at `32 x 64`;
- eight overdispersed chains in the initial grid and four chains in the
  thinned candidate-gap checks;
- three independent records at representative easy and difficult
  largest-lattice points.

The recommendations apply through `32 x 64` on the benchmarked implementation
and are intentionally conservative. Disorder-record variation is material,
especially for Z noise at `p = 0.05` and ZZ noise at `p = 0.25`.

## Recommended starting points

Use 64 saved samples for the base chain, retain the diagnostic multipliers and
replicated chains, and choose the gap and interval by regime:

```toml
[mc]
updates = ["metropolis"]
inner_measurements = 64
inner_budget_multipliers = [1, 2, 4]
diagnostic_outer_records = 4
replicated_chains = 2
```

| Regime | `posterior_decorrelation_gap` | `inner_saving_interval` | Base sweeps | First save | Status |
|---|---:|---:|---:|---:|---|
| clean Z / spin-odd observables | 32768 | 16384 | 1081344 | 49152 | Validated, but TNMC is strongly preferred |
| clean ZZ / gauge-even observables | 1024 | 512 | 33792 | 1536 | Validated for energy and planted bond overlap |
| Z, `p = 0.05` | 16384 | 8192 | 540672 | 24576 | Validated over three records; expensive |
| Z, `p = 0.25, 0.48` | 4096 | 1024 | 69632 | 5120 | Validated at `32 x 64`; three records at `p = 0.48` |
| ZZ, `p = 0.05, 0.25, 0.48` | 4096 | 1024 | 69632 | 5120 | Validated; three records at `p = 0.25` |

The base sweep count is

```text
posterior_decorrelation_gap + 64 * inner_saving_interval.
```

The 2x and 4x diagnostic chains replace 64 by 128 and 256. For example, the
clean-Z totals are 1,081,344, 2,129,920, and 4,227,072 sweeps; the Z `p = 0.05`
totals are 540,672, 1,064,960, and 2,113,536; and the common disordered totals
are 69,632, 135,168, and 266,240.

The implementation takes the first saved measurement only after both the gap
and one saving interval. The `First save` column is therefore the sum of those
two values, not the gap alone.

## Initialization and burn-in

The planted configuration generated with a record is an exact posterior draw.
Consequently, ordinary posterior burn-in is zero when the production chain
starts from that configuration. This removes initialization bias but does not
make later states independent of the planted replica or of one another.

Use `posterior_decorrelation_gap` as a replica-decorrelation budget, not as a
claim that the planted start was out of equilibrium. The saving interval then
controls within-chain correlation. If a chain is initialized from an arbitrary
state instead, the zero-burn-in statement no longer applies; use overdispersed
diagnostics and at least the candidate gap before promotion.

## Largest-lattice mixing and throughput

At `32 x 64`, a sweep makes 2,048 uniformly random single-spin proposals. The
measured sweep rate was approximately 37,600--39,100 sweeps/s in the main grid,
or about 80 million proposals/s. Local acceptance stayed near 0.18--0.20 and
did not predict mixing.

| Noise | p | Records | Tau energy (sweeps) | Tau planted overlap (sweeps) | Effective overlap samples/s |
|---|---:|---:|---:|---:|---:|
| Z | 0 | 3 | 152--199 | 4428--6777 | 2.8--4.2 |
| Z | 0.05 | 3 | 133--253 | 1423--2662 | 7.0--13.4 |
| Z | 0.25 | 1 | 66.6 | 3.86 | 5052 |
| Z | 0.48 | 3 | 63--368 | 1.29--2.07 | 9212--15040 |
| ZZ | 0 | 1 | 158 | 10.9 | 1791 |
| ZZ | 0.05 | 1 | 165 | 15.9 | 1229 |
| ZZ | 0.25 | 3 | 175--298 | 6.52--54.6 | 345--2943 |
| ZZ | 0.48 | 1 | 176 | 0.95 | 20327 |

The effective-sample rate is the unthinned diagnostic value
`sweeps_per_second / (2 tau_overlap)`. The recommended saving intervals are
larger than twice the worst observed relevant autocorrelation time in each
regime, including energy, and trade raw estimator throughput for a simple,
nearly independent fixed inner budget. Saving every sweep is valid and can be
more efficient when downstream analysis explicitly retains autocorrelation.

## Clean critical slowing and spin sectors

The clean zero-field Hamiltonian is the same for the Z and ZZ labels, but their
reported overlap observables differ. Z uses planted spin overlap, which is odd
under the global spin symmetry. ZZ uses planted bond overlap, which is even.

For `L_x = 8, 16, 24, 32`, the default-record clean traces found energy
autocorrelation times of approximately 14, 50, 140, and 158 sweeps and spin
overlap times of 241, 817, 3818, and 5349 sweeps. The extended and independent
`32 x 64` traces put the spin value at 4428--6777 sweeps. The exploratory
four-size log-log slopes are about `L^1.83` for energy and `L^2.35` for spin
overlap; do not treat four sizes as a precision dynamic-exponent fit.

The clean planted bond overlap at `32 x 64` had tau about 10.9 sweeps. Raw
magnetization R-hat remains poor for ZZ because it is gauge odd and is not a
convergence requirement. Use energy and planted bond overlap for ZZ.

## Candidate-gap diagnostics

The recommended settings were tested with four overdispersed chains initialized
all-plus, all-minus, and randomly. The thinned checks used 64 measurements per
chain, except that the two additional Z `p = 0.05` records used 32 because of
their high cost.

| Regime | Records | Energy split R-hat | Relevant-overlap split R-hat |
|---|---:|---:|---:|
| clean Z | 1 | 0.995 | 1.001 |
| clean ZZ | 1 | 1.006 | 1.000 |
| Z, `p = 0.05` | 3 | 0.980--0.991 | 0.989--1.011 |
| Z, `p = 0.25` | 1 | 1.026 | 1.005 |
| Z, `p = 0.48` | 3 | 1.009--1.022 | 0.993--1.008 |
| ZZ, `p = 0.05` | 1 | 0.991 | 1.019 |
| ZZ, `p = 0.25` | 3 | 0.993--0.998 | 1.005--1.030 |
| ZZ, `p = 0.48` | 1 | 1.023 | 1.000 |

These checks support the proposed starting values, not a universal guarantee.
Keep replicated-chain and finite-inner-budget diagnostics enabled in production.

## Size and worker scaling

Sweep rate scaled approximately as `L_x^-1.97`, consistent with the `2 L_x^2`
single-spin proposals per sweep. The default long traces measured about 590k,
150k, 66.8k, and 39.0k clean sweeps/s at `L_x = 8, 16, 24, 32`.

A single Metropolis chain is serial. Four workers parallelize independent
chains. The candidate-gap checks measured 2.84x--3.79x speedup for four chains
relative to the benchmark's serial execution of the same deterministic chains.

## Comparison with TNMC

The documented `chi = 2` TNMC recommendation uses interval 8, a clean gap of
1, and provisional disordered gap 64. At `32 x 64`, TNMC runs only about 203
sweeps/s, versus about 39,000 for Metropolis, but its clean spin-overlap tau is
about 0.55 sweep rather than thousands. Its clean spin effective-sample rate is
therefore about 185/s, roughly 45--65 times the Metropolis rate. Prefer TNMC for
clean Z and spin-sector-sensitive work.

For the older disordered TNMC results, planted-overlap effective-sample rates
were about 68--204/s. Metropolis is slower for Z `p = 0.05` at 7--13/s, but
faster at the tested Z `p = 0.25, 0.48` and ZZ points, often by large factors.
The older TNMC weak-Z overdispersed check failed even after long gaps, whereas
the present Metropolis 16384/8192 setting passed three records. TNMC's
disordered gap results predate its lazy global-flip substep and must be rerun
before making a current head-to-head convergence claim.

Acceptance is not a method-selection criterion by itself. Select by effective
samples per wall time subject to relevant R-hat and numerical diagnostics.

## Benchmark protocol and artifacts

Begin with a light largest-lattice timing pilot. A long-trace check is:

```bash
dcft benchmark updates \
  --config REQUEST_CONFIG \
  --lx 32 --noise z --p 0.05 --measurement heterodyne \
  --update metropolis --workers 4 \
  --warmup-sweeps 128 --speed-sweeps 1024 \
  --probes 131072 --probe-interval 1 \
  --thermalization-sweeps 1 --thermalization-measurements 4 \
  --chains 2 --output OUTPUT.json
```

Run the candidate gap separately with its recommended saving interval so
autocorrelation estimation and thinned R-hat are not conflated. The evidence
and exact configs for this benchmark are under the ignored
`artifacts/metropolis-production-benchmark/` directory.

## Limitations

- Timings are specific to an Apple M4 Mac mini with 16 GB RAM and four workers.
- The maximum tested lattice was `32 x 64`, with heterodyne records only.
- Three records were used only at selected largest-lattice points; other
  disordered points have one record and should retain production diagnostics.
- The Z `p = 0.05` extra-record R-hat checks used 32 saved values per chain and
  are less precise than the 64-value checks.
- The clean dynamic scaling slopes are exploratory four-size summaries.
- No source-code defect was encountered or changed during this benchmark.
