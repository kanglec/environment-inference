# Bouchet qualification record

The unified simulation was cluster-qualified on Yale Bouchet on 2026-08-13.
This was an infrastructure and small-numerics smoke test, not a production
campaign.

- Account/partition: `pi_mc2832` / `day`
- Resources per job: one node, one task, one CPU, 5120 MB per CPU, one hour
- User tools: `uv 0.12.3`; no Python or Rust environment modules
- Locked runtime: CPython 3.12.13 and Rust/Cargo 1.96.1
- Linux bootstrap/test job: `22145751` (19 Rust unit/property tests passed)
- Final clean job: `22146894`
- Final TNMC job: `22146895`
- Final analysis job: `22146896`
- Final validation/root job: `22146897`
- Validation: 16 checks, 0 failures
- Artifacts: seven current, checksum-valid, source-current artifacts
- TNMC production: 96 proposals, 96 acceptances, 0 regularized conditionals
- Resumability: `dcft cluster resume` returned `nothing-to-resume`

After qualification, the renderer was corrected to use one digest-keyed,
read-only environment prepared by an explicit Slurm stage. Production tasks
do not mutate or recreate the virtual environment; their only writable state
is task scratch and caches.

The verified source digest before this status-only promotion was
`cc80e2523843db3e9e3161f9958dc585cb688b0e7d6c21b467729e936ce459fe`.
The immutable smoke artifacts retain their contemporaneous
`cluster-unqualified` label because qualification was granted only after their
successful validation; future artifacts use `cluster-qualified`.
