# Update-method parameter guidance

This directory contains evidence-based parameter suggestions for individual
Monte Carlo update methods. These documents are starting points, not universal
defaults: update cost and mixing must be rebenchmarked when lattice geometry,
disorder, protocol, implementation, or target hardware changes materially.

Available guidance:

- [Plain random-site Metropolis](metropolis.md)
- [TNMC](tnmc.md)

Add future methods as sibling documents, for example `sequential-metropolis.md`
or `corrected-wolff.md`, and link them from this index and
`AGENT_SIMULATION_WORKFLOW.md`.
