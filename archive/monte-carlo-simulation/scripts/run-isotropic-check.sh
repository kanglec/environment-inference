#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mc_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${mc_root}/.." && pwd)"
binary="${mc_root}/target/release/decohered-cft"
result_root="${repo_root}/results/isotropic-check/monte-carlo-square"

seed=2254
samples=500
p_values=(0.0 0.1 0.2 0.3 0.4 0.49)
p_tags=(p000 p010 p020 p030 p040 p049)

for lx in 4 6 8; do
  size_dir="${result_root}/L${lx}"
  clean="${size_dir}/clean.bin"
  mkdir -p "${size_dir}"
  if [[ ! -f "${clean}" ]]; then
    "${binary}" generate-clean \
      --lx "${lx}" --lt "${lx}" --seed "${seed}" --samples "${samples}" \
      --clean-therm-sweeps 10000 --clean-skip-sweeps 100 --out "${clean}"
  fi
  for index in "${!p_values[@]}"; do
    p="${p_values[$index]}"
    tag="${p_tags[$index]}"
    point_dir="${size_dir}/${tag}"
    aggregate="${point_dir}/aggregate.bin"
    mkdir -p "${point_dir}"
    if [[ -e "${aggregate}" ]]; then
      echo "refusing to overwrite existing isotropic output: ${aggregate}" >&2
      exit 1
    fi
    "${binary}" measure \
      --clean "${clean}" --noise z --p "${p}" \
      --sample-start 0 --samples "${samples}" \
      --disorder-update metropolis-global \
      --disorder-therm-sweeps 20000 --measurements 1000 --skip-sweeps 100 \
      --threads 8 --out "${aggregate}"
    "${binary}" analyze \
      --path "${aggregate}" \
      --csv "${point_dir}/observables.csv" \
      --disorder-records-csv "${point_dir}/disorder_records.csv"
  done
done
