#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mc_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${mc_root}/.." && pwd)"
binary="${mc_root}/target/release/decohered-cft"
result_root="${repo_root}/results/monte-carlo"

seed=2254
samples=500
clean_therm_sweeps=10000
clean_skip_sweeps=100
disorder_therm_sweeps=20000
measurements=1000
skip_sweeps=100
threads=8
delta_tau=0.2
p_values=(0.0 0.1 0.2 0.3 0.4 0.49)
p_tags=(p000 p010 p020 p030 p040 p049)

if [[ ! -x "${binary}" ]]; then
  echo "release binary not found: ${binary}" >&2
  exit 1
fi

for lx in 4 6 8; do
  lt=$((16 * lx))
  size_dir="${result_root}/L${lx}"
  clean="${size_dir}/clean.bin"
  mkdir -p "${size_dir}"

  if [[ ! -f "${clean}" ]]; then
    "${binary}" generate-clean \
      --lx "${lx}" \
      --lt "${lt}" \
      --seed "${seed}" \
      --samples "${samples}" \
      --clean-therm-sweeps "${clean_therm_sweeps}" \
      --clean-skip-sweeps "${clean_skip_sweeps}" \
      --delta-tau "${delta_tau}" \
      --out "${clean}"
  fi

  for index in "${!p_values[@]}"; do
    p="${p_values[$index]}"
    tag="${p_tags[$index]}"
    point_dir="${size_dir}/${tag}"
    aggregate="${point_dir}/aggregate.bin"
    mkdir -p "${point_dir}"
    if [[ -e "${aggregate}" ]]; then
      echo "refusing to overwrite existing production output: ${aggregate}" >&2
      exit 1
    fi

    "${binary}" measure \
      --clean "${clean}" \
      --noise z \
      --p "${p}" \
      --sample-start 0 \
      --samples "${samples}" \
      --disorder-update metropolis-global \
      --disorder-therm-sweeps "${disorder_therm_sweeps}" \
      --measurements "${measurements}" \
      --skip-sweeps "${skip_sweeps}" \
      --threads "${threads}" \
      --out "${aggregate}"

    "${binary}" analyze \
      --path "${aggregate}" \
      --csv "${point_dir}/observables.csv" \
      --disorder-records-csv "${point_dir}/disorder_records.csv"
  done
done
