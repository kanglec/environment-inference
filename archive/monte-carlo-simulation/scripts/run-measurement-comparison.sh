#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mc_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${mc_root}/.." && pwd)"
binary="${mc_root}/target/release/decohered-cft"
clean_root="${repo_root}/results/monte-carlo"
result_root="${repo_root}/results/measurement-comparison/monte-carlo"

samples=500
disorder_therm_sweeps=20000
measurements=500
skip_sweeps=50
threads=8
p_values=(0.0 0.1 0.2 0.3 0.4 0.49)
p_tags=(p000 p010 p020 p030 p040 p049)

if [[ ! -x "${binary}" ]]; then
  echo "release binary not found: ${binary}" >&2
  exit 1
fi

for measurement in homodyne local-x; do
  measurement_dir="${result_root}/${measurement}"
  for lx in 4 6 8; do
    clean="${clean_root}/L${lx}/clean.bin"
    if [[ ! -f "${clean}" ]]; then
      echo "missing matched clean sample file: ${clean}" >&2
      exit 1
    fi
    for index in "${!p_values[@]}"; do
      p="${p_values[$index]}"
      tag="${p_tags[$index]}"
      point_dir="${measurement_dir}/L${lx}/${tag}"
      aggregate="${point_dir}/aggregate.bin"
      mkdir -p "${point_dir}"
      if [[ -e "${aggregate}" ]]; then
        echo "refusing to overwrite existing output: ${aggregate}" >&2
        exit 1
      fi
      "${binary}" measure \
        --clean "${clean}" \
        --noise z \
        --measurement "${measurement}" \
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
done
