#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mc_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${mc_root}/.." && pwd)"
binary="${mc_root}/target/release/decohered-cft"
result_root="${repo_root}/results/trotter-check/monte-carlo/L8_dt010"

seed=2254
samples=250
mkdir -p "${result_root}"
clean="${result_root}/clean.bin"

if [[ ! -f "${clean}" ]]; then
  "${binary}" generate-clean \
    --lx 8 --lt 256 --seed "${seed}" --samples "${samples}" \
    --clean-therm-sweeps 10000 --clean-skip-sweeps 100 \
    --delta-tau 0.1 --out "${clean}"
fi

for specification in "0.1:p010" "0.3:p030" "0.49:p049"; do
  p="${specification%%:*}"
  tag="${specification##*:}"
  point_dir="${result_root}/${tag}"
  aggregate="${point_dir}/aggregate.bin"
  mkdir -p "${point_dir}"
  if [[ -e "${aggregate}" ]]; then
    echo "refusing to overwrite existing Trotter-check output: ${aggregate}" >&2
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
