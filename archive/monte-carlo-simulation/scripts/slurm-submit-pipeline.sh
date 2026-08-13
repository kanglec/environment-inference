#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

usage() {
    echo "usage: $0 RUN_SPEC.toml [--dry-run]" >&2
    exit 2
}

get_value() {
    local key=$1
    local line
    local value
    line=$(grep -E "^[[:space:]]*$key[[:space:]]*=" "$SPEC" | head -n 1 || true)
    if [[ -z "$line" ]]; then
        echo "missing key in run spec: $key" >&2
        exit 2
    fi
    value=$(cut -d= -f2- <<< "$line" | sed 's/[[:space:]]*#.*$//' | xargs)
    value=${value#\"}
    value=${value%\"}
    if [[ -z "$value" ]]; then
        echo "missing key in run spec: $key" >&2
        exit 2
    fi
    printf '%s\n' "$value"
}

p_tag() {
    local p=$1
    if [[ ! "$p" =~ ^(0([.][0-9]+)?|[.][0-9]+)$ ]]; then
        echo "invalid p value '$p': expected 0 <= p < 0.5" >&2
        return 2
    fi
    awk -v p="$p" '
        BEGIN {
            if (p < 0 || p >= 0.5) exit 2
            scaled = p * 100
            rounded = int(scaled + 0.5)
            delta = scaled - rounded
            if (delta < 0) delta = -delta
            if (delta > 1e-9) exit 3
            printf "p%03d", rounded
        }
    ' || {
        echo "invalid p value '$p': campaign paths support increments of 0.01 (for example 0.05 -> p005)" >&2
        return 2
    }
}

load_spec() {
    OUTPUT_ROOT=$(get_value output_root)
    BIN=$(get_value bin)
    LX=$(get_value lx)
    LT=$(get_value lt)
    SEED=$(get_value seed)
    SAMPLES=$(get_value samples)
    CHUNKS=$(get_value chunks)
    NOISES_RAW=$(get_value noises)
    P_VALUES_RAW=$(get_value p_values)
    UPDATE=$(get_value update)
    CLEAN_THERM=$(get_value clean_therm_sweeps)
    CLEAN_SKIP=$(get_value clean_skip_sweeps)
    DISORDER_THERM=$(get_value disorder_therm_sweeps)
    MEASUREMENTS=$(get_value measurements)
    SKIP_SWEEPS=$(get_value skip_sweeps)
    PARTITION=$(get_value partition)
    GENERATE_TIME=$(get_value generate_time)
    GENERATE_MEM=$(get_value generate_mem)
    GENERATE_CPUS=$(get_value generate_cpus)
    MEASURE_TIME=$(get_value measure_time)
    MEASURE_MEM=$(get_value measure_mem)
    MEASURE_CPUS=$(get_value measure_cpus)
    MERGE_TIME=$(get_value merge_time)
    MERGE_MEM=$(get_value merge_mem)
    MERGE_CPUS=$(get_value merge_cpus)

    local integer_name integer_value
    for integer_name in LX LT SEED SAMPLES CHUNKS CLEAN_THERM CLEAN_SKIP DISORDER_THERM MEASUREMENTS SKIP_SWEEPS GENERATE_CPUS MEASURE_CPUS MERGE_CPUS; do
        integer_value=${!integer_name}
        if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
            echo "invalid integer for $integer_name: $integer_value" >&2
            exit 2
        fi
    done

    read -r -a NOISES <<< "$NOISES_RAW"
    read -r -a P_VALUES <<< "$P_VALUES_RAW"

    if ((${#NOISES[@]} == 0 || ${#P_VALUES[@]} == 0)); then
        echo "noises and p_values must each contain at least one value" >&2
        exit 2
    fi

    local noise
    local seen_noises=" "
    for noise in "${NOISES[@]}"; do
        if [[ "$noise" != "z" && "$noise" != "zz" ]]; then
            echo "invalid noise '$noise': expected z or zz" >&2
            exit 2
        fi
        if [[ "$seen_noises" == *" $noise "* ]]; then
            echo "duplicate noise in run spec: $noise" >&2
            exit 2
        fi
        seen_noises+="$noise "
    done

    P_TAGS=()
    local p tag
    local seen_tags=" "
    for p in "${P_VALUES[@]}"; do
        tag=$(p_tag "$p")
        if [[ "$seen_tags" == *" $tag "* ]]; then
            echo "duplicate p path tag '$tag' in run spec" >&2
            exit 2
        fi
        seen_tags+="$tag "
        P_TAGS+=("$tag")
    done

    if [[ "$LX" == "$LT" ]]; then
        LATTICE_TAG="L$LX"
    else
        LATTICE_TAG="L${LX}x${LT}"
    fi
    LATTICE_DIR="$OUTPUT_ROOT/$LATTICE_TAG"
    CLEAN_STEM="clean_lt${LT}_iso_seed${SEED}_n${SAMPLES}_therm${CLEAN_THERM}_skip${CLEAN_SKIP}"
    CLEAN_PATH="$LATTICE_DIR/clean/${CLEAN_STEM}.bin"
    CLEAN_METADATA="$LATTICE_DIR/clean/${CLEAN_STEM}.metadata.txt"
    POINT_COUNT=$((${#NOISES[@]} * ${#P_VALUES[@]}))
    MEASURE_TASK_COUNT=$((POINT_COUNT * CHUNKS))
}

point_values() {
    local point_id=$1
    local noise_index=$((point_id / ${#P_VALUES[@]}))
    local p_index=$((point_id % ${#P_VALUES[@]}))
    POINT_NOISE=${NOISES[$noise_index]}
    POINT_P=${P_VALUES[$p_index]}
    POINT_TAG=${P_TAGS[$p_index]}
    POINT_DIR="$LATTICE_DIR/$POINT_NOISE/$POINT_TAG"
}

chunk_values() {
    local task_id=$1
    CHUNK_ID=$((task_id % CHUNKS))
    POINT_ID=$((task_id / CHUNKS))
    local base=$((SAMPLES / CHUNKS))
    local remainder=$((SAMPLES % CHUNKS))
    if ((CHUNK_ID < remainder)); then
        CHUNK_COUNT=$((base + 1))
        CHUNK_START=$((CHUNK_ID * (base + 1)))
    else
        CHUNK_COUNT=$base
        CHUNK_START=$((remainder * (base + 1) + (CHUNK_ID - remainder) * base))
    fi
    point_values "$POINT_ID"
}

clean_field() {
    local metadata=$1
    local key=$2
    awk -v key="$key:" '$1 == key {print $2}' <<< "$metadata"
}

validate_clean() {
    local path=$1
    local metadata
    metadata=$("$BIN" inspect-clean --clean "$path")
    [[ "$(clean_field "$metadata" lx)" == "$LX" ]]
    [[ "$(clean_field "$metadata" lt)" == "$LT" ]]
    [[ "$(clean_field "$metadata" delta_tau)" == "isotropic" ]]
    [[ "$(clean_field "$metadata" seed)" == "$SEED" ]]
    [[ "$(clean_field "$metadata" samples)" == "$SAMPLES" ]]
    [[ "$(clean_field "$metadata" clean_therm_sweeps)" == "$CLEAN_THERM" ]]
    [[ "$(clean_field "$metadata" clean_skip_sweeps)" == "$CLEAN_SKIP" ]]
    printf '%s\n' "$metadata"
}

run_clean_worker() {
    mkdir -p "$(dirname "$CLEAN_PATH")"
    if [[ -f "$CLEAN_PATH" ]]; then
        validate_clean "$CLEAN_PATH" > "$CLEAN_METADATA"
        echo "reused_clean=$CLEAN_PATH"
        return
    fi

    local temp_path="${CLEAN_PATH}.tmp.${SLURM_JOB_ID:-$$}"
    trap 'rm -f "$temp_path" "${temp_path}.metadata"' EXIT
    "$BIN" generate-clean \
        --lx "$LX" --lt "$LT" \
        --seed "$SEED" \
        --samples "$SAMPLES" \
        --clean-therm-sweeps "$CLEAN_THERM" \
        --clean-skip-sweeps "$CLEAN_SKIP" \
        --out "$temp_path"
    validate_clean "$temp_path" > "${temp_path}.metadata"

    if [[ -f "$CLEAN_PATH" ]]; then
        validate_clean "$CLEAN_PATH" > "$CLEAN_METADATA"
        rm -f "$temp_path" "${temp_path}.metadata"
        echo "reused_clean=$CLEAN_PATH"
    else
        mv "$temp_path" "$CLEAN_PATH"
        mv "${temp_path}.metadata" "$CLEAN_METADATA"
        echo "generated_clean=$CLEAN_PATH"
    fi
    trap - EXIT
}

run_measure_worker() {
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
    chunk_values "$SLURM_ARRAY_TASK_ID"
    mkdir -p "$POINT_DIR/chunks"
    "$BIN" measure \
        --clean "$CLEAN_PATH" \
        --noise "$POINT_NOISE" \
        --p "$POINT_P" \
        --sample-start "$CHUNK_START" \
        --samples "$CHUNK_COUNT" \
        --disorder-update "$UPDATE" \
        --disorder-therm-sweeps "$DISORDER_THERM" \
        --measurements "$MEASUREMENTS" \
        --skip-sweeps "$SKIP_SWEEPS" \
        --out "$POINT_DIR/chunks/chunk_${CHUNK_ID}.bin" \
        --threads "${SLURM_CPUS_PER_TASK:-1}"
}

run_merge_worker() {
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
    point_values "$SLURM_ARRAY_TASK_ID"
    local merged="$POINT_DIR/merged/aggregates.bin"
    mkdir -p "$POINT_DIR"/{merged,analysis,manifests}
    printf '%s\n' "$POINT_DIR"/chunks/*.bin > "$POINT_DIR/manifests/chunks.txt"
    "$BIN" merge-aggregates --out "$merged" "$POINT_DIR"/chunks/*.bin
    "$BIN" inspect-aggregate --path "$merged" > "$POINT_DIR/manifests/aggregate_metadata.txt"
    "$BIN" analyze \
        --path "$merged" \
        --csv "$POINT_DIR/analysis/observables.csv" \
        > "$POINT_DIR/analysis/summary.txt"
}

if [[ "${1:-}" == "--worker-clean" || "${1:-}" == "--worker-measure" || "${1:-}" == "--worker-merge" ]]; then
    MODE=$1
    [[ $# == 2 ]] || usage
    SPEC=$2
    load_spec
    case "$MODE" in
        --worker-clean) run_clean_worker ;;
        --worker-measure) run_measure_worker ;;
        --worker-merge) run_merge_worker ;;
    esac
    exit 0
fi

[[ $# -ge 1 && $# -le 2 ]] || usage
SPEC=$1
DRY_RUN=${2:-}
[[ -z "$DRY_RUN" || "$DRY_RUN" == "--dry-run" ]] || usage
SPEC=$(cd "$(dirname "$SPEC")" && pwd)/$(basename "$SPEC")
load_spec

"$BIN" run-spec-dry-run --path "$SPEC" > /dev/null

echo "lattice_dir=$LATTICE_DIR"
echo "clean=$CLEAN_PATH"
echo "points=$POINT_COUNT"
echo "measurement_tasks=$MEASURE_TASK_COUNT"
for ((point_id = 0; point_id < POINT_COUNT; point_id++)); do
    point_values "$point_id"
    echo "point=$point_id noise=$POINT_NOISE p=$POINT_P tag=$POINT_TAG run_dir=$POINT_DIR"
    for ((chunk_id = 0; chunk_id < CHUNKS; chunk_id++)); do
        task_id=$((point_id * CHUNKS + chunk_id))
        chunk_values "$task_id"
        echo "  array_task=$task_id chunk=$CHUNK_ID sample_start=$CHUNK_START samples=$CHUNK_COUNT"
    done
done

if [[ "$DRY_RUN" == "--dry-run" ]]; then
    exit 0
fi

mkdir -p "$LATTICE_DIR"/{config,clean,logs}
SPEC_ID=$(cksum "$SPEC" | awk '{print $1}')
CAMPAIGN_SPEC="$LATTICE_DIR/config/campaign_${SPEC_ID}.toml"
cp "$SPEC" "$CAMPAIGN_SPEC"

for ((point_id = 0; point_id < POINT_COUNT; point_id++)); do
    point_values "$point_id"
    if compgen -G "$POINT_DIR/chunks/*.bin" > /dev/null || [[ -e "$POINT_DIR/merged/aggregates.bin" ]]; then
        echo "refusing to overwrite existing point data: $POINT_DIR" >&2
        echo "move or remove the existing point directory before submitting a replacement" >&2
        exit 2
    fi
    mkdir -p "$POINT_DIR"/{config,chunks,merged,analysis,manifests}
    cp "$CAMPAIGN_SPEC" "$POINT_DIR/config/campaign.toml"
    {
        echo "noise=$POINT_NOISE"
        echo "p=$POINT_P"
        echo "tag=$POINT_TAG"
        echo "clean=$CLEAN_PATH"
    } > "$POINT_DIR/config/point.txt"
done

CLEAN_KEY=$(printf '%s' "$CLEAN_PATH" | cksum | awk '{print $1}')
GENERATE_JOB=$(sbatch --parsable \
    --dependency=singleton \
    --job-name="dcft-clean-$CLEAN_KEY" \
    --partition="$PARTITION" \
    --time="$GENERATE_TIME" \
    --mem="$GENERATE_MEM" \
    --cpus-per-task="$GENERATE_CPUS" \
    --output="$LATTICE_DIR/logs/%x-%j.out" \
    --error="$LATTICE_DIR/logs/%x-%j.err" \
    --wrap="'$SCRIPT_PATH' --worker-clean '$CAMPAIGN_SPEC'")

MEASURE_JOB=$(sbatch --parsable \
    --dependency="afterok:$GENERATE_JOB" \
    --array="0-$((MEASURE_TASK_COUNT - 1))" \
    --job-name="dcft-measure-$LATTICE_TAG" \
    --partition="$PARTITION" \
    --time="$MEASURE_TIME" \
    --mem="$MEASURE_MEM" \
    --cpus-per-task="$MEASURE_CPUS" \
    --output="$LATTICE_DIR/logs/%x-%A_%a.out" \
    --error="$LATTICE_DIR/logs/%x-%A_%a.err" \
    --wrap="'$SCRIPT_PATH' --worker-measure '$CAMPAIGN_SPEC'")

MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:$MEASURE_JOB" \
    --array="0-$((POINT_COUNT - 1))" \
    --job-name="dcft-merge-$LATTICE_TAG" \
    --partition="$PARTITION" \
    --time="$MERGE_TIME" \
    --mem="$MERGE_MEM" \
    --cpus-per-task="$MERGE_CPUS" \
    --output="$LATTICE_DIR/logs/%x-%A_%a.out" \
    --error="$LATTICE_DIR/logs/%x-%A_%a.err" \
    --wrap="'$SCRIPT_PATH' --worker-merge '$CAMPAIGN_SPEC'")

echo "generate_job=$GENERATE_JOB"
echo "measure_job=$MEASURE_JOB"
echo "merge_job=$MERGE_JOB"
